"""Asynchronous job manager.

One job at a time (GPU is exclusive).  A background worker coroutine pops
queued jobs and runs:  prepare (loader + retarget + train-npz) → train
(subprocess) → export (checkpoint + onnx + mp4 + motion files).

Progress is broadcast via `asyncio.Queue`s per job id so the SSE endpoint can
stream updates to the browser.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import time
from collections.abc import AsyncIterator
from pathlib import Path

from sqlalchemy import select

from ..config import settings
from ..db import SessionLocal
from ..loaders import load as load_mocap
from ..models import Job, JobStatus, UploadedFile
from ..retarget import retarget_to_g1, write_training_npz
from .progress import parse_event_file
from .runner import run_rsl_rl_train

log = logging.getLogger(__name__)


class JobManager:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[int] = asyncio.Queue()
        self._listeners: dict[int, list[asyncio.Queue[dict]]] = {}
        self._worker_task: asyncio.Task | None = None
        self._running_proc: int | None = None

    # ----- lifecycle -----
    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker(), name="job-worker")

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    # ----- api -----
    async def enqueue(self, job_id: int) -> None:
        await self._queue.put(job_id)
        await self._emit(job_id, {"event": "queued"})

    async def cancel(self, job_id: int) -> bool:
        if self._running_proc == job_id and self._running_pid:
            try:
                import os
                os.kill(self._running_pid, signal.SIGTERM)
                return True
            except OSError:
                return False
        return False

    async def listen(self, job_id: int) -> AsyncIterator[dict]:
        q: asyncio.Queue[dict] = asyncio.Queue()
        self._listeners.setdefault(job_id, []).append(q)
        try:
            while True:
                msg = await q.get()
                yield msg
                if msg.get("event") in {"completed", "failed", "cancelled"}:
                    break
        finally:
            self._listeners.get(job_id, []).remove(q)

    # ----- internals -----
    _running_pid: int | None = None

    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self._run_job(job_id)
            except Exception as exc:  # noqa: BLE001
                log.exception("Job %s crashed", job_id)
                await self._mark_failed(job_id, repr(exc))

    async def _run_job(self, job_id: int) -> None:
        async with SessionLocal() as sess:
            job = await sess.get(Job, job_id)
            if job is None:
                return
            upload = await sess.get(UploadedFile, job.upload_id)
            if upload is None:
                raise RuntimeError("upload missing")
            workspace = settings.jobs_dir / f"job_{job_id:06d}"
            workspace.mkdir(parents=True, exist_ok=True)
            job.workspace_dir = str(workspace)
            job.status = JobStatus.preparing
            await sess.commit()
        await self._emit(job_id, {"event": "preparing"})

        # 1. Load mocap → canonical
        motion = load_mocap(Path(upload.stored_path))
        log.info("Loaded mocap: %s frames, %s joints, fps=%s",
                 motion.num_frames, motion.num_joints, motion.fps)
        await self._emit(job_id, {"event": "loaded", "frames": motion.num_frames,
                                  "fps": motion.fps, "joints": motion.num_joints})

        # 2. GMR retarget → G1 qpos
        gmr_out = workspace / "g1_motion.npz"
        retarget_to_g1(motion, gmr_out)
        await self._emit(job_id, {"event": "retargeted"})

        # 3. Training npz
        train_npz = workspace / "train_motion.npz"
        write_training_npz(gmr_out, train_npz)
        await self._emit(job_id, {"event": "train_npz_ready", "path": str(train_npz)})

        # 4. Training subprocess
        async with SessionLocal() as sess:
            job = await sess.get(Job, job_id)
            job.status = JobStatus.training
            job.progress_total = job.max_iterations
            from datetime import datetime
            job.started_at = datetime.utcnow()
            await sess.commit()

        await self._emit(job_id, {"event": "training_started"})
        self._running_proc = job_id
        run_dir, pid = run_rsl_rl_train(
            task=job.task, num_envs=job.num_envs,
            max_iterations=job.max_iterations,
            motion_npz=train_npz,
            workspace=workspace,
        )
        self._running_pid = pid

        # Poll training progress from TensorBoard events
        while True:
            await asyncio.sleep(10)
            iter_now = parse_event_file(run_dir)
            async with SessionLocal() as sess:
                job = await sess.get(Job, job_id)
                if iter_now is not None and iter_now != job.progress_iter:
                    job.progress_iter = iter_now
                    await sess.commit()
                    await self._emit(job_id, {"event": "progress", "iter": iter_now})
            # Detect subprocess exit
            try:
                import os
                os.kill(pid, 0)  # probe
            except OSError:
                break

        self._running_proc = None
        self._running_pid = None

        # 5. Export
        async with SessionLocal() as sess:
            job = await sess.get(Job, job_id)
            job.status = JobStatus.exporting
            await sess.commit()
        await self._emit(job_id, {"event": "exporting"})

        from ..exporter import run_exports
        artifacts = await run_exports(job_id, run_dir, workspace)

        async with SessionLocal() as sess:
            job = await sess.get(Job, job_id)
            job.status = JobStatus.completed
            from datetime import datetime
            job.finished_at = datetime.utcnow()
            for k, v in artifacts.items():
                setattr(job, k, str(v) if v else None)
            await sess.commit()
        await self._emit(job_id, {"event": "completed"})

    async def _mark_failed(self, job_id: int, err: str) -> None:
        async with SessionLocal() as sess:
            job = await sess.get(Job, job_id)
            if job is None:
                return
            job.status = JobStatus.failed
            job.error_message = err[:4000]
            await sess.commit()
        await self._emit(job_id, {"event": "failed", "error": err})

    async def _emit(self, job_id: int, msg: dict) -> None:
        msg.setdefault("ts", time.time())
        msg.setdefault("job_id", job_id)
        for q in list(self._listeners.get(job_id, [])):
            await q.put(msg)


manager = JobManager()
