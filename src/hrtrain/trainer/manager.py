"""Orchestrate jobs: local Web → remote training → local artifact downloads."""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path, PurePosixPath

from ..config import settings
from ..db import SessionLocal
from ..loaders import load as load_mocap
from ..models import Job, JobStatus, UploadedFile
from ..remote import Host, get_host
from ..retarget import retarget_to_g1, write_training_npz
from .progress import poll_iter
from .runner import TrainHandle, run_rsl_rl_train

log = logging.getLogger(__name__)


class JobManager:
    """Single-queue, single-worker orchestrator.

    One job at a time because the remote GPU is exclusive.  All remote state
    lives under `{remote_workdir}/jobs/{job_id}/`.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[int] = asyncio.Queue()
        self._listeners: dict[int, list[asyncio.Queue[dict]]] = {}
        self._worker_task: asyncio.Task | None = None
        self._active_handle: TrainHandle | None = None
        self._active_job_id: int | None = None

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
        if self._active_job_id != job_id or self._active_handle is None:
            return False
        host = get_host()
        try:
            await host.exec(f"kill -TERM {self._active_handle.pid} 2>/dev/null", timeout=15)
        except Exception:
            return False
        async with SessionLocal() as sess:
            job = await sess.get(Job, job_id)
            if job:
                job.status = JobStatus.cancelled
                await sess.commit()
        await self._emit(job_id, {"event": "cancelled"})
        return True

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
    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            self._active_job_id = job_id
            try:
                await self._run_job(job_id)
            except Exception as exc:  # noqa: BLE001
                log.exception("Job %s crashed", job_id)
                await self._mark_failed(job_id, repr(exc))
            finally:
                self._active_job_id = None
                self._active_handle = None

    async def _run_job(self, job_id: int) -> None:
        host = get_host()

        async with SessionLocal() as sess:
            job = await sess.get(Job, job_id)
            if job is None:
                return
            upload = await sess.get(UploadedFile, job.upload_id)
            if upload is None:
                raise RuntimeError("upload missing")
            job.status = JobStatus.preparing
            await sess.commit()
        await self._emit(job_id, {"event": "preparing"})

        # Remote job workspace
        remote_job_dir = str(PurePosixPath(settings.remote_workdir) / "jobs" / f"job_{job_id:06d}")
        await host.mkdir(remote_job_dir)

        # Local job workspace (for downloaded artifacts)
        local_job_dir = settings.outputs_dir / f"job_{job_id:06d}"
        local_job_dir.mkdir(parents=True, exist_ok=True)

        # Stamp paths
        async with SessionLocal() as sess:
            job = await sess.get(Job, job_id)
            job.workspace_dir = str(local_job_dir)
            await sess.commit()

        # 1) Load locally (loaders are pure python, no GPU needed)
        motion = load_mocap(Path(upload.stored_path))
        await self._emit(job_id, {
            "event": "loaded",
            "frames": motion.num_frames,
            "joints": motion.num_joints,
            "fps": motion.fps,
        })

        # 2) Retarget on remote
        remote_g1_npz = await retarget_to_g1(host, motion, remote_job_dir)
        await self._emit(job_id, {"event": "retargeted", "path": remote_g1_npz})

        # 3) Build training npz on remote
        remote_train_npz = await write_training_npz(host, remote_g1_npz, remote_job_dir)
        await self._emit(job_id, {"event": "train_npz_ready", "path": remote_train_npz})

        # 4) Start training (detached remote subprocess)
        async with SessionLocal() as sess:
            job = await sess.get(Job, job_id)
            job.status = JobStatus.training
            job.progress_total = job.max_iterations
            job.started_at = datetime.utcnow()
            await sess.commit()
        await self._emit(job_id, {"event": "training_started"})

        handle = await run_rsl_rl_train(
            host,
            task=job.task,
            num_envs=job.num_envs,
            max_iterations=job.max_iterations,
            remote_motion_npz=remote_train_npz,
            remote_workspace=remote_job_dir,
        )
        self._active_handle = handle
        async with SessionLocal() as sess:
            job = await sess.get(Job, job_id)
            job.pid = handle.pid
            await sess.commit()

        # 5) Poll progress
        while True:
            await asyncio.sleep(15)
            alive = await host.is_pid_alive(handle.pid)
            step = await poll_iter(host, handle.remote_run_dir)
            async with SessionLocal() as sess:
                job = await sess.get(Job, job_id)
                if step is not None and step != job.progress_iter:
                    job.progress_iter = step
                    await sess.commit()
                    await self._emit(job_id, {"event": "progress", "iter": step})
            if not alive:
                break

        # 6) Export artifacts on remote, download what we can to local
        async with SessionLocal() as sess:
            job = await sess.get(Job, job_id)
            job.status = JobStatus.exporting
            await sess.commit()
        await self._emit(job_id, {"event": "exporting"})

        from ..exporter import run_exports
        artifacts = await run_exports(host, remote_job_dir, handle.remote_run_dir, local_job_dir)

        async with SessionLocal() as sess:
            job = await sess.get(Job, job_id)
            job.status = JobStatus.completed
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
