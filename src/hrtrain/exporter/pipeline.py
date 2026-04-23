"""Post-training artifact export.

Runs on the remote host:
  - Pick the latest `model_*.pt` from the rsl_rl run dir
  - Invoke ONNX export helper (if present)
  - Render a rollout video via `play.py`
  - Dump rollout qpos → CSV (via a tiny remote one-liner)

Then downloads each produced file to the local Job workspace so the Web UI
can serve it directly.
"""
from __future__ import annotations

import logging
import re
import shlex
from pathlib import Path, PurePosixPath

from ..config import settings
from ..remote import Host

log = logging.getLogger(__name__)


async def run_exports(
    host: Host,
    remote_job_dir: str,
    remote_run_dir: str,
    local_job_dir: Path,
) -> dict[str, Path | None]:
    out: dict[str, Path | None] = {
        "checkpoint_path": None,
        "onnx_path": None,
        "video_path": None,
        "motion_bvh_path": None,
        "motion_fbx_path": None,
        "motion_csv_path": None,
    }

    latest_remote_pt = await _find_latest_checkpoint(host, remote_run_dir)
    if latest_remote_pt:
        local_pt = local_job_dir / Path(latest_remote_pt).name
        try:
            await host.download(latest_remote_pt, local_pt)
            out["checkpoint_path"] = local_pt
        except Exception:
            log.exception("failed to download checkpoint")

        remote_onnx = str(PurePosixPath(remote_job_dir) / "policy.onnx")
        if await _try_export_onnx(host, latest_remote_pt, remote_onnx):
            local_onnx = local_job_dir / "policy.onnx"
            try:
                await host.download(remote_onnx, local_onnx)
                out["onnx_path"] = local_onnx
            except Exception:
                log.exception("failed to download onnx")

    remote_video = await _try_render_rollout(host, remote_run_dir, latest_remote_pt)
    if remote_video:
        local_video = local_job_dir / "rollout.mp4"
        try:
            await host.download(remote_video, local_video)
            out["video_path"] = local_video
        except Exception:
            log.exception("failed to download video")

    remote_csv = await _try_qpos_csv(host, remote_run_dir)
    if remote_csv:
        local_csv = local_job_dir / "motion.csv"
        try:
            await host.download(remote_csv, local_csv)
            out["motion_csv_path"] = local_csv
        except Exception:
            log.exception("failed to download motion csv")

    # BVH/FBX export from rig qpos: placeholder — requires a G1 rig-aware
    # Blender script; scheduled for a follow-up commit.

    return out


async def _find_latest_checkpoint(host: Host, remote_run_dir: str) -> str | None:
    r = await host.exec(
        f"shopt -s globstar nullglob; ls {shlex.quote(remote_run_dir)}/**/model_*.pt 2>/dev/null",
        timeout=15,
    )
    if not r.ok or not r.stdout.strip():
        return None
    candidates = [line.strip() for line in r.stdout.splitlines() if line.strip()]

    def iter_num(p: str) -> int:
        m = re.search(r"model_(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1

    candidates.sort(key=iter_num)
    return candidates[-1] if candidates else None


async def _try_export_onnx(host: Host, pt_path: str, onnx_path: str) -> bool:
    driver = f"{settings.unitree_rl_lab_root}/scripts/rsl_rl/export_onnx.py"
    r = await host.exec(f"test -f {shlex.quote(driver)} && echo yes || echo no", timeout=15)
    if r.stdout.strip() != "yes":
        log.info("no export_onnx.py on remote; skipping")
        return False
    conda = f"source {settings.conda_root}/etc/profile.d/conda.sh && conda activate {settings.conda_env_isaaclab}"
    cmd = (
        f"{conda} && python {shlex.quote(driver)} "
        f"--checkpoint {shlex.quote(pt_path)} --output {shlex.quote(onnx_path)}"
    )
    r = await host.exec(cmd, timeout=300)
    if not r.ok:
        log.warning("onnx export failed: %s", r.stderr[-500:])
        return False
    return True


async def _try_render_rollout(host: Host, remote_run_dir: str, pt_path: str | None) -> str | None:
    if pt_path is None:
        return None
    play = f"{settings.unitree_rl_lab_root}/scripts/rsl_rl/play.py"
    r = await host.exec(f"test -f {shlex.quote(play)} && echo yes || echo no", timeout=15)
    if r.stdout.strip() != "yes":
        return None
    conda = f"source {settings.conda_root}/etc/profile.d/conda.sh && conda activate {settings.conda_env_isaaclab}"
    # Reuse the same task embedded in pt metadata — caller already validated it exists.
    cmd = (
        f"{conda} && cd {shlex.quote(settings.isaaclab_root)} && "
        f"python {shlex.quote(play)} --num_envs 1 --headless --video --video_length 1500 "
        f"--checkpoint {shlex.quote(pt_path)}"
    )
    r = await host.exec(cmd, timeout=900)
    if not r.ok:
        log.warning("rollout video render failed: %s", r.stderr[-500:])
        return None
    # Find the produced mp4 — play.py writes it under the run dir's videos/ subfolder.
    r = await host.exec(
        f"shopt -s globstar nullglob; ls -t {shlex.quote(remote_run_dir)}/**/*.mp4 2>/dev/null | head -1",
        timeout=15,
    )
    path = r.stdout.strip()
    return path or None


async def _try_qpos_csv(host: Host, remote_run_dir: str) -> str | None:
    # play.py may save `qpos.npz` alongside the mp4 if patched; convert to CSV.
    r = await host.exec(
        f"shopt -s globstar nullglob; ls {shlex.quote(remote_run_dir)}/**/*.qpos.npz 2>/dev/null | head -1",
        timeout=15,
    )
    qpos_npz = r.stdout.strip()
    if not qpos_npz:
        return None
    conda = f"source {settings.conda_root}/etc/profile.d/conda.sh && conda activate {settings.conda_env_isaaclab}"
    out_csv = str(PurePosixPath(qpos_npz).with_suffix("").with_suffix(".csv"))
    snippet = (
        "import numpy as np, sys; "
        "p_in=sys.argv[1]; p_out=sys.argv[2]; "
        "q=np.load(p_in)['qpos']; "
        "header=','.join(['bx','by','bz','qw','qx','qy','qz']+[f'j{i:02d}' for i in range(q.shape[1]-7)]); "
        "np.savetxt(p_out, q, delimiter=',', header=header, comments='')"
    )
    cmd = f"{conda} && python -c {shlex.quote(snippet)} {shlex.quote(qpos_npz)} {shlex.quote(out_csv)}"
    r = await host.exec(cmd, timeout=120)
    if not r.ok:
        log.warning("qpos csv export failed: %s", r.stderr[-500:])
        return None
    return out_csv
