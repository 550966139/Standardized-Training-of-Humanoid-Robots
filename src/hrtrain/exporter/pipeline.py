"""Post-training export pipeline.

Collects the latest checkpoint, optionally exports ONNX, renders a rollout
video, and regenerates motion files (BVH/FBX/CSV) from the learned policy.

Each sub-step is a best-effort: failure logs the error but does not abort
the overall export, so users still get partial artifacts.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from ..config import settings

log = logging.getLogger(__name__)


async def run_exports(job_id: int, run_dir: Path, workspace: Path) -> dict[str, Path | None]:
    out: dict[str, Path | None] = {
        "checkpoint_path": None,
        "onnx_path": None,
        "video_path": None,
        "motion_bvh_path": None,
        "motion_fbx_path": None,
        "motion_csv_path": None,
    }
    latest_pt = _find_latest_checkpoint(run_dir)
    if latest_pt is not None:
        out["checkpoint_path"] = latest_pt
        onnx = workspace / "policy.onnx"
        if _export_onnx(latest_pt, onnx):
            out["onnx_path"] = onnx

    video = _render_rollout(run_dir, latest_pt, workspace)
    if video is not None:
        out["video_path"] = video
        bvh, fbx, csv = _extract_motion(video, workspace)
        out["motion_bvh_path"] = bvh
        out["motion_fbx_path"] = fbx
        out["motion_csv_path"] = csv

    return out


def _find_latest_checkpoint(run_dir: Path) -> Path | None:
    cands = sorted(run_dir.rglob("model_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else -1)
    return cands[-1] if cands else None


def _export_onnx(pt_path: Path, onnx_path: Path) -> bool:
    """Invoke rsl_rl's export helper (if available).

    Upstream `rsl_rl` has a `play.py --export_policy_onnx` path.  We shell out
    because the infrastructure is already set up for subprocess training.
    """
    export_script = settings.unitree_rl_lab_root / "scripts" / "rsl_rl" / "export_onnx.py"
    if not export_script.exists():
        log.warning("No export_onnx.py script; skipping ONNX export")
        return False
    conda_sh = settings.conda_root / "etc" / "profile.d" / "conda.sh"
    cmd = (
        f"source {conda_sh} && conda activate {settings.conda_env_isaaclab} && "
        f"python {export_script} --checkpoint {pt_path} --output {onnx_path}"
    )
    r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        log.warning("ONNX export failed: %s", r.stderr[-500:])
        return False
    return onnx_path.exists()


def _render_rollout(run_dir: Path, pt_path: Path | None, workspace: Path) -> Path | None:
    if pt_path is None:
        return None
    play_script = settings.unitree_rl_lab_root / "scripts" / "rsl_rl" / "play.py"
    if not play_script.exists():
        log.warning("No play.py; skipping rollout video")
        return None
    out_mp4 = workspace / "rollout.mp4"
    conda_sh = settings.conda_root / "etc" / "profile.d" / "conda.sh"
    cmd = (
        f"source {conda_sh} && conda activate {settings.conda_env_isaaclab} && "
        f"cd {settings.isaaclab_root} && "
        f"python {play_script} --num_envs 1 --headless --video --video_length 1500 "
        f"--checkpoint {pt_path}"
    )
    r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        log.warning("Rollout video render failed: %s", r.stderr[-500:])
        return None
    vids = sorted(run_dir.rglob("videos/*.mp4"), key=lambda p: p.stat().st_mtime)
    if not vids:
        return None
    vids[-1].replace(out_mp4)
    return out_mp4


def _extract_motion(video_path: Path, workspace: Path) -> tuple[Path | None, Path | None, Path | None]:
    """Convert the learned policy's rollout back to motion files.

    The rollout produces a qpos trajectory (saved alongside the MP4 by play.py
    via a small patch).  We emit CSV directly, and use Blender to export BVH
    and FBX from a reconstructed G1 skeleton.  These last two steps are
    optional — users get the CSV regardless.
    """
    qpos_npz = video_path.with_suffix(".qpos.npz")
    csv_path: Path | None = None
    if qpos_npz.exists():
        csv_path = workspace / "motion.csv"
        try:
            import numpy as np
            data = np.load(qpos_npz)["qpos"]
            header = ",".join(
                ["base_x", "base_y", "base_z", "qw", "qx", "qy", "qz"]
                + [f"j{i:02d}" for i in range(data.shape[1] - 7)]
            )
            np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
        except Exception:
            log.exception("CSV export failed")
            csv_path = None

    # BVH/FBX export requires a separate Blender script (to be added alongside
    # the G1 rig skeleton file).  Stubbed here.
    return (None, None, csv_path)
