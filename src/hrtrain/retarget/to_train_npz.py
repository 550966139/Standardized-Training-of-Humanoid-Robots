"""Convert a G1MotionSequence (qpos-based) into the training-ready npz.

The mimic task in `unitree_rl_lab/tasks/mimic/mdp/commands.py` expects keys:
    fps, joint_pos(T,29), joint_vel(T,29),
    body_pos_w(T,30,3), body_quat_w(T,30,4),
    body_lin_vel_w(T,30,3), body_ang_vel_w(T,30,3)

The upstream `unitree_rl_lab/scripts/mimic/csv_to_npz.py` already does this
(insertion via CSV → interpolation → FK → derivatives), so we reuse it by
first emitting a CSV through the existing `npz2csv.py`.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..config import settings


def _find_csv_to_npz() -> Path:
    p = settings.unitree_rl_lab_root / "scripts" / "mimic" / "csv_to_npz.py"
    if not p.exists():
        raise FileNotFoundError(
            f"Expected unitree_rl_lab helper at {p}. "
            "Install unitree_rl_lab at the configured path."
        )
    return p


def _find_npz2csv() -> Path:
    p = Path("/root/autodl-tmp/npz2csv.py")
    if not p.exists():
        raise FileNotFoundError(
            f"Expected bridge script at {p}. Write it first or adjust the path."
        )
    return p


def write_training_npz(
    gmr_output: Path,
    out_path: Path,
    fps_out: int = 50,
) -> Path:
    """Chain: GMR .npz (qpos wxyz) → CSV → training .npz.

    The first step writes a CSV with xyzw quaternions and joint angles.
    The second uses unitree_rl_lab's `csv_to_npz.py` which resamples to
    `fps_out`, computes velocities, and emits the mimic-compatible npz.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_path.with_suffix(".csv")

    npz2csv = _find_npz2csv()
    csv_to_npz = _find_csv_to_npz()

    conda_sh = settings.conda_root / "etc" / "profile.d" / "conda.sh"

    step1 = (
        f"source {conda_sh} && conda activate {settings.conda_env_isaaclab} && "
        f"python {npz2csv} --input {gmr_output} --output {csv_path}"
    )
    r1 = subprocess.run(["bash", "-lc", step1], capture_output=True, text=True, timeout=120)
    if r1.returncode != 0:
        raise RuntimeError(f"npz→csv step failed:\n{r1.stderr[-1000:]}")

    step2 = (
        f"source {conda_sh} && conda activate {settings.conda_env_isaaclab} && "
        f"python {csv_to_npz} --input_file {csv_path} --output_file {out_path} "
        f"--output_fps {fps_out}"
    )
    r2 = subprocess.run(["bash", "-lc", step2], capture_output=True, text=True, timeout=600)
    if r2.returncode != 0:
        raise RuntimeError(f"csv→training-npz step failed:\n{r2.stderr[-1000:]}")
    return out_path
