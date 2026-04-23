"""Convert a G1 qpos npz on the remote host into the training-ready schema.

The mimic task expects keys:
    fps, joint_pos(T,29), joint_vel(T,29),
    body_pos_w(T,30,3), body_quat_w(T,30,4),
    body_lin_vel_w(T,30,3), body_ang_vel_w(T,30,3)

Strategy: piggyback on the existing `unitree_rl_lab/scripts/mimic/csv_to_npz.py`
which already does interpolation + SO3 derivatives, by:
  1. Convert GMR output → CSV via `/root/autodl-tmp/npz2csv.py`
  2. Run csv_to_npz.py → training npz
Both run remotely in `conda_env_isaaclab`.
"""
from __future__ import annotations

import shlex
from pathlib import PurePosixPath

from ..config import settings
from ..remote import Host


async def write_training_npz(
    host: Host,
    remote_gmr_output: str,
    remote_job_dir: str,
    fps_out: int = 50,
) -> str:
    remote_csv = str(PurePosixPath(remote_job_dir) / "g1_motion.csv")
    remote_train = str(PurePosixPath(remote_job_dir) / "train_motion.npz")

    npz2csv = "/root/autodl-tmp/npz2csv.py"
    csv_to_npz = f"{settings.unitree_rl_lab_root}/scripts/mimic/csv_to_npz.py"

    for p in (npz2csv, csv_to_npz):
        r = await host.exec(f"test -f {shlex.quote(p)} && echo yes || echo no", timeout=15)
        if r.stdout.strip() != "yes":
            raise FileNotFoundError(f"Required remote script missing: {p}")

    conda = f"source {settings.conda_root}/etc/profile.d/conda.sh && conda activate {settings.conda_env_isaaclab}"

    step1 = f"{conda} && python {shlex.quote(npz2csv)} --input {shlex.quote(remote_gmr_output)} --output {shlex.quote(remote_csv)}"
    r = await host.exec(step1, timeout=180)
    if not r.ok:
        raise RuntimeError(f"npz2csv failed:\n{r.stderr[-1500:]}")

    step2 = (
        f"{conda} && python {shlex.quote(csv_to_npz)} "
        f"--input_file {shlex.quote(remote_csv)} "
        f"--output_file {shlex.quote(remote_train)} "
        f"--output_fps {fps_out}"
    )
    r = await host.exec(step2, timeout=600)
    if not r.ok:
        raise RuntimeError(f"csv_to_npz failed:\n{r.stderr[-1500:]}")

    return remote_train
