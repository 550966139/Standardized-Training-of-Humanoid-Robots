"""Call GMR retargeter on the remote host.

Flow:
  1. Upload canonical `.npz` from our local tempdir to `<remote_workdir>/jobs/<id>/`
  2. Invoke humanoid-choreo's `scripts/run_gmr_retarget.py` in `conda_env_gmr`
  3. Return the remote path of the resulting G1 qpos `.npz`

If the upstream driver script is missing we raise a clear error — the caller
should surface that into the Job's status so the user sees it in the UI.
"""
from __future__ import annotations

import shlex
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np

from ..config import settings
from ..loaders.base import MotionData
from ..remote import Host


def _dump_canonical(motion: MotionData, path: Path) -> None:
    np.savez_compressed(
        path,
        positions=motion.positions,
        rotations=motion.rotations,
        fps=np.array([motion.fps], dtype=np.float32),
        joint_names=np.array(motion.skeleton.joint_names, dtype=object),
        parents=motion.skeleton.parents,
        offsets=motion.skeleton.offsets,
    )


async def retarget_to_g1(
    host: Host,
    motion: MotionData,
    remote_job_dir: str,
    *,
    actual_human_height: float = 1.7,
    src_human_profile: str = "bvh_lafan1",
) -> str:
    """Upload canonical motion, run GMR on remote, return remote output path."""
    remote_canonical = str(PurePosixPath(remote_job_dir) / "canonical.npz")
    remote_output = str(PurePosixPath(remote_job_dir) / "g1_motion.npz")

    await host.mkdir(remote_job_dir)

    with tempfile.TemporaryDirectory() as td:
        local_canonical = Path(td) / "canonical.npz"
        _dump_canonical(motion, local_canonical)
        await host.upload(local_canonical, remote_canonical)

    driver = f"{settings.hc_root}/scripts/run_gmr_retarget.py"
    check = await host.exec(f"test -f {shlex.quote(driver)} && echo yes || echo no", timeout=15)
    if check.stdout.strip() != "yes":
        raise FileNotFoundError(
            f"Missing upstream GMR driver on remote: {driver}. "
            "Add scripts/run_gmr_retarget.py to humanoid-choreo that accepts "
            "`--input <canonical.npz> --output <g1.npz> --src <profile> --height <float>`."
        )

    cmd = (
        f"source {settings.conda_root}/etc/profile.d/conda.sh && "
        f"conda activate {settings.conda_env_gmr} && "
        f"python {shlex.quote(driver)} "
        f"--input {shlex.quote(remote_canonical)} "
        f"--output {shlex.quote(remote_output)} "
        f"--src {shlex.quote(src_human_profile)} "
        f"--height {actual_human_height:.3f}"
    )
    r = await host.exec(cmd, timeout=600)
    if not r.ok:
        raise RuntimeError(f"GMR retarget failed:\n{r.stderr[-1500:]}")
    return remote_output
