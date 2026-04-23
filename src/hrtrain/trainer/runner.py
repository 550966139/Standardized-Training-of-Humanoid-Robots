"""Launch `rsl_rl` training from unitree_rl_lab in the `hc-isaac` conda env.

Returns the run directory (so we can tail TensorBoard events) and the PID of
the spawned Python process.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path

from ..config import settings


def run_rsl_rl_train(
    task: str,
    num_envs: int,
    max_iterations: int,
    motion_npz: Path,
    workspace: Path,
) -> tuple[Path, int]:
    """Spawn training as a detached subprocess. Returns (run_dir, pid)."""
    isaac_base = settings.conda_root / "envs" / settings.conda_env_isaaclab / \
        "lib" / "python3.10" / "site-packages" / "isaacsim"
    # Populate LD_LIBRARY_PATH from every `bin/` under isaacsim (same pattern
    # used by the existing train_mimic_dance.sh script).
    extra_libs = ":".join(str(p) for p in isaac_base.glob("**/bin") if p.is_dir())

    conda_sh = settings.conda_root / "etc" / "profile.d" / "conda.sh"
    train_script = settings.unitree_rl_lab_root / "scripts" / "rsl_rl" / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Missing {train_script}")

    run_dir = workspace / "rsl_rl_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = workspace / "train.log"

    # Patch the mimic task to consume our custom motion_npz by writing an env var
    # that the task config can read.  Upstream tasks/mimic/robots/g1_29dof/dance_*/
    # should support `HRTRAIN_MOTION_NPZ` — to be added in that repo.
    env_pairs = {
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "PRIVACY_CONSENT": "Y",
        "HRTRAIN_MOTION_NPZ": str(motion_npz),
        "HRTRAIN_RUN_DIR": str(run_dir),
        "LD_LIBRARY_PATH": f"{extra_libs}:${{LD_LIBRARY_PATH:-}}",
    }
    env_setup = " && ".join(f"export {k}={shlex.quote(v)}" for k, v in env_pairs.items())

    cmd = (
        f"source {conda_sh} && conda activate {settings.conda_env_isaaclab} && "
        f"{env_setup} && cd {settings.isaaclab_root} && "
        f"python {train_script} "
        f"--task {task} --num_envs {num_envs} --headless --logger tensorboard "
        f"--max_iterations {max_iterations} "
        f">> {shlex.quote(str(log_path))} 2>&1"
    )

    proc = subprocess.Popen(
        ["bash", "-lc", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Leave some time for python to boot and materialise the run dir.
    time.sleep(2)
    return run_dir, proc.pid


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
