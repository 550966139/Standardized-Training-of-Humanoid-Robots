"""Start rsl_rl training on the remote host, detached."""
from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath

from ..config import settings
from ..remote import Host


@dataclass
class TrainHandle:
    remote_run_dir: str
    remote_log: str
    pid: int


async def run_rsl_rl_train(
    host: Host,
    *,
    task: str,
    num_envs: int,
    max_iterations: int,
    remote_motion_npz: str,
    remote_workspace: str,
) -> TrainHandle:
    remote_run_dir = str(PurePosixPath(remote_workspace) / "rsl_rl_run")
    remote_log = str(PurePosixPath(remote_workspace) / "train.log")

    conda = f"source {settings.conda_root}/etc/profile.d/conda.sh && conda activate {settings.conda_env_isaaclab}"
    # Emulate the pattern used by existing train_mimic_dance.sh — populate LD_LIBRARY_PATH
    # from every /bin/ under the isaacsim install.
    isaac_lib_setup = (
        f"ISAAC_BASE={shlex.quote(f'{settings.conda_root}/envs/{settings.conda_env_isaaclab}/lib/python3.10/site-packages/isaacsim')}; "
        r"""EXTRA_LIBS=$(find "$ISAAC_BASE" -name bin -type d 2>/dev/null | tr '\n' ':'); """
        r"""export LD_LIBRARY_PATH="${EXTRA_LIBS}${LD_LIBRARY_PATH:-}"; """
    )

    env_vars = {
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "PRIVACY_CONSENT": "Y",
        "HRTRAIN_MOTION_NPZ": remote_motion_npz,
        "HRTRAIN_RUN_DIR": remote_run_dir,
    }
    env_setup = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_vars.items())

    train_script = f"{settings.unitree_rl_lab_root}/scripts/rsl_rl/train.py"
    cmd = (
        f"{conda} && {isaac_lib_setup} "
        f"mkdir -p {shlex.quote(remote_run_dir)} && "
        f"cd {shlex.quote(settings.isaaclab_root)} && "
        f"{env_setup} python {shlex.quote(train_script)} "
        f"--task {shlex.quote(task)} "
        f"--num_envs {num_envs} --headless --logger tensorboard "
        f"--max_iterations {max_iterations}"
    )

    pid = await host.exec_detached(cmd, log_remote=remote_log)
    return TrainHandle(remote_run_dir=remote_run_dir, remote_log=remote_log, pid=pid)
