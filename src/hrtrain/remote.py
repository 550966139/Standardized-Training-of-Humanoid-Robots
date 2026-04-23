"""Host abstraction: run commands locally or against a remote machine over SSH.

The Web UI runs locally on a laptop/PC, but all heavy work (GMR retargeting,
rsl_rl training, ONNX export, video render) needs GPU + CUDA + Isaac Sim that
live on the AutoDL server.  This module wraps `ssh`/`scp` (OpenSSH on PATH)
so every hot code path can say `await host.exec(...)` / `await host.upload(...)`
without caring whether it lands local or remote.
"""
from __future__ import annotations

import asyncio
import logging
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from .config import settings

log = logging.getLogger(__name__)


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Host(Protocol):
    """Protocol every backend honours."""

    kind: str

    async def exec(self, cmd: str, *, timeout: float | None = None) -> ExecResult: ...

    async def exec_detached(self, cmd: str, *, log_remote: str) -> int:
        """Start `cmd` detached; return its process id.  Stdout/stderr appended to log_remote."""

    async def is_pid_alive(self, pid: int) -> bool: ...

    async def read_text(self, remote_path: str, *, tail_lines: int | None = None) -> str: ...

    async def upload(self, local: Path, remote: str) -> None: ...

    async def download(self, remote: str, local: Path) -> None: ...

    async def glob(self, remote_pattern: str) -> list[str]: ...

    async def mkdir(self, remote_path: str) -> None: ...


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _run_blocking(argv: list[str], timeout: float | None, input_bytes: bytes | None) -> ExecResult:
    # Windows + asyncio subprocess + ssh combines badly (stdin/stdout pipes deadlock).
    # Plain blocking subprocess.run inside a thread is both simpler and faster.
    completed = subprocess.run(
        argv,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL if input_bytes is None else None,
        timeout=timeout,
    )
    return ExecResult(
        returncode=completed.returncode,
        stdout=completed.stdout.decode("utf-8", errors="replace"),
        stderr=completed.stderr.decode("utf-8", errors="replace"),
    )


async def _run(argv: list[str], timeout: float | None = None, input_bytes: bytes | None = None) -> ExecResult:
    return await asyncio.to_thread(_run_blocking, argv, timeout, input_bytes)


def _ssh_target() -> str:
    return f"{settings.remote_user}@{settings.remote_host}"


def _ssh_argv() -> list[str]:
    bin_ = shutil.which("ssh") or "ssh"
    return [
        bin_,
        "-p", str(settings.remote_port),
        "-o", "BatchMode=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=4",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        _ssh_target(),
    ]


def _scp_argv(direction: str, src: str, dst: str) -> list[str]:
    """direction: 'up' (local→remote) or 'down' (remote→local)."""
    bin_ = shutil.which("scp") or "scp"
    base = [bin_, "-P", str(settings.remote_port), "-B", "-o", "StrictHostKeyChecking=accept-new", "-q"]
    if direction == "up":
        return [*base, src, f"{_ssh_target()}:{dst}"]
    return [*base, f"{_ssh_target()}:{src}", dst]


# ---------------------------------------------------------------------------
# LocalHost
# ---------------------------------------------------------------------------
class LocalHost:
    kind = "local"

    async def exec(self, cmd: str, *, timeout: float | None = None) -> ExecResult:
        return await _run(["bash", "-lc", cmd], timeout=timeout)

    async def exec_detached(self, cmd: str, *, log_remote: str) -> int:
        full = f"nohup bash -lc {shlex.quote(cmd)} > {shlex.quote(log_remote)} 2>&1 & echo $!"
        r = await _run(["bash", "-lc", full])
        if not r.ok:
            raise RuntimeError(f"local exec_detached failed: {r.stderr}")
        return int(r.stdout.strip().splitlines()[-1])

    async def is_pid_alive(self, pid: int) -> bool:
        r = await _run(["bash", "-lc", f"kill -0 {pid} 2>/dev/null && echo 1 || echo 0"])
        return r.stdout.strip().endswith("1")

    async def read_text(self, remote_path: str, *, tail_lines: int | None = None) -> str:
        if tail_lines:
            r = await _run(["bash", "-lc", f"tail -n {tail_lines} {shlex.quote(remote_path)}"])
        else:
            r = await _run(["cat", remote_path])
        return r.stdout

    async def upload(self, local: Path, remote: str) -> None:
        Path(remote).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, remote)

    async def download(self, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote, local)

    async def glob(self, remote_pattern: str) -> list[str]:
        import glob
        return sorted(glob.glob(remote_pattern, recursive=True))

    async def mkdir(self, remote_path: str) -> None:
        Path(remote_path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# RemoteHost
# ---------------------------------------------------------------------------
class RemoteHost:
    kind = "remote"

    async def exec(self, cmd: str, *, timeout: float | None = None) -> ExecResult:
        # ssh flattens positional args into a single remote command string,
        # so we pre-wrap the cmd in `bash -l -c '<cmd>'` as one argument.
        wrapped = f"bash -l -c {shlex.quote(cmd)}"
        argv = [*_ssh_argv(), wrapped]
        return await _run(argv, timeout=timeout)

    async def exec_detached(self, cmd: str, *, log_remote: str) -> int:
        # Wrap remote shell to daemonise and echo the pid after disowning.
        full = (
            f"mkdir -p $(dirname {shlex.quote(log_remote)}) && "
            f"nohup bash -lc {shlex.quote(cmd)} > {shlex.quote(log_remote)} 2>&1 & "
            f"pid=$!; disown; echo $pid"
        )
        r = await self.exec(full, timeout=30)
        if not r.ok:
            raise RuntimeError(f"remote exec_detached failed: {r.stderr}")
        # Last line of stdout is the pid; earlier lines may be shell banner noise.
        return int(r.stdout.strip().splitlines()[-1])

    async def is_pid_alive(self, pid: int) -> bool:
        r = await self.exec(f"kill -0 {pid} 2>/dev/null && echo 1 || echo 0", timeout=15)
        return r.stdout.strip().endswith("1")

    async def read_text(self, remote_path: str, *, tail_lines: int | None = None) -> str:
        if tail_lines:
            cmd = f"tail -n {tail_lines} {shlex.quote(remote_path)}"
        else:
            cmd = f"cat {shlex.quote(remote_path)}"
        r = await self.exec(cmd, timeout=30)
        return r.stdout

    async def upload(self, local: Path, remote: str) -> None:
        await self.exec(f"mkdir -p {shlex.quote(str(PurePosixPath(remote).parent))}", timeout=15)
        r = await _run(_scp_argv("up", str(local), remote), timeout=600)
        if not r.ok:
            raise RuntimeError(f"scp upload failed {local}→{remote}: {r.stderr}")

    async def download(self, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        r = await _run(_scp_argv("down", remote, str(local)), timeout=600)
        if not r.ok:
            raise RuntimeError(f"scp download failed {remote}→{local}: {r.stderr}")

    async def glob(self, remote_pattern: str) -> list[str]:
        # Use bash globstar for recursive patterns via `shopt -s globstar`.
        cmd = f"shopt -s globstar nullglob; for f in {remote_pattern}; do echo \"$f\"; done"
        r = await self.exec(cmd, timeout=20)
        return [line for line in r.stdout.splitlines() if line.strip()]

    async def mkdir(self, remote_path: str) -> None:
        await self.exec(f"mkdir -p {shlex.quote(remote_path)}", timeout=15)


# ---------------------------------------------------------------------------
# singleton chooser
# ---------------------------------------------------------------------------
_host: Host | None = None


def get_host() -> Host:
    global _host
    if _host is None:
        _host = RemoteHost() if settings.remote_enabled else LocalHost()
    return _host


async def health_check() -> dict:
    """Used by the /health endpoint to surface connectivity + env status."""
    host = get_host()
    info: dict = {"kind": host.kind, "remote_host": settings.remote_host if settings.remote_enabled else None}
    try:
        r = await host.exec("echo ok", timeout=20)
        info["ssh_ok"] = r.ok and r.stdout.strip().endswith("ok")
    except Exception as exc:  # noqa: BLE001
        info["ssh_ok"] = False
        info["ssh_error"] = str(exc)
        return info

    probes = {
        "isaaclab": f"test -d {shlex.quote(settings.isaaclab_root)} && echo yes || echo no",
        "unitree_rl_lab": f"test -d {shlex.quote(settings.unitree_rl_lab_root)} && echo yes || echo no",
        "humanoid_choreo": f"test -d {shlex.quote(settings.hc_root)} && echo yes || echo no",
        "conda_env_isaaclab": f"conda env list 2>/dev/null | awk '{{print $1}}' | grep -x {shlex.quote(settings.conda_env_isaaclab)} || echo missing",
    }
    for name, cmd in probes.items():
        r = await host.exec(cmd, timeout=20)
        info[name] = r.stdout.strip() or "?"
    return info
