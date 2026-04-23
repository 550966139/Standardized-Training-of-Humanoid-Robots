"""Poll TensorBoard event file for latest iteration step.

Strategy:
  1. `host.glob` the event files under the remote run dir.
  2. scp the latest one to a local tempfile (events are KB-scale).
  3. Parse records to extract `step`.
"""
from __future__ import annotations

import struct
import tempfile
from pathlib import Path

from ..remote import Host


def _iter_records(path: Path):
    with path.open("rb") as fh:
        while True:
            header = fh.read(12)
            if len(header) < 12:
                return
            length = struct.unpack("<Q", header[:8])[0]
            data = fh.read(length)
            if len(data) < length:
                return
            fh.read(4)
            yield data


def _max_step_local(path: Path) -> int | None:
    try:
        from tensorboard.compat.proto.event_pb2 import Event
    except Exception:
        return None
    last = None
    for raw in _iter_records(path):
        try:
            e = Event()
            e.ParseFromString(raw)
            if e.step:
                last = int(e.step)
        except Exception:
            continue
    return last


async def poll_iter(host: Host, remote_run_dir: str) -> int | None:
    pattern = f"{remote_run_dir}/**/events.out.tfevents.*"
    files = await host.glob(pattern)
    if not files:
        return None
    latest_remote = files[-1]
    with tempfile.NamedTemporaryFile(suffix=".tfevents", delete=False) as tf:
        local_path = Path(tf.name)
    try:
        await host.download(latest_remote, local_path)
        return _max_step_local(local_path)
    finally:
        local_path.unlink(missing_ok=True)
