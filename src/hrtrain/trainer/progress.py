"""Parse TensorBoard event files to extract the latest iteration.

We intentionally avoid importing `tensorboard` (heavy dep).  The file format
is a sequence of records:
    uint64 length
    uint32 crc of length
    <length> bytes of serialised Event protobuf
    uint32 crc of data

For our purposes we just want the maximum `step` recorded.  We scan tags like
`Train/mean_reward` that every rsl_rl run emits once per iteration.
"""
from __future__ import annotations

import struct
from pathlib import Path


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
            fh.read(4)  # masked crc32 of data
            yield data


def parse_event_file(run_dir: Path) -> int | None:
    events = list(run_dir.rglob("events.out.tfevents.*"))
    if not events:
        return None
    last = max(events, key=lambda p: p.stat().st_mtime)
    last_step: int | None = None
    try:
        from tensorboard.compat.proto.event_pb2 import Event  # lazy import
    except Exception:
        return None
    for raw in _iter_records(last):
        try:
            e = Event()
            e.ParseFromString(raw)
            if e.step:
                last_step = int(e.step)
        except Exception:
            continue
    return last_step
