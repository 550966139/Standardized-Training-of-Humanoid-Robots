"""Factory for picking the right loader based on file sniff."""
from __future__ import annotations

from pathlib import Path

from .amass import AMASSLoader
from .base import DataSource, MotionData
from .bvh import BVHLoader
from .fbx import FBXLoader
from .smpl import SMPLLoader

# Ordering matters — AMASS sniff is more specific than SMPL sniff.
_LOADERS: tuple[type[DataSource], ...] = (
    BVHLoader,
    AMASSLoader,
    SMPLLoader,
    FBXLoader,
)


def detect_format(path: Path) -> str | None:
    """Return the detected format name, or None if no loader matches."""
    for cls in _LOADERS:
        if cls.can_load(path):
            return cls.__name__.replace("Loader", "").lower()
    return None


def load(path: Path) -> MotionData:
    """Dispatch to a matching loader."""
    for cls in _LOADERS:
        if cls.can_load(path):
            return cls().load(path)
    raise ValueError(f"No registered loader can handle {path!s}")
