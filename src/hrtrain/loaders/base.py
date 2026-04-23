"""Canonical motion-data interface shared by every loader.

Every format-specific loader must output a `MotionData` whose `TaggedMotion`
obeys the project's canonical convention:
  - up axis:    Z
  - handedness: right
  - length:     m
  - quat order: wxyz
  - euler:      XYZ extrinsic (when applicable)

Positions/rotations are GLOBAL after forward kinematics, shape
`(T, N, 3)` and `(T, N, 4)` respectively.  Loaders are responsible for
converting their native coordinate system into this canonical form before
returning.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SkeletonSchema:
    """Minimal skeleton topology shared across formats."""
    joint_names: list[str]
    parents: np.ndarray              # (N,) int, parent index; -1 for root
    offsets: np.ndarray              # (N, 3) rest-pose local offsets in metres

    def __post_init__(self) -> None:
        assert self.parents.ndim == 1
        assert self.offsets.shape == (len(self.joint_names), 3)
        assert self.parents.shape == (len(self.joint_names),)


@dataclass
class MotionData:
    """Canonical motion payload returned by every loader."""
    positions: np.ndarray            # (T, N, 3) metres, world
    rotations: np.ndarray            # (T, N, 4) wxyz, world
    fps: float
    skeleton: SkeletonSchema
    source_format: str               # "bvh" / "smpl" / "amass" / "fbx" / ...
    source_path: str
    extras: dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.positions.shape[1])

    @property
    def duration_sec(self) -> float:
        return self.num_frames / float(self.fps)


class DataSource(ABC):
    """Abstract loader. Subclasses implement `load()` returning MotionData."""

    #: extensions this loader claims (lowercase, dot-free)
    extensions: tuple[str, ...] = ()

    @classmethod
    @abstractmethod
    def can_load(cls, path: Path) -> bool:
        """Quick sniff test — extension, magic bytes, or header lookahead."""

    @abstractmethod
    def load(self, path: Path) -> MotionData: ...
