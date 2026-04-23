"""AMASS motion loader (.npz with `trans`, `poses`, `betas`, `gender`, `mocap_framerate`)."""
from __future__ import annotations

from pathlib import Path

from .base import DataSource, MotionData


class AMASSLoader(DataSource):
    extensions = ("npz",)

    @classmethod
    def can_load(cls, path: Path) -> bool:
        if path.suffix.lower() != ".npz":
            return False
        try:
            import numpy as np
            with np.load(path, allow_pickle=True) as zf:
                keys = set(zf.files)
            amass_keys = {"trans", "poses", "mocap_framerate"}
            return amass_keys.issubset(keys)
        except Exception:
            return False

    def load(self, path: Path) -> MotionData:
        raise NotImplementedError(
            "AMASSLoader not yet implemented. Planned: reuse SMPLLoader FK "
            "with body-model lookup keyed on `gender`."
        )
