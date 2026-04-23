"""SMPL .pkl / .npz loader (EasyMocap smplfull, VIBE, SPIN, AMASS).

Reads joint rotations (axis-angle) + root translation, runs SMPL forward
kinematics to produce canonical `MotionData`.  Currently stubbed — will be
expanded once SMPL regressors / body model files are available on the host.
"""
from __future__ import annotations

from pathlib import Path

from .base import DataSource, MotionData


class SMPLLoader(DataSource):
    extensions = ("pkl", "npz")

    @classmethod
    def can_load(cls, path: Path) -> bool:
        ext = path.suffix.lower().lstrip(".")
        if ext not in cls.extensions:
            return False
        # Quick sniff: peek for SMPL-ish keys without fully decoding.
        try:
            if ext == "npz":
                import numpy as np
                with np.load(path, allow_pickle=True) as zf:
                    keys = set(zf.files)
                return bool(keys & {"poses", "trans", "betas", "body_pose"})
            # pkl sniff deferred to full load — cheap to just attempt.
            return True
        except Exception:
            return False

    def load(self, path: Path) -> MotionData:
        raise NotImplementedError(
            "SMPLLoader not yet implemented. Planned: parse {shapes/betas, poses, Rh, Th}, "
            "run SMPL FK via smplx package, emit canonical MotionData."
        )
