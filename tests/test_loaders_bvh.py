"""Smoke tests for the generic BVH loader."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hrtrain.loaders import detect_format, load
from hrtrain.loaders.bvh import BVHLoader

FIX = Path(__file__).parent / "fixtures" / "tiny.bvh"


def test_can_load():
    assert BVHLoader.can_load(FIX)


def test_detect_format_routes_to_bvh():
    assert detect_format(FIX) == "bvh"


def test_load_shapes_and_fk():
    motion = load(FIX)
    assert motion.num_frames == 3
    assert motion.num_joints == 2  # Hips + Spine (End Site not counted)
    assert motion.fps == pytest.approx(30.0, rel=1e-3)
    assert motion.positions.shape == (3, 2, 3)
    assert motion.rotations.shape == (3, 2, 4)

    # Rest-pose sanity: spine above hips (after Y→Z swap, along +Z)
    spine_offset = motion.skeleton.offsets[1]
    assert spine_offset[2] > 0.09  # 10 cm → 0.1 m; tolerate -90° swap
    # Quaternion norms ≈ 1
    norms = np.linalg.norm(motion.rotations, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)
