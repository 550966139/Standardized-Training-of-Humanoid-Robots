"""Thin wrapper over the `general_motion_retargeting` (GMR) package.

GMR lives in the `gmr` conda env (MuJoCo + Mink IK solver).  We do not import
it into the FastAPI process; instead we invoke it as a subprocess with a
serialised intermediate file so we keep env isolation.

Input:  canonical MotionData dumped to a temporary .npz (positions, rotations, fps)
Output: G1MotionSequence .npz with keys {qpos, qvel, fps, ik_residual, source_meta}
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from ..config import settings
from ..loaders.base import MotionData


def _dump_canonical(motion: MotionData, path: Path) -> None:
    np.savez_compressed(
        path,
        positions=motion.positions,
        rotations=motion.rotations,
        fps=np.array([motion.fps], dtype=np.float32),
        joint_names=np.array(motion.skeleton.joint_names, dtype=object),
        parents=motion.skeleton.parents,
        offsets=motion.skeleton.offsets,
    )


def retarget_to_g1(
    motion: MotionData,
    out_path: Path,
    actual_human_height: float = 1.7,
    src_human_profile: str = "bvh_lafan1",
) -> Path:
    """Run GMR in the dedicated conda env and produce a G1 qpos sequence.

    NOTE: when executed during training orchestration we usually receive
    MotionData directly from a loader.  This helper is synchronous and
    blocking; call from an executor if you need async behaviour.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path = out_path.with_suffix(".canonical.npz")
    _dump_canonical(motion, canonical_path)

    script = (settings.hc_root / "scripts" / "run_gmr_retarget.py")
    if not script.exists():
        raise FileNotFoundError(
            f"Missing upstream GMR driver script: {script}. "
            "Install humanoid-choreo and provide scripts/run_gmr_retarget.py that accepts "
            "`--input <canonical.npz> --output <g1.npz> --src <profile> --height <float>`."
        )

    conda_sh = settings.conda_root / "etc" / "profile.d" / "conda.sh"
    cmd = (
        f"source {conda_sh} && conda activate {settings.conda_env_gmr} && "
        f"python {script} --input {canonical_path} --output {out_path} "
        f"--src {src_human_profile} --height {actual_human_height}"
    )
    result = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"GMR retargeting failed:\n{result.stderr[-1000:]}")
    return out_path


def read_gmr_output(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as zf:
        data = {k: zf[k] for k in zf.files}
    # Normalise metadata json
    if "source_meta_json" in data:
        meta_raw = str(data["source_meta_json"][0]) if data["source_meta_json"].shape else ""
        try:
            data["source_meta"] = json.loads(meta_raw)
        except Exception:
            data["source_meta"] = {}
    return data
