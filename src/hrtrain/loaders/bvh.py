"""Generic BVH loader.

Parses hierarchy (joint tree + rest offsets) and motion (euler channels) from
BVH text files.  Converts every pose to canonical: Z-up, right-handed, metres,
wxyz quaternions in global frame via forward-kinematics.

Supports CMU, LAFAN, EasyMocap (SMPL m_avg_*), and any other flat BVH with
rotation channels.  Translation channels are honoured on the root; optionally
on non-root joints if present.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from .base import DataSource, MotionData, SkeletonSchema

log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
_CM_TO_M = 0.01


class BVHLoader(DataSource):
    extensions = ("bvh",)

    @classmethod
    def can_load(cls, path: Path) -> bool:
        if path.suffix.lower().lstrip(".") != "bvh":
            return False
        try:
            with path.open("rb") as fh:
                head = fh.read(128).lstrip()
            return head[:9].upper().startswith(b"HIERARCHY")
        except OSError:
            return False

    def load(self, path: Path) -> MotionData:
        text = path.read_text(errors="replace")
        hierarchy, motion = _split_sections(text)
        joints, channels = _parse_hierarchy(hierarchy)
        fps, frames = _parse_motion(motion)
        expected_cols = sum(len(c) for c in channels)
        if frames.shape[1] != expected_cols:
            raise ValueError(
                f"BVH motion columns ({frames.shape[1]}) do not match channel count ({expected_cols})"
            )

        length_scale = _guess_length_scale(joints)
        up_axis = _guess_up_axis(joints)

        offsets = np.stack([j["offset"] for j in joints]) * length_scale
        parents = np.array([j["parent"] for j in joints], dtype=np.int64)
        names = [j["name"] for j in joints]

        positions, rotations = _forward_kinematics(
            offsets=offsets,
            parents=parents,
            channels=channels,
            frames=frames,
            length_scale=length_scale,
            up_axis=up_axis,
        )

        # Canonicalise skeleton rest offsets to match the motion's Z-up frame.
        if up_axis == "Y":
            R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
            offsets = offsets @ R.T
        schema = SkeletonSchema(joint_names=names, parents=parents, offsets=offsets)

        return MotionData(
            positions=positions.astype(np.float32),
            rotations=rotations.astype(np.float32),
            fps=float(fps),
            skeleton=schema,
            source_format="bvh",
            source_path=str(path),
            extras={"joint_names": names, "up_axis": up_axis, "length_scale": length_scale},
        )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _split_sections(text: str) -> tuple[str, str]:
    up = text.upper()
    idx_h = up.find("HIERARCHY")
    idx_m = up.find("MOTION")
    if idx_h < 0 or idx_m < 0:
        raise ValueError("Missing HIERARCHY or MOTION section")
    return text[idx_h:idx_m], text[idx_m:]


def _parse_hierarchy(block: str) -> tuple[list[dict], list[list[str]]]:
    """Return (joints, channels_per_joint).

    joints[i] = {"name": str, "parent": int, "offset": (3,)}
    channels_per_joint[i] = list of channel names ("Xposition", "Zrotation", ...)
    """
    tokens = block.replace("{", " { ").replace("}", " } ").split()
    joints: list[dict] = []
    channels: list[list[str]] = []
    stack: list[int] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("ROOT", "JOINT"):
            name = tokens[i + 1]
            parent = stack[-1] if stack else -1
            joints.append({"name": name, "parent": parent, "offset": np.zeros(3)})
            channels.append([])
            stack.append(len(joints) - 1)
            i += 2
        elif t == "End":  # End Site
            # Match the offset but do not register as a joint
            depth_before = len(stack)
            while i < len(tokens) and tokens[i] != "{":
                i += 1
            depth = 0
            while i < len(tokens):
                if tokens[i] == "{":
                    depth += 1
                elif tokens[i] == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            assert len(stack) == depth_before  # End Site doesn't push
        elif t == "OFFSET":
            off = np.array([float(tokens[i + 1]), float(tokens[i + 2]), float(tokens[i + 3])])
            joints[stack[-1]]["offset"] = off
            i += 4
        elif t == "CHANNELS":
            n = int(tokens[i + 1])
            ch_list = tokens[i + 2 : i + 2 + n]
            channels[stack[-1]] = ch_list
            i += 2 + n
        elif t == "}":
            stack.pop()
            i += 1
        else:
            i += 1
    return joints, channels


def _parse_motion(block: str) -> tuple[float, np.ndarray]:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    frames = 0
    dt = 0.0
    data_start = 0
    for idx, ln in enumerate(lines):
        low = ln.lower()
        if low.startswith("frames:"):
            frames = int(ln.split(":")[1])
        elif low.startswith("frame time:"):
            dt = float(ln.split(":")[1])
            data_start = idx + 1
            break
    if dt <= 0:
        raise ValueError("Frame Time missing or zero")
    data = np.array(
        [[float(x) for x in _WS.split(ln)] for ln in lines[data_start:]],
        dtype=np.float64,
    )
    if frames and data.shape[0] != frames:
        log.warning("BVH Frames header (%d) disagrees with actual (%d)", frames, data.shape[0])
    fps = 1.0 / dt
    return fps, data


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------
def _guess_length_scale(joints: list[dict]) -> float:
    """BVH offsets may be cm (CMU/EasyMocap) or m (some custom exports).

    Use root-to-first-leaf accumulated offset magnitude as a proxy for the
    actor's height: a value >> 3 strongly suggests centimetres.
    """
    max_offset = max((np.linalg.norm(j["offset"]) for j in joints), default=0.0)
    return _CM_TO_M if max_offset > 3.0 else 1.0


def _guess_up_axis(joints: list[dict]) -> str:
    """Detect Y-up vs Z-up from where the largest vertical rest-offset points."""
    rest = sum((j["offset"] for j in joints), start=np.zeros(3))
    ax = int(np.argmax(np.abs(rest[1:]) )) + 1   # pick between Y (1) and Z (2)
    return "Y" if ax == 1 else "Z"


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------
def _forward_kinematics(
    offsets: np.ndarray,
    parents: np.ndarray,
    channels: list[list[str]],
    frames: np.ndarray,
    length_scale: float,
    up_axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    T = frames.shape[0]
    N = offsets.shape[0]

    # Split motion columns per joint
    col = 0
    joint_cols: list[tuple[int, int, list[str]]] = []
    for ch in channels:
        joint_cols.append((col, col + len(ch), ch))
        col += len(ch)

    local_trans = np.zeros((T, N, 3), dtype=np.float64)
    local_rot_q = np.tile([1.0, 0.0, 0.0, 0.0], (T, N, 1))  # identity wxyz

    for j, (c0, c1, ch) in enumerate(joint_cols):
        slice_ = frames[:, c0:c1]
        pos_idx = {"Xposition": 0, "Yposition": 1, "Zposition": 2}
        rot_axes: list[str] = []
        rot_vals: list[np.ndarray] = []
        local_trans[:, j, :] = offsets[j]
        for ci, name in enumerate(ch):
            col_vals = slice_[:, ci]
            if name in pos_idx:
                local_trans[:, j, pos_idx[name]] = col_vals * length_scale
            elif name.endswith("rotation"):
                rot_axes.append(name[0])
                rot_vals.append(col_vals)
        if rot_axes:
            q = _euler_to_quat(rot_axes, rot_vals)
            local_rot_q[:, j, :] = q

    # Swap Y-up to Z-up if needed (+90° about the X axis).
    if up_axis == "Y":
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        local_trans = local_trans @ R.T
        q_swap = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0])  # +90° about X (wxyz)
        local_rot_q = _quat_mul(np.broadcast_to(q_swap, local_rot_q.shape), local_rot_q)

    positions = np.zeros_like(local_trans)
    rotations = np.zeros_like(local_rot_q)
    for j in range(N):
        p = parents[j]
        if p < 0:
            rotations[:, j] = local_rot_q[:, j]
            positions[:, j] = local_trans[:, j]
        else:
            rotations[:, j] = _quat_mul(rotations[:, p], local_rot_q[:, j])
            rot_offset = _quat_rotate(rotations[:, p], local_trans[:, j])
            positions[:, j] = positions[:, p] + rot_offset

    return positions, rotations


def _euler_to_quat(axes: list[str], values: list[np.ndarray]) -> np.ndarray:
    """Convert per-axis euler channels (degrees) to wxyz quaternion.

    Applies rotations in the *channel order* as extrinsic rotations.
    """
    q = np.tile([1.0, 0.0, 0.0, 0.0], (values[0].shape[0], 1))
    axis_vec = {"X": np.array([1.0, 0.0, 0.0]),
                "Y": np.array([0.0, 1.0, 0.0]),
                "Z": np.array([0.0, 0.0, 1.0])}
    for ax, deg in zip(axes, values):
        rad = np.deg2rad(deg)
        half = rad * 0.5
        s = np.sin(half)
        ax_q = np.zeros((rad.shape[0], 4))
        ax_q[:, 0] = np.cos(half)
        ax_q[:, 1:] = axis_vec[ax][None, :] * s[:, None]
        q = _quat_mul(ax_q, q)
    return q


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([w, x, y, z], axis=-1)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a (..., 3) vector by a (..., 4) wxyz quaternion."""
    qw, qv = q[..., 0:1], q[..., 1:]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)
