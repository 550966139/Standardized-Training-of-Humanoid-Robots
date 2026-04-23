"""FBX loader — delegates to Blender CLI to convert to BVH then reuses BVHLoader.

Blender's Python API is the most portable way to read FBX on Linux servers
that cannot pull the official Autodesk FBX SDK.  The loader writes a helper
script to a temp directory and invokes `blender -b --python <script>`.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from ..config import settings
from .base import DataSource, MotionData
from .bvh import BVHLoader


_BLENDER_SCRIPT = """
import sys, bpy
args = sys.argv[sys.argv.index('--') + 1:]
fbx_in, bvh_out = args[0], args[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx_in)
bpy.ops.export_anim.bvh(filepath=bvh_out)
"""


class FBXLoader(DataSource):
    extensions = ("fbx",)

    @classmethod
    def can_load(cls, path: Path) -> bool:
        return path.suffix.lower() == ".fbx" and shutil.which(settings.blender_bin) is not None

    def load(self, path: Path) -> MotionData:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            script_path = tmp / "fbx2bvh.py"
            bvh_path = tmp / (path.stem + ".bvh")
            script_path.write_text(_BLENDER_SCRIPT)
            result = subprocess.run(
                [settings.blender_bin, "-b", "-noaudio",
                 "--python", str(script_path), "--", str(path), str(bvh_path)],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0 or not bvh_path.exists():
                raise RuntimeError(
                    f"Blender failed to convert FBX → BVH: {result.stderr[-500:]}"
                )
            data = BVHLoader().load(bvh_path)
        data.source_format = "fbx"
        data.source_path = str(path)
        return data
