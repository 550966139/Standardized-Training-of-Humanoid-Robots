# Standardized Training of Humanoid Robots

Web-based pipeline: **import any common mocap format → retarget to humanoid → run simulation training → export deployable artifacts**.

Target robot: **Unitree G1 29-DoF** (first-class), extensible to H1 / Go2.

## Capabilities

### Input formats
- [x] **BVH** — generic parser, supports CMU, LAFAN, SMPL-style, EasyMocap output
- [x] **SMPL** `.pkl` / `.npz` — EasyMocap smplfull, VIBE, SPIN
- [x] **AMASS** `.npz` — academic motion dataset
- [x] **FBX** — via Blender subprocess

### Training orchestration
- Converts any input → canonical skeleton → G1 retargeted motion → training `.npz`
- Delegates to **Isaac Lab + unitree_rl_lab** (rsl_rl PPO) on the same host
- Real-time progress via Server-Sent Events
- Per-job workspace isolation

### Output artifacts
- `.pt` — PyTorch policy checkpoint
- `.onnx` — deployment model (Unitree SDK inference)
- `.mp4` — rollout video per iter + final
- `.bvh` / `.fbx` / `.csv` — learned motion (reimportable into Blender / Unity)
- TensorBoard events — training curves

## Architecture

```
  ┌─────────────────────────────────────────────────┐
  │            FastAPI + HTMX/Alpine web UI         │
  │  upload │ jobs │ progress │ videos │ downloads   │
  └────┬────────┬─────────────────────────┬─────────┘
       │        │                         │
       ▼        ▼                         ▼
  ┌─────────┐ ┌──────────────┐  ┌────────────────┐
  │ Loaders │ │ JobManager    │  │ Exporter       │
  │ (BVH/   │ │ (queue,       │  │ (.pt → .onnx,  │
  │  SMPL/  │ │  subprocess,  │  │  mp4, BVH/FBX) │
  │  FBX/…) │ │  SSE events)  │  │                │
  └────┬────┘ └──────┬───────┘  └─────────┬──────┘
       │             │                    │
       ▼             ▼                    ▼
  ┌─────────────────────────────────────────────┐
  │   Canonical MotionData → GMR retarget       │
  │   → train .npz → rsl_rl train.py (G1)       │
  └─────────────────────────────────────────────┘
```

## Quick start (on AutoDL)

```bash
cd /root/autodl-tmp/hrtrain

# one-time: uv venv + install
uv venv && source .venv/bin/activate
uv pip install -e ".[loaders,dev]"

# launch
./scripts/start.sh       # defaults: 0.0.0.0:6006, reload=false
```

On AutoDL, go to 自定义服务 → 把 6006 映射为公网 URL。

## Dev

```bash
pytest                   # unit tests with bundled fixtures
ruff check .             # lint
mypy src/                # type check
```

## Status

- [x] Scaffold
- [x] BVH generic loader
- [ ] SMPL loader
- [ ] AMASS loader
- [ ] FBX loader (Blender bridge)
- [ ] GMR retarget wrapper
- [ ] Train orchestration
- [ ] Output exporter
- [ ] Web UI

See `docs/` for per-module design.
