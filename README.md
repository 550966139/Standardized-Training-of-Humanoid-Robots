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

## Quick start (本机 Web + 远端训练,推荐)

**Web UI 在本机跑,训练远程在 AutoDL**,本机只需要 Python 3.10+ 和 OpenSSH。

```bash
git clone git@github.com:550966139/Standardized-Training-of-Humanoid-Robots.git
cd Standardized-Training-of-Humanoid-Robots

# 一次性:uv 创建 venv + 清华源装依赖
pip install uv
uv venv -p 3.10 --seed
source .venv/bin/activate   # Windows: .venv\Scripts\activate
UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple uv pip install -e ".[dev]"

# 启动(默认 0.0.0.0:6006 → 本机浏览器打开 http://127.0.0.1:6006/)
python -m hrtrain.main
```

### 配置远端(第一次必填)

本机 `~/.ssh/id_ed25519.pub` 加入 AutoDL 服务器 `~/.ssh/authorized_keys`。
然后写 `.env`(或导出环境变量):

```env
HRTRAIN_REMOTE_HOST=connect.bjb2.seetacloud.com
HRTRAIN_REMOTE_PORT=51501
HRTRAIN_REMOTE_USER=root
HRTRAIN_REMOTE_WORKDIR=/root/autodl-tmp/hrtrain_remote
HRTRAIN_CONDA_ROOT=/root/miniconda3
HRTRAIN_CONDA_ENV_ISAACLAB=hc-isaac
HRTRAIN_ISAACLAB_ROOT=/root/autodl-tmp/IsaacLab
HRTRAIN_UNITREE_RL_LAB_ROOT=/root/autodl-tmp/unitree_rl_lab
HRTRAIN_HC_ROOT=/root/autodl-tmp/humanoid-choreo
```

首页右上角会显示 **后端就绪 · <host>**;若是红色或黄色,点 ↻ 重新检测。

## 也可以整体部署在服务器

详见 `docs/deploy-autodl.md`:所有代码一起搬到 AutoDL,暴露 6006 端口。

## Dev

```bash
pytest                   # unit tests with bundled fixtures
ruff check .             # lint
mypy src/                # type check
```

## Status

- [x] Scaffold — FastAPI + HTMX/Alpine, SQLite-backed jobs
- [x] **Remote execution layer** — SSH/SCP-based `Host` abstraction;本机 Web ↔ 远端 GPU
- [x] `/health` endpoint — 前端实时显示后端连通性
- [x] BVH generic loader (Y-up → Z-up, FK,适配 CMU / LAFAN / EasyMocap)
- [ ] SMPL loader (sniff OK, FK 未实装)
- [ ] AMASS loader (sniff OK, FK 未实装)
- [ ] FBX loader (Blender subprocess)
- [x] GMR retarget wrapper — 等 humanoid-choreo 侧补 `scripts/run_gmr_retarget.py`
- [x] Train orchestration — 通过 ssh 发起 rsl_rl train.py,`HRTRAIN_MOTION_NPZ` 注入自定义动作
- [x] Output exporter — `.pt` / `.onnx` / 渲染 MP4 / qpos CSV 全部 scp 回本机;BVH/FBX 回刻留 TODO
- [x] Web UI — 上传 / 任务列表 / 进度 SSE / 下载

详见 `docs/architecture.md`。
