# Architecture

## Process model

One FastAPI process hosts everything except the training subprocess:

```
              ┌────────────────────── browser ───────────────────────┐
              │                          │                            │
              │  HTMX upload form → POST /upload                      │
              │  HTMX create job   → POST /jobs                       │
              │  SSE progress      ← GET  /sse/jobs/{id}              │
              │  Video playback    ← GET  /jobs/{id}/download/video   │
              └────────────────────┬──────────────────────────────────┘
                                   ▼
              ┌──────────────────── hrtrain FastAPI ──────────────────┐
              │                                                      │
              │  /upload  → save file → detect_format                │
              │  /jobs    → insert Job row → manager.enqueue()       │
              │                                                      │
              │  JobManager (singleton)                              │
              │    - asyncio queue of job ids                        │
              │    - worker coroutine runs one job at a time         │
              │    - per-job listener queues → SSE                   │
              │                                                      │
              └──────────┬──────────────────────────────────┬────────┘
                         │                                   │
                         ▼                                   ▼
           ┌────── loaders ──────┐                ┌─── retarget + train ───┐
           │ BVH / SMPL / AMASS /│                │ gmr_wrapper.py (gmr env)│
           │ FBX                 │                │ to_train_npz.py (isaac env)
           │ → MotionData        │                │ runner.py → rsl_rl train.py │
           └─────────────────────┘                └──────────┬─────────────┘
                                                             ▼
                                                   ┌──── exporter ─────┐
                                                   │ latest .pt        │
                                                   │ export_onnx.py    │
                                                   │ play.py → MP4     │
                                                   │ rollout.qpos→CSV  │
                                                   └───────────────────┘
```

## Local Web + Remote Training 拓扑

```
           本机 (Windows/Mac/Linux)         AutoDL 服务器 (GPU)
 ┌────────────────────────────────┐   ┌──────────────────────────────┐
 │ Browser → http://127.0.0.1:6006│   │ conda env hc-isaac           │
 │                                │   │ IsaacSim 4.5 + Isaac Lab 2.3│
 │ FastAPI (hrtrain)              │SSH│ unitree_rl_lab              │
 │ ├─ Loaders (BVH/SMPL/AMASS/FBX)│━━▶│ humanoid-choreo (GMR)       │
 │ ├─ host = RemoteHost (ssh/scp) │◀━━│ GPU 训练 rsl_rl              │
 │ ├─ SQLite job store            │   │ play.py 渲染 MP4             │
 │ └─ data/outputs/<job>/ ← scp ──┼───┤ checkpoint / ONNX / log      │
 └────────────────────────────────┘   └──────────────────────────────┘
```

只有 Web + 解析器 + 编排代码在本机,GPU 相关全部远程。重启本机 Web 不影响远程训练进程(都是 nohup 守护的)。

## Conda environment contract (on AutoDL)

hrtrain 本身不加载 Isaac Lab / MuJoCo,而是通过 SSH 调用远端脚本:

| Env name    | Used for                                | Path                                  |
|-------------|-----------------------------------------|---------------------------------------|
| `hc-isaac`  | Isaac Lab + unitree_rl_lab + rsl_rl    | /root/miniconda3/envs/hc-isaac        |
| (同上)       | GMR retarget(在 hc-isaac 里直接跑 MuJoCo + mink,省一个 env) | 同上 |

每次远端 exec 都走 `bash -l -c "source conda.sh && conda activate <env> && <cmd>"`。

## Data flow per job

1. **Upload**: binary saved to `data/uploads/<uuid>.<ext>` ; row inserted in
   `uploaded_files`.
2. **Detect**: `loaders.registry.detect_format` picks a subclass based on
   extension + magic-byte sniff.
3. **Load**: selected loader returns canonical `MotionData`
   (Z-up, meters, wxyz, global FK).
4. **Retarget**: `retarget.gmr_wrapper.retarget_to_g1` dumps canonical to a
   tmp npz, calls GMR in `gmr` env, receives G1MotionSequence with `qpos`.
5. **Train-ready**: `retarget.to_train_npz.write_training_npz` goes through
   the existing npz→CSV→train-npz bridge in `unitree_rl_lab`.
6. **Train**: `trainer.runner.run_rsl_rl_train` starts `rsl_rl train.py` with
   the new motion via the `HRTRAIN_MOTION_NPZ` env var.  Upstream mimic task
   needs a tiny patch to honour that env var.
7. **Progress**: `trainer.progress.parse_event_file` tails TensorBoard events
   for the iteration counter; SSE pushes updates.
8. **Export**: `exporter.pipeline.run_exports` collects latest `.pt`, builds
   ONNX via `export_onnx.py`, runs `play.py` to render rollout MP4 and dumps
   learned motion back to CSV.

## Extending to new robots

* Add `tasks/mimic/robots/<robot>/` to `unitree_rl_lab` with the schema that
  mirrors `g1_29dof/dance_102/`.
* Add a mapping in `retarget/gmr_wrapper.py` for the new joint/body count.
* Surface the robot as a dropdown option in `web/templates/partials/upload_card.html`.

## Security notes

* No auth.  Intended for single-user lab deployment behind AutoDL's tunnel.
* Upload size capped at 500 MB (`HRTRAIN_MAX_UPLOAD_BYTES`).
* Downloads streamed straight from disk — `stored_path` values must never be
  user-controllable; they are generated UUIDs in `settings.uploads_dir`.
