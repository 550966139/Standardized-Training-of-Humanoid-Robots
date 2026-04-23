# 在 AutoDL 部署

AutoDL 会为每个容器实例固定暴露 **6006** 端口(原本用于 TensorBoard)——我们直接把 hrtrain 绑到这个端口就能拿到公网 URL。

## 首次部署

```bash
# 在服务器上
git clone git@github.com:550966139/Standardized-Training-of-Humanoid-Robots.git \
          /root/autodl-tmp/hrtrain
cd /root/autodl-tmp/hrtrain
./scripts/dev_install.sh         # 用清华源装 uv venv + deps
```

## 启动

### 方式 A:前台(调试用)

```bash
cd /root/autodl-tmp/hrtrain
source .venv/bin/activate
HRTRAIN_PORT=6006 hrtrain
```

### 方式 B:后台 + 日志(推荐)

```bash
./scripts/serve.sh               # 启动
./scripts/serve.sh status        # 查状态
./scripts/serve.sh stop          # 停止
./scripts/serve.sh logs          # 跟踪日志
```

## 公网访问

1. 登录 [AutoDL 控制台](https://www.autodl.com/console)
2. 进入当前容器实例 → **自定义服务** 栏
3. 复制 `6006` 端口对应的公网 URL(形如 `https://u123-xx.autodl.com/`)
4. 浏览器打开即可看到 hrtrain 主页

> AutoDL 若同时开启了其自带 TensorBoard (默认 6007),两者不冲突——点自定义服务里对应端口的链接即可。

## 常见故障

| 症状 | 排查 |
|---|---|
| 500 Internal Server Error | `./scripts/serve.sh logs` 查 traceback;九成是 import 错误或 DB 初始化失败 |
| 端口被占用 | `ss -tlnp \| grep 6006` 找冲突进程;必要时 `pkill -f hrtrain` |
| uv pip 卡住 | 改 `UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` 后重试 |
| 上传失败 413 | `HRTRAIN_MAX_UPLOAD_BYTES` 环境变量调大,默认 500 MB |
| 训练触发但 `hc-isaac` 找不到 | 确认 `/root/miniconda3/envs/hc-isaac` 存在,或改 `HRTRAIN_CONDA_ENV_ISAACLAB` |

## 上游依赖路径约定

hrtrain **不复制任何训练栈**,仅通过子进程调用现有的目录:

| 变量 | 默认值 | 说明 |
|---|---|---|
| `HRTRAIN_ISAACLAB_ROOT` | `/root/autodl-tmp/IsaacLab` | Isaac Lab 仓库根 |
| `HRTRAIN_UNITREE_RL_LAB_ROOT` | `/root/autodl-tmp/unitree_rl_lab` | Unitree RL Lab |
| `HRTRAIN_HC_ROOT` | `/root/autodl-tmp/humanoid-choreo` | 动捕/重定向上游 |
| `HRTRAIN_CONDA_ENV_ISAACLAB` | `hc-isaac` | 跑 Isaac Lab 的 conda env |
| `HRTRAIN_CONDA_ENV_GMR` | `gmr` | 跑 GMR 重定向的 conda env |

可以把所有覆盖项写进 `.env`:

```env
HRTRAIN_ISAACLAB_ROOT=/path/to/IsaacLab
HRTRAIN_CONDA_ENV_GMR=hc-isaac   # 如果合并到单一 env
```
