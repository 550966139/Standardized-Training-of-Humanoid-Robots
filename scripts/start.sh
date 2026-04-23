#!/usr/bin/env bash
# Launch the HR Train web server.
set -euo pipefail
cd "$(dirname "$0")/.."

# Load venv if present
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Bindings tuned for AutoDL 自定义服务端口 6006
export HRTRAIN_HOST="${HRTRAIN_HOST:-0.0.0.0}"
export HRTRAIN_PORT="${HRTRAIN_PORT:-6006}"

exec hrtrain
