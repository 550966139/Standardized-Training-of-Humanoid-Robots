#!/usr/bin/env bash
# Create uv venv and install hrtrain in editable mode.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv -p 3.10
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e ".[dev]"

echo
echo "Done.  Launch with:  ./scripts/start.sh"
