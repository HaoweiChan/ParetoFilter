#!/usr/bin/env bash

set -euo pipefail

# Simple uv-based environment setup for ParetoFilter
# - Ensures uv is installed
# - Creates/uses .venv with Python 3.11 if available
# - Installs project dependencies from pyproject.toml via `uv sync`

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing uv..."
  # Official installer (macOS/Linux)
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Ensure the typical install location is on PATH for this session
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    echo "✗ Failed to install uv or add it to PATH. Please add \"$HOME/.local/bin\" to your PATH and re-run."
    exit 1
  fi
fi

echo "✓ Using uv at: $(command -v uv)"

# Create a local virtual environment if it does not exist
if [[ ! -d .venv ]]; then
  echo "Creating .venv (Python 3.11 preferred)..."
  if uv venv --python 3.11 .venv >/dev/null 2>&1; then
    echo "✓ Created .venv with Python 3.11"
  else
    echo "Python 3.11 not available to uv; creating .venv with default Python"
    uv venv .venv
  fi
else
  echo ".venv already exists — skipping creation"
fi

echo "Syncing dependencies from pyproject.toml..."
UV_PROJECT_ENVIRONMENT=".venv" uv sync

echo "✓ Environment ready. You can now run:"
echo "  uv run python main.py --config runs/sample_run_1/config.yaml"



