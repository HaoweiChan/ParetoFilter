#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT_DIR"

CONFIG_PATH=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config=*)
      CONFIG_PATH="${1#*=}"
      shift
      ;;
    --config)
      shift
      CONFIG_PATH="$1"
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$CONFIG_PATH" ]]; then
  CONFIG_PATH="runs/sample_run_1/config.yaml"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "✗ Config file not found: $CONFIG_PATH"
  exit 1
fi

# Ensure environment is prepared
"$REPO_ROOT_DIR/scripts/setup_uv.sh"

echo "Running ParetoFilter with config: $CONFIG_PATH"
UV_PROJECT_ENVIRONMENT=".venv" uv run python main.py --config "$CONFIG_PATH" "${EXTRA_ARGS[@]}"



