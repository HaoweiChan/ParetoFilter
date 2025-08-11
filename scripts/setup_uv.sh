#!/bin/tcsh

# Simple uv-based environment setup for ParetoFilter (tcsh version)
# - Ensures uv is installed
# - Creates/uses .venv with Python 3.11 if available
# - Installs project dependencies from pyproject.toml via `uv sync`

# Move to repo root (one level above this script)
set script_dir = `dirname $0`
cd "$script_dir/.."
set REPO_ROOT_DIR = `pwd`

# Check for uv; install if missing
which uv >& /dev/null
if ( $status != 0 ) then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    setenv PATH "$HOME/.local/bin:$PATH"
    which uv >& /dev/null
    if ( $status != 0 ) then
        echo "✗ Failed to install uv or add it to PATH. Please add \"$HOME/.local/bin\" to your PATH and re-run."
        exit 1
    endif
endif

set uv_path = `which uv`
echo "✓ Using uv at: $uv_path"

# Create a local virtual environment if it does not exist
if ( ! -d .venv ) then
    echo "Creating .venv (Python 3.11 preferred)..."
    uv venv --python 3.11 .venv >& /dev/null
    if ( $status == 0 ) then
        echo "✓ Created .venv with Python 3.11"
    else
        echo "Python 3.11 not available to uv; creating .venv with default Python"
        uv venv .venv
        if ( $status != 0 ) then
            echo "✗ Failed to create .venv"
            exit 1
        endif
    endif
else
    echo ".venv already exists — skipping creation"
endif

echo "Syncing dependencies from pyproject.toml..."
setenv UV_PROJECT_ENVIRONMENT .venv
uv sync
if ( $status != 0 ) then
    echo "✗ uv sync failed"
    exit 1
endif

echo "✓ Environment ready. You can now run:"
echo "  uv run python main.py --config runs/sample_run_1/config.yaml"



