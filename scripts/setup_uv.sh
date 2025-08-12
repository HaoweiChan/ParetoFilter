#!/bin/tcsh

# Simple uv-based environment setup for ParetoFilter (tcsh version)
# - Ensures uv is installed
# - Creates/uses .venv with Python 3.11 if available
# - Installs project dependencies from pyproject.toml via `uv sync`

# Move to repo root (one level above this script)
set script_dir = `dirname $0`
cd "$script_dir/.."
set REPO_ROOT_DIR = `pwd`

# Load a Python module first (align with setup_env.sh) if the module system is available
set loaded = 0
if ( $?MODULEPATH || $?MODULESHOME ) then
    (module load Python3/3.11.1) >& /dev/null
    if ( $status == 0 ) then
        set loaded = 1
        echo "Module load Python3/3.11.1"
    else
        (module load Python3/3.11.8_gpu_torch251) >& /dev/null
        if ( $status == 0 ) then
            set loaded = 1
            echo "Module load Python3/3.11.8_gpu_torch251"
        else
            (module load Python/3.11) >& /dev/null
            if ( $status == 0 ) then
                set loaded = 1
                echo "Module load Python/3.11"
            endif
        endif
    endif
else
    echo "Environment modules not detected; proceeding without module load"
endif

# Capture python3 path if available (prefer module-provided python)
set python_bin = `which python3` 

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
    echo "Creating .venv (prefer module python if available)..."
    if ( -x "$python_bin" ) then
        uv venv --python "$python_bin" .venv >& /dev/null
        if ( $status == 0 ) then
            echo "✓ Created .venv with $python_bin"
        else
            echo "uv could not use $python_bin; trying Python 3.11 toolchain"
            uv venv --python 3.11 .venv >& /dev/null
            if ( $status == 0 ) then
                echo "✓ Created .venv with Python 3.11"
            else
                echo "uv could not provision Python 3.11; creating .venv with default"
                uv venv .venv
                if ( $status != 0 ) then
                    echo "✗ Failed to create .venv"
                    exit 1
                endif
            endif
        endif
    else
        echo "python3 not found on PATH; trying uv-managed Python 3.11"
        uv venv --python 3.11 .venv >& /dev/null
        if ( $status == 0 ) then
            echo "✓ Created .venv with Python 3.11"
        else
            echo "uv could not provision Python 3.11; creating .venv with default"
            uv venv .venv
            if ( $status != 0 ) then
                echo "✗ Failed to create .venv"
                exit 1
            endif
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



