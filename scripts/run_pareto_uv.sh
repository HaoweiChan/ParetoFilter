#!/bin/tcsh

# Prepare uv-based environment using tcsh script
if ( -f scripts/setup_uv.sh ) then
    source scripts/setup_uv.sh
else
    echo "✗ scripts/setup_uv.sh not found. Please run from repo root or install uv manually."
    exit 1
endif

######################################################### USER UPDATE #########################################################
# Configure user settings file here
set config = "runs/sample_run_1/config.yaml"

# Check if user config file exists
if ( ! -f "$config" ) then
    echo "✗ User config file not found: $config"
    echo "Please create the file or update the config variable in this script."
    exit 1
endif

echo "Reading run settings from: $config"

# Build uv command with user config
set python_cmd = "env UV_PROJECT_ENVIRONMENT=.venv uv run python main.py --config $config"

######################################################### USER UPDATE #########################################################

eval $python_cmd
