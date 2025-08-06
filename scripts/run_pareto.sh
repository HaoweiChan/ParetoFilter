#!/bin/tcsh

# Source the environment setup script
source scripts/setup_env.sh

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

# Build python command with user config
set python_cmd = "python3 main.py --config $config"

######################################################### USER UPDATE #########################################################

eval $python_cmd