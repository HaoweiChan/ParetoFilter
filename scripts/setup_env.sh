#!/bin/tcsh

# This script sets up the environment for running Python scripts.
# It handles Python module loading, virtual environment setup,
# and package installation. It also parses a '--bsub' flag
# for job submission.

# Module load Python3
set tmp_file = "module.tmp"
(module load Python3/3.11.1) >&! $tmp_file
set output = `cat $tmp_file`
if ("$output:q" =~ "*ERROR*" || "$output:q" =~ "*failed*") then
    module load Python3/3.11.8_gpu_torch251
    echo "Module load Python3/3.11.8_gpu_torch251"
else
    module load Python3/3.11.1
    echo "Module load Python3/3.11.1"
endif
rm -f $tmp_file

# Check if virtual environment exists, create it if not
set new_env = 0
if ( ! -d .venv/pareto ) then
    echo "Setting up virtual environment..."
    set new_env = 1
    python3 -m venv .venv/pareto
endif
source .venv/pareto/bin/activate.csh
if ($new_env == 1) then
    eval "pip3 install --upgrade pip setuptools wheel cmake==3.25"
    eval "pip3 install pyarrow==18.1.0"
    eval "pip3 install -r requirements.txt"
endif

# Set execution mode based on --bsub or --utilq flag
set bsub_flag = 0
set utilq_flag = 0
foreach arg ($argv)
    if ( "$arg" == "--bsub") then
        set bsub_flag = 1
    else if ( "$arg" == "--utilq") then
        set utilq_flag = 1
    endif
end

if ( $bsub_flag == 1 && $utilq_flag == 1 ) then
    echo "✗ Error: --bsub and --utilq flags cannot be used at the same time."
    exit 1
endif
