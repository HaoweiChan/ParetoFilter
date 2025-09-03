#!/bin/tcsh

# Debug script for tolerance behavior analysis
# Usage: ./scripts/debug_tolerance.sh [config_file]

# Source the environment setup script
source scripts/setup_env.sh

######################################################### USER UPDATE #########################################################
# Configure user settings file here (or pass as first argument)
if ( $#argv >= 1 ) then
    set config = "$1"
else
    set config = "runs/sample_run_2/config.yaml"
endif

# Check if user config file exists
if ( ! -f "$config" ) then
    echo "✗ Config file not found: $config"
    echo "Usage: ./scripts/debug_tolerance.sh [config_file]"
    exit 1
endif

echo "=== Tolerance Debugging Script ==="
echo "Using config: $config"
echo ""

######################################################### DEBUG RUNS #########################################################

echo "1. Running with DEBUG mode to see detailed tolerance calculations..."
echo "=================================================="
set debug_cmd = "python3 main.py --config $config --debug --verbose"
echo "Command: $debug_cmd"
eval "$debug_cmd" |& tee debug_run.log

echo ""
echo "2. Running tolerance comparison script..."
echo "=================================================="
set comparison_cmd = "python3 tests/tolerance_comparison.py --config $config --tolerances 0.05 0.1 0.15 0.2 0.25 --debug"
echo "Command: $comparison_cmd"
eval "$comparison_cmd" |& tee comparison_run.log

######################################################### ANALYSIS OUTPUT #########################################################

echo ""
echo "=== Analysis Complete ==="
echo "Check the log files:"
echo "- debug_run.log: Full debug run with tolerance details"
echo "- comparison_run.log: Side-by-side tolerance comparison"
echo ""
echo "Look for:"
echo "- 'TOLERANCE DEBUG' sections showing actual tolerance calculations per group"
echo "- 'EPSILON vs DATA RANGE' ratios in the logs"
echo "- 'COUNTERINTUITIVE' warnings in the comparison output"
echo ""
echo "Next steps:"
echo "1. Check debug_run.log for per-group tolerance calculations"
echo "2. Check comparison_run.log for tolerance behavior patterns"
echo "3. Look for groups with very small data ranges that get tiny tolerances"
