
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
    eval "pip3 install -r requirements.txt"
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

# Build python command with user config
set python_cmd = "python3 main.py --config $config"

######################################################### USER UPDATE #########################################################

eval $python_cmd