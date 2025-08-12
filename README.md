# Pareto Frontier Selection Tool

A Python tool for multi-dimensional Pareto frontier selection with advanced data processing (folder-based CSV loading, groupby processing, IDX-based multi-value selection) and an optional interactive Plotly dashboard.

## Features

- **Folder-Based Data Loading**: Load one or many CSV files from a directory and concatenate automatically
- **Groupby Processing**: Optimize Pareto frontier per group (e.g., by `FP`, `STD`)
- **Multi-Value Variables with IDXs**: Select from list-of-lists using IDX1/IDX2 bin edges (value- or index-based)
- **Per-Feature Tolerances**: Absolute or relative epsilon values
- **Pareto Selection**: Platypus EpsilonBoxArchive
- **Optional Dashboard**: 2D/3D scatter, tolerance overlays, interactive tooltips

## Installation

Preferred: uv (fast Python package manager)

1. Clone the repository
2. Install `uv` and sync the environment:
```bash
# One-time: install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Prepare local .venv and install deps from pyproject.toml
scripts/setup_uv.sh
```

Alternative: pip

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Install dependencies**:
   ```bash
   # Using uv (recommended)
   scripts/setup_uv.sh

   # Or with pip
   pip install -r requirements.txt
   ```

2. **Run with sample data (no dashboard by default)**:
   ```bash
   # Using uv
   uv run python main.py --config runs/sample_run_1/config.yaml

   # Or via helper script
   scripts/run_pareto_uv.sh --config runs/sample_run_1/config.yaml

   # Using system/python directly
   python main.py --config runs/sample_run_1/config.yaml
   ```

3. **Launch dashboard after processing (opt-in)**:
   ```bash
   python main.py --config runs/sample_run_1/config.yaml --visualize
   ```

4. **Dashboard-only mode (re-uses vis_data.npz)**:
   ```bash
   python main.py --dashboard-only --config runs/sample_run_1/config.yaml
   ```

5. **Run with custom output directory**:
   ```bash
   python main.py --config runs/sample_run_1/config.yaml --run-dir custom_output_dir
   ```

## Usage

```bash
# Using uv
uv run python main.py --config CONFIG_FILE [OPTIONS]

# Or directly
python main.py --config CONFIG_FILE [OPTIONS]
```

### Arguments

- `--config`: Path to configuration file (YAML or JSON) - **Required**

### Options

- `--processed-output`: Output file for processed data
- `--run-dir`: Run directory for outputs (defaults to config file directory)
- `--verbose, -v`: Enable verbose logging
- `--visualize`: Launch dashboard after processing (never auto-starts without this flag)
- `--dashboard-only`: Skip processing; load `vis_data.npz` next to the config and start dashboard
- `--bind-all`: Bind dashboard to `0.0.0.0` (use with `--visualize` or `--dashboard-only`)

### Examples

```bash
# Basic usage (no dashboard)
python main.py --config runs/sample_run_1/config.yaml

# With custom run directory
python main.py --config runs/sample_run_1/config.yaml --run-dir runs/custom_run

# With custom processed data output
python main.py --config runs/sample_run_1/config.yaml --processed-output my_processed.csv

# With verbose logging
python main.py --config runs/sample_run_1/config.yaml --verbose

# Launch dashboard after processing on all interfaces
python main.py --config runs/sample_run_1/config.yaml --visualize --bind-all

# Dashboard-only mode
python main.py --dashboard-only --config runs/sample_run_1/config.yaml
```

## Configuration File Format

```yaml
run:
  # Input path: a single CSV file or a directory of CSV files
  input_dir: path/to/data_or_dir

  # Use input filename as prefix for output files
  use_input_prefix: true

  # Output settings
  output_processed_data: true

# Optional: dashboard settings (server does NOT auto-start from config)
visualization:
  dashboard_port: 8050
  dashboard_host: localhost

# Optional: groupby columns for grouped Pareto selection
groupby_columns:
  - FP
  - STD

data:
  AREA:
    objective: minimize
    variable:
      type: single
    tolerance:
      type: absolute
      value: 1.0

  DYNAMIC:
    objective: minimize
    variable:
      type: multi
      selection_strategy: value   # or: index
      idx1_value: 0.25            # value- or index-based depending on strategy
      idx2_value: 2.5
    tolerance:
      type: relative
      value: 0.15
```

Notes:
- The dashboard never auto-starts from config; use `--visualize` or `--dashboard-only`.
- `run.input_dir` may point to a directory; all CSVs within are loaded and concatenated.
- For legacy configs, `run.input_file` is still accepted.

## Data Format

- Expected CSV columns (example): `FP,STD,AREA,OUT_CAP,LEAKAGE,DYNAMIC_IDX1,DYNAMIC_IDX2,DYNAMIC,TRANSITION_IDX1,TRANSITION_IDX2,TRANSITION,DELAY_IDX1,DELAY_IDX2,DELAY`
- **FP, STD**: Groupby columns (strings)
- **AREA, OUT_CAP, LEAKAGE**: Single-value variables (numeric)
- **DYNAMIC, TRANSITION, DELAY**: Multi-value variables (list of lists)
- **_IDX1, _IDX2**: Bin edge arrays for IDX-based value selection

### Multi-Value Variable Processing

- IDX-based selection uses IDX1/IDX2 bin edges to find the appropriate element `variable[i][j]`
- Supports both index and value strategies (`selection_strategy`)

## Outputs

- `vis_data.npz`: Saved next to the config or in `--run-dir`; used by `--dashboard-only`
- `pareto_results.csv`: Pareto front summary with columns ordered as:
  - `candidate_index`
  - For each feature in `feature_names`: `FEATURE`, `FEATURE_IDX1`, `FEATURE_IDX2` (IDX columns appear immediately after their feature when applicable)
- `processed_data.csv`: When `--processed-output` is provided or `run.output_processed_data` is true

## Dashboard Features

- 2D/3D scatter plots, feature selection, tolerance visualization, interactive tooltips, statistics panel
- Use `--bind-all` to expose the dashboard on your network (0.0.0.0)

## Project Structure

```
ParetoFilter/
├── requirements.txt
├── README.md
├── main.py                       # CLI and entry point
├── src/
│   ├── preprocess.py             # Data preprocessing and tolerance handling
│   ├── pareto.py                 # Pareto frontier selection logic
│   ├── visualization.py          # Plotly dashboard implementation
│   └── utils.py                  # Utility functions
├── runs/
│   ├── sample_run_1/
│   │   ├── config.yaml           # Sample config
│   │   ├── sample_run_1_processed_data.csv
│   │   ├── sample_run_1_pareto_results.csv
│   │   └── vis_data.npz
│   └── ...
```

## Development

This project targets Python 3.11 and uses type hints throughout. Follow the coding standards in the Cursor Rules for consistency.