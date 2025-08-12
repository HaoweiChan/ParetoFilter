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

No manual setup is required. The tcsh scripts in `scripts/` will create and manage the environment for you.

## Quick Start (tcsh scripts only)

1. Open `scripts/run_pareto.sh` and set the `config` variable to your config path.
2. Run the pipeline using tcsh:
   ```bash
   tcsh scripts/run_pareto.sh
   ```

Notes:
- `scripts/run_pareto.sh` is the stable entry point. It sets up the Python environment and executes the pipeline for you.
- `scripts/run_pareto_uv.sh` is experimental/unstable at the moment; do not use it.
- If you need to enable the dashboard (e.g., `--visualize` or `--dashboard-only`), edit the `python_cmd` line inside the script to append the desired flags.

## Usage

Run only via tcsh scripts:
```bash
tcsh scripts/run_pareto.sh
```

Optional flags like `--visualize`, `--dashboard-only`, or `--bind-all` should be appended inside the script by editing its `python_cmd` line. Direct `python main.py` commands are not recommended and have been removed from this README.

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