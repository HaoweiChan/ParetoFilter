# Pareto Frontier Selection Tool

A Python tool for multi-dimensional Pareto frontier selection with tolerance handling, supporting both single-value and multi-value variables, and an interactive Plotly visualization server.

## Features

- **Data Loading**: Support for CSV and parquet files with single-value (N,) and multi-value (N, K) variables
- **Configuration**: YAML/JSON config files specifying variable types, tolerances, and selection strategies
- **Preprocessing**: Multi-value variable reduction and tolerance computation
- **Pareto Selection**: Using Platypus EpsilonBoxArchive with per-feature epsilon
- **Visualization**: Interactive Plotly dashboard with 2D/3D scatter plots and tolerance visualization

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run with sample data**:
   ```bash
   python main.py --config runs/sample_run_1/config.yaml
   ```

3. **Run with custom output directory**:
   ```bash
   python main.py --config runs/sample_run_1/config.yaml --run-dir custom_output_dir
   ```

## Usage

```bash
python main.py --config CONFIG_FILE [OPTIONS]
```

### Arguments

- `--config`: Path to configuration file (YAML or JSON) - **Required**

### Options

- `--processed-output`: Output file for processed data
- `--run-dir`: Run directory for outputs (defaults to config file directory)
- `--verbose, -v`: Enable verbose logging

### Examples

```bash
# Basic usage
python main.py --config runs/sample_run_1/config.yaml

# With custom run directory
python main.py --config runs/sample_run_1/config.yaml --run-dir runs/custom_run

# With custom processed data output
python main.py --config runs/sample_run_1/config.yaml --processed-output my_processed.csv

# With verbose logging
python main.py --config runs/sample_run_1/config.yaml --verbose
```

### Configuration File Format

```yaml
run:
  # Use input CSV filename as prefix for output files
  use_input_prefix: true
  
  # Visualization settings
  generate_visualization: true
  dashboard_port: 8050
  dashboard_host: localhost
  
  # Output settings
  output_processed_data: true
  # Note: Output format is always CSV for Pareto results

data:
  variable_name:
    objective: "minimize" | "maximize"
    variable:
      type: "single" | "multi"
      # For multi-value variables:
      selection_strategy: "index" | "value"
      # index: use integer bin indices; value: use numeric values to locate bins
      idx1_value: int | float
      idx2_value: int | float
    tolerance:
      type: "absolute" | "relative"
      value: float
```

### Data Format

- **Single-value variables**: Shape (N,) - direct tolerance application
- **Multi-value variables**: Shape (N, K) - requires selection strategy
  - Index selection: use integer bin indices (0-based) over IDX1/IDX2
  - Value selection: use numeric values to find bins in IDX1/IDX2 and select from list of lists

### Folder-Based Data Loading

The tool now supports loading multiple CSV files from a folder:

```yaml
run:
  input_file: /path/to/csv/folder  # Folder containing multiple CSV files
```

All CSV files in the folder will be automatically loaded and concatenated into a single DataFrame.

### Groupby Processing

Process data in groups based on categorical columns (e.g., FP, STD):

```yaml
groupby_columns:
  - FP
  - STD
```

Pareto frontier selection will be performed separately for each unique combination of groupby column values.

### IDX-Based Multi-Value Variables

For multi-value variables stored as list of lists with corresponding IDX columns:

```yaml
data:
  DYNAMIC:  # Multi-value variable
    objective: minimize
    variable:
      type: multi
      selection_strategy: value
      idx1_value: 0.25
      idx2_value: 2.5
    tolerance:
      type: relative
      value: 0.15
```

Expected CSV columns:
- `DYNAMIC`: List of lists containing the actual values
- `DYNAMIC_IDX1`: List of bin edge values for first dimension
- `DYNAMIC_IDX2`: List of bin edge values for second dimension

The tool will find the appropriate bin indices and extract `DYNAMIC[i][j]` where `i` and `j` are determined by the bin locations of `selected_idx1` and `selected_idx2`.

## Dashboard Features

The interactive dashboard provides:
- **2D/3D Scatter Plots**: Visualize Pareto frontier candidates
- **Feature Selection**: Choose up to 3 features to plot
- **Tolerance Visualization**: See tolerance ranges around points
- **Interactive Tooltips**: Display candidate values and metadata
- **Statistics Panel**: View selection metrics and reduction ratios
- **Real-time Updates**: Dynamic plotting based on feature selection

## Project Structure

```
ParetoFilter/
├── requirements.txt
├── README.md
├── config.yaml                   # Example/global config for reference
├── main.py                       # Main CLI interface and entry point
├── src/
│   ├── preprocess.py             # Data preprocessing and tolerance handling
│   ├── pareto.py                 # Pareto frontier selection logic
│   ├── visualization.py          # Plotly dashboard implementation
│   └── utils.py                  # Utility functions
├── runs/
│   ├── sample_run/               # For example/sample/demo usage
│   │   ├── config.yaml           # Sample config for this data
│   │   ├── sample_data.csv       # Sample input data
│   │   ├── output_front.json     # Output: Pareto front results
│   │   ├── output_processed.csv  # Output: Preprocessed data
│   │   └── viz/                  # (optional) Saved figures/plots
│   └── ...                      # Other experiment runs
```

## Examples

See the `runs/sample_run/` directory for sample data and configuration files. Example usage is provided below.

## Development

This project targets Python 3.11 and uses type hints throughout. Follow the coding standards in the Cursor Rules for consistency. 