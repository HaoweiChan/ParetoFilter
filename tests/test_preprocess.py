#!/usr/bin/env python3
"""
Test script for data preprocessing module.

Generates sample test data with 1000 rows and tests the preprocessing functionality.
"""

import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from preprocess import DataPreprocessor


def generate_sample_run_data(n_rows: int = 1000) -> pd.DataFrame:
    """Generate sample data matching the new CSV format with FP, STD, and multi-value variables."""
    
    np.random.seed(42)  # For reproducible results
    
    # Generate groupby columns (FP and STD)
    fp_values = ['FP1', 'FP2', 'FP3']
    std_values = ['STD_A', 'STD_B', 'STD_C']
    
    data = {
        'FP': np.random.choice(fp_values, n_rows),
        'STD': np.random.choice(std_values, n_rows),
        
        # Single-value variables
        'AREA': np.random.uniform(100, 1000, n_rows),
        'OUT_CAP': np.random.uniform(10, 50, n_rows),
        'LEAKAGE': np.random.uniform(1e-12, 1e-9, n_rows),
        
        # Multi-value variables with IDX columns
        'DYNAMIC_IDX1': [],
        'DYNAMIC_IDX2': [],
        'DYNAMIC': [],
        'TRANSITION_IDX1': [],
        'TRANSITION_IDX2': [],
        'TRANSITION': [],
        'DELAY_IDX1': [],
        'DELAY_IDX2': [],
        'DELAY': []
    }
    
    # Generate multi-value data with list of lists format
    for i in range(n_rows):
        # DYNAMIC variable
        dynamic_idx1 = [0.1, 0.2, 0.3, 0.4, 0.5]
        dynamic_idx2 = [1.0, 2.0, 3.0, 4.0, 5.0]
        dynamic_values = [[np.random.uniform(1, 10) for _ in range(4)] for _ in range(4)]
        
        data['DYNAMIC_IDX1'].append(dynamic_idx1)
        data['DYNAMIC_IDX2'].append(dynamic_idx2)
        data['DYNAMIC'].append(dynamic_values)
        
        # TRANSITION variable
        transition_idx1 = [0.05, 0.1, 0.15, 0.2, 0.25]
        transition_idx2 = [0.5, 1.0, 1.5, 2.0, 2.5]
        transition_values = [[np.random.uniform(0.1, 2.0) for _ in range(4)] for _ in range(4)]
        
        data['TRANSITION_IDX1'].append(transition_idx1)
        data['TRANSITION_IDX2'].append(transition_idx2)
        data['TRANSITION'].append(transition_values)
        
        # DELAY variable
        delay_idx1 = [10, 20, 30, 40, 50]
        delay_idx2 = [100, 200, 300, 400, 500]
        delay_values = [[np.random.uniform(5, 50) for _ in range(4)] for _ in range(4)]
        
        data['DELAY_IDX1'].append(delay_idx1)
        data['DELAY_IDX2'].append(delay_idx2)
        data['DELAY'].append(delay_values)
    
    return pd.DataFrame(data)


def create_sample_run_config() -> dict:
    """Create a configuration matching the new CSV format with groupby and IDX-based variables."""
    
    config = {
        'run': {
            'input_file': 'sample_data.csv',
            'use_input_prefix': True,
            'output_processed_data': True
        },
        'groupby_columns': ['FP', 'STD'],  # Enable groupby processing
        'visualization': {
            'generate_visualization': False,
            'dashboard_port': 8050,
            'dashboard_host': 'localhost'
        },
        'data': {
            # Single-value variables
            'AREA': {
                'objective': 'minimize',
                'variable': {
                    'type': 'single'
                },
                'tolerance': {
                    'type': 'relative',
                    'value': 0.05
                }
            },
            'OUT_CAP': {
                'objective': 'maximize',
                'variable': {
                    'type': 'single'
                },
                'tolerance': {
                    'type': 'absolute',
                    'value': 1.0
                }
            },
            'LEAKAGE': {
                'objective': 'minimize',
                'variable': {
                    'type': 'single'
                },
                'tolerance': {
                    'type': 'relative',
                    'value': 0.1
                }
            },
            
            # Multi-value variables with simplified selection strategies
            'DYNAMIC': {
                'objective': 'minimize',
                'variable': {
                    'type': 'multi',
                    'selection_strategy': 'index',
                    'idx1_value': 1,
                    'idx2_value': 2
                },
                'tolerance': {
                    'type': 'relative',
                    'value': 0.15
                }
            },
            'TRANSITION': {
                'objective': 'minimize',
                'variable': {
                    'type': 'multi',
                    'selection_strategy': 'value',
                    'idx1_value': 0.12,
                    'idx2_value': 1.2
                },
                'tolerance': {
                    'type': 'absolute',
                    'value': 0.5
                }
            },
            'DELAY': {
                'objective': 'minimize',
                'variable': {
                    'type': 'multi',
                    'selection_strategy': 'value',
                    'idx1_value': 25,
                    'idx2_value': 250
                },
                'tolerance': {
                    'type': 'relative',
                    'value': 0.1
                }
            }
        }
    }
    
    return config


def save_sample_run_files(data: pd.DataFrame, config: dict, output_dir: str = "runs/sample_run_1"):
    """Save sample run data and configuration files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clean up multi-value columns for CSV (convert lists to string representation)
    df = data.copy()
    
    # Convert multi-value columns to string representation
    multi_value_columns = ['DYNAMIC', 'TRANSITION', 'DELAY']
    idx_columns = ['DYNAMIC_IDX1', 'DYNAMIC_IDX2', 'TRANSITION_IDX1', 'TRANSITION_IDX2', 'DELAY_IDX1', 'DELAY_IDX2']
    
    for col in multi_value_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x))
    
    for col in idx_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x))
    
    # Save sample data
    data_path = output_path / "sample_data.csv"
    df.to_csv(data_path, index=False)
    print(f"Saved sample data to: {data_path}")
    
    # Save sample configuration only if it doesn't exist
    config_path = output_path / "config.yaml"
    if not config_path.exists():
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"Saved sample config to: {config_path}")
    else:
        print(f"Config file already exists, skipping save: {config_path}")
    
    return data_path, config_path


def test_sample_run_preprocessing(data_path: str, config_path: str):
    """Test the preprocessing functionality for sample_run_1."""
    print("\n" + "="*50)
    print("TESTING SAMPLE RUN PREPROCESSING FUNCTIONALITY")
    print("="*50)
    
    # Load config from file to ensure it's the latest version
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Process data
    print("Processing data...")
    processed_data, tolerances = preprocessor.process(data_path)
    
    # Display results
    print(f"\nResults:")
    print(f"- Original data shape: {pd.read_csv(data_path).shape}")
    print(f"- Processed data shape: {processed_data.shape}")
    print(f"- Tolerance array shape: {tolerances.shape}")
    
    # Show group information if available
    if hasattr(preprocessor, 'group_info') and preprocessor.group_info:
        print(f"- Group info available: {len(preprocessor.group_info)} entries")
        groups = {}
        for group in preprocessor.group_info:
            key = tuple(sorted(group.items()))
            groups[key] = groups.get(key, 0) + 1
        
        print("\nGroups found:")
        for group_key, count in groups.items():
            group_dict = dict(group_key)
            print(f"  {group_dict}: {count} candidates")
    
    # Show feature statistics
    variables_section = config.get('data', config.get('variables', {}))
    feature_names = list(variables_section.keys())
    print(f"\nFeature statistics:")
    for i, name in enumerate(feature_names):
        values = processed_data[:, i]
        print(f"- {name}: mean={values.mean():.2f}, std={values.std():.2f}, min={values.min():.2f}, max={values.max():.2f}")
    
    # Show tolerance values
    print(f"\nTolerance values:")
    for i, name in enumerate(feature_names):
        tol = tolerances[0, i] if tolerances.ndim == 2 else tolerances[i]
        print(f"- {name}: {tol:.4f}")
    
    return processed_data, tolerances


def main():
    """Main function to generate sample_run_1."""
    print("Generating sample_run_1 data...")
    
    # Generate sample data
    sample_data = generate_sample_run_data(1000)
    print(f"Generated sample data with {sample_data.shape[0]} rows and {sample_data.shape[1]} columns")
    
    # Show data info
    print(f"\nData columns:")
    single_value_cols = ['AREA', 'OUT_CAP', 'LEAKAGE']
    multi_value_cols = ['DYNAMIC', 'TRANSITION', 'DELAY']
    groupby_cols = ['FP', 'STD']
    
    for col in sample_data.columns:
        if col in groupby_cols:
            print(f"- {col}: groupby column")
        elif col in single_value_cols:
            print(f"- {col}: single-value variable")
        elif col in multi_value_cols:
            print(f"- {col}: multi-value variable (list of lists)")
        elif col.endswith('_IDX1') or col.endswith('_IDX2'):
            print(f"- {col}: index column for multi-value variable")
    
    # Create sample configuration
    sample_config = create_sample_run_config()
    variables_section = sample_config.get('data', sample_config.get('variables', {}))
    print(f"\nCreated sample configuration with {len(variables_section)} variables")
    
    # Save sample files
    data_path, config_path = save_sample_run_files(sample_data, sample_config)
    
    # Test preprocessing
    processed_data, tolerances = test_sample_run_preprocessing(data_path, config_path)
    
    print(f"\n" + "="*50)
    print("SAMPLE RUN CREATED SUCCESSFULLY!")
    print("="*50)
    print(f"Sample run files saved in: {Path('runs/sample_run_1')}")
    print(f"Use these files to test the main application:")
    print(f"python main.py --config {config_path} --data {data_path} --no-viz")


if __name__ == '__main__':
    main() 