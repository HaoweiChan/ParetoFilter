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

from src.preprocess import DataPreprocessor


def generate_sample_run_data(n_rows: int = 1000) -> pd.DataFrame:
    """Generate sample data for sample_run_1 with 3 variables."""
    
    # Define column structure
    single_value_cols = ['single_value_1', 'single_value_2']
    multi_value_cols = ['multiple_value_1']  # Will have 10 values per row
    
    # Generate single-value data
    data = {}
    
    # Single-value variables with different distributions
    data['single_value_1'] = np.random.normal(100, 20, n_rows)  # Normal distribution
    data['single_value_2'] = np.random.exponential(50, n_rows)  # Exponential distribution
    
    # Multi-value variable with 10 conditions
    for col in multi_value_cols:
        # Generate base values
        base_values = np.random.normal(50, 15, n_rows)
        
        # Create multi-value arrays with 10 conditions
        multi_values = []
        for i in range(n_rows):
            # Add some variation across conditions
            condition_values = []
            for j in range(10):  # 10 conditions
                condition_values.append(base_values[i] + np.random.normal(0, 5))
            multi_values.append(condition_values)
        
        data[col] = multi_values
    
    return pd.DataFrame(data)


def create_sample_run_config() -> dict:
    """Create a configuration for sample_run_1 with 3 variables."""
    
    config = {
        'run': {
            'input_file': 'sample_data.csv',
            'use_input_prefix': True,
            'output_processed_data': True
        },
        'visualization': {
            'generate_visualization': False,
            'dashboard_port': 8050,
            'dashboard_host': 'localhost'
        },
        'data': {
            # Single-value variables
            'single_value_1': {
                'objective': 'minimize',
                'variable': {
                    'type': 'single'
                },
                'tolerance': {
                    'type': 'absolute',
                    'value': 2.0
                }
            },
            'single_value_2': {
                'objective': 'maximize',
                'variable': {
                    'type': 'single'
                },
                'tolerance': {
                    'type': 'relative',
                    'value': 0.05  # 5% relative tolerance
                }
            },
            
            # Multi-value variable with 10 conditions
            'multiple_value_1': {
                'objective': 'minimize',
                'variable': {
                    'type': 'multi',
                    'selection_strategy': 'index',
                    'selection_value': 0  # Use first condition
                },
                'tolerance': {
                    'type': 'absolute',
                    'value': 1.5
                }
            }
        }
    }
    
    return config


def save_sample_run_files(data: pd.DataFrame, config: dict, output_dir: str = "runs/sample_run_1"):
    """Save sample run data and configuration files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clean up multi-value columns for CSV
    df = data.copy()
    if 'multiple_value_1' in df.columns:
        df['multiple_value_1'] = df['multiple_value_1'].apply(lambda arr: '[' + ', '.join(f'{float(x):.8f}' for x in arr) + ']')
    
    # Save sample data
    data_path = output_path / "sample_data.csv"
    df.to_csv(data_path, index=False)
    print(f"Saved sample data to: {data_path}")
    
    # Save sample configuration
    config_path = output_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    print(f"Saved sample config to: {config_path}")
    
    return data_path, config_path


def test_sample_run_preprocessing(data_path: str, config: dict):
    """Test the preprocessing functionality for sample_run_1."""
    print("\n" + "="*50)
    print("TESTING SAMPLE RUN PREPROCESSING FUNCTIONALITY")
    print("="*50)
    
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
    for col in sample_data.columns:
        if col.startswith('single_value'):
            print(f"- {col}: single-value variable")
        else:
            print(f"- {col}: multi-value variable (10 conditions)")
    
    # Create sample configuration
    sample_config = create_sample_run_config()
    variables_section = sample_config.get('data', sample_config.get('variables', {}))
    print(f"\nCreated sample configuration with {len(variables_section)} variables")
    
    # Save sample files
    data_path, config_path = save_sample_run_files(sample_data, sample_config)
    
    # Test preprocessing
    processed_data, tolerances = test_sample_run_preprocessing(data_path, sample_config)
    
    print(f"\n" + "="*50)
    print("SAMPLE RUN CREATED SUCCESSFULLY!")
    print("="*50)
    print(f"Sample run files saved in: {Path('runs/sample_run_1')}")
    print(f"Use these files to test the main application:")
    print(f"python main.py --config {config_path} --data {data_path} --no-viz")


if __name__ == '__main__':
    main() 