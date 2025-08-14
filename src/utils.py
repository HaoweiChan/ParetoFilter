"""
Utility functions for Pareto Frontier Selection Tool.

Common functions for configuration handling, validation, and logging.
"""

import sys
import json
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return config

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration structure and values."""
    # Handle both old and new config structure
    if 'data' in config:
        variables_section = config['data']
    elif 'variables' in config:
        variables_section = config['variables']
    else:
        raise ValueError("Configuration must contain either 'data' or 'variables' section")
    
    for var_name, var_config in variables_section.items():
        # Check required fields
        required_fields = ['tolerance', 'objective']
        for field in required_fields:
            if field not in var_config:
                raise ValueError(f"Variable '{var_name}' missing required field: {field}")
        
        # Handle both old and new structure
        if 'variable' in var_config:
            # New structure with nested variable config
            variable_config = var_config['variable']
            if 'type' not in variable_config:
                raise ValueError(f"Variable '{var_name}' missing 'type' in variable section")
        else:
            # Old structure with flat config
            variable_config = var_config
            if 'type' not in var_config:
                raise ValueError(f"Variable '{var_name}' missing required field: type")
        
        # Validate variable type
        if variable_config['type'] not in ['single', 'multi']:
            raise ValueError(f"Variable '{var_name}' has invalid type: {variable_config['type']}")
        
        # Validate tolerance
        tolerance = var_config['tolerance']
        if 'type' not in tolerance or 'value' not in tolerance:
            raise ValueError(f"Variable '{var_name}' has invalid tolerance configuration")
        if tolerance['type'] not in ['absolute', 'relative']:
            raise ValueError(f"Variable '{var_name}' has invalid tolerance type: {tolerance['type']}")
        if tolerance['value'] <= 0:
            raise ValueError(f"Variable '{var_name}' tolerance value must be positive")
        
        # Validate objective
        if var_config['objective'] not in ['minimize', 'maximize']:
            raise ValueError(f"Variable '{var_name}' has invalid objective: {var_config['objective']}")
        
        # Validate multi-value specific fields
        if variable_config['type'] == 'multi':
            if 'selection_strategy' not in variable_config:
                raise ValueError(f"Multi-value variable '{var_name}' missing selection_strategy")
            
            strategy = variable_config['selection_strategy']
            if strategy not in ['index', 'value']:
                raise ValueError(f"Variable '{var_name}' has invalid selection_strategy: {strategy}")

            # Both strategies require idx1_value and idx2_value
            required_fields = ['idx1_value', 'idx2_value']
            for field in required_fields:
                if field not in variable_config:
                    raise ValueError(f"Multi-value variable '{var_name}' missing required field: {field}")

            idx1_value = variable_config['idx1_value']
            idx2_value = variable_config['idx2_value']

            if strategy == 'index':
                if not isinstance(idx1_value, int) or idx1_value < 0:
                    raise ValueError(f"Variable '{var_name}' idx1_value must be non-negative integer for index strategy")
                if not isinstance(idx2_value, int) or idx2_value < 0:
                    raise ValueError(f"Variable '{var_name}' idx2_value must be non-negative integer for index strategy")
            else:  # value
                if not isinstance(idx1_value, (int, float)):
                    raise ValueError(f"Variable '{var_name}' idx1_value must be numeric for value strategy")
                if not isinstance(idx2_value, (int, float)):
                    raise ValueError(f"Variable '{var_name}' idx2_value must be numeric for value strategy")

def setup_logging(verbose: bool = False, log_file: str = None) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def save_results(data: Any, output_path: str, format: str = 'csv') -> None:
    """Save results to file in specified format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'csv':
        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'pareto_indices' in data:
            # Handle Pareto results dictionary
            pareto_indices = data['pareto_indices']
            pareto_values = np.array(data['pareto_values'])
            feature_names = data['feature_names']
            
            # Create DataFrame columns, ensuring IDX columns immediately follow their variable
            df_data = {}
            
            # Add groupby columns if they exist
            if 'groupby_columns' in data and 'grouped_results' in data:
                groupby_columns = data['groupby_columns']
                grouped_results = data['grouped_results']
                
                # Create a mapping from candidate_index to group values
                index_to_group = {}
                for group in grouped_results:
                    group_values = group['group']
                    for idx in group['pareto_indices']:
                        index_to_group[idx] = group_values
                
                # Add group columns to the DataFrame
                for col_name in groupby_columns:
                    df_data[col_name] = [index_to_group.get(idx, {}).get(col_name, None) for idx in pareto_indices]

            if 'passthrough_data' in data and 'passthrough_columns' in data:
                passthrough_df = data['passthrough_data'].iloc[pareto_indices]
                for col_name in data['passthrough_columns']:
                    df_data[col_name] = passthrough_df[col_name].values

            pareto_indices_arr = np.asarray(pareto_indices, dtype=int)

            idx_values = data.get('idx_values', {})

            for i, feature_name in enumerate(feature_names):
                # Base feature values
                df_data[feature_name] = pareto_values[:, i]

                # If this feature has idx1/idx2 values, append them right after
                idx_data = idx_values.get(feature_name)
                if isinstance(idx_data, dict) and 'idx1' in idx_data and 'idx2' in idx_data:
                    idx1_arr = np.asarray(idx_data['idx1'])
                    idx2_arr = np.asarray(idx_data['idx2'])

                    # Guard against out-of-bounds if arrays are shorter than total candidates
                    max_index = int(pareto_indices_arr.max()) if pareto_indices_arr.size > 0 else -1
                    if idx1_arr.shape[0] <= max_index or idx2_arr.shape[0] <= max_index:
                        # Align by padding with NaN up to required length
                        target_len = max_index + 1

                        def _pad(arr: np.ndarray) -> np.ndarray:
                            if arr.shape[0] >= target_len:
                                return arr
                            pad = np.full(target_len - arr.shape[0], np.nan)
                            return np.concatenate([arr, pad])

                        idx1_arr = _pad(idx1_arr)
                        idx2_arr = _pad(idx2_arr)

                    # Get idx values for the selected Pareto candidates
                    idx1_vals = idx1_arr[pareto_indices_arr]
                    idx2_vals = idx2_arr[pareto_indices_arr]

                    df_data[f'{feature_name}_IDX1'] = idx1_vals
                    df_data[f'{feature_name}_IDX2'] = idx2_vals
            
            # Add metadata as separate rows or in a comment
            df = pd.DataFrame(df_data)
            
            # Add metadata as a comment at the top
            metadata = f"# Total candidates: {data['total_candidates']}\n"
            metadata += f"# Pareto candidates: {data['pareto_candidates']}\n"
            metadata += f"# Reduction ratio: {data['pareto_candidates']/data['total_candidates']:.3f}\n"
            
            # Write metadata and data
            with open(output_path, 'w') as f:
                f.write(metadata)
                df.to_csv(f, index=False)
            return
        else:
            df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
    elif format.lower() == 'json':
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    else:
        raise ValueError(f"Unsupported output format: {format}")