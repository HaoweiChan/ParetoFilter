"""
Utility functions for Pareto Frontier Selection Tool.

Common functions for configuration handling, validation, and logging.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            import yaml
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
            if strategy not in ['index', 'percentile', 'idx_based']:
                raise ValueError(f"Variable '{var_name}' has invalid selection_strategy: {strategy}")
            
            if strategy == 'idx_based':
                # Validate idx_based strategy fields
                required_idx_fields = ['idx1_values', 'idx2_values', 'selected_idx1', 'selected_idx2']
                for field in required_idx_fields:
                    if field not in variable_config:
                        raise ValueError(f"IDX-based variable '{var_name}' missing required field: {field}")
            else:
                # Validate legacy strategies (index, percentile)
                if 'selection_value' not in variable_config:
                    raise ValueError(f"Multi-value variable '{var_name}' missing selection_value")
                
                value = variable_config['selection_value']
                if strategy == 'index' and (not isinstance(value, int) or value < 0):
                    raise ValueError(f"Variable '{var_name}' index must be non-negative integer")
                if strategy == 'percentile' and (not isinstance(value, (int, float)) or value < 0 or value > 100):
                    raise ValueError(f"Variable '{var_name}' percentile must be between 0 and 100")


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
        import pandas as pd
        import numpy as np
        
        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'pareto_indices' in data:
            # Handle Pareto results dictionary
            pareto_indices = data['pareto_indices']
            pareto_values = np.array(data['pareto_values'])
            feature_names = data['feature_names']
            
            # Create DataFrame with indices and values
            df_data = {'candidate_index': pareto_indices}
            for i, feature_name in enumerate(feature_names):
                df_data[feature_name] = pareto_values[:, i]
            
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


def create_run_directory(run_name: str = None) -> Path:
    """Create a new run directory with timestamp."""
    from datetime import datetime
    
    if run_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        run_name = f"run_{timestamp}"
    
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (run_dir / "viz").mkdir(exist_ok=True)
    
    return run_dir 