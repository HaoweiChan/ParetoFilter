"""
Data preprocessing module for Pareto Frontier Selection Tool.

Handles data loading, multi-value variable processing, and tolerance computation.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Tuple


class DataPreprocessor:
    """Handles data loading, preprocessing, and tolerance computation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize preprocessor with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Handle both old and new config structure
        if 'data' in config:
            self.variable_configs = config['data']
        else:
            self.variable_configs = config['variables']
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from CSV or parquet file."""
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        if data_path.suffix.lower() == '.csv':
            data = pd.read_csv(data_path)
        elif data_path.suffix.lower() in ['.parquet', '.pq']:
            data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported data file format: {data_path.suffix}")
        
        self.logger.info(f"Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
        return data
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate that data contains all required variables."""
        missing_vars = []
        for var_name in self.variable_configs.keys():
            if var_name not in data.columns:
                missing_vars.append(var_name)
        
        if missing_vars:
            raise ValueError(f"Missing variables in data: {missing_vars}")
        
        # Check for NaN values
        nan_counts = data[self.variable_configs.keys()].isna().sum()
        if nan_counts.any():
            self.logger.warning(f"Found NaN values: {nan_counts[nan_counts > 0].to_dict()}")
    
    def process_multi_value_variable(self, data: pd.DataFrame, var_name: str, 
                                   var_config: Dict[str, Any]) -> np.ndarray:
        """Process multi-value variable using specified selection strategy."""
        variable_config = var_config.get('variable', var_config)  # Handle both old and new structure
        strategy = variable_config['selection_strategy']
        value = variable_config['selection_value']
        
        # Get the multi-value data
        var_data = data[var_name].values
        
        # Handle different data formats
        if isinstance(var_data[0], (list, np.ndarray)):
            # Data is already in list/array format
            var_array = np.array([np.array(x) if isinstance(x, list) else x for x in var_data])
        else:
            # Data might be string representation of arrays
            try:
                var_array = np.array([eval(x) if isinstance(x, str) else x for x in var_data])
            except:
                raise ValueError(f"Unable to parse multi-value data for variable {var_name}")
        
        if strategy == 'index':
            if value >= var_array.shape[1]:
                raise ValueError(f"Index {value} out of bounds for variable {var_name} (max: {var_array.shape[1]-1})")
            return var_array[:, value]
        
        elif strategy == 'percentile':
            if not (0 <= value <= 100):
                raise ValueError(f"Percentile {value} must be between 0 and 100")
            
            # Compute percentile across all conditions for each candidate
            percentiles = np.percentile(var_array, value, axis=1)
            return percentiles
        
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
    
    def compute_tolerance(self, data: pd.DataFrame, var_name: str, 
                         var_config: Dict[str, Any]) -> np.ndarray:
        """Compute tolerance values for a variable."""
        tolerance_config = var_config['tolerance']
        tolerance_type = tolerance_config['type']
        tolerance_value = tolerance_config['value']
        
        variable_config = var_config.get('variable', var_config)  # Handle both old and new structure
        if variable_config['type'] == 'single':
            var_data = data[var_name].values
        else:
            # For multi-value variables, use the processed single value
            var_data = self.process_multi_value_variable(data, var_name, var_config)
        
        if tolerance_type == 'absolute':
            return np.full_like(var_data, tolerance_value, dtype=float)
        
        elif tolerance_type == 'relative':
            max_val = np.max(np.abs(var_data))
            if max_val == 0:
                self.logger.warning(f"Variable {var_name} has all zero values, using small absolute tolerance")
                return np.full_like(var_data, 1e-6, dtype=float)
            return np.full_like(var_data, tolerance_value * max_val, dtype=float)
        
        else:
            raise ValueError(f"Unknown tolerance type: {tolerance_type}")
    
    def process(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Process data and return processed features and tolerances."""
        # Load and validate data
        data = self.load_data(data_path)
        self.validate_data(data)
        
        processed_features = []
        tolerance_values = []
        feature_names = []
        
        # Process each variable according to configuration
        for var_name, var_config in self.variable_configs.items():
            self.logger.debug(f"Processing variable: {var_name}")
            
            variable_config = var_config.get('variable', var_config)  # Handle both old and new structure
            
            if variable_config['type'] == 'single':
                # Single-value variable: use directly
                feature_data = data[var_name].values
                tolerance_data = self.compute_tolerance(data, var_name, var_config)
            else:
                # Multi-value variable: process to single value
                feature_data = self.process_multi_value_variable(data, var_name, var_config)
                tolerance_data = self.compute_tolerance(data, var_name, var_config)
            
            # Validate processed data
            if np.any(np.isnan(feature_data)):
                raise ValueError(f"NaN values found in processed variable {var_name}")
            
            processed_features.append(feature_data)
            tolerance_values.append(tolerance_data)
            feature_names.append(var_name)
        
        # Combine all features
        processed_array = np.column_stack(processed_features)
        tolerance_array = np.column_stack(tolerance_values)
        
        self.logger.info(f"Processed {len(feature_names)} variables: {feature_names}")
        self.logger.info(f"Final data shape: {processed_array.shape}")
        
        return processed_array, tolerance_array 