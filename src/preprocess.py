"""
Data preprocessing module for Pareto Frontier Selection Tool.

Handles data loading, multi-value variable processing, and tolerance computation.
"""

import ast
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple


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
        """Load data from CSV files or folder containing CSV files."""
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        if data_path.is_file():
            # Single file mode (backward compatibility)
            if data_path.suffix.lower() == '.csv':
                data = pd.read_csv(data_path)
            elif data_path.suffix.lower() in ['.parquet', '.pq']:
                data = pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported data file format: {data_path.suffix}")
            
            self.logger.info(f"Loaded single file: {data.shape[0]} rows, {data.shape[1]} columns")
            return data
        
        elif data_path.is_dir():
            # Folder mode - load and concatenate all CSV files
            csv_files = list(data_path.glob("*.csv"))
            if not csv_files:
                raise ValueError(f"No CSV files found in directory: {data_path}")
            
            dataframes = []
            total_rows = 0
            
            for csv_file in csv_files:
                self.logger.info(f"Loading CSV file: {csv_file.name}")
                df = pd.read_csv(csv_file)
                dataframes.append(df)
                total_rows += len(df)
            
            # Concatenate all dataframes
            data = pd.concat(dataframes, ignore_index=True)
            self.logger.info(f"Loaded {len(csv_files)} CSV files: {total_rows} total rows, {data.shape[1]} columns")
            return data
        
        else:
            raise ValueError(f"Data path must be a file or directory: {data_path}")
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate that data contains all required variables."""
        # Check for required groupby columns if they exist in config
        groupby_columns = self.config.get('groupby_columns', ['FP', 'STD'])
        missing_groupby = [col for col in groupby_columns if col in data.columns]
        if len(missing_groupby) != len(groupby_columns):
            missing = [col for col in groupby_columns if col not in data.columns]
            self.logger.warning(f"Some groupby columns not found: {missing}")
        
        # Check for configured variables
        missing_vars = []
        for var_name in self.variable_configs.keys():
            if var_name not in data.columns:
                missing_vars.append(var_name)
        
        if missing_vars:
            raise ValueError(f"Missing variables in data: {missing_vars}")
        
        # Check for IDX columns for multi-value variables
        for var_name, var_config in self.variable_configs.items():
            variable_config = var_config.get('variable', var_config)
            if variable_config.get('type') == 'multi' and variable_config.get('selection_strategy') == 'idx_based':
                idx1_col = f"{var_name}_IDX1"
                idx2_col = f"{var_name}_IDX2"
                if idx1_col not in data.columns:
                    raise ValueError(f"Missing IDX1 column for multi-value variable {var_name}: {idx1_col}")
                if idx2_col not in data.columns:
                    raise ValueError(f"Missing IDX2 column for multi-value variable {var_name}: {idx2_col}")
        
        # Check for NaN values
        check_columns = list(self.variable_configs.keys())
        if check_columns:
            nan_counts = data[check_columns].isna().sum()
            if nan_counts.any():
                self.logger.warning(f"Found NaN values: {nan_counts[nan_counts > 0].to_dict()}")
    
    def process_multi_value_variable(self, data: pd.DataFrame, var_name: str, 
                                   var_config: Dict[str, Any]) -> np.ndarray:
        """Process multi-value variable using specified selection strategy."""
        variable_config = var_config.get('variable', var_config)  # Handle both old and new structure
        strategy = variable_config['selection_strategy']
        
        if strategy == 'idx_based':
            return self._process_idx_based_variable(data, var_name, var_config)
        
        # Handle legacy strategies
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
    
    def _process_idx_based_variable(self, data: pd.DataFrame, var_name: str, 
                                  var_config: Dict[str, Any]) -> np.ndarray:
        """Process multi-value variable using IDX1/IDX2-based selection."""
        variable_config = var_config.get('variable', var_config)
        
        # Get IDX values and configuration
        idx1_col = f"{var_name}_IDX1"
        idx2_col = f"{var_name}_IDX2"
        idx1_values = variable_config['idx1_values']
        idx2_values = variable_config['idx2_values']
        selected_idx1 = variable_config['selected_idx1']
        selected_idx2 = variable_config['selected_idx2']
        
        # Get the multi-value data as list of lists
        var_data = data[var_name].values
        idx1_data = data[idx1_col].values
        idx2_data = data[idx2_col].values
        
        # Parse the multi-value data if it's string representation
        if isinstance(var_data[0], str):
            try:
                var_data = [ast.literal_eval(x) for x in var_data]
            except:
                raise ValueError(f"Unable to parse list of lists for variable {var_name}")
        
        # Parse IDX data if string representation
        if isinstance(idx1_data[0], str):
            try:
                idx1_data = [ast.literal_eval(x) for x in idx1_data]
            except:
                raise ValueError(f"Unable to parse IDX1 data for variable {var_name}")
        
        if isinstance(idx2_data[0], str):
            try:
                idx2_data = [ast.literal_eval(x) for x in idx2_data]
            except:
                raise ValueError(f"Unable to parse IDX2 data for variable {var_name}")
        
        # Process each row to extract the correct value
        results = []
        for i in range(len(var_data)):
            # Find the bin indices for the selected values
            i_bin = self._find_bin_index(selected_idx1, idx1_data[i])
            j_bin = self._find_bin_index(selected_idx2, idx2_data[i])
            
            # Extract value from the list of lists
            try:
                value = var_data[i][i_bin][j_bin]
                results.append(value)
            except (IndexError, TypeError):
                raise ValueError(f"Unable to extract value at indices [{i_bin}][{j_bin}] for variable {var_name}, row {i}")
        
        return np.array(results)
    
    def _find_bin_index(self, selected_value: float, bin_edges: List[float]) -> int:
        """Find the bin index for a selected value given bin edges."""
        bin_edges = sorted(bin_edges)
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= selected_value < bin_edges[i + 1]:
                return i
        # Handle edge case where selected_value equals the last bin edge
        if selected_value == bin_edges[-1]:
            return len(bin_edges) - 2
        raise ValueError(f"Selected value {selected_value} is outside bin range {bin_edges}")
    
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
        
        # Check if groupby processing is enabled
        groupby_columns = self.config.get('groupby_columns')
        if groupby_columns:
            return self._process_with_groupby(data, groupby_columns)
        else:
            return self._process_single_group(data)
    
    def _process_with_groupby(self, data: pd.DataFrame, groupby_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Process data with groupby functionality."""
        # Verify groupby columns exist
        missing_cols = [col for col in groupby_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Groupby columns not found in data: {missing_cols}")
        
        # Group data
        grouped = data.groupby(groupby_columns)
        self.logger.info(f"Found {len(grouped)} groups based on columns: {groupby_columns}")
        
        all_processed_features = []
        all_tolerance_values = []
        group_info = []
        
        for group_key, group_data in grouped:
            self.logger.info(f"Processing group: {group_key} ({len(group_data)} rows)")
            
            # Process this group
            group_features, group_tolerances = self._process_single_group(group_data)
            
            # Store group information
            group_dict = dict(zip(groupby_columns, group_key if isinstance(group_key, tuple) else [group_key]))
            group_info.extend([group_dict] * len(group_features))
            
            all_processed_features.append(group_features)
            all_tolerance_values.append(group_tolerances)
        
        # Combine all groups
        if all_processed_features:
            processed_array = np.vstack(all_processed_features)
            tolerance_array = np.vstack(all_tolerance_values)
        else:
            raise ValueError("No data processed from any group")
        
        # Store group information for later use
        self.group_info = group_info
        
        self.logger.info(f"Processed {len(grouped)} groups, total shape: {processed_array.shape}")
        return processed_array, tolerance_array
    
    def _process_single_group(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single group or all data without groupby."""
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
        if processed_features:
            processed_array = np.column_stack(processed_features)
            tolerance_array = np.column_stack(tolerance_values)
        else:
            raise ValueError("No variables processed")
        
        self.logger.debug(f"Processed {len(feature_names)} variables: {feature_names}")
        self.logger.debug(f"Group data shape: {processed_array.shape}")
        
        return processed_array, tolerance_array 