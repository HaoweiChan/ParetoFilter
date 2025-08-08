"""
Data preprocessing module for Pareto Frontier Selection Tool.

Handles data loading, multi-value variable processing, and tolerance computation.
"""

import re
import ast
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Tuple, Union
from pathlib import Path


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
            # Exclude processed and results files
            excluded_patterns = ['*processed_data.csv', '*pareto_results.csv']
            
            csv_files = [
                f for f in data_path.glob("*.csv")
                if not any(f.match(p) for p in excluded_patterns)
            ]
            if not csv_files:
                raise ValueError(f"No valid CSV files found in directory: {data_path}")
            
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
            if variable_config.get('type') == 'multi' and variable_config.get('selection_strategy') in ['index', 'value']:
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
                                   var_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process multi-value variable using 'index' or 'value' strategy.
        
        Returns tuple: (variable_values, idx1_values, idx2_values)
        """
        variable_config = var_config.get('variable', var_config)  # Handle both old and new structure
        strategy = variable_config['selection_strategy']
        if strategy in ['index', 'value']:
            return self._process_index_value_variable(data, var_name, var_config)
        raise ValueError(f"Unknown selection strategy: {strategy}")
    
    def _process_index_value_variable(self, data: pd.DataFrame, var_name: str, 
                                      var_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process multi-value variable using index (integer) or value (bin) selection over IDX1/IDX2 grids.
        
        Returns tuple: (variable_values, idx1_values, idx2_values)
        """
        variable_config = var_config.get('variable', var_config)
        strategy = variable_config['selection_strategy']
        
        # Get IDX values and configuration
        idx1_col = f"{var_name}_IDX1"
        idx2_col = f"{var_name}_IDX2"
        idx1_value = variable_config['idx1_value']
        idx2_value = variable_config['idx2_value']
        
        # Get the multi-value data as list of lists
        var_data = data[var_name].values
        idx1_data = data[idx1_col].values
        idx2_data = data[idx2_col].values
        
                # A more robust parser for string-formatted lists
        def _parse_list(s):
            if not isinstance(s, str):
                return s
            
            # Add commas between numbers and after brackets
            s = re.sub(r'(\d)\s+(\d)', r'\1, \2', s)
            s = re.sub(r'\]\s+\[', r'], [', s)
            
            # Standardize to use commas
            s = s.replace(' ', ', ')
            
            # Remove duplicate commas
            s = re.sub(r',,', ',', s)
            
            # Ensure outer brackets are present
            if not s.startswith('['): s = '[' + s
            if not s.endswith(']'):   s = s + ']'

            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                raise ValueError(f"Unable to parse list: {s}")

        # Parse the multi-value data if it's string representation
        if isinstance(var_data[0], str):
            try:
                var_data = [_parse_list(x) for x in var_data]
            except (ValueError, SyntaxError):
                raise ValueError(f"Unable to parse list of lists for variable {var_name}")
        
        # Parse IDX data if string representation
        if isinstance(idx1_data[0], str):
            try:
                idx1_data = [_parse_list(x) for x in idx1_data]
            except (ValueError, SyntaxError):
                raise ValueError(f"Unable to parse IDX1 data for variable {var_name}")
        
        if isinstance(idx2_data[0], str):
            try:
                idx2_data = [_parse_list(x) for x in idx2_data]
            except (ValueError, SyntaxError):
                raise ValueError(f"Unable to parse IDX2 data for variable {var_name}")
        
        # Process each row to extract the correct value
        results = []
        idx1_results = []
        idx2_results = []
        
        for i in range(len(var_data)):
            if strategy == 'index':
                # Use direct indices
                i_bin = int(idx1_value)
                j_bin = int(idx2_value)
                
                # Get the actual idx1/idx2 values at these indices
                actual_idx1 = idx1_data[i][i_bin] if isinstance(idx1_data[i], list) else idx1_data[i]
                actual_idx2 = idx2_data[i][j_bin] if isinstance(idx2_data[i], list) else idx2_data[i]
                
            else:  # value
                # Find the bin indices for the selected values
                i_bin = self._find_bin_index(idx1_value, idx1_data[i])
                j_bin = self._find_bin_index(idx2_value, idx2_data[i])
                
                # The actual values are the ones we're looking for
                actual_idx1 = idx1_value
                actual_idx2 = idx2_value
            
            # Extract value from the list of lists
            try:
                value = var_data[i][i_bin][j_bin]
                results.append(value)
                idx1_results.append(actual_idx1)
                idx2_results.append(actual_idx2)
            except (IndexError, TypeError):
                raise ValueError(f"Unable to extract value at indices [{i_bin}][{j_bin}] for variable {var_name}, row {i}")
        
        return np.array(results), np.array(idx1_results), np.array(idx2_results)
    
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
            result = self.process_multi_value_variable(data, var_name, var_config)
            if isinstance(result, tuple):  # index/value returns tuple
                var_data = result[0]
            else:  # legacy strategies return array directly
                var_data = result
        
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
        
        # Initialize combined idx_values storage
        combined_idx_values = {}
        
        for group_key, group_data in grouped:
            self.logger.info(f"Processing group: {group_key} ({len(group_data)} rows)")
            
            # Process this group
            group_features, group_tolerances = self._process_single_group(group_data)
            
            # Store group information
            group_dict = dict(zip(groupby_columns, group_key if isinstance(group_key, tuple) else [group_key]))
            group_info.extend([group_dict] * len(group_features))
            
            all_processed_features.append(group_features)
            all_tolerance_values.append(group_tolerances)
            
            # Collect idx_values from this group
            if hasattr(self, 'idx_values') and self.idx_values:
                for var_name, idx_data in self.idx_values.items():
                    if var_name not in combined_idx_values:
                        combined_idx_values[var_name] = {'idx1': [], 'idx2': []}
                    combined_idx_values[var_name]['idx1'].extend(idx_data['idx1'])
                    combined_idx_values[var_name]['idx2'].extend(idx_data['idx2'])
        
        # Combine all groups
        if all_processed_features:
            processed_array = np.vstack(all_processed_features)
            tolerance_array = np.vstack(all_tolerance_values)
        else:
            raise ValueError("No data processed from any group")
        
        # Convert combined idx_values to numpy arrays
        if combined_idx_values:
            for var_name in combined_idx_values:
                combined_idx_values[var_name]['idx1'] = np.array(combined_idx_values[var_name]['idx1'])
                combined_idx_values[var_name]['idx2'] = np.array(combined_idx_values[var_name]['idx2'])
            self.idx_values = combined_idx_values
        
        # Store group information for later use
        self.group_info = group_info
        
        self.logger.info(f"Processed {len(grouped)} groups, total shape: {processed_array.shape}")
        return processed_array, tolerance_array
    
    def _process_single_group(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single group or all data without groupby."""
        processed_features = []
        tolerance_values = []
        feature_names = []
        
        # Store idx values for multi-value variables
        if not hasattr(self, 'idx_values'):
            self.idx_values = {}
        
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
                if variable_config.get('selection_strategy') in ['index', 'value']:
                    feature_data, idx1_vals, idx2_vals = self.process_multi_value_variable(data, var_name, var_config)
                    # Store idx values for later use in results
                    self.idx_values[var_name] = {
                        'idx1': idx1_vals,
                        'idx2': idx2_vals
                    }
                else:
                    feature_data, _, _ = self.process_multi_value_variable(data, var_name, var_config)
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