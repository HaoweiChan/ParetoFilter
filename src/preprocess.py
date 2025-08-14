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
                    # Allow 1D variables which only have IDX1; downstream logic will ignore IDX2
                    self.logger.warning(f"IDX2 column not found for {var_name} ({idx2_col}). Assuming 1D variable.")
        
        # Check for NaN values
        check_columns = list(self.variable_configs.keys())
        if check_columns:
            nan_counts = data[check_columns].isna().sum()
            if nan_counts.any():
                self.logger.warning(f"Found NaN values: {nan_counts[nan_counts > 0].to_dict()}")

    def _robust_parse_list(self, value: Any) -> List[Any]:
        """Parse string-formatted list (or list of lists) into a Python list.

        - Accepts strings like "[1 2 3]" or "[1, 2, 3]" and normalizes them
        - Returns lists as-is
        - Returns empty list for clearly empty content like "[]" or "[ ]"
        - Raises ValueError if parsing fails
        """
        if isinstance(value, list):
            return value
        if not isinstance(value, str):
            return value

        s = value
        # Handle explicit empty list like "[]" or with spaces inside brackets
        if re.fullmatch(r"\s*\[\s*\]\s*", s):
            return []
        # Normalize whitespace separated numbers and bracket boundaries
        s = re.sub(r'(\d)\s+(\d)', r'\1, \2', s)
        s = re.sub(r'\]\s+\[', r'], [', s)
        # Insert commas between adjacent numbers separated by whitespace
        s = re.sub(r'(?<=\d)\s+(?=[\d\-\.])', ', ', s)
        # Normalize remaining spaces to commas (handles mixed formatting)
        s = s.replace(' ', ', ')
        s = re.sub(r',,', ',', s)

        if not s.startswith('['):
            s = '[' + s
        if not s.endswith(']'):
            s = s + ']'
        # If there are multiple top-level lists like "[...], [...]" ensure an outer wrapper
        if s.count('[') > 1 and not s.strip().startswith('[['):
            s = '[' + s + ']'

        try:
            result = ast.literal_eval(s)
            return self._to_list_recursive(result)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Unable to parse list: {value}") from e

    def _to_list_recursive(self, obj: Any) -> Any:
        """Recursively convert tuples to lists for consistent downstream handling."""
        if isinstance(obj, tuple):
            return [self._to_list_recursive(x) for x in obj]
        if isinstance(obj, list):
            return [self._to_list_recursive(x) for x in obj]
        return obj

    def _find_float_bin_index(self, selected_value: float, bin_edges: List[float]) -> float:
        """Find the floating-point bin index for a selected value given bin edges for interpolation."""
        if not isinstance(bin_edges, list) or not bin_edges:
            raise ValueError("bin_edges must be a non-empty list.")

        if len(bin_edges) == 1:
            if selected_value == bin_edges[0]:
                return 0.0
            raise ValueError(f"Selected value {selected_value} does not match the only bin edge {bin_edges[0]}")

        bin_edges_arr = np.array(sorted(bin_edges))

        if not (bin_edges_arr[0] <= selected_value <= bin_edges_arr[-1]):
            raise ValueError(f"Selected value {selected_value} is outside bin range [{bin_edges_arr[0]}, {bin_edges_arr[-1]}]")

        pos = np.searchsorted(bin_edges_arr, selected_value, side='right') - 1
        pos = max(0, min(pos, len(bin_edges_arr) - 2))

        lower_bound = bin_edges_arr[pos]
        upper_bound = bin_edges_arr[pos + 1]

        if np.isclose(upper_bound, lower_bound):
            fraction = 0.0
        else:
            fraction = (selected_value - lower_bound) / (upper_bound - lower_bound)
        
        return pos + fraction

    def _interpolate_1d(self, arr: List[float], f_i: float) -> float:
        """Performs linear interpolation on a 1D array."""
        num_elements = len(arr)
        if num_elements == 0:
            return np.nan

        i_low = int(f_i)
        i_low = max(0, min(i_low, num_elements - 1))
        i_high = min(i_low + 1, num_elements - 1)
        
        di = f_i - i_low

        v_low = arr[i_low]
        v_high = arr[i_high]
        
        return v_low * (1 - di) + v_high * di

    def _interpolate_2d(self, grid: List[List[float]], f_i: float, f_j: float) -> float:
        """Performs bilinear interpolation on a 2D grid."""
        num_rows = len(grid)
        if num_rows == 0 or not isinstance(grid[0], list):
            return np.nan
        num_cols = len(grid[0])
        if num_cols == 0:
            return np.nan

        i_low = int(f_i)
        j_low = int(f_j)
        
        i_low = max(0, min(i_low, num_rows - 1))
        j_low = max(0, min(j_low, num_cols - 1))
        
        i_high = min(i_low + 1, num_rows - 1)
        j_high = min(j_low + 1, num_cols - 1)
        
        di = f_i - i_low
        dj = f_j - j_low

        v00 = grid[i_low][j_low]
        v10 = grid[i_high][j_low]
        v01 = grid[i_low][j_high]
        v11 = grid[i_high][j_high]

        v_j_low = v00 * (1 - di) + v10 * di
        v_j_high = v01 * (1 - di) + v11 * di
        
        return v_j_low * (1 - dj) + v_j_high * dj

    def _filter_invalid_rows(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Drop rows with empty/unparseable multi-value or IDX data and log warnings.
        
        This checks configured multi-value variables that use IDX-based selection.
        Rows failing parsing or basic validation will be removed with a WARNING.
        
        Returns a tuple of (cleaned_dataframe, invalid_rows_dataframe).
        """
        invalid_indices = set()

        for var_name, var_config in self.variable_configs.items():
            variable_config = var_config.get('variable', var_config)
            if variable_config.get('type') != 'multi':
                continue
            if variable_config.get('selection_strategy') not in ['index', 'value']:
                continue

            idx1_col = f"{var_name}_IDX1"
            idx2_col = f"{var_name}_IDX2"
            # For index strategy we don't need IDX columns at all
            if variable_config.get('selection_strategy') != 'index':
                if idx1_col not in data.columns or idx2_col not in data.columns:
                    # Column presence is validated elsewhere; skip here to avoid duplication
                    continue

            idx1_value = variable_config['idx1_value']
            idx2_value = variable_config['idx2_value']
            strategy = variable_config['selection_strategy']

            if strategy == 'index':
                # Only the variable column is needed for index strategy
                for row_idx, row in data[[var_name]].iterrows():
                    try:
                        parsed_var = self._robust_parse_list(row[var_name])
                        # Determine dimensionality
                        if not isinstance(parsed_var, list):
                            raise ValueError(f"{var_name} is not a list or list-of-lists")
                        is_2d = len(parsed_var) > 0 and isinstance(parsed_var[0], list)
                        i_bin = int(idx1_value)
                        j_bin = int(idx2_value) if is_2d else 0
                        # Access value directly; if out-of-range, an exception will be raised
                        _ = parsed_var[i_bin][j_bin] if is_2d else parsed_var[i_bin]
                    except Exception as e:
                        invalid_indices.add(row_idx)
                        groupby_columns = self.config.get('groupby_columns', [])
                        group_context = {col: data.at[row_idx, col] for col in groupby_columns if col in data.columns}
                        self.logger.warning(
                            f"Dropping row {row_idx} for variable {var_name} due to invalid index access: {e}. Context: {group_context}"
                        )
            else:
                # value strategy requires idx arrays to map values to bins
                for row_idx, row in data[[var_name, idx1_col, idx2_col]].iterrows():
                    try:
                        parsed_var = self._robust_parse_list(row[var_name])
                        parsed_idx1 = self._robust_parse_list(row[idx1_col])
                        parsed_idx2 = self._robust_parse_list(row[idx2_col])

                        if not isinstance(parsed_var, list):
                            raise ValueError(f"{var_name} is not a list or list-of-lists")
                        is_2d = len(parsed_var) > 0 and isinstance(parsed_var[0], list)

                        # Ensure selected values fall within bin ranges
                        _ = self._find_float_bin_index(float(idx1_value), parsed_idx1)
                        if is_2d:
                            _ = self._find_float_bin_index(float(idx2_value), parsed_idx2)
                    except Exception as e:
                        invalid_indices.add(row_idx)
                        groupby_columns = self.config.get('groupby_columns', [])
                        group_context = {col: data.at[row_idx, col] for col in groupby_columns if col in data.columns}
                        self.logger.warning(
                            f"Dropping row {row_idx} for variable {var_name} due to invalid data: {e}. Context: {group_context}"
                        )

        if invalid_indices:
            invalid_rows_df = data.loc[sorted(list(invalid_indices))].copy()
            cleaned_df = data.drop(index=sorted(list(invalid_indices))).reset_index(drop=True)
            self.logger.warning(
                f"Dropped {len(invalid_indices)} row(s) with invalid multi-value/IDX data. "
                "These rows have been saved to a separate file."
            )
            return cleaned_df, invalid_rows_df
            
        return data, pd.DataFrame()

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
                return self._to_list_recursive(s)
            
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
                parsed = ast.literal_eval(s)
                return self._to_list_recursive(parsed)
            except (ValueError, SyntaxError):
                raise ValueError(f"Unable to parse list: {s}")

        # Parse the multi-value data if it's string representation
        if isinstance(var_data[0], str):
            try:
                var_data = [_parse_list(x) for x in var_data]
            except (ValueError, SyntaxError):
                raise ValueError(f"Unable to parse list/list-of-lists for variable {var_name}")
        
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
                # Allow empty second-dimension in case of 1D variables
                idx2_data = [[] for _ in range(len(idx1_data))]
        
        # Process each row to extract the correct value
        results = []
        idx1_results = []
        idx2_results = []
        
        for i in range(len(var_data)):
            # Determine dimensionality for this row (lists or tuples normalized to lists already)
            is_2d = isinstance(var_data[i], list) and (len(var_data[i]) > 0 and isinstance(var_data[i][0], list))
            try:
                if strategy == 'index':
                    # Use direct indices
                    i_bin = int(idx1_value)
                    # If 1D array, use 0 for second index
                    j_bin = int(idx2_value) if is_2d else 0
                    
                    # Get the actual idx1/idx2 values at these indices
                    actual_idx1 = idx1_data[i][i_bin] if isinstance(idx1_data[i], list) else idx1_data[i]
                    actual_idx2 = (idx2_data[i][j_bin] if (is_2d and isinstance(idx2_data[i], list) and len(idx2_data[i])>j_bin)
                                else (idx2_data[i] if is_2d else None))
                    
                    if is_2d:
                        value = var_data[i][i_bin][j_bin]
                    else:
                        value = var_data[i][i_bin]
                
                else:  # value strategy with interpolation
                    actual_idx1 = float(idx1_value)
                    actual_idx2 = float(idx2_value) if is_2d else None
                    
                    if is_2d:
                        f_i = self._find_float_bin_index(actual_idx1, idx1_data[i])
                        f_j = self._find_float_bin_index(actual_idx2, idx2_data[i])
                        value = self._interpolate_2d(var_data[i], f_i, f_j)
                    else: # 1D
                        f_i = self._find_float_bin_index(actual_idx1, idx1_data[i])
                        value = self._interpolate_1d(var_data[i], f_i)
                
                results.append(value)
                idx1_results.append(actual_idx1)
                # For 1D case, store placeholder for idx2
                idx2_results.append(actual_idx2 if is_2d else np.nan)
            except Exception as e:
                raise ValueError(f"Unable to extract value for variable {var_name}, row {i}: {e}")
        
        return np.array(results), np.array(idx1_results), np.array(idx2_results)
    
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
        data.rename(columns={'DYNAMIN_IDX2': 'DYNAMIC_IDX2'}, inplace=True)
        self.validate_data(data)
        # Drop invalid rows upfront to avoid parse errors later
        data, self.invalid_rows = self._filter_invalid_rows(data)
        
        # Check if groupby processing is enabled
        groupby_columns = self.config.get('groupby_columns')
        if groupby_columns:
            # Identify passthrough columns
            variable_names = list(self.variable_configs.keys())
            self.passthrough_columns = [
                col for col in data.columns 
                if col not in groupby_columns and col not in variable_names and not col.endswith(('_IDX1', '_IDX2'))
            ]
            if self.passthrough_columns:
                self.logger.info(f"Identified passthrough columns: {self.passthrough_columns}")
            
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
        passthrough_data = []
        
        # Initialize combined idx_values storage
        combined_idx_values = {}
        
        for group_key, group_data in grouped:
            self.logger.info(f"Processing group: {group_key} ({len(group_data)} rows)")
            group_dict = dict(zip(groupby_columns, group_key if isinstance(group_key, tuple) else [group_key]))
            try:
                # Process this group
                group_features, group_tolerances = self._process_single_group(group_data)
            except Exception as e:
                self.logger.warning(f"Skipping group {group_dict} due to processing error: {e}")
                continue

            # Store group information
            group_len = len(group_features)
            group_info.extend([group_dict] * group_len)
            all_processed_features.append(group_features)
            all_tolerance_values.append(group_tolerances)
            
            # Store passthrough data
            if self.passthrough_columns:
                passthrough_data.append(group_data[self.passthrough_columns])
            
            # Collect idx_values from this group safely
            if hasattr(self, 'idx_values') and self.idx_values:
                # Build set of expected multi-value variables
                multi_vars = [
                    name for name, cfg in self.variable_configs.items()
                    if (cfg.get('variable', cfg).get('type') == 'multi' and
                        cfg.get('variable', cfg).get('selection_strategy') in ['index', 'value'])
                ]
                for var_name in multi_vars:
                    idx_data = self.idx_values.get(var_name)
                    if var_name not in combined_idx_values:
                        combined_idx_values[var_name] = {'idx1': [], 'idx2': []}
                    try:
                        if not idx_data:
                            raise ValueError('missing idx data for variable')
                        idx1_vals = np.asarray(idx_data['idx1'])
                        idx2_vals = np.asarray(idx_data['idx2'])
                        if idx1_vals.ndim != 1 or idx2_vals.ndim != 1:
                            raise ValueError(f"inhomogeneous idx shapes: idx1.ndim={idx1_vals.ndim}, idx2.ndim={idx2_vals.ndim}")
                        if len(idx1_vals) != group_len or len(idx2_vals) != group_len:
                            raise ValueError(f"idx length mismatch with group size {group_len}")
                        combined_idx_values[var_name]['idx1'].extend(idx1_vals.tolist())
                        combined_idx_values[var_name]['idx2'].extend(idx2_vals.tolist())
                    except Exception as e:
                        self.logger.warning(
                            f"Filling idx placeholders for group {group_dict}, variable {var_name} due to issue: {e}"
                        )
                        # Keep alignment by appending NaNs of group length
                        combined_idx_values[var_name]['idx1'].extend([np.nan] * group_len)
                        combined_idx_values[var_name]['idx2'].extend([np.nan] * group_len)
        
        # Combine all groups
        if all_processed_features:
            processed_array = np.vstack(all_processed_features)
            tolerance_array = np.vstack(all_tolerance_values)
            if passthrough_data:
                self.passthrough_data = pd.concat(passthrough_data).reset_index(drop=True)
        else:
            raise ValueError("No data processed from any group")
        
        # Convert combined idx_values to numpy arrays
        if combined_idx_values:
            sanitized_idx_values = {}
            for var_name in combined_idx_values:
                try:
                    idx1_arr = np.array(combined_idx_values[var_name]['idx1'], dtype=float)
                    idx2_arr = np.array(combined_idx_values[var_name]['idx2'], dtype=float)
                    # Sanity check alignment with processed rows
                    if idx1_arr.shape[0] != processed_array.shape[0] or idx2_arr.shape[0] != processed_array.shape[0]:
                        self.logger.warning(
                            f"Variable {var_name} idx lengths ({idx1_arr.shape[0]}, {idx2_arr.shape[0]}) do not match processed rows ({processed_array.shape[0]}). "
                            "Truncating/padding with NaN to align."
                        )
                        target_len = processed_array.shape[0]
                        def _pad_or_truncate(arr):
                            if arr.shape[0] > target_len:
                                return arr[:target_len]
                            if arr.shape[0] < target_len:
                                pad = np.full(target_len - arr.shape[0], np.nan)
                                return np.concatenate([arr, pad])
                            return arr
                        idx1_arr = _pad_or_truncate(idx1_arr)
                        idx2_arr = _pad_or_truncate(idx2_arr)
                    sanitized_idx_values[var_name] = {'idx1': idx1_arr, 'idx2': idx2_arr}
                except Exception as e:
                    self.logger.warning(
                        f"Skipping aggregated idx values for variable {var_name} due to inhomogeneous shapes: {e}"
                    )
            if sanitized_idx_values:
                self.idx_values = sanitized_idx_values
        
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