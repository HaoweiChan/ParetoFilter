"""
Pareto frontier selection module using Platypus.

Handles multi-objective optimization with tolerance-based epsilon configuration.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from platypus import EpsilonBoxArchive, Problem, Solution


class ParetoSelector:
    """Handles Pareto frontier selection using Platypus EpsilonBoxArchive."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize selector with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Handle both old and new config structure
        if 'data' in config:
            self.variable_configs = config['data']
        else:
            self.variable_configs = config['variables']
        self.feature_names = list(self.variable_configs.keys())
    
    def select(self, data: np.ndarray, tolerances: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """Select Pareto frontier candidates using Platypus EpsilonBoxArchive."""
        if data.shape[0] == 0:
            raise ValueError("No data provided for Pareto selection")
        
        # Align incoming data/tolerances with variables declared in config
        data, tolerances = self._align_data_to_config(data, tolerances)
        
        self.logger.info(f"Starting Pareto selection with {data.shape[0]} candidates and {data.shape[1]} objectives")
        
        # Get directions and tolerances
        directions = [var_config['objective'] for var_config in self.variable_configs.values()]
        
        # Normalize tolerances to 1D array
        if tolerances.ndim == 2:
            tol_array = tolerances[0, :]
        else:
            tol_array = tolerances
        
        self.logger.debug(f"Tolerance values: {tol_array.tolist()}")
        
        # Create Platypus Problem
        problem = Problem(1, len(self.feature_names))  # 1 variable (for index), n objectives
        
        # Set problem directions (minimize = 1, maximize = -1)
        for i, direction in enumerate(directions):
            if direction == 'minimize':
                problem.directions[i] = Problem.MINIMIZE
            else:  # maximize
                problem.directions[i] = Problem.MAXIMIZE
        
        # Create EpsilonBoxArchive with epsilon values
        archive = EpsilonBoxArchive(tol_array.tolist())
        
        # Add all candidates to the archive
        for i, candidate in enumerate(data):
            solution = Solution(problem)
            solution.objectives[:] = candidate.tolist()
            solution.variables[:] = [float(i)]  # Store original index as variable
            
            # Try to add to archive
            archive.add(solution)
        
        # Extract Pareto frontier candidates
        pareto_indices = []
        pareto_values = []
        
        for solution in archive:
            original_index = int(solution.variables[0])
            pareto_indices.append(original_index)
            pareto_values.append(solution.objectives)
        
        pareto_values = np.array(pareto_values)
        
        self.logger.info(f"Pareto frontier selected: {len(pareto_indices)} candidates from {data.shape[0]} total")
        
        return pareto_indices, pareto_values

    def _align_data_to_config(self, data: np.ndarray, tolerances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure data and tolerances match the number of configured variables.

        - If there are extra columns in `data`/`tolerances`, slice to the first N (in order of preprocessing)
        - If there are fewer columns than expected, raise an error
        """
        expected_n = len(self.variable_configs)
        actual_n = data.shape[1]
        if actual_n < expected_n:
            raise ValueError(
                f"Data has {actual_n} features but config has {expected_n} variables"
            )
        if actual_n > expected_n:
            self.logger.warning(
                f"Data has {actual_n} features but config declares {expected_n}. "
                f"Using the first {expected_n} features in preprocessing order."
            )
            data = data[:, :expected_n]
            if tolerances.ndim == 2 and tolerances.shape[1] >= expected_n:
                tolerances = tolerances[:, :expected_n]
            elif tolerances.ndim == 1 and tolerances.shape[0] >= expected_n:
                tolerances = tolerances[:expected_n]
        return data, tolerances
    
    def get_selection_metadata(self, data: np.ndarray, tolerances: np.ndarray, 
                             pareto_indices: List[int]) -> Dict[str, Any]:
        """Get metadata about the Pareto selection process."""
        metadata = {
            'total_candidates': data.shape[0],
            'pareto_candidates': len(pareto_indices),
            'reduction_ratio': len(pareto_indices) / data.shape[0],
            'epsilon_values': tolerances[0, :].tolist() if tolerances.ndim == 2 else tolerances.tolist(),
            'feature_names': self.feature_names,
            'objectives': [var_config['objective'] for var_config in self.variable_configs.values()]
        }
        
        return metadata
    
    def select_grouped(self, data: np.ndarray, tolerances: np.ndarray, 
                      group_info: List[Dict[str, Any]]) -> Tuple[List[int], np.ndarray, List[Dict[str, Any]]]:
        """Select Pareto frontier candidates with grouped processing."""
        if len(group_info) != data.shape[0]:
            raise ValueError(f"Group info length ({len(group_info)}) doesn't match data rows ({data.shape[0]})")
        
        # Group data by group_info
        groups = {}
        for i, group_dict in enumerate(group_info):
            group_key = tuple(sorted(group_dict.items()))
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(i)
        
        self.logger.info(f"Processing {len(groups)} groups for Pareto selection")
        
        all_pareto_indices = []
        all_pareto_values = []
        group_metadata = []
        
        for group_key, indices in groups.items():
            group_dict = dict(group_key)
            self.logger.info(f"Processing group: {group_dict} ({len(indices)} candidates)")
            
            # Extract group data
            group_data = data[indices]
            group_tolerances = tolerances[indices] if tolerances.ndim == 2 else tolerances
            
            # Perform Pareto selection for this group
            group_pareto_indices, group_pareto_values = self.select(group_data, group_tolerances)
            
            # Map back to original indices
            original_pareto_indices = [indices[i] for i in group_pareto_indices]
            
            # Store results
            all_pareto_indices.extend(original_pareto_indices)
            all_pareto_values.extend(group_pareto_values)
            
            # Store group metadata
            group_meta = {
                'group': group_dict,
                'total_candidates': len(indices),
                'pareto_candidates': len(group_pareto_indices),
                'reduction_ratio': len(group_pareto_indices) / len(indices),
                'pareto_indices': original_pareto_indices
            }
            group_metadata.append(group_meta)
        
        # Combine all Pareto values
        combined_pareto_values = np.array(all_pareto_values) if all_pareto_values else np.array([])
        
        self.logger.info(f"Grouped Pareto selection complete: {len(all_pareto_indices)} total candidates selected from {len(groups)} groups")
        
        return all_pareto_indices, combined_pareto_values, group_metadata 