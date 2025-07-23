"""
Pareto frontier selection module using Platypus.

Handles multi-objective optimization with tolerance-based epsilon configuration.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, List
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
        
        if data.shape[1] != len(self.variable_configs):
            raise ValueError(f"Data has {data.shape[1]} features but config has {len(self.variable_configs)} variables")
        
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