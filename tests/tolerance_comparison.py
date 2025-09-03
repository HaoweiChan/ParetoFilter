#!/usr/bin/env python3
"""
Tolerance comparison script for debugging Pareto selection behavior.

This script runs the same dataset with different tolerance values to compare results.
"""

import argparse
import json
import logging
import os
import sys
import yaml
from copy import deepcopy
from pathlib import Path

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from preprocess import DataPreprocessor
from pareto import ParetoSelector
from utils import setup_logging


def compare_tolerances(config_path: str, tolerance_values: list, target_variables: list = None):
    """Compare Pareto selection results with different tolerance values."""
    
    # Load base configuration
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            base_config = yaml.safe_load(f)
        else:
            base_config = json.load(f)
    
    results = {}
    
    for tolerance_val in tolerance_values:
        print(f"\n{'='*60}")
        print(f"TESTING TOLERANCE: {tolerance_val}")
        print(f"{'='*60}")
        
        # Create modified config
        test_config = deepcopy(base_config)
        
        # Update tolerance for specified variables (or all if none specified)
        variables_to_update = target_variables or list(test_config['data'].keys())
        
        for var_name in variables_to_update:
            if var_name in test_config['data'] and 'tolerance' in test_config['data'][var_name]:
                if test_config['data'][var_name]['tolerance']['type'] == 'relative':
                    test_config['data'][var_name]['tolerance']['value'] = tolerance_val
                    print(f"Updated {var_name} tolerance to: {tolerance_val}")
        
        # Initialize preprocessor and selector
        preprocessor = DataPreprocessor(test_config)
        selector = ParetoSelector(test_config)
        
        # Load and process data
        input_path = test_config['run']['input_dir']
        processed_data, tolerance_data = preprocessor.process(input_path)
        
        # Perform Pareto selection
        if hasattr(preprocessor, 'group_info'):
            pareto_indices, pareto_values, group_metadata = selector.select_grouped(
                processed_data, tolerance_data, preprocessor.group_info
            )
        else:
            pareto_indices, pareto_values = selector.select(processed_data, tolerance_data)
            group_metadata = None
        
        # Store results
        results[tolerance_val] = {
            'total_candidates': processed_data.shape[0],
            'pareto_candidates': len(pareto_indices),
            'selection_ratio': len(pareto_indices) / processed_data.shape[0],
            'group_metadata': group_metadata
        }
        
        print(f"\nRESULT SUMMARY for tolerance {tolerance_val}:")
        print(f"  Total candidates: {results[tolerance_val]['total_candidates']}")
        print(f"  Pareto selected: {results[tolerance_val]['pareto_candidates']}")
        print(f"  Selection ratio: {results[tolerance_val]['selection_ratio']:.3f}")
        
        if group_metadata:
            print(f"  Group breakdown:")
            for meta in group_metadata:
                group_ratio = meta['reduction_ratio']
                print(f"    {meta['group']}: {meta['pareto_candidates']}/{meta['total_candidates']} = {group_ratio:.3f}")
    
    # Final comparison
    print(f"\n{'='*60}")
    print("TOLERANCE COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Tolerance':<12} {'Total':<8} {'Selected':<10} {'Ratio':<8} {'Groups':<15}")
    print(f"{'-'*60}")
    
    for tol_val in tolerance_values:
        result = results[tol_val]
        groups_info = ""
        if result['group_metadata']:
            group_counts = [f"{meta['pareto_candidates']}" for meta in result['group_metadata']]
            groups_info = f"[{', '.join(group_counts)}]"
        
        print(f"{tol_val:<12.3f} {result['total_candidates']:<8} {result['pareto_candidates']:<10} "
              f"{result['selection_ratio']:<8.3f} {groups_info:<15}")
    
    # Highlight unexpected behavior
    print(f"\nUNEXPECTED BEHAVIOR ANALYSIS:")
    for i in range(1, len(tolerance_values)):
        curr_tol = tolerance_values[i]
        prev_tol = tolerance_values[i-1]
        curr_selected = results[curr_tol]['pareto_candidates']
        prev_selected = results[prev_tol]['pareto_candidates']
        
        if curr_tol > prev_tol and curr_selected > prev_selected:
            print(f"⚠️  COUNTERINTUITIVE: Tolerance {curr_tol} > {prev_tol} but selected {curr_selected} > {prev_selected}")
        elif curr_tol < prev_tol and curr_selected < prev_selected:
            print(f"✅ Expected: Tolerance {curr_tol} < {prev_tol} and selected {curr_selected} < {prev_selected}")


def main():
    parser = argparse.ArgumentParser(description="Compare Pareto selection with different tolerance values")
    parser.add_argument('--config', '-c', type=str, required=True, help="Path to configuration file")
    parser.add_argument('--tolerances', '-t', nargs='+', type=float, 
                       default=[0.05, 0.1, 0.15, 0.2, 0.25],
                       help="List of tolerance values to test (default: 0.05 0.1 0.15 0.2 0.25)")
    parser.add_argument('--variables', '-v', nargs='+', type=str,
                       help="Specific variables to modify tolerance for (default: all relative tolerance variables)")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(verbose=True, log_file=None)
    logging.getLogger().setLevel(level)
    
    print(f"Comparing tolerance values: {args.tolerances}")
    if args.variables:
        print(f"For variables: {args.variables}")
    else:
        print("For all relative tolerance variables")
    
    compare_tolerances(args.config, args.tolerances, args.variables)


if __name__ == "__main__":
    main()
