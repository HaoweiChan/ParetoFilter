#!/usr/bin/env python3
"""
Main entry point for Pareto Frontier Selection Tool.

Provides CLI interface for data processing, Pareto selection, and visualization.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import yaml
import numpy as np

from pareto import ParetoSelector
from visualization import Dashboard
from preprocess import DataPreprocessor
from utils import load_config, validate_config, setup_logging, save_results, create_run_directory


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pareto Frontier Selection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config runs/sample_run_1/config.yaml
  python main.py --config runs/sample_run_1/config.yaml --run-dir custom_run_dir
  python main.py --config runs/sample_run_1/config.yaml --processed-output processed.csv
  python main.py --dashboard-only --config runs/sample_run_1/config.yaml
        """
    )
    
    # Required arguments
    parser.add_argument('--config', required=True, help='Path to configuration file (YAML or JSON)')
    
    # Optional arguments
    parser.add_argument('--processed-output', help='Output file for processed data')
    parser.add_argument('--run-dir', help='Run directory for outputs (defaults to config file directory)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--dashboard-only', action='store_true', help='Only launch dashboard using vis_data.npz and config (skip processing)')
    parser.add_argument('--bind-all', action='store_true', help='Bind dashboard to all network interfaces (0.0.0.0), making it accessible externally.')
    
    args = parser.parse_args()
    
    # Load and validate configuration first
    logger = logging.getLogger(__name__)
    setup_logging(args.verbose)
    
    try:
        if args.dashboard_only:
            # Load config
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            # Find vis_data.npz in same directory as config
            config_dir = os.path.dirname(args.config)
            vis_data_path = os.path.join(config_dir, 'vis_data.npz')
            if not os.path.exists(vis_data_path):
                logger.error(f"vis_data.npz not found in {config_dir}. Run the full pipeline first.")
                sys.exit(1)
            npz = np.load(vis_data_path)
            data = npz['data']
            pareto_indices = npz['pareto_indices']
            tolerances = npz['tolerances']
            dashboard = Dashboard(config, data, pareto_indices, tolerances)
            
            host = '0.0.0.0' if args.bind_all else config.get('visualization', {}).get('dashboard_host', 'localhost')
            port = config.get('visualization', {}).get('dashboard_port', 8050)
            
            dashboard.run(host=host, port=port)
            return
        
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        validate_config(config)
        logger.info("Configuration validated successfully")
        
        # Get configuration settings
        run_config = config.get('run', {})
        viz_config = config.get('visualization', {})
        
        # Determine input file path
        input_file = run_config.get('input_file')
        if not input_file:
            raise ValueError("Configuration must specify 'input_file' in the 'run' section")
        
        # Determine config directory and data file path
        config_path = Path(args.config)
        config_dir = config_path.parent
        data_path = config_dir / input_file
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Determine run directory (default to config directory if not specified)
        if args.run_dir:
            run_dir = Path(args.run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "viz").mkdir(exist_ok=True)
            log_file = run_dir / "logs.txt"
        else:
            run_dir = config_dir
            log_file = run_dir / "logs.txt"
        
        # Setup file logging
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Load and preprocess data
        logger.info(f"Loading data from {data_path}")
        preprocessor = DataPreprocessor(config)
        processed_data, tolerances = preprocessor.process(str(data_path))
        logger.info(f"Data processed: {processed_data.shape[0]} candidates, {processed_data.shape[1]} features")
        
        # Get run configuration settings
        use_input_prefix = run_config.get('use_input_prefix', False)
        output_processed_data = run_config.get('output_processed_data', False)
        
        # Generate output filenames
        input_stem = Path(input_file).stem if use_input_prefix else None
        
        # Save processed data if requested
        if args.processed_output or output_processed_data:
            if args.processed_output:
                output_path = args.processed_output
            else:
                # Auto-generate filename
                prefix = f"{input_stem}_" if input_stem else ""
                output_path = f"{prefix}processed_data.csv"
            
            if run_dir and not Path(output_path).is_absolute():
                output_path = run_dir / output_path
            logger.info(f"Saving processed data to {output_path}")
            save_results(processed_data, output_path, 'csv')
        
        # Perform Pareto selection
        logger.info("Performing Pareto frontier selection")
        selector = ParetoSelector(config)
        pareto_indices, pareto_values = selector.select(processed_data, tolerances)
        logger.info(f"Pareto frontier selected: {len(pareto_indices)} candidates")
        
        # Save vis_data.npz for dashboard-only mode
        vis_data_path = run_dir / "vis_data.npz"
        np.savez(vis_data_path, data=processed_data, pareto_indices=pareto_indices, tolerances=tolerances)
        logger.info(f"Saved visualization data for dashboard-only mode to {vis_data_path}")
        
        # Auto-save Pareto results
        prefix = f"{input_stem}_" if input_stem else ""
        pareto_output_path = run_dir / f"{prefix}pareto_results.csv"
        logger.info(f"Saving Pareto frontier results to {pareto_output_path}")
        
        # Create results dictionary
        variables_section = config.get('data', config.get('variables', {}))
        results = {
            'pareto_indices': pareto_indices,
            'pareto_values': pareto_values.tolist(),
            'total_candidates': processed_data.shape[0],
            'pareto_candidates': len(pareto_indices),
            'feature_names': list(variables_section.keys())
        }
        save_results(results, pareto_output_path, 'csv')
        
        # Launch dashboard if requested
        generate_viz = viz_config.get('generate_visualization', False)
        if generate_viz:
            dashboard_host = '0.0.0.0' if args.bind_all else viz_config.get('dashboard_host', 'localhost')
            dashboard_port = viz_config.get('dashboard_port', 8050)
            logger.info(f"Launching dashboard on {dashboard_host}:{dashboard_port}")
            dashboard = Dashboard(config, processed_data, pareto_indices, tolerances)
            dashboard.run(host=dashboard_host, port=dashboard_port)
        else:
            logger.info("Visualization skipped")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()