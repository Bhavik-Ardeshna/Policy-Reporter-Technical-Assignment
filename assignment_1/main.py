#!/usr/bin/env python3
"""
Main script for finding the best threshold with minimum recall.

Example usage:
    python main.py --data_path data/model_metrics.csv --min_recall 0.9 --optimization_metric precision
    python main.py --chunk_size 1000  # For processing large datasets in chunks
    python main.py --run_tests  # Run unit tests
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

from assignment_1.core.logger import setup_logger
from assignment_1.core.threshold_evaluator import find_best_threshold_with_min_recall
from assignment_1.core.exception import (
    ThresholdEvaluatorError,
    FileOperationError,
    DataValidationError,
    ThresholdNotFoundError,
    InvalidParameterError
)
from assignment_1.core.constant import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_PATH,
    MIN_RECALL_THRESHOLD,
    DEFAULT_OPTIMIZATION_METRIC
)


def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Find the best threshold with minimum recall requirement"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the metrics data file (default: {DEFAULT_DATA_PATH})"
    )
    
    parser.add_argument(
        "--min_recall",
        type=float,
        default=MIN_RECALL_THRESHOLD,
        help=f"Minimum acceptable recall value (default: {MIN_RECALL_THRESHOLD})"
    )
    
    parser.add_argument(
        "--optimization_metric",
        type=str,
        default=DEFAULT_OPTIMIZATION_METRIC,
        choices=["precision", "f1", "specificity"],
        help=f"Metric to optimize (default: {DEFAULT_OPTIMIZATION_METRIC})"
    )
    
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Number of rows to process at once for large files (default: None = load all at once)"
    )
    
    parser.add_argument(
        "--run_tests",
        action="store_true",
        help="Run unit tests instead of normal execution"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,  # We'll set the default in the main function to allow for timestamp
        help=f"Output file path for results (default: {DEFAULT_OUTPUT_PATH})"
    )
    
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save results to a file"
    )
    
    return parser.parse_args()


def run_tests():
    """Run unit tests for the application."""
    import unittest
    from assignment_1.test.test_threshold_evaluator import TestThresholdEvaluator
    
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestThresholdEvaluator)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


def get_default_output_path():
    """Get default output path with timestamp.
    
    Returns:
        Default output path with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.dirname(DEFAULT_OUTPUT_PATH)
    base_name = os.path.basename(DEFAULT_OUTPUT_PATH)
    name, ext = os.path.splitext(base_name)
    
    return os.path.join(output_dir, f"{name}_{timestamp}{ext}")


def save_results(output_path, result, logger):
    """Save results to the specified output path.
    
    Args:
        output_path: Path to save results to
        result: Results to save
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        print(f"Results saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error writing to output file: {str(e)}")
        print(f"Error writing to output file: {str(e)}")
        return False


def main():
    """Main function to run the application."""
    logger = setup_logger("main")
    args = parse_arguments()
    
    # Run tests if requested
    if args.run_tests:
        return run_tests()
    
    # Set default output path if none specified and not explicitly disabled
    if not args.no_save and args.output is None:
        args.output = get_default_output_path()
    
    # Ensure data file exists - this check is now redundant with our custom exceptions,
    # but keeping it for early user feedback
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return 1
    
    try:
        # Find best threshold
        logger.info(f"Finding best threshold with min_recall={args.min_recall}")
        if args.chunk_size:
            logger.info(f"Processing data in chunks of {args.chunk_size} rows")
            
        best_threshold = find_best_threshold_with_min_recall(
            data_path=args.data_path,
            min_recall=args.min_recall,
            optimization_metric=args.optimization_metric,
            chunk_size=args.chunk_size
        )
        
        if best_threshold is None:
            logger.warning(f"No threshold found with recall >= {args.min_recall}")
            print(f"No threshold found with recall >= {args.min_recall}")
            return 1
        
        # Output result
        result = {
            "best_threshold": best_threshold,
            "parameters": {
                "data_path": args.data_path,
                "min_recall": args.min_recall,
                "optimization_metric": args.optimization_metric,
                "chunk_size": args.chunk_size
            }
        }
        
        print(f"Best threshold: {best_threshold}")
        
        # Write to output file if specified and not disabled
        if args.output and not args.no_save:
            if not save_results(args.output, result, logger):
                return 1
        
        return 0
        
    except FileOperationError as e:
        logger.error(f"File error: {e.message}")
        print(f"File error: {e.message}")
        if e.file_path:
            print(f"File path: {e.file_path}")
        return 1
        
    except DataValidationError as e:
        logger.error(f"Data validation error: {e.message}")
        print(f"Data validation error: {e.message}")
        if e.invalid_columns:
            print(f"Invalid/missing columns: {', '.join(e.invalid_columns)}")
        return 1
        
    except InvalidParameterError as e:
        logger.error(f"Invalid parameter: {e.message}")
        print(f"Invalid parameter: {e.message}")
        if e.parameter_name and e.valid_values:
            print(f"Parameter '{e.parameter_name}' must be one of: {e.valid_values}")
        return 1
        
    except ThresholdNotFoundError as e:
        # This should be handled by the find_best_threshold_with_min_recall function
        # which returns None when no threshold meets the criteria
        # But keeping this just in case it bubbles up
        logger.warning(f"Threshold not found: {e.message}")
        if e.max_available_recall:
            logger.info(f"Maximum available recall: {e.max_available_recall:.4f}")
        print(f"No threshold found with recall >= {args.min_recall}")
        return 1
        
    except ThresholdEvaluatorError as e:
        # Generic handler for all other threshold evaluator errors
        logger.error(f"Error: {e.message}")
        print(f"Error: {e.message}")
        return 1
        
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error: {str(e)}")
        print(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 