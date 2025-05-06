#!/usr/bin/env python3
"""Main entry point for the ModThree FSM application.
"""

import argparse
import json
import logging
import sys
from typing import List, Dict, Any, Union

from assignment_2.core.logger import setup_logger
from assignment_2.core.mod_three_fsm import ModThreeFSM
from assignment_2.core.exception import FSMError, InvalidInputError

# Set up logger
logger = setup_logger("main")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Calculate the remainder when a binary number is divided by 3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    parser.add_argument(
        "-b", "--binary",
        type=str,
        help="Binary string to process"
    )
    
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Path to file containing binary strings (one per line)"
    )
    
    # Output options
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to output file (JSON format)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output except errors"
    )
    
    return parser.parse_args()


def configure_logging(verbose: bool, quiet: bool) -> None:
    """Configure logging based on command-line options.
    
    Args:
        verbose: Enable verbose output
        quiet: Suppress all output except errors
    """
    root_logger = logging.getLogger()
    
    if quiet:
        root_logger.setLevel(logging.ERROR)
    elif verbose:
        root_logger.setLevel(logging.DEBUG)
    else:
        root_logger.setLevel(logging.INFO)


def read_binary_strings_from_file(file_path: str) -> List[str]:
    """Read binary strings from a file, one per line.
    
    Args:
        file_path: Path to the file
    
    Returns:
        List of binary strings
        
    Raises:
        FileNotFoundError: If the file does not exist
        PermissionError: If the file cannot be read
    """
    try:
        with open(file_path, "r") as file:
            # Strip whitespace and filter out empty lines
            binary_strings = [line.strip() for line in file if line.strip()]
        
        logger.info(f"Read {len(binary_strings)} binary strings from {file_path}")
        return binary_strings
    
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise


def write_results_to_file(results: List[Dict[str, Any]], file_path: str) -> None:
    """Write processing results to a JSON file.
    
    Args:
        results: List of processing results
        file_path: Path to the output file
        
    Raises:
        PermissionError: If the file cannot be written
    """
    try:
        with open(file_path, "w") as file:
            json.dump(results, file, indent=2)
        
        logger.info(f"Results written to {file_path}")
    
    except PermissionError as e:
        logger.error(f"Error writing to file {file_path}: {str(e)}")
        raise


def process_binary_strings(
    fsm: ModThreeFSM,
    binary_strings: List[str]
) -> List[Dict[str, Union[str, int]]]:
    """Process a list of binary strings.
    
    Args:
        fsm: ModThreeFSM instance
        binary_strings: List of binary strings to process
    
    Returns:
        List of processing results
    """
    results: List[Dict[str, Union[str, int]]] = []
    
    for binary_string in binary_strings:
        try:
            result = fsm.process_binary_string(binary_string)
            results.append(result)
            
            if logger.level <= logging.INFO:
                print(f"{binary_string} → {result['remainder']}")
        
        except InvalidInputError as e:
            logger.warning(f"Invalid binary string '{binary_string}': {e.message}")
            results.append({
                "binary_string": binary_string,
                "error": e.message
            })
    
    return results


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_args()
    
    # Configure logging
    configure_logging(args.verbose, args.quiet)
    
    # Input validation
    if not args.binary and not args.file:
        logger.error("No input provided. Please specify either a binary string or a file.")
        return 1
    
    try:
        # Initialize FSM
        mod_three_fsm = ModThreeFSM()
        
        # Collect binary strings to process
        binary_strings: List[str] = []
        
        if args.binary:
            binary_strings.append(args.binary)
        
        if args.file:
            try:
                file_strings = read_binary_strings_from_file(args.file)
                binary_strings.extend(file_strings)
            except (FileNotFoundError, PermissionError) as e:
                logger.error(f"Error reading file: {str(e)}")
                return 1
        
        # Process binary strings
        results = process_binary_strings(mod_three_fsm, binary_strings)
        
        # Write results to file if specified
        if args.output:
            try:
                write_results_to_file(results, args.output)
            except PermissionError as e:
                logger.error(f"Error writing to file: {str(e)}")
                return 1
        
        return 0
    
    except FSMError as e:
        logger.error(f"FSM error: {e.message}")
        if hasattr(e, 'details') and e.details and not args.quiet:
            logger.error(f"Details: {e.details}")
        return 1
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if args.verbose:
            logger.exception("Exception details:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
