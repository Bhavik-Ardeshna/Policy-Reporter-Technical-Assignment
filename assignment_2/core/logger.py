"""Logger module for the application."""

import logging
import os
from pathlib import Path

from assignment_2.core.constant import LOG_FORMAT, LOG_LEVEL, LOG_FILE


def setup_logger(name: str = "policy_reporter") -> logging.Logger:
    """Set up and configure logger.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set log level from constant
    log_level = getattr(logging, LOG_LEVEL)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if LOG_FILE is defined
    if LOG_FILE:
        # Create directory for log file if it doesn't exist
        log_dir = os.path.dirname(LOG_FILE)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 