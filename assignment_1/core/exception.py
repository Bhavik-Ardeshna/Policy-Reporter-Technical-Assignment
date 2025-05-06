"""Custom exceptions for the threshold evaluator module."""

from typing import Optional, Any, List, Dict


class ThresholdEvaluatorError(Exception):
    """Base exception class for all threshold evaluator errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DataValidationError(ThresholdEvaluatorError):
    """Exception raised for errors in data validation."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        invalid_columns: Optional[List[str]] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
            invalid_columns: List of invalid or missing columns
        """
        details = details or {}
        if invalid_columns:
            details["invalid_columns"] = invalid_columns
        super().__init__(message, details)
        self.invalid_columns = invalid_columns


class DataProcessingError(ThresholdEvaluatorError):
    """Exception raised for errors during data processing."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        chunk_index: Optional[int] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
            chunk_index: Index of the chunk that failed processing
        """
        details = details or {}
        if chunk_index is not None:
            details["chunk_index"] = chunk_index
        super().__init__(message, details)
        self.chunk_index = chunk_index


class MetricCalculationError(ThresholdEvaluatorError):
    """Exception raised for errors in metric calculations."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        metric_name: Optional[str] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
            metric_name: Name of the metric that failed calculation
        """
        details = details or {}
        if metric_name:
            details["metric_name"] = metric_name
        super().__init__(message, details)
        self.metric_name = metric_name


class ThresholdNotFoundError(ThresholdEvaluatorError):
    """Exception raised when no threshold meets the specified criteria."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        min_recall: Optional[float] = None,
        max_available_recall: Optional[float] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
            min_recall: Minimum recall requirement that couldn't be met
            max_available_recall: Maximum recall value available in the data
        """
        details = details or {}
        if min_recall is not None:
            details["min_recall"] = min_recall
        if max_available_recall is not None:
            details["max_available_recall"] = max_available_recall
        super().__init__(message, details)
        self.min_recall = min_recall
        self.max_available_recall = max_available_recall


class InvalidParameterError(ThresholdEvaluatorError):
    """Exception raised when a parameter is invalid."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        parameter_name: Optional[str] = None,
        parameter_value: Optional[Any] = None,
        valid_values: Optional[List[Any]] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
            parameter_name: Name of the invalid parameter
            parameter_value: Invalid value of the parameter
            valid_values: List of valid values for the parameter
        """
        details = details or {}
        if parameter_name:
            details["parameter_name"] = parameter_name
        if parameter_value is not None:
            details["parameter_value"] = parameter_value
        if valid_values:
            details["valid_values"] = valid_values
        super().__init__(message, details)
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.valid_values = valid_values


class FileOperationError(ThresholdEvaluatorError):
    """Exception raised for errors in file operations."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None,
        operation: Optional[str] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
            file_path: Path of the file that caused the error
            operation: Type of operation that failed (read, write, etc.)
        """
        details = details or {}
        if file_path:
            details["file_path"] = file_path
        if operation:
            details["operation"] = operation
        super().__init__(message, details)
        self.file_path = file_path
        self.operation = operation 