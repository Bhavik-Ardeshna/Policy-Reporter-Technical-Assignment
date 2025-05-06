"""Custom exceptions for the FSM module."""

from typing import Optional, Any, List, Dict


class FSMError(Exception):
    """Base exception class for all FSM errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class InvalidStateError(FSMError):
    """Exception raised when an invalid state is provided."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        state: Optional[str] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
            state: The invalid state
        """
        details = details or {}
        if state is not None:
            details["state"] = state
        super().__init__(message, details)
        self.state = state


class InvalidInputError(FSMError):
    """Exception raised when an invalid input is provided."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        input_symbol: Optional[str] = None,
        valid_inputs: Optional[List[str]] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
            input_symbol: The invalid input symbol
            valid_inputs: List of valid input symbols
        """
        details = details or {}
        if input_symbol is not None:
            details["input_symbol"] = input_symbol
        if valid_inputs is not None:
            details["valid_inputs"] = valid_inputs
        super().__init__(message, details)
        self.input_symbol = input_symbol
        self.valid_inputs = valid_inputs


class TransitionError(FSMError):
    """Exception raised when a transition is not defined."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        current_state: Optional[str] = None,
        input_symbol: Optional[str] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
            current_state: Current state where transition failed
            input_symbol: Input symbol that caused the failure
        """
        details = details or {}
        if current_state is not None:
            details["current_state"] = current_state
        if input_symbol is not None:
            details["input_symbol"] = input_symbol
        super().__init__(message, details)
        self.current_state = current_state
        self.input_symbol = input_symbol


class ConfigurationError(FSMError):
    """Exception raised when the FSM configuration is invalid."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        config_error: Optional[str] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
            config_error: Description of the configuration error
        """
        details = details or {}
        if config_error is not None:
            details["config_error"] = config_error
        super().__init__(message, details)
        self.config_error = config_error 