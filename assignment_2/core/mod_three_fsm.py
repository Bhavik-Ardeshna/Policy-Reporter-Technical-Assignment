"""Implementation of FSM to solve the Modulo-3 problem."""

from typing import Dict, Union

from assignment_2.core.fsm import FSM
from assignment_2.core.constant import (
    MOD_THREE_STATES,
    MOD_THREE_ALPHABET,
    MOD_THREE_INITIAL_STATE,
    MOD_THREE_FINAL_STATES,
    MOD_THREE_TRANSITIONS,
    MOD_THREE_REMAINDERS,
)
from assignment_2.core.exception import InvalidInputError
from assignment_2.core.logger import setup_logger

# Set up logger
logger = setup_logger(__name__)


class ModThreeFSM:
    """FSM implementation for solving the modulo-3 problem.
    
    This class provides a specialized implementation to determine
    the remainder when a binary number is divided by 3.
    """
    
    def __init__(self):
        """Initialize the ModThreeFSM with predefined states and transitions."""
        # Convert lists to sets for the FSM configuration
        states = set(MOD_THREE_STATES)
        alphabet = set(MOD_THREE_ALPHABET)
        final_states = set(MOD_THREE_FINAL_STATES)
        
        # Create the underlying FSM
        self.fsm = FSM[str, str, int](
            states=states,
            alphabet=alphabet,
            transitions=MOD_THREE_TRANSITIONS,
            initial_state=MOD_THREE_INITIAL_STATE,
            final_states=final_states,
            output_map=MOD_THREE_REMAINDERS
        )
        
        logger.info("ModThreeFSM initialized")
    
    def calculate_remainder(self, binary_string: str) -> int:
        """Calculate the remainder when the binary number is divided by 3.
        
        Args:
            binary_string: Binary number as a string of '0's and '1's
            
        Returns:
            Remainder when the binary number is divided by 3 (0, 1, or 2)
            
        Raises:
            InvalidInputError: If the input string contains characters other than '0' and '1'
        """
        # Validate input
        if not all(char in MOD_THREE_ALPHABET for char in binary_string):
            invalid_chars = set(char for char in binary_string if char not in MOD_THREE_ALPHABET)
            raise InvalidInputError(
                f"Invalid characters in binary string: {invalid_chars}",
                input_symbol=str(invalid_chars),
                valid_inputs=MOD_THREE_ALPHABET
            )
        
        if not binary_string:
            logger.warning("Empty binary string provided, returning 0")
            return 0
        
        # Process the binary string through the FSM
        input_sequence = list(binary_string)
        self.fsm.process_sequence(input_sequence)
        
        # Get the remainder from the output map
        remainder = self.fsm.get_output()
        if remainder is None:
            # This should never happen with our configuration, but to satisfy the type checker
            logger.error("Unexpected None remainder value, defaulting to 0")
            remainder = 0
            
        logger.info(f"Binary string '{binary_string}' has remainder {remainder} when divided by 3")
        
        return remainder
    
    def process_binary_string(self, binary_string: str) -> Dict[str, Union[str, int]]:
        """Process a binary string and return detailed information.
        
        This method provides more detailed information about the processing
        of the binary string through the FSM.
        
        Args:
            binary_string: Binary number as a string of '0's and '1's
            
        Returns:
            Dictionary with:
                - binary_string: Original binary string
                - decimal_value: Decimal representation of the binary string
                - remainder: Remainder when divided by 3
                - final_state: Final state of the FSM
        """
        remainder = self.calculate_remainder(binary_string)
        
        # Calculate decimal value for reference
        decimal_value = int(binary_string, 2) if binary_string else 0
        
        result = {
            "binary_string": binary_string,
            "decimal_value": decimal_value,
            "remainder": remainder,
            "final_state": self.fsm.get_current_state()
        }
        
        return result 