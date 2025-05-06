"""Generic Finite State Machine (FSM) implementation."""

from typing import Dict, List, Set, Optional, TypeVar, Generic

from assignment_2.core.exception import (
    InvalidInputError,
    TransitionError,
    ConfigurationError,
)
from assignment_2.core.logger import setup_logger

# Type variables for generic FSM implementation
StateType = TypeVar('StateType')
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')

# Set up logger
logger = setup_logger(__name__)


class FSM(Generic[StateType, InputType, OutputType]):
    """Generic Finite State Machine implementation.
    
    This class implements a generic Finite State Machine (FSM) that can be
    configured for any FSM problem. It manages the state transitions and
    provides methods to process input sequences.
    
    Attributes:
        states: Set of valid states in the FSM
        alphabet: Set of valid input symbols
        transitions: Dictionary mapping (state, input) to next state
        initial_state: Starting state of the FSM
        final_states: Set of final/accepting states
        current_state: Current state of the FSM
    """
    
    def __init__(
        self,
        states: Set[StateType],
        alphabet: Set[InputType],
        transitions: Dict[StateType, Dict[InputType, StateType]],
        initial_state: StateType,
        final_states: Set[StateType],
        output_map: Optional[Dict[StateType, OutputType]] = None
    ):
        """Initialize the FSM.
        
        Args:
            states: Set of all possible states
            alphabet: Set of valid input symbols
            transitions: Dictionary mapping (state, input) to next state
            initial_state: Starting state of the FSM
            final_states: Set of final/accepting states
            output_map: Optional mapping from states to output values
            
        Raises:
            ConfigurationError: If the FSM configuration is invalid
        """
        # Validate the configuration
        if not states:
            raise ConfigurationError("States set cannot be empty")
        
        if not alphabet:
            raise ConfigurationError("Alphabet set cannot be empty")
        
        if initial_state not in states:
            raise ConfigurationError(
                f"Initial state '{initial_state}' not in states set",
                config_error=f"Initial state: {initial_state}, Available states: {states}"
            )
        
        if not all(state in states for state in final_states):
            invalid_finals = [state for state in final_states if state not in states]
            raise ConfigurationError(
                f"Some final states are not in states set: {invalid_finals}",
                config_error=f"Invalid final states: {invalid_finals}"
            )
        
        # Check that transitions are valid
        for state, trans in transitions.items():
            if state not in states:
                raise ConfigurationError(
                    f"Transition defined for non-existent state: {state}",
                    config_error=f"Invalid state in transitions: {state}"
                )
            
            for input_symbol, next_state in trans.items():
                if input_symbol not in alphabet:
                    raise ConfigurationError(
                        f"Transition uses invalid input symbol: {input_symbol}",
                        config_error=f"Invalid input in transitions: {input_symbol}"
                    )
                if next_state not in states:
                    raise ConfigurationError(
                        f"Transition leads to non-existent state: {next_state}",
                        config_error=f"Invalid next state in transitions: {next_state}"
                    )
        
        # Check that all state-input combinations have transitions
        for state in states:
            if state not in transitions:
                raise ConfigurationError(
                    f"No transitions defined for state: {state}",
                    config_error=f"Missing transitions for state: {state}"
                )
            
            for input_symbol in alphabet:
                if input_symbol not in transitions[state]:
                    raise ConfigurationError(
                        f"No transition defined for (state={state}, input={input_symbol})",
                        config_error=f"Missing transition: ({state}, {input_symbol})"
                    )
        
        # Check output map if provided
        if output_map is not None and not all(state in states for state in output_map):
            invalid_outputs = [state for state in output_map if state not in states]
            raise ConfigurationError(
                f"Output map contains invalid states: {invalid_outputs}",
                config_error=f"Invalid states in output map: {invalid_outputs}"
            )
        
        # Store configuration
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.initial_state = initial_state
        self.final_states = final_states
        self.output_map = output_map or {}
        
        # Initialize current state
        self.current_state = initial_state
        
        logger.info(f"FSM initialized with {len(states)} states and {len(alphabet)} input symbols")
    
    def reset(self) -> None:
        """Reset the FSM to its initial state."""
        self.current_state = self.initial_state
        logger.debug("FSM reset to initial state")
    
    def get_current_state(self) -> StateType:
        """Get the current state of the FSM.
        
        Returns:
            Current state
        """
        return self.current_state
    
    def is_in_final_state(self) -> bool:
        """Check if the FSM is in a final/accepting state.
        
        Returns:
            True if current state is a final state, False otherwise
        """
        return self.current_state in self.final_states
    
    def process_input(self, input_symbol: InputType) -> StateType:
        """Process a single input symbol and transition to the next state.
        
        Args:
            input_symbol: Input symbol to process
        
        Returns:
            Next state after processing the input
            
        Raises:
            InvalidInputError: If the input symbol is not in the alphabet
            TransitionError: If no transition is defined for the current state and input
        """
        if input_symbol not in self.alphabet:
            raise InvalidInputError(
                f"Invalid input symbol: {input_symbol}",
                input_symbol=str(input_symbol),
                valid_inputs=[str(symbol) for symbol in self.alphabet]
            )
        
        if self.current_state not in self.transitions:
            raise TransitionError(
                f"No transitions defined for current state: {self.current_state}",
                current_state=str(self.current_state)
            )
        
        if input_symbol not in self.transitions[self.current_state]:
            raise TransitionError(
                f"No transition defined for input {input_symbol} in state {self.current_state}",
                current_state=str(self.current_state),
                input_symbol=str(input_symbol)
            )
        
        next_state = self.transitions[self.current_state][input_symbol]
        logger.debug(f"Transition: {self.current_state} --({input_symbol})--> {next_state}")
        self.current_state = next_state
        return self.current_state
    
    def process_sequence(self, input_sequence: List[InputType]) -> StateType:
        """Process a sequence of input symbols.
        
        Args:
            input_sequence: List of input symbols to process in order
        
        Returns:
            Final state after processing the entire sequence
            
        Raises:
            InvalidInputError: If any input symbol is not in the alphabet
            TransitionError: If any transition is undefined
        """
        self.reset()
        logger.info(f"Processing input sequence of length {len(input_sequence)}")
        
        for input_symbol in input_sequence:
            self.process_input(input_symbol)
        
        logger.info(f"Sequence processed, final state: {self.current_state}")
        return self.current_state
    
    def get_output(self) -> Optional[OutputType]:
        """Get the output value for the current state if defined.
        
        Returns:
            Output value for the current state, or None if not defined
        """
        return self.output_map.get(self.current_state) 