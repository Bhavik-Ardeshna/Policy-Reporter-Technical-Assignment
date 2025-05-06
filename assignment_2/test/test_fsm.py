"""Tests for the generic FSM class."""

import unittest

from assignment_2.core.fsm import FSM
from assignment_2.core.exception import (
    InvalidInputError,
    ConfigurationError,
)


class TestFSM(unittest.TestCase):
    """Test cases for the generic FSM class."""
    
    def test_simple_fsm(self):
        """Test a simple FSM with two states."""
        # Create a simple FSM that accepts strings that end with '1'
        fsm = FSM[str, str, bool](
            states={"q0", "q1"},
            alphabet={"0", "1"},
            transitions={
                "q0": {"0": "q0", "1": "q1"},
                "q1": {"0": "q0", "1": "q1"}
            },
            initial_state="q0",
            final_states={"q1"},
            output_map={"q0": False, "q1": True}
        )
        
        # Test with strings that end with '1'
        fsm.process_sequence(list("1"))
        self.assertTrue(fsm.get_output())
        
        fsm.process_sequence(list("01"))
        self.assertTrue(fsm.get_output())
        
        fsm.process_sequence(list("001"))
        self.assertTrue(fsm.get_output())
        
        # Test with strings that don't end with '1'
        fsm.process_sequence(list("0"))
        self.assertFalse(fsm.get_output())
        
        fsm.process_sequence(list("10"))
        self.assertFalse(fsm.get_output())
        
        fsm.process_sequence(list("1110"))
        self.assertFalse(fsm.get_output())
    
    def test_invalid_input(self):
        """Test that invalid input symbols raise an exception."""
        fsm = FSM[str, str, None](
            states={"q0", "q1"},
            alphabet={"0", "1"},
            transitions={
                "q0": {"0": "q0", "1": "q1"},
                "q1": {"0": "q0", "1": "q1"}
            },
            initial_state="q0",
            final_states={"q1"}
        )
        
        with self.assertRaises(InvalidInputError):
            fsm.process_input("2")
    
    def test_invalid_configuration(self):
        """Test that invalid FSM configurations raise exceptions."""
        # Test with empty states
        with self.assertRaises(ConfigurationError):
            FSM[str, str, None](
                states=set(),
                alphabet={"0", "1"},
                transitions={},
                initial_state="q0",
                final_states={"q1"}
            )
        
        # Test with empty alphabet
        with self.assertRaises(ConfigurationError):
            FSM[str, str, None](
                states={"q0", "q1"},
                alphabet=set(),
                transitions={
                    "q0": {"0": "q0", "1": "q1"},
                    "q1": {"0": "q0", "1": "q1"}
                },
                initial_state="q0",
                final_states={"q1"}
            )
        
        # Test with invalid initial state
        with self.assertRaises(ConfigurationError):
            FSM[str, str, None](
                states={"q0", "q1"},
                alphabet={"0", "1"},
                transitions={
                    "q0": {"0": "q0", "1": "q1"},
                    "q1": {"0": "q0", "1": "q1"}
                },
                initial_state="q2",  # Not in states
                final_states={"q1"}
            )
        
        # Test with invalid final state
        with self.assertRaises(ConfigurationError):
            FSM[str, str, None](
                states={"q0", "q1"},
                alphabet={"0", "1"},
                transitions={
                    "q0": {"0": "q0", "1": "q1"},
                    "q1": {"0": "q0", "1": "q1"}
                },
                initial_state="q0",
                final_states={"q2"}  # Not in states
            )
        
        # Test with invalid transition state
        with self.assertRaises(ConfigurationError):
            FSM[str, str, None](
                states={"q0", "q1"},
                alphabet={"0", "1"},
                transitions={
                    "q0": {"0": "q0", "1": "q1"},
                    "q1": {"0": "q0", "1": "q1"},
                    "q2": {"0": "q0", "1": "q1"}  # q2 not in states
                },
                initial_state="q0",
                final_states={"q1"}
            )
        
        # Test with invalid transition input
        with self.assertRaises(ConfigurationError):
            FSM[str, str, None](
                states={"q0", "q1"},
                alphabet={"0", "1"},
                transitions={
                    "q0": {"0": "q0", "1": "q1", "2": "q0"},  # 2 not in alphabet
                    "q1": {"0": "q0", "1": "q1"}
                },
                initial_state="q0",
                final_states={"q1"}
            )
        
        # Test with invalid transition target
        with self.assertRaises(ConfigurationError):
            FSM[str, str, None](
                states={"q0", "q1"},
                alphabet={"0", "1"},
                transitions={
                    "q0": {"0": "q0", "1": "q2"},  # q2 not in states
                    "q1": {"0": "q0", "1": "q1"}
                },
                initial_state="q0",
                final_states={"q1"}
            )
    
    def test_reset(self):
        """Test the reset method."""
        fsm = FSM[str, str, None](
            states={"q0", "q1"},
            alphabet={"0", "1"},
            transitions={
                "q0": {"0": "q0", "1": "q1"},
                "q1": {"0": "q0", "1": "q1"}
            },
            initial_state="q0",
            final_states={"q1"}
        )
        
        # Move to q1
        fsm.process_input("1")
        self.assertEqual(fsm.get_current_state(), "q1")
        
        # Reset and check that we're back to q0
        fsm.reset()
        self.assertEqual(fsm.get_current_state(), "q0")
    
    def test_is_in_final_state(self):
        """Test the is_in_final_state method."""
        fsm = FSM[str, str, None](
            states={"q0", "q1"},
            alphabet={"0", "1"},
            transitions={
                "q0": {"0": "q0", "1": "q1"},
                "q1": {"0": "q0", "1": "q1"}
            },
            initial_state="q0",
            final_states={"q1"}
        )
        
        # Initial state is not final
        self.assertFalse(fsm.is_in_final_state())
        
        # Move to q1
        fsm.process_input("1")
        self.assertTrue(fsm.is_in_final_state())
        
        # Move back to q0
        fsm.process_input("0")
        self.assertFalse(fsm.is_in_final_state())


if __name__ == "__main__":
    unittest.main() 