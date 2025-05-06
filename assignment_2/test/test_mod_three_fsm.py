"""Tests for the ModThreeFSM class."""

import unittest

from assignment_2.core.mod_three_fsm import ModThreeFSM
from assignment_2.core.exception import InvalidInputError


class TestModThreeFSM(unittest.TestCase):
    """Test cases for the ModThreeFSM class."""
    
    def setUp(self):
        """Set up the test case."""
        self.fsm = ModThreeFSM()
    
    def test_empty_string(self):
        """Test that an empty string returns 0."""
        self.assertEqual(self.fsm.calculate_remainder(""), 0)
    
    def test_zero(self):
        """Test that '0' returns 0."""
        self.assertEqual(self.fsm.calculate_remainder("0"), 0)
    
    def test_one(self):
        """Test that '1' returns 1."""
        self.assertEqual(self.fsm.calculate_remainder("1"), 1)
    
    def test_example_1(self):
        """Test the example from the assignment: '1101' should return 1."""
        self.assertEqual(self.fsm.calculate_remainder("1101"), 1)
    
    def test_example_2(self):
        """Test the example from the assignment: '1110' should return 2."""
        self.assertEqual(self.fsm.calculate_remainder("1110"), 2)
    
    def test_example_3(self):
        """Test the example from the assignment: '1111' should return 0."""
        self.assertEqual(self.fsm.calculate_remainder("1111"), 0)
    
    def test_fsm_example_1(self):
        """Test the FSM example from the assignment: '110' should return 0."""
        self.assertEqual(self.fsm.calculate_remainder("110"), 0)
    
    def test_fsm_example_2(self):
        """Test the FSM example from the assignment: '1010' should return 1."""
        self.assertEqual(self.fsm.calculate_remainder("1010"), 1)
    
    def test_long_binary(self):
        """Test a longer binary string."""
        # 1010101010101010 = 43690 in decimal, 43690 % 3 = 1
        self.assertEqual(self.fsm.calculate_remainder("1010101010101010"), 1)
    
    def test_invalid_input(self):
        """Test that an invalid binary string raises an exception."""
        with self.assertRaises(InvalidInputError):
            self.fsm.calculate_remainder("1201")
    
    def test_process_binary_string(self):
        """Test the process_binary_string method."""
        result = self.fsm.process_binary_string("1101")
        self.assertEqual(result["binary_string"], "1101")
        self.assertEqual(result["decimal_value"], 13)
        self.assertEqual(result["remainder"], 1)
        self.assertEqual(result["final_state"], "S1")


if __name__ == "__main__":
    unittest.main() 