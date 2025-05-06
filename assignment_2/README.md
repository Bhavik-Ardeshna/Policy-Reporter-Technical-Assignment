# Assignment 2: Finite State Machine (FSM) for Modulo-3 Problem

This project implements a generic Finite State Machine (FSM) and uses it to solve the modulo-3 problem: determining the remainder when a binary number is divided by 3.

## Project Structure

```
assignment_2/
├── core/                   # Core functionality
│   ├── __init__.py         # Package initialization
│   ├── fsm.py              # Generic FSM implementation
│   ├── mod_three_fsm.py    # FSM for modulo-3 problem
│   ├── constant.py         # Constants used in the project
│   ├── exception.py        # Custom exceptions
│   └── logger.py           # Logging configuration
├── test/                   # Test modules
│   ├── __init__.py         # Test package initialization
│   ├── test_fsm.py         # Tests for the generic FSM
│   └── test_mod_three_fsm.py # Tests for the modulo-3 FSM
├── logs/                   # Log output directory
│   └── policy_reporter.log # Application logs
├── main.py                 # Command-line interface
└── README.md               # This file
```

## Logging

The project implements a comprehensive logging system to track FSM operations and aid in debugging:

- **Log Location**: All logs are stored in the `assignment_2/logs/` directory
- **Log Format**: Timestamp, module, log level, and message
- **Log Levels**: INFO, WARNING, ERROR, DEBUG
- **Log Content**: FSM state transitions, binary string processing, and results

### Log File Preview

```
2025-05-05 23:07:46,090 - assignment_2.core.fsm - INFO - FSM initialized with 3 states and 2 input symbols
2025-05-05 23:07:46,090 - assignment_2.core.mod_three_fsm - INFO - ModThreeFSM initialized
2025-05-05 23:07:46,090 - assignment_2.core.fsm - INFO - Processing input sequence of length 4
2025-05-05 23:07:46,090 - assignment_2.core.fsm - INFO - Sequence processed, final state: S1
2025-05-05 23:07:46,091 - assignment_2.core.mod_three_fsm - INFO - Binary string '1101' has remainder 1 when divided by 3
2025-05-06 09:10:30,671 - assignment_2.core.mod_three_fsm - WARNING - Empty binary string provided, returning 0
2025-05-06 09:17:16,844 - main - INFO - Read 2 binary strings from assignment_2/data/binary_string.txt
2025-05-06 09:17:16,844 - assignment_2.core.mod_three_fsm - INFO - Binary string '111010010101' has remainder 1 when divided by 3
2025-05-06 09:17:16,844 - assignment_2.core.mod_three_fsm - INFO - Binary string '111000100' has remainder 2 when divided by 3
2025-05-06 09:17:16,844 - main - INFO - Results written to assignment_2/output/results.json
```

## Features

- Generic FSM implementation that can be used for any finite state automata problem
- Modulo-3 FSM implementation using the generic FSM
- Comprehensive error handling
- Detailed logging
- Unit tests

## How It Works

The modulo-3 FSM is configured with three states (S0, S1, S2) that represent the remainder when divided by 3:

- S0: Remainder 0
- S1: Remainder 1
- S2: Remainder 2

As binary digits are processed from left to right (most significant bit first), the FSM transitions between states according to the FSM transition table. After processing all digits, the final state indicates the remainder when the binary number is divided by 3.

### Transitions

| Current State | Input 0 | Input 1 |
| ------------- | ------- | ------- |
| S0            | S0      | S1      |
| S1            | S2      | S0      |
| S2            | S1      | S2      |

## Usage

### Command-Line Interface

The `main.py` script provides a command-line interface to use the modulo-3 FSM:

```bash
python -m assignment_2.main -b <binary_string>
```

Or:

```bash
python -m assignment_2.main -f <file_path>
```

#### Options

- `-b, --binary`: Binary string to process
- `-f, --file`: Path to file containing binary strings (one per line)
- `-o, --output`: Path to output file (JSON format)
- `-v, --verbose`: Enable verbose output
- `-q, --quiet`: Suppress all output except errors

### Examples

Process a single binary string:

```bash
python -m assignment_2.main -b 1101
```

Output:

```
1101 → 1
```

Process multiple binary strings from a file:

```bash
python -m assignment_2.main -f binary_strings.txt -o results.json
```

### Using the FSM programmatically

You can also use the FSM programmatically in your Python code:

```python
from assignment_2.core.mod_three_fsm import ModThreeFSM

# Create an instance of the ModThreeFSM
fsm = ModThreeFSM()

# Calculate the remainder
remainder = fsm.calculate_remainder("1101")
print(f"Remainder: {remainder}")  # Output: Remainder: 1

# Get more detailed information
result = fsm.process_binary_string("1101")
print(result)
# Output: {'binary_string': '1101', 'decimal_value': 13, 'remainder': 1, 'final_state': 'S1'}
```

## Running Tests

You can run the tests using the `unittest` module:

```bash
python -m unittest discover -s assignment_2/test
```

Or run individual test modules:

```bash
python -m unittest assignment_2.test.test_fsm
python -m unittest assignment_2.test.test_mod_three_fsm
```
