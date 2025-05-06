"""Constants for the FSM module."""

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
LOG_FILE = "assignment_2/logs/policy_reporter.log"

# ModThree FSM constants
MOD_THREE_STATES = ["S0", "S1", "S2"]
MOD_THREE_ALPHABET = ["0", "1"]
MOD_THREE_INITIAL_STATE = "S0"
MOD_THREE_FINAL_STATES = ["S0", "S1", "S2"]
MOD_THREE_TRANSITIONS = {
    "S0": {"0": "S0", "1": "S1"},
    "S1": {"0": "S2", "1": "S0"},
    "S2": {"0": "S1", "1": "S2"}
}
MOD_THREE_REMAINDERS = {
    "S0": 0,
    "S1": 1,
    "S2": 2
}

