"""Constants for the threshold evaluator."""

# Default data path
DEFAULT_DATA_PATH = "assignment_1/data/model_metrics.csv"

# Default output path
DEFAULT_OUTPUT_PATH = "assignment_1/output/threshold_results.json"

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
LOG_FILE = "assignment_1/logs/threshold_evaluator.log"

# Threshold evaluation constants
MIN_RECALL_THRESHOLD = 0.9
METRICS_COLUMNS = ["threshold", "true_positives", "true_negatives", "false_positives", "false_negatives"]

# Default optimization metric (can be "precision", "f1" or "specificity")
DEFAULT_OPTIMIZATION_METRIC = "precision" 