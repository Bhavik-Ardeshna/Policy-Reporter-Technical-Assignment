# Assignment 1: Binary Classification Threshold Evaluator

This project evaluates binary classification model thresholds to find the best threshold that yields a recall >= 0.9 (or other specified minimum recall value). The threshold that maximizes precision (or other specified metric) while meeting the recall requirement will be selected.

## Data Structure and Design

### Input Data Structure

The application uses a structured approach to efficiently handle classification metrics:

- **Input Format**: CSV with columns for threshold, true_positives, true_negatives, false_positives, and false_negatives
- Each row represents metrics at a specific threshold value (0.1, 0.2, 0.3, etc.)
- This tabular format allows for efficient vectorized operations on multiple threshold values

### Optimization Strategies

Different strategies are employed based on dataset size:

1. **Small Datasets**:

   - Loaded entirely into memory as pandas DataFrame
   - Memory-mapped access for better performance
   - Single-pass vectorized calculations

2. **Large Datasets (Millions of Rows)**:
   - Processed in configurable chunks to minimize memory usage
   - Early type conversion using Pandas series to primitive types to reduce memoery overhead.

## Error Handling Strategy

The application implements a comprehensive error handling system:

**Exception Hierarchy**:

- `ThresholdEvaluatorError`: Base class for all application errors
- `DataValidationError`: For data format/content issues
- `DataProcessingError`: For errors during processing
- `MetricCalculationError`: For calculation-related errors
- `ThresholdNotFoundError`: When no threshold meets criteria
- `InvalidParameterError`: For parameter validation failures
- `FileOperationError`: For file system issues

## Logging

- **Log Location**: All logs are stored in the `assignment_1/logs/` directory
- **Log Format**: Timestamp, module, log level, and message
- **Log Levels**: INFO, WARNING, ERROR, DEBUG
- **Log Rotation**: Logs are automatically rotated when they reach 5MB in size

### Log File Preview

```
2025-05-05 21:47:11,159 - main - INFO - Finding best threshold with min_recall=0.9
2025-05-05 21:47:11,163 - threshold_evaluator - INFO - Successfully loaded data from assignment_1/data/model_metrics.csv
2025-05-05 21:47:11,165 - threshold_evaluator - INFO - Successfully calculated metrics for all thresholds
2025-05-05 21:47:11,166 - threshold_evaluator - INFO - Found best threshold 0.2 with precision=0.6087, recall=0.9032
2025-05-05 21:59:06,807 - main - INFO - Finding best threshold with min_recall=0.9
2025-05-05 21:59:06,819 - threshold_evaluator - INFO - Successfully loaded data from assignment_1/data/model_metrics.csv
2025-05-05 21:59:06,825 - threshold_evaluator - INFO - Found best threshold 0.2 with precision=0.6087, recall=0.9032
2025-05-05 21:59:06,825 - main - INFO - Results written to results.json
2025-05-05 22:10:14,985 - find_best_threshold - ERROR - FileOperationError: Data file not found: nonexistent_file.csv
```

## Project Structure

```
assignment_1/
├── core/
│   ├── __init__.py
│   ├── constant.py    # Configuration constants
│   ├── exception.py   # Custom exception hierarchy
│   ├── logger.py      # Logging setup
│   └── threshold_evaluator.py  # Core evaluation logic
├── data/
│   └── model_metrics.csv  # Sample metrics data
├── output/
│   └── # Directory for results output
├── logs/
│   └── threshold_evaluator.log  # Application logs
├── test/
│   ├── __init__.py
│   └── test_threshold_evaluator.py  # Unit tests
├── main.py  # Command-line interface
└── README.md
```

## Requirements

- Python 3.6+
- pandas
- numpy

## Usage

### Running with Default Parameters

```bash
python -m assignment_1.main
```

### Customizing Parameters

```bash
python -m assignment_1.main \
  --data_path assignment_1/data/model_metrics.csv \
  --min_recall 0.9 \
  --optimization_metric precision \
  --output custom_output.json
```

### Processing Large Datasets

For efficiently processing large datasets (millions of rows), use the chunk_size parameter:

```bash
python -m assignment_1.main \
  --data_path large_dataset.csv \
  --chunk_size 10000
```

This processes the data in chunks of 10,000 rows at a time, significantly reducing memory usage for large files.

### Output Files

By default, results are saved to the `assignment_1/output/` directory with a timestamp in the filename:

```
assignment_1/output/threshold_results_YYYYMMDD_HHMMSS.json
```

To disable saving results to a file:

```bash
python -m assignment_1.main --no_save
```

### Command-line Arguments

- `--data_path`: Path to the metrics data CSV file (default: assignment_1/data/model_metrics.csv)
- `--min_recall`: Minimum acceptable recall value (default: 0.9)
- `--optimization_metric`: Metric to optimize; choices are "precision", "f1", or "specificity" (default: precision)
- `--chunk_size`: Number of rows to process at once for large files (default: None = load all at once)
- `--output`: Custom output path for results (default: assignment*1/output/threshold_results*[timestamp].json)
- `--no_save`: Don't save results to a file
- `--run_tests`: Run unit tests instead of normal execution

## Input Data Format

The input CSV file should have the following columns:

- `threshold`: Classification threshold value
- `true_positives`: Number of true positives at that threshold
- `true_negatives`: Number of true negatives at that threshold
- `false_positives`: Number of false positives at that threshold
- `false_negatives`: Number of false negatives at that threshold

Example:

```
threshold,true_positives,true_negatives,false_positives,false_negatives
0.1,145,780,120,10
0.2,140,810,90,15
...
```

## Performance Considerations

The implementation includes several optimizations for handling large datasets:

1. **Chunked Processing**: Data can be processed in manageable chunks to avoid loading the entire dataset into memory
2. **Memory-Mapped I/O**: Uses memory mapping for efficient file access with large CSV files
3. **Vectorized Operations**: Employs pandas vectorized operations for fast metric calculations
4. **Early Type Conversion**: Converts pandas Series to primitive types to reduce memory overhead

## Running Tests

```bash
python -m assignment_1.main --run_tests
```

Or directly with unittest:

```bash
python -m unittest discover -s assignment_1/test
```
