# Policy Reporter Technical Assignment

1. [Assignment 1: Binary Classification Threshold Evaluator](assignment_1/README.md)
2. [Assignment 2: Finite State Machine (FSM) for Modulo-3 Problem](assignment_2/README.md)

## Installation

```bash
# Clone the repository
git clone https://github.com/Bhavik-Ardeshna/Policy-Reporter-Technical-Assignment.git
cd Policy-Reporter-Technical-Assignments
```

## Assignment 1: Binary Classification Threshold Evaluator

A tool for evaluating binary classification model thresholds to find the best threshold that yields a recall >= 0.9 (or other specified minimum recall value) while maximizing precision or another metric.

### Running Assignment 1

Run with default parameters:

```bash
python -m assignment_1.main
```

Run with custom parameters:

```bash
python -m assignment_1.main --data_path assignment_1/data/model_metrics.csv --min_recall 0.9 --optimization_metric precision
```

For large datasets:

> Use this feature when working with millions of rows to optimize memory I/O.

```bash
python -m assignment_1.main --data_path assignment_1/data/large_dataset.csv --chunk_size 30
```

### Testing Assignment 1

Run tests using the main script:

```bash
python -m assignment_1.main --run_tests
```

Or directly with unittest:

```bash
python -m unittest discover -s assignment_1/test
```

## Assignment 2: Finite State Machine (FSM) for Modulo-3 Problem

An implementation of a generic Finite State Machine (FSM) used to solve the modulo-3 problem: determining the remainder when a binary number is divided by 3.

### Running Assignment 2

Process a single binary string:

```bash
python -m assignment_2.main -b 1101
```

Process multiple binary strings from a file:

```bash
python -m assignment_2.main -f assignment_2/data/binary_string.txt -o assignment_2/output/results.json
```

With additional options:

```bash
python -m assignment_2.main -b 1101 -v  # Verbose output
```

### Testing Assignment 2

Run all tests:

```bash
python -m unittest discover -s assignment_2/test
```

Run specific test modules:

```bash
python -m unittest assignment_2.test.test_fsm
python -m unittest assignment_2.test.test_mod_three_fsm
```

## Requirements

See individual assignment READMEs for specific requirements.
