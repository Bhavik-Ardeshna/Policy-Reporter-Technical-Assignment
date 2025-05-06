"""Unit tests for threshold evaluator.

This module provides comprehensive test coverage for the threshold evaluator 
functionality, including edge cases and performance optimizations.

Edge Cases Covered:
- Empty datasets
- Datasets with only one threshold
- Thresholds with zero denominators (divide by zero scenarios)
- No thresholds meeting recall criteria
- All thresholds meeting recall criteria
- Negative values in metrics
- Missing columns in data
- Invalid file paths
- Invalid parameters
- Very large datasets (via chunked processing simulation)
"""

import os
import unittest
import tempfile
import pandas as pd
from typing import Dict
from unittest.mock import patch

from assignment_1.core.threshold_evaluator import (
    ThresholdEvaluator,
    find_best_threshold_with_min_recall
)
from assignment_1.core.exception import (
    ThresholdEvaluatorError,
    DataValidationError,
    InvalidParameterError,
    FileOperationError,
    ThresholdNotFoundError
)
from assignment_1.core.constant import DEFAULT_DATA_PATH


class TestThresholdEvaluator(unittest.TestCase):
    """Test cases for ThresholdEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use the actual model_metrics.csv file for most tests
        self.data_path: str = DEFAULT_DATA_PATH
        
        # Load the data for creating test variants
        self.test_data: pd.DataFrame = pd.read_csv(self.data_path)
        
        # Create temporary files for edge cases
        self.temp_files: Dict[str, str] = {}
        
        # Test data with negative values (invalid)
        negative_data = self.test_data.copy()
        negative_data.loc[1, "false_positives"] = -90  # Modify one value to be negative
        
        # Test data with missing columns
        missing_columns_data = self.test_data.copy()
        missing_columns_data = missing_columns_data.drop(columns=["true_negatives"])
        
        # Test data with division by zero scenario
        zero_denominator_data = self.test_data.copy()
        zero_denominator_data.loc[2, ["true_positives", "false_positives", "false_negatives"]] = 0
        
        # Test data with high recall
        high_recall_data = self.test_data.copy()
        high_recall_data["false_negatives"] = 5  # Set all FN to 5 for high recall
        
        # Test data with low recall
        low_recall_data = self.test_data.copy()
        low_recall_data["false_negatives"] = 200  # Set all FN to 200 for low recall
        
        # Test data with only one threshold
        single_threshold_data = self.test_data.iloc[[4]].copy()  # Just take one row
        
        # Save temporary files for edge cases
        edge_cases: Dict[str, pd.DataFrame] = {
            "negative": negative_data,
            "missing_columns": missing_columns_data,
            "zero_denominator": zero_denominator_data,
            "high_recall": high_recall_data,
            "low_recall": low_recall_data,
            "single_threshold": single_threshold_data
        }
        
        for case_name, data in edge_cases.items():
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{case_name}.csv")
            data.to_csv(temp_file.name, index=False)
            temp_file.close()
            self.temp_files[case_name] = temp_file.name
            
        # Create a truly empty file for testing empty dataset
        self.empty_file = tempfile.NamedTemporaryFile(delete=False, suffix="_empty.csv")
        self.empty_file.close()
        # Create file with just a header row (another type of empty)
        empty_data = pd.DataFrame(columns=self.test_data.columns)
        self.empty_file_with_header = tempfile.NamedTemporaryFile(delete=False, suffix="_empty_with_header.csv")
        empty_data.to_csv(self.empty_file_with_header.name, index=False)
        self.empty_file_with_header.close()
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove all temporary files
        temp_files = list(self.temp_files.values()) + [self.empty_file.name, self.empty_file_with_header.name]
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    # ====== DATA LOADING TESTS ======
    
    def test_initialization(self):
        """Test that the evaluator initializes correctly."""
        evaluator = ThresholdEvaluator(data_path=self.data_path)
        self.assertIsNotNone(evaluator.metrics_df)
        if evaluator.metrics_df is not None:  # Type checking for linter
            self.assertEqual(len(evaluator.metrics_df), 9)
        
        # Test initialization with custom parameters
        evaluator = ThresholdEvaluator(
            data_path=self.data_path,
            min_recall=0.8,
            optimization_metric="f1"
        )
        self.assertEqual(evaluator.min_recall, 0.8)
        self.assertEqual(evaluator.optimization_metric, "f1")
    
    def test_file_not_found(self):
        """Test handling of non-existent file."""
        with self.assertRaises(FileOperationError):
            ThresholdEvaluator(data_path="nonexistent_file.csv")
    
    def test_empty_file(self):
        """Test handling of empty file."""
        # Test with attached empty file - should raise DataValidationError during initialization
        with self.assertRaises(DataValidationError):
            ThresholdEvaluator(data_path="assignment_1/data/empty_metrics.csv")
        
        # Test with file that has only headers - should also raise DataValidationError during initialization
        with self.assertRaises(DataValidationError):
            ThresholdEvaluator(data_path=self.empty_file_with_header.name)
    
    def test_missing_columns(self):
        """Test handling of data with missing columns."""
        with self.assertRaises(DataValidationError) as context:
            ThresholdEvaluator(data_path=self.temp_files["missing_columns"])
        
        # Verify that the exception contains information about the missing column
        error_message = context.exception.message.lower()
        self.assertTrue("true_negatives" in error_message or "missing" in error_message)
    
    # ====== PARAMETER VALIDATION TESTS ======
    
    def test_invalid_min_recall(self):
        """Test validation of min_recall parameter."""
        # Test negative min_recall
        with self.assertRaises(InvalidParameterError) as context:
            ThresholdEvaluator(
                data_path=self.data_path,
                min_recall=-0.1
            )
        self.assertTrue("min_recall" in str(context.exception))
        
        # Test min_recall > 1
        with self.assertRaises(InvalidParameterError):
            ThresholdEvaluator(
                data_path=self.data_path,
                min_recall=1.1
            )
    
    def test_invalid_optimization_metric(self):
        """Test validation of optimization_metric parameter."""
        with self.assertRaises(InvalidParameterError) as context:
            ThresholdEvaluator(
                data_path=self.data_path,
                optimization_metric="invalid_metric"
            )
        self.assertTrue("optimization_metric" in str(context.exception))
    
    def test_invalid_chunk_size(self):
        """Test validation of chunk_size parameter."""
        with self.assertRaises(InvalidParameterError) as context:
            ThresholdEvaluator(
                data_path=self.data_path,
                chunk_size=-10
            )
        self.assertTrue("chunk_size" in str(context.exception))
    
    # ====== METRIC CALCULATION TESTS ======
    
    def test_calculate_metrics(self):
        """Test that metrics are calculated correctly."""
        evaluator = ThresholdEvaluator(data_path=self.data_path)
        metrics = evaluator.calculate_metrics()
        
        # Check that all required metrics exist
        for col in ["recall", "precision", "specificity", "f1"]:
            self.assertIn(col, metrics.columns)
        
        # Check calculations for first threshold (0.1)
        row_data = metrics[metrics["threshold"] == 0.1].iloc[0].to_dict()
        
        # Get the original values for threshold 0.1
        original_data = self.test_data[self.test_data["threshold"] == 0.1].iloc[0]
        tp = original_data["true_positives"]
        tn = original_data["true_negatives"]
        fp = original_data["false_positives"]
        fn = original_data["false_negatives"]
        
        # Calculate expected metrics
        expected_recall = tp / (tp + fn)
        expected_precision = tp / (tp + fp)
        expected_specificity = tn / (tn + fp)
        expected_f1 = 2 * (expected_precision * expected_recall) / (expected_precision + expected_recall)
        
        # Compare calculated metrics with expected values
        self.assertAlmostEqual(row_data["recall"], expected_recall)
        self.assertAlmostEqual(row_data["precision"], expected_precision)
        self.assertAlmostEqual(row_data["specificity"], expected_specificity)
        self.assertAlmostEqual(row_data["f1"], expected_f1)
    
    def test_zero_denominator_handling(self):
        """Test handling of zero denominators in metric calculations."""
        # This should warn but not error
        evaluator = ThresholdEvaluator(data_path=self.temp_files["zero_denominator"])
        
        # Capture warning logs
        with self.assertLogs(level='WARNING') as log:
            metrics = evaluator.calculate_metrics()
            
            # Check for the warning message about zero denominators
            self.assertTrue(any("zero" in msg.lower() for msg in log.output))
            
            # The metrics should contain NaN values for the threshold with zero denominators
            threshold_03_row = metrics[metrics["threshold"] == 0.3]
            self.assertTrue(pd.isna(threshold_03_row["recall"].iloc[0]) or 
                           pd.isna(threshold_03_row["precision"].iloc[0]))
    
    def test_negative_values(self):
        """Test handling of negative values in metrics."""
        with self.assertRaises(DataValidationError) as context:
            evaluator = ThresholdEvaluator(data_path=self.temp_files["negative"])
            evaluator.calculate_metrics()
        
        error_message = str(context.exception).lower()
        self.assertTrue("negative" in error_message)
    
    def test_single_threshold(self):
        """Test handling of data with only one threshold."""
        evaluator = ThresholdEvaluator(data_path=self.temp_files["single_threshold"])
        metrics = evaluator.calculate_metrics()
        
        # Should have only one row
        self.assertEqual(len(metrics), 1)
        
        # Verify that metrics are calculated correctly
        original_row = self.test_data.iloc[4]
        tp = original_row["true_positives"]
        fp = original_row["false_positives"]
        fn = original_row["false_negatives"]
        
        # Calculate expected metrics
        expected_recall = tp / (tp + fn)
        expected_precision = tp / (tp + fp)
        
        # Compare with calculated metrics
        row_data = metrics.iloc[0].to_dict()
        self.assertAlmostEqual(row_data["recall"], expected_recall)
        self.assertAlmostEqual(row_data["precision"], expected_precision)
    
    # ====== THRESHOLD SELECTION TESTS ======
    
    def test_find_best_threshold_min_recall_0_9(self):
        """Test finding the best threshold with min_recall=0.9."""
        # Check if any thresholds meet the criteria
        evaluator = ThresholdEvaluator(
            data_path=self.data_path,
            min_recall=0.9,
            optimization_metric="precision"
        )
        
        # Calculate metrics to determine if any threshold meets min_recall=0.9
        df = evaluator.calculate_metrics()
        valid_thresholds = df[df["recall"] >= 0.9]
        
        if not valid_thresholds.empty:
            # At least one threshold meets the criteria
            best_threshold, metrics = evaluator.find_best_threshold()
            self.assertIsNotNone(best_threshold)
            
            # Get the threshold with highest precision among those with recall >= 0.9
            expected_threshold = valid_thresholds.loc[valid_thresholds["precision"].idxmax()]["threshold"]
            self.assertEqual(best_threshold, expected_threshold)
        else:
            # No threshold meets the criteria
            with self.assertRaises(ThresholdNotFoundError):
                evaluator.find_best_threshold()
    
    def test_find_best_threshold_min_recall_lower(self):
        """Test finding the best threshold with lower min_recall."""
        # Use min_recall=0.7 which should definitely be met by some thresholds
        evaluator = ThresholdEvaluator(
            data_path=self.data_path,
            min_recall=0.7,
            optimization_metric="precision"
        )
        
        best_threshold, metrics = evaluator.find_best_threshold()
        
        # We expect a valid result (not None)
        self.assertIsNotNone(best_threshold)
        
        # Verify it selected the correct threshold
        df = evaluator.calculate_metrics()
        valid_thresholds = df[df["recall"] >= 0.7]
        
        if not valid_thresholds.empty:
            expected_threshold = valid_thresholds.loc[valid_thresholds["precision"].idxmax()]["threshold"]
            self.assertEqual(best_threshold, expected_threshold)
    
    def test_no_valid_thresholds(self):
        """Test when no thresholds meet the min_recall criteria."""
        # Use the low recall dataset with high min_recall
        evaluator = ThresholdEvaluator(
            data_path=self.temp_files["low_recall"],
            min_recall=0.95,  # Higher than any available recall
            optimization_metric="precision"
        )
        
        # Should raise ThresholdNotFoundError
        with self.assertRaises(ThresholdNotFoundError) as context:
            evaluator.find_best_threshold()
            
        # Exception should contain info about max available recall
        self.assertIsNotNone(context.exception.max_available_recall)
        
        # Standalone function should handle this gracefully by returning None
        result = find_best_threshold_with_min_recall(
            data_path=self.temp_files["low_recall"],
            min_recall=0.95
        )
        self.assertIsNone(result)
    
    def test_all_valid_thresholds(self):
        """Test when all thresholds meet the min_recall criteria."""
        # Use the high recall dataset
        evaluator = ThresholdEvaluator(
            data_path=self.temp_files["high_recall"],
            min_recall=0.8,  # Low enough that all thresholds should meet it
            optimization_metric="precision"
        )
        
        best_threshold, metrics = evaluator.find_best_threshold()
        
        # Verify it selected the threshold with best precision
        df = evaluator.calculate_metrics()
        expected_threshold = df.loc[df["precision"].idxmax()]["threshold"]
        self.assertEqual(best_threshold, expected_threshold)
    
    def test_different_optimization_metrics(self):
        """Test optimization with different metrics."""
        # Using a minimum recall that's definitely achievable
        min_recall = 0.7
        metrics_to_test = ["precision", "f1", "specificity"]
        
        for metric in metrics_to_test:
            evaluator = ThresholdEvaluator(
                data_path=self.data_path,
                min_recall=min_recall,
                optimization_metric=metric
            )
            
            best_threshold, metrics_result = evaluator.find_best_threshold()
            
            # Verify that the chosen threshold maximizes the chosen metric
            df = evaluator.calculate_metrics()
            valid_thresholds = df[df["recall"] >= min_recall]
            
            if not valid_thresholds.empty:
                expected_threshold = valid_thresholds.loc[valid_thresholds[metric].idxmax()]["threshold"]
                self.assertEqual(best_threshold, expected_threshold)
    
    # ====== CHUNKED PROCESSING TESTS ======
    
    def test_chunked_processing(self):
        """Test that chunked processing produces the same results as whole-file processing."""
        # Create a larger test dataset by repeating the test data multiple times
        larger_data = pd.concat([self.test_data] * 5, ignore_index=True)
        
        # Create a temporary file for the larger dataset
        larger_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        larger_data.to_csv(larger_temp_file.name, index=False)
        larger_temp_file.close()
        
        try:
            # Process without chunking
            evaluator_whole = ThresholdEvaluator(
                data_path=larger_temp_file.name,
                min_recall=0.8,
                optimization_metric="precision"
            )
            
            # Process with chunking (chunk size of 3 rows)
            evaluator_chunked = ThresholdEvaluator(
                data_path=larger_temp_file.name,
                min_recall=0.8,
                optimization_metric="precision",
                chunk_size=3
            )
            
            # Calculate metrics and compare results
            metrics_whole = evaluator_whole.calculate_metrics().sort_values("threshold")
            metrics_chunked = evaluator_chunked.calculate_metrics().sort_values("threshold")
            
            # Check that we have the same thresholds
            self.assertEqual(
                list(metrics_whole["threshold"]),
                list(metrics_chunked["threshold"])
            )
            
            # Check that calculated metrics are the same within floating point precision
            for col in ["recall", "precision", "specificity", "f1"]:
                # Using almost equal for float comparison
                pd.testing.assert_series_equal(
                    metrics_whole[col],
                    metrics_chunked[col],
                    rtol=1e-5  # Relative tolerance for comparing floats
                )
            
            # Check that best thresholds are the same
            best_threshold_whole, _ = evaluator_whole.find_best_threshold()
            best_threshold_chunked, _ = evaluator_chunked.find_best_threshold()
            
            self.assertEqual(best_threshold_whole, best_threshold_chunked)
            
        finally:
            # Clean up the larger temporary file
            if os.path.exists(larger_temp_file.name):
                os.unlink(larger_temp_file.name)
    
    def test_very_large_dataset_simulation(self):
        """Test processing of a simulated very large dataset."""
        # Create a test dataset similar to model_metrics.csv but as a pandas DataFrame
        test_data = pd.DataFrame({
            'threshold': [0.1, 0.2, 0.3, 0.4, 0.5],
            'true_positives': [100, 90, 80, 70, 60],
            'true_negatives': [60, 70, 80, 90, 100],
            'false_positives': [40, 30, 20, 10, 5],
            'false_negatives': [10, 20, 30, 40, 50]
        })
        
        # Create a custom iterator that will yield this test data
        def mock_data_iterator(*args, **kwargs):
            yield test_data
        
        # Create the evaluator with a real file path
        evaluator = ThresholdEvaluator(
            data_path=self.data_path,
            min_recall=0.7,  # Set min recall to test threshold selection
            chunk_size=2  # Small chunk size to test chunked processing
        )
        
        # Replace the _get_data_iterator method with our mock
        with patch.object(evaluator, '_get_data_iterator', side_effect=mock_data_iterator):
            # Calculate metrics
            metrics = evaluator.calculate_metrics()
            
            # Check that all metrics were processed
            self.assertEqual(len(metrics), 5)
            
            # Find best threshold
            best_threshold, best_metrics = evaluator.find_best_threshold()
            
            # Verify the best threshold meets the min_recall criteria
            self.assertIsNotNone(best_threshold)
            self.assertGreaterEqual(best_metrics['recall'], 0.7)
            
            # Verify the best threshold is 0.3 (based on our test data and optimization metrics)
            self.assertEqual(best_threshold, 0.3)
    
    # ====== STANDALONE FUNCTION TESTS ======
    
    def test_find_best_threshold_with_function(self):
        """Test the standalone function for finding best threshold."""
        result = find_best_threshold_with_min_recall(
            data_path=self.data_path,
            min_recall=0.8,
            optimization_metric="precision"
        )
        
        # We expect a valid result (not None) for this recall threshold
        self.assertIsNotNone(result)
        
        # Compare with direct calculation
        evaluator = ThresholdEvaluator(
            data_path=self.data_path,
            min_recall=0.8,
            optimization_metric="precision"
        )
        expected_threshold, _ = evaluator.find_best_threshold()
        self.assertEqual(result, expected_threshold)
    
    def test_find_best_threshold_error_handling(self):
        """Test error handling in the standalone function."""
        # The standalone function should return None for both file not found and
        # no valid thresholds, since it catches exceptions
        
        # Test with non-existent file
        nonexistent_result = find_best_threshold_with_min_recall(
            data_path="nonexistent_file.csv"
        )
        self.assertIsNone(nonexistent_result)
        
        # Test with impossible recall threshold
        impossible_result = find_best_threshold_with_min_recall(
            data_path=self.data_path,
            min_recall=0.99  # Higher than any threshold in our test data
        )
        self.assertIsNone(impossible_result)
        
        # Test with invalid parameter - should raise exception
        with self.assertRaises(ThresholdEvaluatorError):
            find_best_threshold_with_min_recall(
                data_path=self.data_path,
                min_recall=-0.1  # Invalid recall
            )
    
    def test_find_best_threshold_with_chunks(self):
        """Test the standalone function with chunked processing."""
        # Create a larger dataset
        larger_data = pd.concat([self.test_data] * 5, ignore_index=True)
        larger_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        larger_data.to_csv(larger_temp_file.name, index=False)
        larger_temp_file.close()
        
        try:
            # Process with and without chunking and compare results
            result_whole = find_best_threshold_with_min_recall(
                data_path=larger_temp_file.name,
                min_recall=0.8
            )
            
            result_chunked = find_best_threshold_with_min_recall(
                data_path=larger_temp_file.name,
                min_recall=0.8,
                chunk_size=3
            )
            
            # Results should be the same
            self.assertEqual(result_whole, result_chunked)
            
        finally:
            # Clean up
            if os.path.exists(larger_temp_file.name):
                os.unlink(larger_temp_file.name)


if __name__ == "__main__":
    unittest.main() 