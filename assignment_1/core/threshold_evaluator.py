"""Module for evaluating binary classification model thresholds."""

import os
import pandas as pd
from typing import Dict, Optional, Tuple, Iterator, Any

from assignment_1.core.logger import setup_logger
from assignment_1.core.constant import (
    DEFAULT_DATA_PATH, 
    MIN_RECALL_THRESHOLD,
    METRICS_COLUMNS,
    DEFAULT_OPTIMIZATION_METRIC
)
from assignment_1.core.exception import (
    ThresholdEvaluatorError,
    DataValidationError,
    DataProcessingError,
    MetricCalculationError,
    ThresholdNotFoundError,
    InvalidParameterError,
    FileOperationError
)


class ThresholdEvaluator:
    """Class for evaluating classification thresholds."""
    
    def __init__(
        self,
        data_path: str = DEFAULT_DATA_PATH,
        min_recall: float = MIN_RECALL_THRESHOLD,
        optimization_metric: str = DEFAULT_OPTIMIZATION_METRIC,
        chunk_size: Optional[int] = None
    ):
        """Initialize the evaluator.
        
        Args:
            data_path: Path to the metrics data file
            min_recall: Minimum acceptable recall value
            optimization_metric: Metric to optimize (precision, f1, or specificity)
            chunk_size: Number of rows to process at once for large files (None = load all at once)
            
        Raises:
            FileOperationError: If the data file doesn't exist or can't be accessed
            InvalidParameterError: If any of the parameters are invalid
        """
        self.logger = setup_logger()
        self.data_path = data_path
        self.min_recall = min_recall
        self.optimization_metric = optimization_metric
        self.chunk_size = chunk_size
        self.metrics_df = None
        
        # Validate parameters
        self._validate_parameters()
        
        # For small datasets, load everything at once
        if not self.chunk_size:
            self._load_data()
    
    def _validate_parameters(self) -> None:
        """Validate the input parameters.
        
        Raises:
            FileOperationError: If the data file doesn't exist
            InvalidParameterError: If min_recall or optimization_metric is invalid
        """
        if not os.path.exists(self.data_path):
            raise FileOperationError(
                f"Data file not found: {self.data_path}",
                file_path=self.data_path,
                operation="read"
            )
        
        if not 0 <= self.min_recall <= 1:
            raise InvalidParameterError(
                f"min_recall must be between 0 and 1, got {self.min_recall}",
                parameter_name="min_recall",
                parameter_value=self.min_recall,
                valid_values=["Value between 0 and 1"]
            )
        
        valid_metrics = ["precision", "f1", "specificity"]
        if self.optimization_metric not in valid_metrics:
            raise InvalidParameterError(
                f"optimization_metric must be one of {valid_metrics}, got {self.optimization_metric}",
                parameter_name="optimization_metric",
                parameter_value=self.optimization_metric,
                valid_values=valid_metrics
            )
        
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise InvalidParameterError(
                f"chunk_size must be positive, got {self.chunk_size}",
                parameter_name="chunk_size",
                parameter_value=self.chunk_size,
                valid_values=["Positive integer"]
            )
    
    def _load_data(self) -> None:
        """Load and validate the metrics data.
        
        Raises:
            FileOperationError: If there's an error reading the file
            DataValidationError: If the data doesn't have the required columns
        """
        try:
            # Check if file is empty before trying to memory map it
            if os.path.getsize(self.data_path) == 0:
                raise DataValidationError(
                    f"The file {self.data_path} is empty",
                    details={"file_path": self.data_path}
                )
            
            # Try reading with memory_map=False first for empty files
            try:
                self.metrics_df = pd.read_csv(self.data_path, memory_map=True)
            except ValueError as e:
                # Handle the specific error for memory-mapping empty files
                if "cannot mmap an empty file" in str(e):
                    # Try again without memory mapping
                    self.metrics_df = pd.read_csv(self.data_path, memory_map=False)
                else:
                    # Re-raise other ValueError exceptions
                    raise
                
            self.logger.info(f"Successfully loaded data from {self.data_path}")
            
            # Validate data columns
            missing_cols = set(METRICS_COLUMNS) - set(self.metrics_df.columns)
            if missing_cols:
                raise DataValidationError(
                    f"Missing required columns in data: {missing_cols}",
                    invalid_columns=list(missing_cols)
                )
            
            # Check if dataframe has no rows (header only)
            if len(self.metrics_df) == 0:
                raise DataValidationError(
                    f"The file {self.data_path} contains headers but no data",
                    details={"file_path": self.data_path}
                )
            
            # Ensure threshold is sorted
            self.metrics_df = self.metrics_df.sort_values("threshold")
            
        except pd.errors.EmptyDataError:
            raise DataValidationError(
                f"The file {self.data_path} is empty",
                details={"file_path": self.data_path}
            )
        except pd.errors.ParserError as e:
            raise DataValidationError(
                f"Error parsing file {self.data_path}: {str(e)}",
                details={"file_path": self.data_path, "error": str(e)}
            )
        except Exception as e:
            if isinstance(e, ThresholdEvaluatorError):
                raise
            raise FileOperationError(
                f"Error loading data from {self.data_path}: {str(e)}",
                file_path=self.data_path,
                operation="read",
                details={"error": str(e)}
            )
    
    def _get_data_iterator(self) -> Iterator[pd.DataFrame]:
        """Get an iterator over data chunks.
        
        Returns:
            Iterator yielding chunks of the dataset
            
        Raises:
            FileOperationError: If there's an error reading the file
            DataValidationError: If the data doesn't have the required columns
        """
        if self.metrics_df is not None:
            # If data is already loaded, yield it as a single chunk
            yield self.metrics_df
            return
        
        # For large datasets, process in chunks
        try:
            chunk_idx = 0
            for chunk in pd.read_csv(self.data_path, chunksize=self.chunk_size, memory_map=True):
                # Validate columns in the first chunk
                if chunk_idx == 0:
                    missing_cols = set(METRICS_COLUMNS) - set(chunk.columns)
                    if missing_cols:
                        raise DataValidationError(
                            f"Missing required columns in data: {missing_cols}",
                            invalid_columns=list(missing_cols)
                        )
                
                self.logger.debug(f"Processing chunk {chunk_idx+1}")
                yield chunk
                chunk_idx += 1
                
        except pd.errors.EmptyDataError:
            raise DataValidationError(
                f"The file {self.data_path} is empty",
                details={"file_path": self.data_path}
            )
        except pd.errors.ParserError as e:
            raise DataValidationError(
                f"Error parsing file {self.data_path}: {str(e)}",
                details={"file_path": self.data_path, "error": str(e)}
            )
        except Exception as e:
            if isinstance(e, ThresholdEvaluatorError):
                raise
            raise DataProcessingError(
                f"Error processing data chunks: {str(e)}",
                chunk_index=chunk_idx,
                details={"error": str(e)}
            )
    
    def calculate_metrics(self) -> pd.DataFrame:
        """Calculate performance metrics for all thresholds.
        
        Returns:
            DataFrame with threshold and performance metrics
            
        Raises:
            DataProcessingError: If there's an error processing the data
            MetricCalculationError: If there's an error calculating metrics
        """
        try:
            # For small datasets already loaded in memory
            if self.metrics_df is not None:
                return self._calculate_metrics_single_chunk(self.metrics_df)
            
            # For large datasets, process in chunks and combine results
            all_results = []
            for i, chunk in enumerate(self._get_data_iterator()):
                try:
                    chunk_results = self._calculate_metrics_single_chunk(chunk)
                    all_results.append(chunk_results)
                except Exception as e:
                    if isinstance(e, ThresholdEvaluatorError):
                        raise
                    raise MetricCalculationError(
                        f"Error calculating metrics for chunk {i}: {str(e)}",
                        details={"chunk_index": i, "error": str(e)}
                    )
            
            # Combine all chunks
            if not all_results:
                raise DataProcessingError(
                    "No data processed - dataset may be empty",
                    details={"file_path": self.data_path}
                )
                
            combined_results = pd.concat(all_results)
            
            # Ensure threshold is sorted
            combined_results = combined_results.sort_values("threshold")
            
            self.logger.info("Successfully calculated metrics for all thresholds")
            return combined_results
            
        except Exception as e:
            if isinstance(e, ThresholdEvaluatorError):
                raise
            raise DataProcessingError(
                f"Error processing data: {str(e)}",
                details={"error": str(e)}
            )
    
    def _calculate_metrics_single_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate metrics for a single chunk of data.
        
        Args:
            df: DataFrame chunk to process
            
        Returns:
            DataFrame with calculated metrics
            
        Raises:
            MetricCalculationError: If there's an error calculating metrics
        """
        try:
            # Make a copy to avoid modifying the original
            results = df.copy()
            
            # Validate that required columns have valid values
            for col in ["true_positives", "true_negatives", "false_positives", "false_negatives"]:
                if (results[col] < 0).any():
                    raise DataValidationError(
                        f"Column '{col}' contains negative values",
                        invalid_columns=[col]
                    )
            
            # Check for division by zero
            zero_denominators = (
                (results["true_positives"] + results["false_negatives"] == 0) |
                (results["true_positives"] + results["false_positives"] == 0) |
                (results["true_negatives"] + results["false_positives"] == 0)
            )
            
            if zero_denominators.any():
                self.logger.warning(
                    f"Some rows have zero denominators for metric calculations. "
                    f"These will result in NaN values."
                )
            
            # Use vectorized operations for efficiency
            tp = results["true_positives"]
            tn = results["true_negatives"]
            fp = results["false_positives"]
            fn = results["false_negatives"]
            
            # Calculate metrics
            results["recall"] = tp / (tp + fn)
            results["precision"] = tp / (tp + fp)
            results["specificity"] = tn / (tn + fp)
            
            # Calculate F1 score directly to avoid potential division by zero
            results["f1"] = 2 * tp / (2 * tp + fp + fn)
            
            return results
            
        except Exception as e:
            if isinstance(e, ThresholdEvaluatorError):
                raise
            raise MetricCalculationError(
                f"Error calculating metrics: {str(e)}",
                details={"error": str(e)}
            )
    
    def find_best_threshold(self) -> Tuple[Optional[float], Dict[str, Any]]:
        """Find the best threshold with recall >= min_recall.
        
        Returns:
            Tuple containing:
                - Best threshold value (or None if no valid threshold found)
                - Dictionary of metrics for the best threshold (empty if none found)
                
        Raises:
            DataProcessingError: If there's an error processing the data
            MetricCalculationError: If there's an error calculating metrics
        """
        try:
            metrics_df = self.calculate_metrics()
            
            # Filter thresholds with recall >= min_recall
            valid_thresholds = metrics_df[metrics_df["recall"] >= self.min_recall]
            
            if valid_thresholds.empty:
                max_recall = metrics_df["recall"].max()
                error_msg = f"No thresholds found with recall >= {self.min_recall}"
                self.logger.warning(error_msg)
                
                raise ThresholdNotFoundError(
                    error_msg,
                    min_recall=self.min_recall,
                    max_available_recall=float(max_recall)
                )
            
            # Find the threshold with the best optimization metric
            best_idx = valid_thresholds[self.optimization_metric].idxmax()
            best_row = valid_thresholds.loc[best_idx]
            
            # Extract the threshold as a float
            best_threshold = float(best_row["threshold"])
            
            # Create metrics dictionary - convert Series values to Python types
            metrics = {
                "threshold": best_threshold,
                "recall": float(best_row["recall"]),
                "precision": float(best_row["precision"]),
                "specificity": float(best_row["specificity"]),
                "f1": float(best_row["f1"]),
                "true_positives": int(best_row["true_positives"]),
                "true_negatives": int(best_row["true_negatives"]),
                "false_positives": int(best_row["false_positives"]),
                "false_negatives": int(best_row["false_negatives"])
            }
            
            self.logger.info(
                f"Found best threshold {best_threshold} with "
                f"{self.optimization_metric}={metrics[self.optimization_metric]:.4f}, "
                f"recall={metrics['recall']:.4f}"
            )
            
            return best_threshold, metrics
            
        except ThresholdNotFoundError:
            # Re-raise the ThresholdNotFoundError to be handled by the caller
            raise
        except Exception as e:
            if isinstance(e, ThresholdEvaluatorError):
                raise
            raise DataProcessingError(
                f"Error finding best threshold: {str(e)}",
                details={"error": str(e)}
            )


def find_best_threshold_with_min_recall(
    data_path: str = DEFAULT_DATA_PATH,
    min_recall: float = MIN_RECALL_THRESHOLD,
    optimization_metric: str = DEFAULT_OPTIMIZATION_METRIC,
    chunk_size: Optional[int] = None
) -> Optional[float]:
    """Find the best threshold that yields a recall >= min_recall.
    
    Args:
        data_path: Path to the metrics data file
        min_recall: Minimum acceptable recall value
        optimization_metric: Metric to optimize (precision, f1, or specificity)
        chunk_size: Number of rows to process at once for large files
        
    Returns:
        Best threshold value or None if no threshold meets criteria
        
    Raises:
        ThresholdEvaluatorError: Base class for all threshold evaluator errors
    """
    logger = setup_logger("find_best_threshold")
    
    try:
        evaluator = ThresholdEvaluator(
            data_path=data_path,
            min_recall=min_recall,
            optimization_metric=optimization_metric,
            chunk_size=chunk_size
        )
        
        best_threshold, _ = evaluator.find_best_threshold()
        return best_threshold
        
    except ThresholdNotFoundError as e:
        logger.warning(f"{e.message} (max available recall: {e.max_available_recall:.4f})")
        return None
    except FileOperationError as e:
        logger.error(f"File operation error: {e.message}")
        return None
    except ThresholdEvaluatorError as e:
        logger.error(f"{e.__class__.__name__}: {e.message}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise ThresholdEvaluatorError(f"Unexpected error: {str(e)}", {"error": str(e)}) 