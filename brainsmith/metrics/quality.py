"""
Quality Metrics Framework
Accuracy, precision, reliability assessment, and validation metrics for FINN implementations.
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import statistics

from .core import MetricsCollector, MetricValue, MetricCollection, MetricType, MetricScope

logger = logging.getLogger(__name__)


@dataclass
class AccuracyMetrics:
    """Accuracy assessment metrics."""
    top1_accuracy: float
    top5_accuracy: Optional[float] = None
    mean_absolute_error: Optional[float] = None
    mean_squared_error: Optional[float] = None
    root_mean_squared_error: Optional[float] = None
    mean_absolute_percentage_error: Optional[float] = None
    r_squared: Optional[float] = None
    classification_accuracy: Optional[float] = None
    confusion_matrix: Optional[Dict[str, Any]] = None


@dataclass
class PrecisionMetrics:
    """Precision and recall metrics."""
    precision: float
    recall: float
    f1_score: float
    specificity: Optional[float] = None
    sensitivity: Optional[float] = None
    area_under_curve: Optional[float] = None
    precision_recall_curve: Optional[Dict[str, List[float]]] = None
    average_precision: Optional[float] = None


@dataclass
class ReliabilityMetrics:
    """Reliability assessment metrics."""
    output_consistency: float  # Consistency across multiple runs
    numerical_stability: float  # Stability of numerical computations
    bit_accuracy: float  # Bit-exact accuracy vs reference
    error_rate: float  # Overall error rate
    mean_time_between_failures: Optional[float] = None
    availability: Optional[float] = None
    fault_tolerance: Optional[float] = None


@dataclass
class ValidationResults:
    """Comprehensive validation results."""
    validation_passed: bool
    accuracy_metrics: AccuracyMetrics
    precision_metrics: Optional[PrecisionMetrics] = None
    reliability_metrics: Optional[ReliabilityMetrics] = None
    test_coverage: float = 0.0
    validation_time: float = 0.0
    test_cases_passed: int = 0
    test_cases_total: int = 0
    validation_errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class AccuracyAnalyzer:
    """Analyze accuracy of FINN implementations against reference."""
    
    def __init__(self):
        self.tolerance_config = {
            'fp32_tolerance': 1e-6,
            'fp16_tolerance': 1e-3,
            'int8_tolerance': 0.01,
            'int4_tolerance': 0.05
        }
    
    def analyze_accuracy(self,
                        predicted_outputs: Union[np.ndarray, List[float]],
                        reference_outputs: Union[np.ndarray, List[float]],
                        task_type: str = 'classification',
                        data_type: str = 'fp32') -> AccuracyMetrics:
        """Analyze accuracy between predicted and reference outputs."""
        
        try:
            # Convert to numpy arrays
            if not isinstance(predicted_outputs, np.ndarray):
                predicted_outputs = np.array(predicted_outputs)
            if not isinstance(reference_outputs, np.ndarray):
                reference_outputs = np.array(reference_outputs)
            
            # Ensure same shape
            if predicted_outputs.shape != reference_outputs.shape:
                logger.warning(f"Shape mismatch: predicted {predicted_outputs.shape} vs reference {reference_outputs.shape}")
                # Try to reshape or truncate
                min_size = min(predicted_outputs.size, reference_outputs.size)
                predicted_outputs = predicted_outputs.flatten()[:min_size]
                reference_outputs = reference_outputs.flatten()[:min_size]
            
            if task_type == 'classification':
                return self._analyze_classification_accuracy(predicted_outputs, reference_outputs)
            elif task_type == 'regression':
                return self._analyze_regression_accuracy(predicted_outputs, reference_outputs, data_type)
            else:
                return self._analyze_general_accuracy(predicted_outputs, reference_outputs, data_type)
            
        except Exception as e:
            logger.error(f"Accuracy analysis failed: {e}")
            return AccuracyMetrics(
                top1_accuracy=0.0,
                mean_absolute_error=float('inf'),
                mean_squared_error=float('inf')
            )
    
    def _analyze_classification_accuracy(self, predicted: np.ndarray, reference: np.ndarray) -> AccuracyMetrics:
        """Analyze classification accuracy."""
        
        # For classification, assume outputs are class probabilities or logits
        if predicted.ndim > 1:
            # Multi-class classification
            predicted_classes = np.argmax(predicted, axis=-1)
            reference_classes = np.argmax(reference, axis=-1) if reference.ndim > 1 else reference.astype(int)
            
            # Top-1 accuracy
            top1_accuracy = np.mean(predicted_classes == reference_classes)
            
            # Top-5 accuracy (if applicable)
            top5_accuracy = None
            if predicted.shape[-1] >= 5:
                top5_predictions = np.argsort(predicted, axis=-1)[:, -5:]
                top5_accuracy = np.mean([ref in top5_pred for ref, top5_pred in zip(reference_classes, top5_predictions)])
            
            # Generate confusion matrix for small number of classes
            num_classes = predicted.shape[-1]
            confusion_matrix = None
            if num_classes <= 20:  # Only for manageable number of classes
                confusion_matrix = self._generate_confusion_matrix(predicted_classes, reference_classes, num_classes)
            
        else:
            # Binary classification or single values
            if np.all((predicted >= 0) & (predicted <= 1)):
                # Probabilities - threshold at 0.5
                predicted_classes = (predicted > 0.5).astype(int)
            else:
                # Logits or other values - threshold at 0
                predicted_classes = (predicted > 0).astype(int)
            
            reference_classes = reference.astype(int)
            top1_accuracy = np.mean(predicted_classes == reference_classes)
            top5_accuracy = None
            
            # Binary confusion matrix
            confusion_matrix = self._generate_confusion_matrix(predicted_classes, reference_classes, 2)
        
        # Calculate additional regression-like metrics for the raw outputs
        mae = np.mean(np.abs(predicted - reference))
        mse = np.mean((predicted - reference) ** 2)
        rmse = np.sqrt(mse)
        
        return AccuracyMetrics(
            top1_accuracy=float(top1_accuracy),
            top5_accuracy=float(top5_accuracy) if top5_accuracy is not None else None,
            mean_absolute_error=float(mae),
            mean_squared_error=float(mse),
            root_mean_squared_error=float(rmse),
            classification_accuracy=float(top1_accuracy),
            confusion_matrix=confusion_matrix
        )
    
    def _analyze_regression_accuracy(self, predicted: np.ndarray, reference: np.ndarray, data_type: str) -> AccuracyMetrics:
        """Analyze regression accuracy."""
        
        # Calculate error metrics
        errors = predicted - reference
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error (avoid division by zero)
        reference_nonzero = reference[reference != 0]
        predicted_nonzero = predicted[reference != 0]
        if len(reference_nonzero) > 0:
            mape = np.mean(np.abs((reference_nonzero - predicted_nonzero) / reference_nonzero)) * 100
        else:
            mape = float('inf')
        
        # R-squared
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((reference - np.mean(reference)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        # Calculate "accuracy" based on tolerance
        tolerance = self.tolerance_config.get(f"{data_type}_tolerance", 1e-6)
        within_tolerance = np.abs(errors) <= tolerance
        accuracy = np.mean(within_tolerance)
        
        return AccuracyMetrics(
            top1_accuracy=float(accuracy),
            mean_absolute_error=float(mae),
            mean_squared_error=float(mse),
            root_mean_squared_error=float(rmse),
            mean_absolute_percentage_error=float(mape),
            r_squared=float(r_squared)
        )
    
    def _analyze_general_accuracy(self, predicted: np.ndarray, reference: np.ndarray, data_type: str) -> AccuracyMetrics:
        """Analyze general accuracy (when task type is unknown)."""
        
        # Use regression-style analysis as default
        return self._analyze_regression_accuracy(predicted, reference, data_type)
    
    def _generate_confusion_matrix(self, predicted: np.ndarray, reference: np.ndarray, num_classes: int) -> Dict[str, Any]:
        """Generate confusion matrix."""
        
        confusion = np.zeros((num_classes, num_classes), dtype=int)
        
        for pred, ref in zip(predicted, reference):
            if 0 <= pred < num_classes and 0 <= ref < num_classes:
                confusion[int(ref), int(pred)] += 1
        
        return {
            'matrix': confusion.tolist(),
            'num_classes': num_classes,
            'total_samples': len(predicted)
        }


class PrecisionTracker:
    """Track precision and recall metrics."""
    
    def analyze_precision_recall(self,
                                predicted_outputs: np.ndarray,
                                reference_outputs: np.ndarray,
                                threshold: float = 0.5) -> PrecisionMetrics:
        """Analyze precision and recall metrics."""
        
        try:
            # Convert to binary predictions if needed
            if predicted_outputs.ndim > 1:
                # Multi-class: use max probability class
                predicted_binary = np.max(predicted_outputs, axis=-1) > threshold
                reference_binary = np.max(reference_outputs, axis=-1) > threshold if reference_outputs.ndim > 1 else reference_outputs > threshold
            else:
                predicted_binary = predicted_outputs > threshold
                reference_binary = reference_outputs > threshold
            
            # Calculate confusion matrix elements
            tp = np.sum((predicted_binary == 1) & (reference_binary == 1))
            fp = np.sum((predicted_binary == 1) & (reference_binary == 0))
            tn = np.sum((predicted_binary == 0) & (reference_binary == 0))
            fn = np.sum((predicted_binary == 0) & (reference_binary == 1))
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensitivity = recall  # Same as recall
            
            return PrecisionMetrics(
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1_score),
                specificity=float(specificity),
                sensitivity=float(sensitivity)
            )
            
        except Exception as e:
            logger.error(f"Precision/recall analysis failed: {e}")
            return PrecisionMetrics(
                precision=0.0,
                recall=0.0,
                f1_score=0.0
            )


class ReliabilityAssessment:
    """Assess reliability of FINN implementations."""
    
    def assess_reliability(self,
                          multiple_runs_outputs: List[np.ndarray],
                          reference_output: np.ndarray,
                          bit_exact_required: bool = False) -> ReliabilityMetrics:
        """Assess reliability across multiple runs."""
        
        try:
            if len(multiple_runs_outputs) < 2:
                logger.warning("Need at least 2 runs for reliability assessment")
                return ReliabilityMetrics(
                    output_consistency=0.0,
                    numerical_stability=0.0,
                    bit_accuracy=0.0,
                    error_rate=1.0
                )
            
            # Convert to numpy arrays
            runs = [np.array(output) for output in multiple_runs_outputs]
            reference = np.array(reference_output)
            
            # Assess output consistency across runs
            consistency = self._assess_output_consistency(runs)
            
            # Assess numerical stability
            stability = self._assess_numerical_stability(runs)
            
            # Assess bit-exact accuracy
            bit_accuracy = self._assess_bit_accuracy(runs, reference, bit_exact_required)
            
            # Calculate overall error rate
            error_rate = self._calculate_error_rate(runs, reference)
            
            return ReliabilityMetrics(
                output_consistency=float(consistency),
                numerical_stability=float(stability),
                bit_accuracy=float(bit_accuracy),
                error_rate=float(error_rate)
            )
            
        except Exception as e:
            logger.error(f"Reliability assessment failed: {e}")
            return ReliabilityMetrics(
                output_consistency=0.0,
                numerical_stability=0.0,
                bit_accuracy=0.0,
                error_rate=1.0
            )
    
    def _assess_output_consistency(self, runs: List[np.ndarray]) -> float:
        """Assess consistency of outputs across multiple runs."""
        
        if len(runs) < 2:
            return 1.0
        
        # Calculate pairwise similarities
        similarities = []
        
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                run1, run2 = runs[i], runs[j]
                
                # Ensure same shape
                if run1.shape != run2.shape:
                    min_size = min(run1.size, run2.size)
                    run1 = run1.flatten()[:min_size]
                    run2 = run2.flatten()[:min_size]
                
                # Calculate similarity (1 - normalized difference)
                diff = np.abs(run1 - run2)
                max_diff = np.maximum(np.abs(run1), np.abs(run2))
                normalized_diff = np.mean(diff / (max_diff + 1e-8))  # Add small epsilon
                similarity = 1.0 - normalized_diff
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _assess_numerical_stability(self, runs: List[np.ndarray]) -> float:
        """Assess numerical stability by measuring variance across runs."""
        
        if len(runs) < 2:
            return 1.0
        
        # Stack all runs
        try:
            stacked = np.stack(runs)
            
            # Calculate coefficient of variation for each output element
            means = np.mean(stacked, axis=0)
            stds = np.std(stacked, axis=0)
            
            # Coefficient of variation (avoid division by zero)
            cv = stds / (np.abs(means) + 1e-8)
            
            # Stability is inverse of average coefficient of variation
            avg_cv = np.mean(cv)
            stability = 1.0 / (1.0 + avg_cv)
            
            return float(stability)
            
        except Exception:
            # If stacking fails due to shape mismatch, use pairwise analysis
            return self._assess_output_consistency(runs)
    
    def _assess_bit_accuracy(self, runs: List[np.ndarray], reference: np.ndarray, bit_exact_required: bool) -> float:
        """Assess bit-exact accuracy against reference."""
        
        if bit_exact_required:
            # For bit-exact requirements, check exact equality
            exact_matches = []
            
            for run in runs:
                if run.shape != reference.shape:
                    min_size = min(run.size, reference.size)
                    run_flat = run.flatten()[:min_size]
                    ref_flat = reference.flatten()[:min_size]
                else:
                    run_flat = run.flatten()
                    ref_flat = reference.flatten()
                
                exact_match = np.mean(run_flat == ref_flat)
                exact_matches.append(exact_match)
            
            return np.mean(exact_matches)
        
        else:
            # For floating-point, use tolerance-based accuracy
            tolerance = 1e-6
            tolerance_matches = []
            
            for run in runs:
                if run.shape != reference.shape:
                    min_size = min(run.size, reference.size)
                    run_flat = run.flatten()[:min_size]
                    ref_flat = reference.flatten()[:min_size]
                else:
                    run_flat = run.flatten()
                    ref_flat = reference.flatten()
                
                within_tolerance = np.abs(run_flat - ref_flat) <= tolerance
                tolerance_match = np.mean(within_tolerance)
                tolerance_matches.append(tolerance_match)
            
            return np.mean(tolerance_matches)
    
    def _calculate_error_rate(self, runs: List[np.ndarray], reference: np.ndarray) -> float:
        """Calculate overall error rate."""
        
        # Use the most accurate run as baseline
        accuracies = []
        
        for run in runs:
            if run.shape != reference.shape:
                min_size = min(run.size, reference.size)
                run_flat = run.flatten()[:min_size]
                ref_flat = reference.flatten()[:min_size]
            else:
                run_flat = run.flatten()
                ref_flat = reference.flatten()
            
            # Calculate relative error
            errors = np.abs(run_flat - ref_flat)
            relative_errors = errors / (np.abs(ref_flat) + 1e-8)
            accuracy = 1.0 - np.mean(relative_errors)
            accuracies.append(max(0.0, accuracy))
        
        # Error rate is 1 - best accuracy
        best_accuracy = max(accuracies) if accuracies else 0.0
        return 1.0 - best_accuracy


class ValidationMetrics:
    """Comprehensive validation metrics framework."""
    
    def __init__(self):
        self.accuracy_analyzer = AccuracyAnalyzer()
        self.precision_tracker = PrecisionTracker()
        self.reliability_assessment = ReliabilityAssessment()
    
    def validate_implementation(self,
                              predicted_outputs: Union[np.ndarray, List[float]],
                              reference_outputs: Union[np.ndarray, List[float]],
                              multiple_runs: Optional[List[np.ndarray]] = None,
                              task_type: str = 'classification',
                              data_type: str = 'fp32',
                              accuracy_threshold: float = 0.95,
                              precision_threshold: float = 0.90,
                              reliability_threshold: float = 0.95) -> ValidationResults:
        """Perform comprehensive validation of FINN implementation."""
        
        start_time = time.time()
        validation_errors = []
        recommendations = []
        
        try:
            # Accuracy analysis
            accuracy_metrics = self.accuracy_analyzer.analyze_accuracy(
                predicted_outputs, reference_outputs, task_type, data_type
            )
            
            # Precision analysis (for classification tasks)
            precision_metrics = None
            if task_type == 'classification':
                precision_metrics = self.precision_tracker.analyze_precision_recall(
                    np.array(predicted_outputs), np.array(reference_outputs)
                )
            
            # Reliability analysis (if multiple runs provided)
            reliability_metrics = None
            if multiple_runs and len(multiple_runs) > 1:
                reliability_metrics = self.reliability_assessment.assess_reliability(
                    multiple_runs, reference_outputs
                )
            
            # Determine if validation passed
            validation_passed = True
            test_cases_passed = 0
            test_cases_total = 0
            
            # Check accuracy threshold
            test_cases_total += 1
            if accuracy_metrics.top1_accuracy >= accuracy_threshold:
                test_cases_passed += 1
            else:
                validation_passed = False
                validation_errors.append(f"Accuracy {accuracy_metrics.top1_accuracy:.3f} below threshold {accuracy_threshold}")
                recommendations.append(f"Improve accuracy: current {accuracy_metrics.top1_accuracy:.3f}, target {accuracy_threshold}")
            
            # Check precision threshold (if applicable)
            if precision_metrics:
                test_cases_total += 1
                if precision_metrics.f1_score >= precision_threshold:
                    test_cases_passed += 1
                else:
                    validation_passed = False
                    validation_errors.append(f"F1-score {precision_metrics.f1_score:.3f} below threshold {precision_threshold}")
                    recommendations.append(f"Improve precision/recall: current F1 {precision_metrics.f1_score:.3f}, target {precision_threshold}")
            
            # Check reliability threshold (if applicable)
            if reliability_metrics:
                test_cases_total += 1
                if reliability_metrics.output_consistency >= reliability_threshold:
                    test_cases_passed += 1
                else:
                    validation_passed = False
                    validation_errors.append(f"Output consistency {reliability_metrics.output_consistency:.3f} below threshold {reliability_threshold}")
                    recommendations.append(f"Improve consistency: current {reliability_metrics.output_consistency:.3f}, target {reliability_threshold}")
            
            # Calculate test coverage
            test_coverage = (test_cases_passed / test_cases_total * 100) if test_cases_total > 0 else 0.0
            
            # Add performance recommendations
            if accuracy_metrics.top1_accuracy < 0.9:
                recommendations.append("Consider increasing model capacity or improving training data")
            
            if precision_metrics and precision_metrics.precision < precision_metrics.recall:
                recommendations.append("Consider adjusting decision threshold to balance precision/recall")
            
            if reliability_metrics and reliability_metrics.numerical_stability < 0.9:
                recommendations.append("Consider using higher precision arithmetic or improving numerical stability")
            
            validation_time = time.time() - start_time
            
            return ValidationResults(
                validation_passed=validation_passed,
                accuracy_metrics=accuracy_metrics,
                precision_metrics=precision_metrics,
                reliability_metrics=reliability_metrics,
                test_coverage=test_coverage,
                validation_time=validation_time,
                test_cases_passed=test_cases_passed,
                test_cases_total=test_cases_total,
                validation_errors=validation_errors,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_time = time.time() - start_time
            
            return ValidationResults(
                validation_passed=False,
                accuracy_metrics=AccuracyMetrics(top1_accuracy=0.0),
                test_coverage=0.0,
                validation_time=validation_time,
                test_cases_passed=0,
                test_cases_total=1,
                validation_errors=[f"Validation error: {str(e)}"],
                recommendations=["Fix validation errors and retry"]
            )


class QualityMetricsCollector(MetricsCollector):
    """Main quality metrics collector that implements MetricsCollector interface."""
    
    def __init__(self, name: str = "QualityMetricsCollector", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.validation_metrics = ValidationMetrics()
    
    def collect_metrics(self, context: Dict[str, Any]) -> MetricCollection:
        """Collect quality metrics."""
        
        collection = MetricCollection(
            collection_id=f"quality_{int(time.time())}",
            name="Quality Metrics",
            description="Accuracy, precision, reliability, and validation metrics"
        )
        
        try:
            # Extract context information
            predicted_outputs = context.get('predicted_outputs')
            reference_outputs = context.get('reference_outputs')
            multiple_runs = context.get('multiple_runs')
            task_type = context.get('task_type', 'classification')
            data_type = context.get('data_type', 'fp32')
            
            # Generate mock data if not provided (for testing)
            if predicted_outputs is None or reference_outputs is None:
                predicted_outputs, reference_outputs, multiple_runs = self._generate_mock_data()
            
            # Perform validation
            validation_results = self.validation_metrics.validate_implementation(
                predicted_outputs=predicted_outputs,
                reference_outputs=reference_outputs,
                multiple_runs=multiple_runs,
                task_type=task_type,
                data_type=data_type
            )
            
            # Add validation metrics to collection
            self._add_validation_metrics(collection, validation_results)
            
        except Exception as e:
            logger.error(f"Failed to collect quality metrics: {e}")
            collection.add_metric(MetricValue(
                name="collection_error",
                value=str(e),
                metric_type=MetricType.QUALITY,
                scope=MetricScope.BUILD
            ))
        
        return collection
    
    def get_supported_metrics(self) -> List[str]:
        """Get list of supported metrics."""
        return [
            "accuracy", "precision", "recall", "f1_score",
            "output_consistency", "numerical_stability", "bit_accuracy",
            "validation_passed", "test_coverage"
        ]
    
    def _generate_mock_data(self) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Generate mock data for testing."""
        
        # Generate mock classification outputs
        num_samples = 100
        num_classes = 10
        
        # Reference outputs (ground truth)
        reference = np.random.randint(0, num_classes, num_samples)
        reference_onehot = np.eye(num_classes)[reference]
        
        # Predicted outputs (with some noise)
        predicted = reference_onehot + np.random.normal(0, 0.1, (num_samples, num_classes))
        predicted = np.maximum(0, predicted)  # Ensure non-negative
        predicted = predicted / np.sum(predicted, axis=1, keepdims=True)  # Normalize
        
        # Multiple runs (with slight variations)
        multiple_runs = []
        for i in range(3):
            run = predicted + np.random.normal(0, 0.05, predicted.shape)
            run = np.maximum(0, run)
            run = run / np.sum(run, axis=1, keepdims=True)
            multiple_runs.append(run)
        
        return predicted, reference_onehot, multiple_runs
    
    def _add_validation_metrics(self, collection: MetricCollection, validation: ValidationResults):
        """Add validation metrics to collection."""
        
        # Overall validation metrics
        collection.add_metric(MetricValue(
            "validation_passed", validation.validation_passed, metric_type=MetricType.QUALITY
        ))
        
        collection.add_metric(MetricValue(
            "test_coverage", validation.test_coverage, "%", metric_type=MetricType.QUALITY
        ))
        
        collection.add_metric(MetricValue(
            "validation_time", validation.validation_time, "s", metric_type=MetricType.PERFORMANCE
        ))
        
        collection.add_metric(MetricValue(
            "test_cases_passed", validation.test_cases_passed, "count", metric_type=MetricType.QUALITY
        ))
        
        collection.add_metric(MetricValue(
            "test_cases_total", validation.test_cases_total, "count", metric_type=MetricType.QUALITY
        ))
        
        # Accuracy metrics
        accuracy = validation.accuracy_metrics
        collection.add_metric(MetricValue(
            "accuracy", accuracy.top1_accuracy, "ratio", metric_type=MetricType.ACCURACY
        ))
        
        if accuracy.top5_accuracy is not None:
            collection.add_metric(MetricValue(
                "top5_accuracy", accuracy.top5_accuracy, "ratio", metric_type=MetricType.ACCURACY
            ))
        
        if accuracy.mean_absolute_error is not None:
            collection.add_metric(MetricValue(
                "mean_absolute_error", accuracy.mean_absolute_error, metric_type=MetricType.ACCURACY
            ))
        
        if accuracy.root_mean_squared_error is not None:
            collection.add_metric(MetricValue(
                "rmse", accuracy.root_mean_squared_error, metric_type=MetricType.ACCURACY
            ))
        
        if accuracy.r_squared is not None:
            collection.add_metric(MetricValue(
                "r_squared", accuracy.r_squared, "ratio", metric_type=MetricType.ACCURACY
            ))
        
        # Precision metrics
        if validation.precision_metrics:
            precision = validation.precision_metrics
            collection.add_metric(MetricValue(
                "precision", precision.precision, "ratio", metric_type=MetricType.PRECISION
            ))
            
            collection.add_metric(MetricValue(
                "recall", precision.recall, "ratio", metric_type=MetricType.PRECISION
            ))
            
            collection.add_metric(MetricValue(
                "f1_score", precision.f1_score, "ratio", metric_type=MetricType.PRECISION
            ))
            
            if precision.specificity is not None:
                collection.add_metric(MetricValue(
                    "specificity", precision.specificity, "ratio", metric_type=MetricType.PRECISION
                ))
        
        # Reliability metrics
        if validation.reliability_metrics:
            reliability = validation.reliability_metrics
            collection.add_metric(MetricValue(
                "output_consistency", reliability.output_consistency, "ratio", metric_type=MetricType.QUALITY
            ))
            
            collection.add_metric(MetricValue(
                "numerical_stability", reliability.numerical_stability, "ratio", metric_type=MetricType.QUALITY
            ))
            
            collection.add_metric(MetricValue(
                "bit_accuracy", reliability.bit_accuracy, "ratio", metric_type=MetricType.ACCURACY
            ))
            
            collection.add_metric(MetricValue(
                "error_rate", reliability.error_rate, "ratio", metric_type=MetricType.QUALITY
            ))
        
        # Store detailed results in metadata
        collection.metadata['validation_errors'] = validation.validation_errors
        collection.metadata['recommendations'] = validation.recommendations
        
        if accuracy.confusion_matrix:
            collection.metadata['confusion_matrix'] = accuracy.confusion_matrix