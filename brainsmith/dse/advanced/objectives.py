"""
Metrics-Driven Objective Functions and Constraint Satisfaction
Integration with Week 2 metrics framework for intelligent optimization.
"""

import os
import sys
import time
import logging
import math
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

# Import Week 2 metrics components
from ...metrics import (
    MetricsManager, MetricsConfiguration, MetricType, MetricScope,
    AdvancedPerformanceMetrics, ResourceUtilizationTracker,
    QualityMetricsCollector, HistoricalAnalysisEngine
)

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveDefinition:
    """Definition of an optimization objective."""
    name: str
    metric_name: str
    optimization_direction: str  # 'minimize' or 'maximize'
    weight: float = 1.0
    target_value: Optional[float] = None
    importance: float = 1.0
    constraint_type: Optional[str] = None  # 'hard', 'soft', or None
    threshold: Optional[float] = None


@dataclass
class ConstraintDefinition:
    """Definition of an optimization constraint."""
    name: str
    constraint_type: str  # 'resource', 'performance', 'quality'
    parameter: str
    operator: str  # '<=', '>=', '==', '<', '>'
    threshold: float
    penalty_weight: float = 1.0
    violation_tolerance: float = 0.0


@dataclass
class OptimizationContext:
    """Context for optimization evaluation."""
    design_parameters: Dict[str, Any]
    finn_model_path: Optional[str] = None
    build_config: Dict[str, Any] = field(default_factory=dict)
    device_constraints: Dict[str, Any] = field(default_factory=dict)
    performance_targets: Dict[str, float] = field(default_factory=dict)
    quality_requirements: Dict[str, float] = field(default_factory=dict)


class MetricsObjectiveFunction:
    """Objective function that uses Week 2 metrics for evaluation."""
    
    def __init__(self, 
                 metrics_manager: Any,  # MetricsManager from Week 2
                 objectives: List[ObjectiveDefinition],
                 constraints: List[ConstraintDefinition] = None):
        self.metrics_manager = metrics_manager
        self.objectives = objectives
        self.constraints = constraints or []
        self.evaluation_cache = {}
        self.evaluation_count = 0
        
        # Setup metrics collectors if not already configured
        self._setup_metrics_collectors()
    
    def _setup_metrics_collectors(self):
        """Setup metrics collectors for optimization."""
        try:
            # Ensure required collectors are registered
            required_collectors = [
                'AdvancedPerformanceMetrics',
                'ResourceUtilizationTracker', 
                'QualityMetricsCollector'
            ]
            
            for collector_name in required_collectors:
                if not hasattr(self.metrics_manager.registry, 'collectors'):
                    continue
                
                if collector_name not in self.metrics_manager.registry.collectors:
                    if collector_name == 'AdvancedPerformanceMetrics':
                        self.metrics_manager.registry.register_collector(AdvancedPerformanceMetrics)
                    elif collector_name == 'ResourceUtilizationTracker':
                        self.metrics_manager.registry.register_collector(ResourceUtilizationTracker)
                    elif collector_name == 'QualityMetricsCollector':
                        self.metrics_manager.registry.register_collector(QualityMetricsCollector)
                    
                    # Create collector instance
                    self.metrics_manager.registry.create_collector(collector_name)
            
        except Exception as e:
            logger.warning(f"Failed to setup metrics collectors: {e}")
    
    def evaluate(self, context: OptimizationContext) -> Tuple[List[float], List[float]]:
        """Evaluate objectives and constraints for given design context."""
        
        # Check cache first
        cache_key = self._create_cache_key(context.design_parameters)
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        try:
            # Collect metrics for this design
            metrics_context = self._create_metrics_context(context)
            metrics_collections = self.metrics_manager.collect_manual(metrics_context)
            
            # Extract metrics into a flat dictionary
            metrics_dict = self._extract_metrics_dict(metrics_collections)
            
            # Evaluate objectives
            objective_values = []
            for objective in self.objectives:
                value = self._evaluate_objective(objective, metrics_dict, context)
                objective_values.append(value)
            
            # Evaluate constraints
            constraint_violations = []
            for constraint in self.constraints:
                violation = self._evaluate_constraint(constraint, metrics_dict, context)
                constraint_violations.append(violation)
            
            # Cache result
            result = (objective_values, constraint_violations)
            self.evaluation_cache[cache_key] = result
            self.evaluation_count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Objective evaluation failed: {e}")
            # Return worst-case values
            objective_values = [float('-inf') if obj.optimization_direction == 'maximize' else float('inf') 
                             for obj in self.objectives]
            constraint_violations = [float('inf')] * len(self.constraints)
            return objective_values, constraint_violations
    
    def evaluate_single_objective(self, design_parameters: Dict[str, Any]) -> float:
        """Evaluate single aggregated objective (for single-objective optimizers)."""
        context = OptimizationContext(design_parameters=design_parameters)
        objective_values, constraint_violations = self.evaluate(context)
        
        # Aggregate objectives using weighted sum
        aggregated_objective = 0.0
        for obj_val, objective in zip(objective_values, self.objectives):
            if objective.optimization_direction == 'maximize':
                aggregated_objective += objective.weight * obj_val
            else:
                aggregated_objective -= objective.weight * obj_val
        
        # Apply constraint penalties
        penalty = sum(max(0, violation * constraint.penalty_weight) 
                     for violation, constraint in zip(constraint_violations, self.constraints))
        
        return aggregated_objective - penalty
    
    def _create_cache_key(self, design_parameters: Dict[str, Any]) -> str:
        """Create cache key from design parameters."""
        # Simple hash of sorted parameter items
        items = sorted(design_parameters.items())
        return str(hash(tuple(items)))
    
    def _create_metrics_context(self, context: OptimizationContext) -> Dict[str, Any]:
        """Create context for metrics collection."""
        metrics_context = {
            'design_parameters': context.design_parameters,
            'build_config': context.build_config,
            'device_constraints': context.device_constraints,
            'performance_targets': context.performance_targets,
            'quality_requirements': context.quality_requirements
        }
        
        # Add mock build result for metrics collection
        # In practice, this would involve actual FINN builds
        mock_build_result = self._create_mock_build_result(context)
        metrics_context['build_result'] = mock_build_result
        
        return metrics_context
    
    def _create_mock_build_result(self, context: OptimizationContext) -> Dict[str, Any]:
        """Create mock build result for metrics evaluation."""
        # This would be replaced with actual FINN build in production
        design_params = context.design_parameters
        
        # Extract key parameters
        parallelism = design_params.get('pe_parallelism', 1)
        memory_width = design_params.get('memory_width', 32)
        clock_freq = design_params.get('clock_frequency_mhz', 100.0)
        precision = design_params.get('weight_precision', 8)
        
        # Estimate resource usage based on parameters
        base_luts = 1000
        base_dsps = 10
        base_brams = 5
        
        estimated_luts = int(base_luts * parallelism * (memory_width / 32) * (precision / 8))
        estimated_dsps = int(base_dsps * parallelism)
        estimated_brams = int(base_brams * (memory_width / 32))
        
        # Estimate performance
        base_throughput = 1000000  # ops/sec
        estimated_throughput = base_throughput * parallelism * (clock_freq / 100.0)
        estimated_latency = max(1, int(100 / parallelism))
        
        # Estimate power
        base_power = 500  # mW
        estimated_power = base_power * (parallelism ** 0.7) * (clock_freq / 100.0) * ((precision / 8) ** 0.5)
        
        return {
            'resource_utilization': {
                'lut_count': estimated_luts,
                'dsp_count': estimated_dsps,
                'bram_count': estimated_brams,
                'uram_count': 0
            },
            'performance_metrics': {
                'throughput_ops_per_sec': estimated_throughput,
                'latency_cycles': estimated_latency,
                'clock_frequency_mhz': clock_freq
            },
            'power_metrics': {
                'total_power_mw': estimated_power,
                'dynamic_power_mw': estimated_power * 0.6,
                'static_power_mw': estimated_power * 0.4
            },
            'timing_metrics': {
                'critical_path_delay_ns': 10.0 - (clock_freq - 100) * 0.05,
                'timing_slack_ns': max(-2.0, 2.0 - (clock_freq - 100) * 0.02),
                'timing_met': True
            },
            'device': context.device_constraints.get('device', 'xczu7ev'),
            'success': True
        }
    
    def _extract_metrics_dict(self, metrics_collections: List[Any]) -> Dict[str, float]:
        """Extract metrics into flat dictionary."""
        metrics_dict = {}
        
        for collection in metrics_collections:
            for metric in collection.metrics:
                if isinstance(metric.value, (int, float)):
                    metrics_dict[metric.name] = float(metric.value)
        
        return metrics_dict
    
    def _evaluate_objective(self, objective: ObjectiveDefinition, 
                           metrics_dict: Dict[str, float], context: OptimizationContext) -> float:
        """Evaluate single objective."""
        
        if objective.metric_name in metrics_dict:
            value = metrics_dict[objective.metric_name]
        else:
            # Try to find similar metric name
            similar_metrics = [name for name in metrics_dict.keys() 
                             if objective.metric_name.lower() in name.lower()]
            if similar_metrics:
                value = metrics_dict[similar_metrics[0]]
            else:
                logger.warning(f"Metric {objective.metric_name} not found in metrics collection")
                return float('-inf') if objective.optimization_direction == 'maximize' else float('inf')
        
        # Apply target-based transformation if specified
        if objective.target_value is not None:
            # Convert to distance from target (closer is better)
            distance = abs(value - objective.target_value)
            if objective.optimization_direction == 'maximize':
                value = -distance  # Minimize distance (maximize negative distance)
            else:
                value = distance   # Minimize distance
        
        return value
    
    def _evaluate_constraint(self, constraint: ConstraintDefinition,
                           metrics_dict: Dict[str, float], context: OptimizationContext) -> float:
        """Evaluate single constraint and return violation amount."""
        
        # Get constraint value
        if constraint.constraint_type == 'resource':
            value = self._get_resource_constraint_value(constraint, metrics_dict, context)
        elif constraint.constraint_type == 'performance':
            value = self._get_performance_constraint_value(constraint, metrics_dict, context)
        elif constraint.constraint_type == 'quality':
            value = self._get_quality_constraint_value(constraint, metrics_dict, context)
        else:
            # Generic constraint - look in metrics
            value = metrics_dict.get(constraint.parameter, 0.0)
        
        # Calculate violation
        violation = 0.0
        threshold = constraint.threshold
        tolerance = constraint.violation_tolerance
        
        if constraint.operator == '<=':
            if value > threshold + tolerance:
                violation = value - threshold - tolerance
        elif constraint.operator == '>=':
            if value < threshold - tolerance:
                violation = threshold - tolerance - value
        elif constraint.operator == '<':
            if value >= threshold:
                violation = value - threshold + 1.0
        elif constraint.operator == '>':
            if value <= threshold:
                violation = threshold - value + 1.0
        elif constraint.operator == '==':
            if abs(value - threshold) > tolerance:
                violation = abs(value - threshold) - tolerance
        
        return max(0.0, violation)
    
    def _get_resource_constraint_value(self, constraint: ConstraintDefinition,
                                     metrics_dict: Dict[str, float], 
                                     context: OptimizationContext) -> float:
        """Get resource constraint value."""
        param = constraint.parameter.lower()
        
        if 'lut' in param:
            return metrics_dict.get('lut_utilization', metrics_dict.get('lut_count', 0.0))
        elif 'dsp' in param:
            return metrics_dict.get('dsp_utilization', metrics_dict.get('dsp_count', 0.0))
        elif 'bram' in param:
            return metrics_dict.get('bram_utilization', metrics_dict.get('bram_count', 0.0))
        elif 'uram' in param:
            return metrics_dict.get('uram_utilization', metrics_dict.get('uram_count', 0.0))
        elif 'power' in param:
            return metrics_dict.get('power_total_mw', metrics_dict.get('total_power_mw', 0.0))
        else:
            return metrics_dict.get(constraint.parameter, 0.0)
    
    def _get_performance_constraint_value(self, constraint: ConstraintDefinition,
                                        metrics_dict: Dict[str, float],
                                        context: OptimizationContext) -> float:
        """Get performance constraint value."""
        param = constraint.parameter.lower()
        
        if 'throughput' in param:
            return metrics_dict.get('throughput_ops_per_sec', 0.0)
        elif 'latency' in param:
            return metrics_dict.get('latency_cycles', 0.0)
        elif 'frequency' in param or 'clock' in param:
            return metrics_dict.get('clock_frequency_mhz', 0.0)
        elif 'timing' in param:
            return metrics_dict.get('timing_slack_ns', 0.0)
        else:
            return metrics_dict.get(constraint.parameter, 0.0)
    
    def _get_quality_constraint_value(self, constraint: ConstraintDefinition,
                                    metrics_dict: Dict[str, float],
                                    context: OptimizationContext) -> float:
        """Get quality constraint value."""
        param = constraint.parameter.lower()
        
        if 'accuracy' in param:
            return metrics_dict.get('accuracy', 0.0)
        elif 'precision' in param:
            return metrics_dict.get('precision', 0.0)
        elif 'recall' in param:
            return metrics_dict.get('recall', 0.0)
        elif 'f1' in param:
            return metrics_dict.get('f1_score', 0.0)
        else:
            return metrics_dict.get(constraint.parameter, 0.0)


class ConstraintSatisfactionEngine:
    """Advanced constraint handling for FPGA design optimization."""
    
    def __init__(self, device_constraints: Dict[str, Any] = None):
        self.device_constraints = device_constraints or {}
        self.constraint_handlers = {
            'resource': self._handle_resource_constraints,
            'performance': self._handle_performance_constraints,
            'quality': self._handle_quality_constraints,
            'architectural': self._handle_architectural_constraints
        }
        self.repair_strategies = {
            'scale_down': self._scale_down_repair,
            'parameter_adjust': self._parameter_adjust_repair,
            'architecture_change': self._architecture_change_repair
        }
    
    def check_feasibility(self, design_parameters: Dict[str, Any], 
                         constraints: List[ConstraintDefinition]) -> Tuple[bool, List[str]]:
        """Check if design satisfies all constraints."""
        
        violations = []
        
        for constraint in constraints:
            handler = self.constraint_handlers.get(constraint.constraint_type, self._handle_generic_constraint)
            is_satisfied, violation_msg = handler(design_parameters, constraint)
            
            if not is_satisfied:
                violations.append(violation_msg)
        
        return len(violations) == 0, violations
    
    def repair_infeasible_solution(self, design_parameters: Dict[str, Any],
                                 constraints: List[ConstraintDefinition],
                                 strategy: str = 'scale_down') -> Dict[str, Any]:
        """Attempt to repair constraint violations."""
        
        repaired_params = design_parameters.copy()
        
        # Check current violations
        is_feasible, violations = self.check_feasibility(repaired_params, constraints)
        
        if is_feasible:
            return repaired_params
        
        # Apply repair strategy
        repair_func = self.repair_strategies.get(strategy, self._scale_down_repair)
        repaired_params = repair_func(repaired_params, constraints, violations)
        
        # Verify repair
        is_feasible_after, remaining_violations = self.check_feasibility(repaired_params, constraints)
        
        if not is_feasible_after:
            logger.warning(f"Could not fully repair solution: {len(remaining_violations)} violations remain")
        
        return repaired_params
    
    def _handle_resource_constraints(self, design_parameters: Dict[str, Any],
                                   constraint: ConstraintDefinition) -> Tuple[bool, str]:
        """Handle resource-based constraints."""
        
        # Estimate resource usage
        resource_usage = self._estimate_resource_usage(design_parameters)
        
        param = constraint.parameter.lower()
        if 'lut' in param:
            current_usage = resource_usage.get('lut_count', 0)
            available = self.device_constraints.get('max_luts', 100000)
        elif 'dsp' in param:
            current_usage = resource_usage.get('dsp_count', 0)
            available = self.device_constraints.get('max_dsps', 1000)
        elif 'bram' in param:
            current_usage = resource_usage.get('bram_count', 0)
            available = self.device_constraints.get('max_brams', 500)
        else:
            return True, ""
        
        # Check constraint
        threshold = constraint.threshold
        if constraint.operator == '<=':
            if isinstance(threshold, float) and threshold <= 1.0:
                # Percentage constraint
                max_allowed = available * threshold
            else:
                # Absolute constraint
                max_allowed = threshold
            
            if current_usage <= max_allowed:
                return True, ""
            else:
                return False, f"Resource violation: {param} usage {current_usage} exceeds limit {max_allowed}"
        
        return True, ""
    
    def _handle_performance_constraints(self, design_parameters: Dict[str, Any],
                                      constraint: ConstraintDefinition) -> Tuple[bool, str]:
        """Handle performance-based constraints."""
        
        # Estimate performance
        performance = self._estimate_performance(design_parameters)
        
        param = constraint.parameter.lower()
        if 'throughput' in param:
            current_value = performance.get('throughput_ops_per_sec', 0)
        elif 'latency' in param:
            current_value = performance.get('latency_cycles', float('inf'))
        elif 'frequency' in param:
            current_value = performance.get('clock_frequency_mhz', 0)
        else:
            return True, ""
        
        # Check constraint
        threshold = constraint.threshold
        if constraint.operator == '>=':
            if current_value >= threshold:
                return True, ""
            else:
                return False, f"Performance violation: {param} {current_value} below minimum {threshold}"
        elif constraint.operator == '<=':
            if current_value <= threshold:
                return True, ""
            else:
                return False, f"Performance violation: {param} {current_value} exceeds maximum {threshold}"
        
        return True, ""
    
    def _handle_quality_constraints(self, design_parameters: Dict[str, Any],
                                  constraint: ConstraintDefinition) -> Tuple[bool, str]:
        """Handle quality-based constraints."""
        
        # Estimate quality metrics
        quality = self._estimate_quality(design_parameters)
        
        param = constraint.parameter.lower()
        if 'accuracy' in param:
            current_value = quality.get('accuracy', 0.0)
        elif 'precision' in param:
            current_value = quality.get('precision', 0.0)
        else:
            return True, ""
        
        # Check constraint
        threshold = constraint.threshold
        if constraint.operator == '>=':
            if current_value >= threshold:
                return True, ""
            else:
                return False, f"Quality violation: {param} {current_value} below minimum {threshold}"
        
        return True, ""
    
    def _handle_architectural_constraints(self, design_parameters: Dict[str, Any],
                                        constraint: ConstraintDefinition) -> Tuple[bool, str]:
        """Handle architectural constraints."""
        
        # Check architectural validity
        is_valid, message = self._check_architectural_validity(design_parameters)
        
        if is_valid:
            return True, ""
        else:
            return False, f"Architectural violation: {message}"
    
    def _handle_generic_constraint(self, design_parameters: Dict[str, Any],
                                 constraint: ConstraintDefinition) -> Tuple[bool, str]:
        """Handle generic constraints."""
        
        param_value = design_parameters.get(constraint.parameter, 0)
        threshold = constraint.threshold
        
        if constraint.operator == '<=':
            satisfied = param_value <= threshold
        elif constraint.operator == '>=':
            satisfied = param_value >= threshold
        elif constraint.operator == '==':
            satisfied = abs(param_value - threshold) <= constraint.violation_tolerance
        else:
            satisfied = True
        
        if satisfied:
            return True, ""
        else:
            return False, f"Generic constraint violation: {constraint.parameter} {constraint.operator} {threshold}"
    
    def _scale_down_repair(self, design_parameters: Dict[str, Any],
                          constraints: List[ConstraintDefinition],
                          violations: List[str]) -> Dict[str, Any]:
        """Repair by scaling down resource-intensive parameters."""
        
        repaired = design_parameters.copy()
        
        # Identify resource-intensive parameters
        scale_params = ['pe_parallelism', 'memory_width', 'pipeline_depth']
        
        for param in scale_params:
            if param in repaired:
                # Scale down by 20%
                current_value = repaired[param]
                if isinstance(current_value, (int, float)) and current_value > 1:
                    repaired[param] = max(1, int(current_value * 0.8))
        
        return repaired
    
    def _parameter_adjust_repair(self, design_parameters: Dict[str, Any],
                               constraints: List[ConstraintDefinition],
                               violations: List[str]) -> Dict[str, Any]:
        """Repair by adjusting specific parameters."""
        
        repaired = design_parameters.copy()
        
        for constraint in constraints:
            if constraint.constraint_type == 'resource':
                if 'parallelism' in constraint.parameter.lower():
                    # Reduce parallelism
                    parallelism_params = [k for k in repaired.keys() if 'parallelism' in k.lower()]
                    for param in parallelism_params:
                        if isinstance(repaired[param], (int, float)) and repaired[param] > 1:
                            repaired[param] = max(1, int(repaired[param] * 0.7))
        
        return repaired
    
    def _architecture_change_repair(self, design_parameters: Dict[str, Any],
                                  constraints: List[ConstraintDefinition],
                                  violations: List[str]) -> Dict[str, Any]:
        """Repair by changing architectural parameters."""
        
        repaired = design_parameters.copy()
        
        # Change precision to reduce resource usage
        if 'weight_precision' in repaired and repaired['weight_precision'] > 4:
            repaired['weight_precision'] = max(4, repaired['weight_precision'] - 2)
        
        if 'activation_precision' in repaired and repaired['activation_precision'] > 4:
            repaired['activation_precision'] = max(4, repaired['activation_precision'] - 2)
        
        return repaired
    
    def _estimate_resource_usage(self, design_parameters: Dict[str, Any]) -> Dict[str, int]:
        """Estimate resource usage for design parameters."""
        
        parallelism = design_parameters.get('pe_parallelism', 1)
        memory_width = design_parameters.get('memory_width', 32)
        precision = design_parameters.get('weight_precision', 8)
        
        # Simple resource estimation model
        base_luts = 1000
        base_dsps = 10
        base_brams = 5
        
        estimated_luts = int(base_luts * parallelism * (memory_width / 32) * (precision / 8))
        estimated_dsps = int(base_dsps * parallelism)
        estimated_brams = int(base_brams * (memory_width / 32))
        
        return {
            'lut_count': estimated_luts,
            'dsp_count': estimated_dsps,
            'bram_count': estimated_brams,
            'uram_count': 0
        }
    
    def _estimate_performance(self, design_parameters: Dict[str, Any]) -> Dict[str, float]:
        """Estimate performance for design parameters."""
        
        parallelism = design_parameters.get('pe_parallelism', 1)
        clock_freq = design_parameters.get('clock_frequency_mhz', 100.0)
        
        base_throughput = 1000000  # ops/sec
        estimated_throughput = base_throughput * parallelism * (clock_freq / 100.0)
        estimated_latency = max(1, int(100 / parallelism))
        
        return {
            'throughput_ops_per_sec': estimated_throughput,
            'latency_cycles': estimated_latency,
            'clock_frequency_mhz': clock_freq
        }
    
    def _estimate_quality(self, design_parameters: Dict[str, Any]) -> Dict[str, float]:
        """Estimate quality metrics for design parameters."""
        
        precision = design_parameters.get('weight_precision', 8)
        
        # Quality typically decreases with lower precision
        base_accuracy = 0.95
        precision_factor = min(1.0, precision / 8.0)
        estimated_accuracy = base_accuracy * precision_factor
        
        return {
            'accuracy': estimated_accuracy,
            'precision': estimated_accuracy * 0.9,
            'recall': estimated_accuracy * 0.95
        }
    
    def _check_architectural_validity(self, design_parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if design parameters form a valid architecture."""
        
        # Check parameter compatibility
        parallelism = design_parameters.get('pe_parallelism', 1)
        memory_width = design_parameters.get('memory_width', 32)
        
        # Memory width should be compatible with parallelism
        if memory_width % parallelism != 0:
            return False, f"Memory width {memory_width} not compatible with parallelism {parallelism}"
        
        # Check precision constraints
        weight_precision = design_parameters.get('weight_precision', 8)
        activation_precision = design_parameters.get('activation_precision', 8)
        
        if weight_precision < 1 or weight_precision > 32:
            return False, f"Invalid weight precision: {weight_precision}"
        
        if activation_precision < 1 or activation_precision > 32:
            return False, f"Invalid activation precision: {activation_precision}"
        
        return True, ""


class ObjectiveRegistry:
    """Registry for managing optimization objectives and constraints."""
    
    def __init__(self):
        self.objectives = {}
        self.constraints = {}
        self.predefined_objectives = self._create_predefined_objectives()
        self.predefined_constraints = self._create_predefined_constraints()
    
    def register_objective(self, name: str, objective: ObjectiveDefinition):
        """Register a custom objective."""
        self.objectives[name] = objective
    
    def register_constraint(self, name: str, constraint: ConstraintDefinition):
        """Register a custom constraint."""
        self.constraints[name] = constraint
    
    def get_objective(self, name: str) -> Optional[ObjectiveDefinition]:
        """Get objective by name."""
        return self.objectives.get(name) or self.predefined_objectives.get(name)
    
    def get_constraint(self, name: str) -> Optional[ConstraintDefinition]:
        """Get constraint by name."""
        return self.constraints.get(name) or self.predefined_constraints.get(name)
    
    def get_objectives_by_type(self, objective_type: str) -> List[ObjectiveDefinition]:
        """Get all objectives of a specific type."""
        all_objectives = {**self.predefined_objectives, **self.objectives}
        return [obj for obj in all_objectives.values() if objective_type.lower() in obj.name.lower()]
    
    def create_pareto_objectives(self, objective_names: List[str]) -> List[ObjectiveDefinition]:
        """Create list of objectives for Pareto optimization."""
        objectives = []
        for name in objective_names:
            obj = self.get_objective(name)
            if obj:
                objectives.append(obj)
            else:
                logger.warning(f"Objective {name} not found in registry")
        return objectives
    
    def _create_predefined_objectives(self) -> Dict[str, ObjectiveDefinition]:
        """Create predefined optimization objectives."""
        return {
            'maximize_throughput': ObjectiveDefinition(
                name='maximize_throughput',
                metric_name='throughput_ops_per_sec',
                optimization_direction='maximize',
                weight=1.0,
                importance=1.0
            ),
            'minimize_latency': ObjectiveDefinition(
                name='minimize_latency', 
                metric_name='latency_cycles',
                optimization_direction='minimize',
                weight=1.0,
                importance=1.0
            ),
            'minimize_power': ObjectiveDefinition(
                name='minimize_power',
                metric_name='power_total_mw',
                optimization_direction='minimize',
                weight=1.0,
                importance=0.8
            ),
            'maximize_efficiency': ObjectiveDefinition(
                name='maximize_efficiency',
                metric_name='ops_per_lut',
                optimization_direction='maximize',
                weight=1.0,
                importance=0.9
            ),
            'minimize_area': ObjectiveDefinition(
                name='minimize_area',
                metric_name='total_area_lut_equiv',
                optimization_direction='minimize',
                weight=1.0,
                importance=0.7
            ),
            'maximize_accuracy': ObjectiveDefinition(
                name='maximize_accuracy',
                metric_name='accuracy',
                optimization_direction='maximize',
                weight=1.0,
                importance=1.0
            ),
            'target_frequency': ObjectiveDefinition(
                name='target_frequency',
                metric_name='clock_frequency_mhz',
                optimization_direction='maximize',
                target_value=150.0,
                weight=1.0,
                importance=0.8
            )
        }
    
    def _create_predefined_constraints(self) -> Dict[str, ConstraintDefinition]:
        """Create predefined optimization constraints."""
        return {
            'lut_budget': ConstraintDefinition(
                name='lut_budget',
                constraint_type='resource',
                parameter='lut_utilization',
                operator='<=',
                threshold=0.8,  # 80% utilization
                penalty_weight=10.0
            ),
            'dsp_budget': ConstraintDefinition(
                name='dsp_budget',
                constraint_type='resource', 
                parameter='dsp_utilization',
                operator='<=',
                threshold=0.8,
                penalty_weight=10.0
            ),
            'bram_budget': ConstraintDefinition(
                name='bram_budget',
                constraint_type='resource',
                parameter='bram_utilization', 
                operator='<=',
                threshold=0.8,
                penalty_weight=10.0
            ),
            'power_budget': ConstraintDefinition(
                name='power_budget',
                constraint_type='resource',
                parameter='power_total_mw',
                operator='<=',
                threshold=3000.0,  # 3W
                penalty_weight=5.0
            ),
            'min_throughput': ConstraintDefinition(
                name='min_throughput',
                constraint_type='performance',
                parameter='throughput_ops_per_sec',
                operator='>=',
                threshold=500000.0,
                penalty_weight=5.0
            ),
            'max_latency': ConstraintDefinition(
                name='max_latency',
                constraint_type='performance',
                parameter='latency_cycles',
                operator='<=',
                threshold=1000,
                penalty_weight=5.0
            ),
            'min_accuracy': ConstraintDefinition(
                name='min_accuracy',
                constraint_type='quality',
                parameter='accuracy',
                operator='>=',
                threshold=0.90,
                penalty_weight=20.0  # High penalty for accuracy violations
            ),
            'timing_closure': ConstraintDefinition(
                name='timing_closure',
                constraint_type='performance',
                parameter='timing_slack_ns',
                operator='>=',
                threshold=0.0,
                penalty_weight=15.0
            )
        }


class ConstraintHandler:
    """Handler for specific constraint types with repair strategies."""
    
    def __init__(self, constraint_satisfaction_engine: ConstraintSatisfactionEngine):
        self.cse = constraint_satisfaction_engine
        self.repair_history = []
    
    def handle_resource_constraints(self, design_parameters: Dict[str, Any],
                                  resource_budget: Dict[str, float]) -> Dict[str, Any]:
        """Handle resource constraint violations with intelligent repair."""
        
        constraints = []
        for resource, budget in resource_budget.items():
            constraint = ConstraintDefinition(
                name=f"{resource}_budget",
                constraint_type='resource',
                parameter=resource,
                operator='<=',
                threshold=budget,
                penalty_weight=10.0
            )
            constraints.append(constraint)
        
        is_feasible, violations = self.cse.check_feasibility(design_parameters, constraints)
        
        if not is_feasible:
            logger.info(f"Resource constraint violations detected: {len(violations)}")
            repaired_params = self.cse.repair_infeasible_solution(
                design_parameters, constraints, 'scale_down'
            )
            
            # Record repair
            self.repair_history.append({
                'original_params': design_parameters.copy(),
                'repaired_params': repaired_params.copy(),
                'violations': violations,
                'repair_strategy': 'scale_down'
            })
            
            return repaired_params
        
        return design_parameters
    
    def handle_performance_constraints(self, design_parameters: Dict[str, Any],
                                     performance_targets: Dict[str, float]) -> Dict[str, Any]:
        """Handle performance constraint violations."""
        
        constraints = []
        for metric, target in performance_targets.items():
            if 'min_' in metric:
                operator = '>='
                threshold = target
            elif 'max_' in metric:
                operator = '<='
                threshold = target
            else:
                operator = '>='
                threshold = target
            
            constraint = ConstraintDefinition(
                name=f"{metric}_constraint",
                constraint_type='performance',
                parameter=metric.replace('min_', '').replace('max_', ''),
                operator=operator,
                threshold=threshold,
                penalty_weight=5.0
            )
            constraints.append(constraint)
        
        is_feasible, violations = self.cse.check_feasibility(design_parameters, constraints)
        
        if not is_feasible:
            logger.info(f"Performance constraint violations detected: {len(violations)}")
            repaired_params = self.cse.repair_infeasible_solution(
                design_parameters, constraints, 'parameter_adjust'
            )
            return repaired_params
        
        return design_parameters
    
    def get_repair_statistics(self) -> Dict[str, Any]:
        """Get statistics about constraint repairs."""
        if not self.repair_history:
            return {'total_repairs': 0}
        
        total_repairs = len(self.repair_history)
        repair_strategies = {}
        
        for repair in self.repair_history:
            strategy = repair['repair_strategy']
            repair_strategies[strategy] = repair_strategies.get(strategy, 0) + 1
        
        return {
            'total_repairs': total_repairs,
            'repair_strategies': repair_strategies,
            'success_rate': 1.0  # Simplified - would track actual success
        }