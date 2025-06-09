"""
Parameter Sensitivity Monitor

Monitor and analyze parameter sensitivity for optimization guidance.
Track parameter changes and their impact on performance.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import statistics
import math

from .types import (
    ParameterChangeRecord,
    ImpactAnalysis,
    ImpactMeasurement,
    InteractionEffect,
    SensitivityInsight,
    SensitivityData,
    ParameterChangeType
)

logger = logging.getLogger(__name__)

class SensitivityDatabase:
    """Database for storing parameter sensitivity data"""
    
    def __init__(self):
        self.changes = []  # List of ParameterChangeRecord
        self.impacts = {}  # parameter -> List[ImpactMeasurement]
        self.correlations = {}  # (param1, param2) -> correlation
    
    def store_change(self, change: ParameterChangeRecord) -> None:
        """Store parameter change record"""
        self.changes.append(change)
        logger.debug(f"Stored parameter change at iteration {change.iteration_number}")
    
    def get_parameter_history(self, parameter: str) -> List[ParameterChangeRecord]:
        """Get history of changes for specific parameter"""
        return [c for c in self.changes if parameter in c.parameter_changes]
    
    def get_recent_changes(self, hours: int = 24) -> List[ParameterChangeRecord]:
        """Get recent parameter changes within time window"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [c for c in self.changes if c.timestamp >= cutoff]

class StatisticalAnalyzer:
    """Statistical analysis of parameter impacts"""
    
    def __init__(self):
        self.significance_threshold = 0.05
    
    def compute_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Compute Pearson correlation coefficient"""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        try:
            # Compute means
            mean_x = statistics.mean(x_values)
            mean_y = statistics.mean(y_values)
            
            # Compute correlation
            numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
            
            sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
            sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)
            
            denominator = math.sqrt(sum_sq_x * sum_sq_y)
            
            if denominator == 0:
                return 0.0
                
            return numerator / denominator
            
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def test_significance(self, correlation: float, sample_size: int) -> float:
        """Test statistical significance of correlation"""
        if sample_size < 3:
            return 1.0  # Not significant
        
        # Simple t-test approximation
        if abs(correlation) < 0.001:
            return 1.0
        
        t_stat = correlation * math.sqrt((sample_size - 2) / (1 - correlation**2))
        
        # Approximate p-value (simplified)
        p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + math.sqrt(sample_size - 2)))
        return max(0.0, min(1.0, p_value))
    
    def compute_sensitivity_coefficient(self, 
                                      parameter_changes: List[float],
                                      performance_changes: List[float]) -> float:
        """Compute sensitivity coefficient (dy/dx)"""
        if len(parameter_changes) != len(performance_changes) or len(parameter_changes) < 2:
            return 0.0
        
        try:
            # Simple finite difference approximation
            sensitivities = []
            for i in range(len(parameter_changes) - 1):
                dx = parameter_changes[i+1] - parameter_changes[i]
                dy = performance_changes[i+1] - performance_changes[i]
                
                if abs(dx) > 1e-10:
                    sensitivities.append(dy / dx)
            
            return statistics.mean(sensitivities) if sensitivities else 0.0
            
        except (ValueError, ZeroDivisionError):
            return 0.0

class CorrelationAnalyzer:
    """Analyze correlations between parameters and performance"""
    
    def __init__(self):
        self.statistical_analyzer = StatisticalAnalyzer()
    
    def analyze_parameter_interactions(self, 
                                     changes: List[ParameterChangeRecord]) -> List[InteractionEffect]:
        """Analyze interactions between parameters"""
        interactions = []
        
        if len(changes) < 3:
            return interactions
        
        # Get all unique parameters
        all_parameters = set()
        for change in changes:
            all_parameters.update(change.parameter_changes.keys())
        
        parameters = list(all_parameters)
        
        # Analyze pairwise interactions
        for i, param1 in enumerate(parameters):
            for param2 in parameters[i+1:]:
                interaction = self._analyze_parameter_pair(param1, param2, changes)
                if interaction.strength > 0.1:  # Threshold for meaningful interaction
                    interactions.append(interaction)
        
        return interactions
    
    def _analyze_parameter_pair(self, 
                               param1: str, 
                               param2: str,
                               changes: List[ParameterChangeRecord]) -> InteractionEffect:
        """Analyze interaction between two parameters"""
        
        # Extract values for both parameters
        param1_values = []
        param2_values = []
        
        for change in changes:
            if param1 in change.parameter_changes and param2 in change.parameter_changes:
                param1_values.append(change.parameter_changes[param1])
                param2_values.append(change.parameter_changes[param2])
        
        if len(param1_values) < 3:
            return InteractionEffect(
                parameters=[param1, param2],
                strength=0.0,
                type="neutral",
                confidence=0.0
            )
        
        # Compute correlation
        correlation = self.statistical_analyzer.compute_correlation(param1_values, param2_values)
        
        # Determine interaction type and strength
        strength = abs(correlation)
        
        if correlation > 0.3:
            interaction_type = "synergistic"
        elif correlation < -0.3:
            interaction_type = "antagonistic"
        else:
            interaction_type = "neutral"
        
        # Compute confidence
        p_value = self.statistical_analyzer.test_significance(correlation, len(param1_values))
        confidence = 1.0 - p_value
        
        return InteractionEffect(
            parameters=[param1, param2],
            strength=strength,
            type=interaction_type,
            confidence=confidence
        )

class ParameterSensitivityMonitor:
    """Monitor and analyze parameter sensitivity for optimization guidance"""
    
    def __init__(self):
        self.sensitivity_database = SensitivityDatabase()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.performance_history = []  # List of (timestamp, performance_value)
    
    def track_parameter_changes(self, parameter_changes: Dict[str, Any]) -> None:
        """Track parameter changes during optimization"""
        
        change_record = ParameterChangeRecord(
            timestamp=datetime.now(),
            parameter_changes=parameter_changes,
            change_context=self._capture_change_context(),
            change_magnitude=self._compute_change_magnitude(parameter_changes),
            change_type=self._classify_change_type(parameter_changes),
            iteration_number=len(self.sensitivity_database.changes)
        )
        
        self.sensitivity_database.store_change(change_record)
        logger.debug(f"Tracked parameter changes: {list(parameter_changes.keys())}")
    
    def record_performance(self, performance_value: float) -> None:
        """Record performance value for correlation analysis"""
        self.performance_history.append((datetime.now(), performance_value))
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def measure_performance_impact(self, 
                                 parameter: str,
                                 window_size: int = 10) -> ImpactAnalysis:
        """Measure performance impact of parameter changes"""
        
        analysis = ImpactAnalysis()
        
        # Get parameter history
        param_history = self.sensitivity_database.get_parameter_history(parameter)
        
        if len(param_history) < 2 or len(self.performance_history) < 2:
            return analysis
        
        # Direct impact measurement
        analysis.direct_impact[parameter] = self._measure_direct_impact(parameter, param_history)
        
        # Interaction effects
        analysis.interaction_effects = self.correlation_analyzer.analyze_parameter_interactions(
            self.sensitivity_database.changes[-window_size:]
        )
        
        # Sensitivity coefficients
        analysis.sensitivity_coefficients[parameter] = self._compute_sensitivity_coefficient(
            parameter, param_history
        )
        
        # Statistical significance
        analysis.statistical_significance[parameter] = self._test_statistical_significance(
            parameter, param_history
        )
        
        logger.debug(f"Measured performance impact for parameter: {parameter}")
        return analysis
    
    def identify_critical_parameters(self, 
                                   sensitivity_data: SensitivityData = None) -> List[str]:
        """Identify parameters with highest impact on performance"""
        
        if sensitivity_data is None:
            sensitivity_data = self._collect_sensitivity_data()
        
        # Compute parameter importance scores
        importance_scores = {}
        
        for param_name in sensitivity_data.parameters:
            # Sensitivity-based importance
            sensitivity_score = self._compute_sensitivity_importance(param_name, sensitivity_data)
            
            # Frequency-based importance
            frequency_score = self._compute_frequency_importance(param_name, sensitivity_data)
            
            # Interaction-based importance
            interaction_score = self._compute_interaction_importance(param_name, sensitivity_data)
            
            # Combined importance score
            importance_scores[param_name] = (
                0.5 * sensitivity_score +
                0.3 * frequency_score +
                0.2 * interaction_score
            )
        
        # Return top critical parameters
        sorted_params = sorted(
            importance_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        critical_params = [param for param, score in sorted_params[:10]]
        logger.info(f"Identified {len(critical_params)} critical parameters")
        return critical_params
    
    def generate_sensitivity_insights(self, analysis: ImpactAnalysis) -> List[SensitivityInsight]:
        """Generate actionable insights from sensitivity analysis"""
        
        insights = []
        
        # High-impact parameter insights
        for param, impact in analysis.direct_impact.items():
            if impact.magnitude > 0.1:  # 10% impact threshold
                insights.append(SensitivityInsight(
                    type="high_impact_parameter",
                    parameter=param,
                    impact_magnitude=impact.magnitude,
                    recommendation=f"Focus optimization effort on {param}",
                    confidence=impact.confidence
                ))
        
        # Interaction effect insights
        for interaction in analysis.interaction_effects:
            if interaction.strength > 0.05:  # 5% interaction threshold
                insights.append(SensitivityInsight(
                    type="parameter_interaction",
                    parameters=interaction.parameters,
                    interaction_strength=interaction.strength,
                    recommendation=f"Consider joint optimization of {interaction.parameters}",
                    confidence=interaction.confidence
                ))
        
        # Low sensitivity parameter insights
        for param, coeff in analysis.sensitivity_coefficients.items():
            if abs(coeff) < 0.01:  # Very low sensitivity
                insights.append(SensitivityInsight(
                    type="low_sensitivity_parameter",
                    parameter=param,
                    impact_magnitude=abs(coeff),
                    recommendation=f"Parameter {param} has low impact - consider fixing or removing",
                    confidence=0.8
                ))
        
        logger.debug(f"Generated {len(insights)} sensitivity insights")
        return insights
    
    def _capture_change_context(self) -> Dict[str, Any]:
        """Capture context when parameter changes occur"""
        return {
            'optimization_iteration': len(self.sensitivity_database.changes),
            'timestamp': datetime.now().isoformat(),
            'recent_performance': self.performance_history[-5:] if self.performance_history else []
        }
    
    def _compute_change_magnitude(self, parameter_changes: Dict[str, Any]) -> float:
        """Compute magnitude of parameter changes"""
        if not parameter_changes:
            return 0.0
        
        # Simple L2 norm of normalized changes
        try:
            normalized_changes = []
            for param, value in parameter_changes.items():
                if isinstance(value, (int, float)):
                    # Normalize by assuming reasonable parameter ranges
                    normalized_value = abs(value) / 100.0  # Simple normalization
                    normalized_changes.append(normalized_value)
            
            if normalized_changes:
                return math.sqrt(sum(x**2 for x in normalized_changes))
            else:
                return 0.0
                
        except (ValueError, TypeError):
            return 0.0
    
    def _classify_change_type(self, parameter_changes: Dict[str, Any]) -> ParameterChangeType:
        """Classify the type of parameter change"""
        # Simple heuristic classification
        iteration = len(self.sensitivity_database.changes)
        
        if iteration == 0:
            return ParameterChangeType.INITIALIZATION
        elif iteration < 10:
            return ParameterChangeType.EXPLORATION
        elif iteration < 50:
            return ParameterChangeType.EXPLOITATION
        else:
            return ParameterChangeType.REFINEMENT
    
    def _measure_direct_impact(self, 
                             parameter: str,
                             param_history: List[ParameterChangeRecord]) -> ImpactMeasurement:
        """Measure direct impact of parameter on performance"""
        
        if len(param_history) < 2 or len(self.performance_history) < 2:
            return ImpactMeasurement()
        
        # Extract parameter values and corresponding performance
        param_values = []
        perf_values = []
        
        for change in param_history[-10:]:  # Use recent history
            if parameter in change.parameter_changes:
                param_values.append(change.parameter_changes[parameter])
                
                # Find corresponding performance value
                closest_perf = self._find_closest_performance(change.timestamp)
                if closest_perf is not None:
                    perf_values.append(closest_perf)
        
        if len(param_values) < 2 or len(perf_values) < 2:
            return ImpactMeasurement()
        
        # Compute correlation
        correlation = self.statistical_analyzer.compute_correlation(param_values, perf_values)
        
        # Determine impact direction
        if correlation > 0.1:
            direction = "positive"
        elif correlation < -0.1:
            direction = "negative"
        else:
            direction = "neutral"
        
        # Statistical significance
        p_value = self.statistical_analyzer.test_significance(correlation, len(param_values))
        
        return ImpactMeasurement(
            magnitude=abs(correlation),
            direction=direction,
            confidence=1.0 - p_value,
            statistical_significance=1.0 - p_value
        )
    
    def _compute_sensitivity_coefficient(self, 
                                       parameter: str,
                                       param_history: List[ParameterChangeRecord]) -> float:
        """Compute sensitivity coefficient for parameter"""
        
        if len(param_history) < 2:
            return 0.0
        
        # Extract parameter and performance values
        param_values = []
        perf_values = []
        
        for change in param_history:
            if parameter in change.parameter_changes:
                param_values.append(change.parameter_changes[parameter])
                closest_perf = self._find_closest_performance(change.timestamp)
                if closest_perf is not None:
                    perf_values.append(closest_perf)
        
        if len(param_values) < 2:
            return 0.0
        
        return self.statistical_analyzer.compute_sensitivity_coefficient(param_values, perf_values)
    
    def _test_statistical_significance(self, 
                                     parameter: str,
                                     param_history: List[ParameterChangeRecord]) -> float:
        """Test statistical significance of parameter impact"""
        
        if len(param_history) < 3:
            return 0.0
        
        impact = self._measure_direct_impact(parameter, param_history)
        return impact.statistical_significance
    
    def _find_closest_performance(self, timestamp: datetime) -> Optional[float]:
        """Find performance value closest to given timestamp"""
        if not self.performance_history:
            return None
        
        # Find closest performance measurement
        closest_perf = None
        min_time_diff = float('inf')
        
        for perf_time, perf_value in self.performance_history:
            time_diff = abs((timestamp - perf_time).total_seconds())
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_perf = perf_value
        
        # Only return if within reasonable time window (e.g., 1 hour)
        if min_time_diff < 3600:
            return closest_perf
        
        return None
    
    def _collect_sensitivity_data(self) -> SensitivityData:
        """Collect comprehensive sensitivity data"""
        
        # Get all unique parameters
        all_parameters = set()
        for change in self.sensitivity_database.changes:
            all_parameters.update(change.parameter_changes.keys())
        
        parameters = list(all_parameters)
        
        # Analyze each parameter
        impacts = []
        for param in parameters:
            param_history = self.sensitivity_database.get_parameter_history(param)
            impact = self.measure_performance_impact(param)
            impacts.append(impact)
        
        # Generate insights
        insights = []
        for impact in impacts:
            insights.extend(self.generate_sensitivity_insights(impact))
        
        return SensitivityData(
            parameters=parameters,
            measurements=self.sensitivity_database.changes,
            impacts=impacts,
            insights=insights
        )
    
    def _compute_sensitivity_importance(self, 
                                      param_name: str,
                                      sensitivity_data: SensitivityData) -> float:
        """Compute sensitivity-based importance score"""
        
        # Find impact analysis for this parameter
        for impact in sensitivity_data.impacts:
            if param_name in impact.direct_impact:
                measurement = impact.direct_impact[param_name]
                return measurement.magnitude * measurement.confidence
        
        return 0.0
    
    def _compute_frequency_importance(self, 
                                    param_name: str,
                                    sensitivity_data: SensitivityData) -> float:
        """Compute frequency-based importance score"""
        
        total_changes = len(sensitivity_data.measurements)
        if total_changes == 0:
            return 0.0
        
        param_changes = len([m for m in sensitivity_data.measurements 
                           if param_name in m.parameter_changes])
        
        return param_changes / total_changes
    
    def _compute_interaction_importance(self, 
                                      param_name: str,
                                      sensitivity_data: SensitivityData) -> float:
        """Compute interaction-based importance score"""
        
        interaction_score = 0.0
        interaction_count = 0
        
        for impact in sensitivity_data.impacts:
            for interaction in impact.interaction_effects:
                if param_name in interaction.parameters:
                    interaction_score += interaction.strength * interaction.confidence
                    interaction_count += 1
        
        if interaction_count > 0:
            return interaction_score / interaction_count
        else:
            return 0.0
    
    def get_parameter_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive parameter statistics"""
        stats = {}
        
        # Get all unique parameters
        all_parameters = set()
        for change in self.sensitivity_database.changes:
            all_parameters.update(change.parameter_changes.keys())
        
        for param in all_parameters:
            param_history = self.sensitivity_database.get_parameter_history(param)
            impact = self._measure_direct_impact(param, param_history)
            
            stats[param] = {
                'change_frequency': len(param_history),
                'impact_magnitude': impact.magnitude,
                'impact_direction': impact.direction,
                'confidence': impact.confidence,
                'sensitivity_coefficient': self._compute_sensitivity_coefficient(param, param_history),
                'last_changed': param_history[-1].timestamp if param_history else None
            }
        
        return stats