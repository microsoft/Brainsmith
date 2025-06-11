"""
FINN Kernel Performance Modeling
Analytical and empirical models for predicting kernel performance.
"""

import math
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Platform:
    """Target platform specification for performance modeling."""
    name: str
    fpga_part: str
    clock_frequency_mhz: float
    resource_limits: Dict[str, int] = field(default_factory=dict)
    memory_bandwidth_gbps: float = 100.0  # Default memory bandwidth
    dsp_frequency_mhz: Optional[float] = None  # DSP-specific frequency
    
    def __post_init__(self):
        if self.dsp_frequency_mhz is None:
            self.dsp_frequency_mhz = self.clock_frequency_mhz


@dataclass
class PerformanceEstimate:
    """Performance estimation results."""
    throughput_ops_sec: float = 0.0
    latency_cycles: int = 0
    frequency_mhz: float = 0.0
    efficiency_ratio: float = 0.0
    resource_usage: Dict[str, int] = field(default_factory=dict)
    power_estimate_w: float = 0.0
    confidence: float = 1.0  # Confidence in estimate (0-1)
    notes: List[str] = field(default_factory=list)


class FINNPerformanceModel(ABC):
    """Abstract base class for FINN kernel performance models."""
    
    def __init__(self, kernel_name: str, operator_type: str):
        self.kernel_name = kernel_name
        self.operator_type = operator_type
        
    @abstractmethod
    def estimate_performance(self, parameters: Dict[str, Any], 
                           platform: Platform) -> PerformanceEstimate:
        """Estimate performance for given parameters and platform."""
        pass
    
    @abstractmethod
    def estimate_throughput(self, parameters: Dict[str, Any], 
                          platform: Platform) -> float:
        """Estimate throughput in operations per second."""
        pass
    
    @abstractmethod
    def estimate_latency(self, parameters: Dict[str, Any], 
                       platform: Platform) -> int:
        """Estimate latency in clock cycles."""
        pass
    
    @abstractmethod
    def estimate_resource_usage(self, parameters: Dict[str, Any]) -> Dict[str, int]:
        """Estimate FPGA resource usage."""
        pass


class AnalyticalModel(FINNPerformanceModel):
    """Analytical performance model based on mathematical relationships."""
    
    def __init__(self, kernel_name: str, operator_type: str):
        super().__init__(kernel_name, operator_type)
        self.model_coefficients = self._get_model_coefficients(operator_type)
    
    def estimate_performance(self, parameters: Dict[str, Any], 
                           platform: Platform) -> PerformanceEstimate:
        """Comprehensive performance estimation."""
        estimate = PerformanceEstimate()
        
        # Core performance metrics
        estimate.throughput_ops_sec = self.estimate_throughput(parameters, platform)
        estimate.latency_cycles = self.estimate_latency(parameters, platform)
        estimate.frequency_mhz = platform.clock_frequency_mhz
        estimate.resource_usage = self.estimate_resource_usage(parameters)
        estimate.power_estimate_w = self.estimate_power_consumption(parameters, platform)
        
        # Calculate efficiency ratio
        estimate.efficiency_ratio = self._calculate_efficiency(parameters, platform)
        
        # Add model-specific notes
        estimate.notes = self._generate_performance_notes(parameters, platform)
        
        # Set confidence based on parameter validity
        estimate.confidence = self._calculate_confidence(parameters)
        
        return estimate
    
    def estimate_throughput(self, parameters: Dict[str, Any], 
                          platform: Platform) -> float:
        """Estimate throughput based on operator type and parameters."""
        if self.operator_type == 'MatMul':
            return self._estimate_matmul_throughput(parameters, platform)
        elif self.operator_type == 'Thresholding':
            return self._estimate_thresholding_throughput(parameters, platform)
        elif self.operator_type == 'LayerNorm':
            return self._estimate_layernorm_throughput(parameters, platform)
        else:
            # Generic estimation
            return self._estimate_generic_throughput(parameters, platform)
    
    def estimate_latency(self, parameters: Dict[str, Any], 
                       platform: Platform) -> int:
        """Estimate latency in clock cycles."""
        if self.operator_type == 'MatMul':
            return self._estimate_matmul_latency(parameters, platform)
        elif self.operator_type == 'Thresholding':
            return self._estimate_thresholding_latency(parameters, platform)
        elif self.operator_type == 'LayerNorm':
            return self._estimate_layernorm_latency(parameters, platform)
        else:
            # Generic estimation
            return self._estimate_generic_latency(parameters, platform)
    
    def estimate_resource_usage(self, parameters: Dict[str, Any]) -> Dict[str, int]:
        """Estimate FPGA resource usage."""
        if self.operator_type == 'MatMul':
            return self._estimate_matmul_resources(parameters)
        elif self.operator_type == 'Thresholding':
            return self._estimate_thresholding_resources(parameters)
        elif self.operator_type == 'LayerNorm':
            return self._estimate_layernorm_resources(parameters)
        else:
            return self._estimate_generic_resources(parameters)
    
    def estimate_power_consumption(self, parameters: Dict[str, Any], 
                                 platform: Platform) -> float:
        """Estimate power consumption in watts."""
        resource_usage = self.estimate_resource_usage(parameters)
        
        # Power model based on resource usage and frequency
        # These are rough estimates - actual values depend on specific FPGA
        base_power = 2.0  # Base static power
        
        # Dynamic power based on resource usage
        lut_power = resource_usage.get('lut_count', 0) * 0.01e-3  # mW per LUT
        dsp_power = resource_usage.get('dsp_count', 0) * 1.0e-3   # mW per DSP
        bram_power = resource_usage.get('bram_count', 0) * 5.0e-3 # mW per BRAM
        
        # Frequency scaling factor
        freq_factor = platform.clock_frequency_mhz / 100.0  # Normalized to 100MHz
        
        total_power = base_power + (lut_power + dsp_power + bram_power) * freq_factor
        
        return total_power
    
    def _estimate_matmul_throughput(self, parameters: Dict[str, Any], 
                                  platform: Platform) -> float:
        """Estimate MatMul kernel throughput."""
        pe = parameters.get('PE', 1)
        simd = parameters.get('SIMD', 1)
        
        # MatMul throughput = PE * SIMD * frequency
        # Each PE can process SIMD operations per cycle
        ops_per_cycle = pe * simd
        throughput = ops_per_cycle * platform.clock_frequency_mhz * 1e6
        
        return throughput
    
    def _estimate_matmul_latency(self, parameters: Dict[str, Any], 
                               platform: Platform) -> int:
        """Estimate MatMul kernel latency."""
        # Matrix dimensions
        m = parameters.get('M', 256)  # Default matrix size
        n = parameters.get('N', 256)
        k = parameters.get('K', 256)
        
        pe = parameters.get('PE', 1)
        simd = parameters.get('SIMD', 1)
        
        # Simplified latency model: cycles = (M * N * K) / (PE * SIMD) + overhead
        compute_cycles = (m * n * k) // (pe * simd)
        overhead_cycles = 100  # Pipeline fill and memory latency
        
        return compute_cycles + overhead_cycles
    
    def _estimate_matmul_resources(self, parameters: Dict[str, Any]) -> Dict[str, int]:
        """Estimate MatMul resource usage."""
        pe = parameters.get('PE', 1)
        simd = parameters.get('SIMD', 1)
        
        # Resource estimation based on PE and SIMD
        # These are rough estimates based on typical FINN implementations
        lut_per_pe = 150
        dsp_per_pe = simd  # Typically one DSP per SIMD element
        bram_per_pe = max(1, simd // 8)  # Memory for weights and activations
        
        return {
            'lut_count': pe * lut_per_pe,
            'dsp_count': pe * dsp_per_pe,
            'bram_count': pe * bram_per_pe
        }
    
    def _estimate_thresholding_throughput(self, parameters: Dict[str, Any], 
                                        platform: Platform) -> float:
        """Estimate Thresholding kernel throughput."""
        pe = parameters.get('PE', 1)
        
        # Thresholding is typically memory-bound
        # Each PE can process one element per cycle
        ops_per_cycle = pe
        throughput = ops_per_cycle * platform.clock_frequency_mhz * 1e6
        
        return throughput
    
    def _estimate_thresholding_latency(self, parameters: Dict[str, Any], 
                                     platform: Platform) -> int:
        """Estimate Thresholding kernel latency."""
        num_elements = parameters.get('NumElements', 1000)
        pe = parameters.get('PE', 1)
        
        # Latency = elements / PE + pipeline overhead
        compute_cycles = num_elements // pe
        overhead_cycles = 20  # Minimal overhead for simple operation
        
        return compute_cycles + overhead_cycles
    
    def _estimate_thresholding_resources(self, parameters: Dict[str, Any]) -> Dict[str, int]:
        """Estimate Thresholding resource usage."""
        pe = parameters.get('PE', 1)
        
        # Thresholding is LUT-intensive but doesn't use many DSPs
        lut_per_pe = 80
        dsp_per_pe = 0  # No DSPs needed for simple thresholding
        bram_per_pe = 1  # Small amount for thresholds
        
        return {
            'lut_count': pe * lut_per_pe,
            'dsp_count': pe * dsp_per_pe,
            'bram_count': pe * bram_per_pe
        }
    
    def _estimate_layernorm_throughput(self, parameters: Dict[str, Any], 
                                     platform: Platform) -> float:
        """Estimate LayerNorm kernel throughput."""
        pe = parameters.get('PE', 1)
        simd = parameters.get('SIMD', 1)
        
        # LayerNorm involves mean, variance, and normalization
        # More complex than simple operations
        ops_per_cycle = pe * simd * 0.5  # Reduced due to complexity
        throughput = ops_per_cycle * platform.clock_frequency_mhz * 1e6
        
        return throughput
    
    def _estimate_layernorm_latency(self, parameters: Dict[str, Any], 
                                  platform: Platform) -> int:
        """Estimate LayerNorm kernel latency."""
        num_elements = parameters.get('NumElements', 1000)
        pe = parameters.get('PE', 1)
        
        # LayerNorm requires multiple passes: mean, variance, normalization
        passes = 3
        cycles_per_pass = num_elements // pe
        overhead_cycles = 50
        
        return passes * cycles_per_pass + overhead_cycles
    
    def _estimate_layernorm_resources(self, parameters: Dict[str, Any]) -> Dict[str, int]:
        """Estimate LayerNorm resource usage."""
        pe = parameters.get('PE', 1)
        simd = parameters.get('SIMD', 1)
        
        # LayerNorm uses DSPs for arithmetic operations
        lut_per_pe = 200
        dsp_per_pe = max(2, simd // 2)  # DSPs for multiply-add operations
        bram_per_pe = 2  # Memory for intermediate values
        
        return {
            'lut_count': pe * lut_per_pe,
            'dsp_count': pe * dsp_per_pe,
            'bram_count': pe * bram_per_pe
        }
    
    def _estimate_generic_throughput(self, parameters: Dict[str, Any], 
                                   platform: Platform) -> float:
        """Generic throughput estimation for unknown operator types."""
        pe = parameters.get('PE', 1)
        simd = parameters.get('SIMD', 1)
        
        # Conservative estimate
        ops_per_cycle = pe * simd * 0.8
        throughput = ops_per_cycle * platform.clock_frequency_mhz * 1e6
        
        return throughput
    
    def _estimate_generic_latency(self, parameters: Dict[str, Any], 
                                platform: Platform) -> int:
        """Generic latency estimation."""
        return 1000  # Conservative default
    
    def _estimate_generic_resources(self, parameters: Dict[str, Any]) -> Dict[str, int]:
        """Generic resource estimation."""
        pe = parameters.get('PE', 1)
        
        return {
            'lut_count': pe * 100,
            'dsp_count': pe * 1,
            'bram_count': pe * 1
        }
    
    def _calculate_efficiency(self, parameters: Dict[str, Any], 
                            platform: Platform) -> float:
        """Calculate efficiency ratio."""
        resource_usage = self.estimate_resource_usage(parameters)
        throughput = self.estimate_throughput(parameters, platform)
        
        # Efficiency = throughput / resource_cost
        total_resources = sum(resource_usage.values())
        if total_resources > 0:
            return throughput / total_resources
        else:
            return 0.0
    
    def _generate_performance_notes(self, parameters: Dict[str, Any], 
                                  platform: Platform) -> List[str]:
        """Generate performance analysis notes."""
        notes = []
        
        # Check for potential bottlenecks
        resource_usage = self.estimate_resource_usage(parameters)
        
        if platform.resource_limits:
            for resource, usage in resource_usage.items():
                limit = platform.resource_limits.get(resource.replace('_count', ''), float('inf'))
                utilization = usage / limit if limit > 0 else 0
                
                if utilization > 0.8:
                    notes.append(f"High {resource} utilization: {utilization:.1%}")
                elif utilization > 0.5:
                    notes.append(f"Moderate {resource} utilization: {utilization:.1%}")
        
        # Check frequency feasibility
        if platform.clock_frequency_mhz > 500:
            notes.append("High clock frequency may require timing optimization")
        
        return notes
    
    def _calculate_confidence(self, parameters: Dict[str, Any]) -> float:
        """Calculate confidence in performance estimate."""
        confidence = 1.0
        
        # Reduce confidence for missing parameters
        required_params = ['PE']
        for param in required_params:
            if param not in parameters:
                confidence *= 0.8
        
        # Reduce confidence for unusual parameter values
        pe = parameters.get('PE', 1)
        if pe > 64:  # Very high parallelism
            confidence *= 0.9
        
        return max(0.1, confidence)  # Minimum 10% confidence
    
    def _get_model_coefficients(self, operator_type: str) -> Dict[str, float]:
        """Get model coefficients for operator type."""
        coefficients = {
            'MatMul': {
                'throughput_factor': 1.0,
                'latency_overhead': 100,
                'lut_per_pe': 150,
                'dsp_per_pe': 1.0
            },
            'Thresholding': {
                'throughput_factor': 1.0,
                'latency_overhead': 20,
                'lut_per_pe': 80,
                'dsp_per_pe': 0.0
            },
            'LayerNorm': {
                'throughput_factor': 0.5,
                'latency_overhead': 50,
                'lut_per_pe': 200,
                'dsp_per_pe': 2.0
            }
        }
        
        return coefficients.get(operator_type, coefficients['MatMul'])


class EmpiricalModel(FINNPerformanceModel):
    """Empirical performance model based on historical data."""
    
    def __init__(self, kernel_name: str, operator_type: str):
        super().__init__(kernel_name, operator_type)
        self.training_data = []
        self.model_parameters = {}
        
    def add_training_data(self, parameters: Dict[str, Any], 
                         platform: Platform, 
                         measured_performance: PerformanceEstimate):
        """Add measured performance data for model training."""
        training_point = {
            'parameters': parameters.copy(),
            'platform': platform,
            'performance': measured_performance
        }
        self.training_data.append(training_point)
        
        # Retrain model with new data
        self._train_model()
    
    def estimate_performance(self, parameters: Dict[str, Any], 
                           platform: Platform) -> PerformanceEstimate:
        """Estimate performance using empirical model."""
        if not self.training_data:
            # Fall back to analytical model if no training data
            analytical_model = AnalyticalModel(self.kernel_name, self.operator_type)
            estimate = analytical_model.estimate_performance(parameters, platform)
            estimate.confidence = 0.5  # Lower confidence for fallback
            estimate.notes.append("Using analytical fallback (no empirical data)")
            return estimate
        
        # Find most similar training point
        best_match = self._find_best_match(parameters, platform)
        
        if best_match:
            # Interpolate or use best match
            estimate = self._interpolate_performance(parameters, platform, best_match)
            estimate.confidence = self._calculate_empirical_confidence(parameters, platform)
            return estimate
        else:
            # No good match found, use analytical fallback
            analytical_model = AnalyticalModel(self.kernel_name, self.operator_type)
            estimate = analytical_model.estimate_performance(parameters, platform)
            estimate.confidence = 0.3
            estimate.notes.append("No similar empirical data found")
            return estimate
    
    def estimate_throughput(self, parameters: Dict[str, Any], 
                          platform: Platform) -> float:
        """Estimate throughput using empirical data."""
        estimate = self.estimate_performance(parameters, platform)
        return estimate.throughput_ops_sec
    
    def estimate_latency(self, parameters: Dict[str, Any], 
                       platform: Platform) -> int:
        """Estimate latency using empirical data."""
        estimate = self.estimate_performance(parameters, platform)
        return estimate.latency_cycles
    
    def estimate_resource_usage(self, parameters: Dict[str, Any]) -> Dict[str, int]:
        """Estimate resource usage using empirical data."""
        if not self.training_data:
            # Fall back to analytical model
            analytical_model = AnalyticalModel(self.kernel_name, self.operator_type)
            return analytical_model.estimate_resource_usage(parameters)
        
        # Find best match and return its resource usage
        best_match = self._find_best_match(parameters, None)
        if best_match:
            return best_match['performance'].resource_usage
        else:
            analytical_model = AnalyticalModel(self.kernel_name, self.operator_type)
            return analytical_model.estimate_resource_usage(parameters)
    
    def _train_model(self):
        """Train empirical model with available data."""
        if len(self.training_data) < 3:
            return  # Need minimum data for training
        
        # Simple statistical model for now
        # Can be enhanced with machine learning algorithms
        self.model_parameters = self._calculate_statistical_parameters()
    
    def _calculate_statistical_parameters(self) -> Dict[str, Any]:
        """Calculate statistical parameters from training data."""
        parameters = {}
        
        # Calculate mean and std dev for each metric
        throughputs = [d['performance'].throughput_ops_sec for d in self.training_data]
        latencies = [d['performance'].latency_cycles for d in self.training_data]
        
        parameters['throughput_mean'] = sum(throughputs) / len(throughputs)
        parameters['throughput_std'] = math.sqrt(
            sum((x - parameters['throughput_mean'])**2 for x in throughputs) / len(throughputs)
        )
        
        parameters['latency_mean'] = sum(latencies) / len(latencies)
        parameters['latency_std'] = math.sqrt(
            sum((x - parameters['latency_mean'])**2 for x in latencies) / len(latencies)
        )
        
        return parameters
    
    def _find_best_match(self, parameters: Dict[str, Any], 
                        platform: Optional[Platform]) -> Optional[Dict[str, Any]]:
        """Find best matching training data point."""
        if not self.training_data:
            return None
        
        best_match = None
        best_distance = float('inf')
        
        for data_point in self.training_data:
            distance = self._calculate_parameter_distance(
                parameters, data_point['parameters']
            )
            
            if platform:
                platform_distance = self._calculate_platform_distance(
                    platform, data_point['platform']
                )
                distance += platform_distance
            
            if distance < best_distance:
                best_distance = distance
                best_match = data_point
        
        return best_match
    
    def _calculate_parameter_distance(self, params1: Dict[str, Any], 
                                    params2: Dict[str, Any]) -> float:
        """Calculate distance between parameter sets."""
        distance = 0.0
        all_keys = set(params1.keys()) | set(params2.keys())
        
        for key in all_keys:
            val1 = params1.get(key, 0)
            val2 = params2.get(key, 0)
            
            # Normalize by parameter type
            if key in ['PE', 'SIMD']:
                # These parameters have multiplicative effect
                if val2 > 0:
                    distance += abs(math.log(val1 + 1) - math.log(val2 + 1))
                else:
                    distance += 1.0
            else:
                # Linear parameters
                distance += abs(val1 - val2) / max(abs(val1), abs(val2), 1.0)
        
        return distance
    
    def _calculate_platform_distance(self, platform1: Platform, 
                                   platform2: Platform) -> float:
        """Calculate distance between platforms."""
        distance = 0.0
        
        # Frequency difference
        freq_diff = abs(platform1.clock_frequency_mhz - platform2.clock_frequency_mhz)
        distance += freq_diff / max(platform1.clock_frequency_mhz, platform2.clock_frequency_mhz)
        
        # FPGA part difference (simple string comparison)
        if platform1.fpga_part != platform2.fpga_part:
            distance += 0.5
        
        return distance
    
    def _interpolate_performance(self, parameters: Dict[str, Any], 
                               platform: Platform,
                               best_match: Dict[str, Any]) -> PerformanceEstimate:
        """Interpolate performance from best match."""
        # Simple approach: use best match with scaling
        base_performance = best_match['performance']
        
        # Scale based on parameter differences
        scale_factor = self._calculate_scale_factor(
            parameters, best_match['parameters']
        )
        
        estimate = PerformanceEstimate(
            throughput_ops_sec=base_performance.throughput_ops_sec * scale_factor,
            latency_cycles=int(base_performance.latency_cycles / scale_factor),
            frequency_mhz=platform.clock_frequency_mhz,
            efficiency_ratio=base_performance.efficiency_ratio,
            resource_usage=base_performance.resource_usage.copy(),
            power_estimate_w=base_performance.power_estimate_w * scale_factor
        )
        
        return estimate
    
    def _calculate_scale_factor(self, target_params: Dict[str, Any], 
                              base_params: Dict[str, Any]) -> float:
        """Calculate scaling factor for performance interpolation."""
        # Simple scaling based on PE and SIMD
        target_pe = target_params.get('PE', 1)
        target_simd = target_params.get('SIMD', 1)
        base_pe = base_params.get('PE', 1)
        base_simd = base_params.get('SIMD', 1)
        
        pe_scale = target_pe / base_pe if base_pe > 0 else 1.0
        simd_scale = target_simd / base_simd if base_simd > 0 else 1.0
        
        return pe_scale * simd_scale
    
    def _calculate_empirical_confidence(self, parameters: Dict[str, Any], 
                                      platform: Platform) -> float:
        """Calculate confidence in empirical estimate."""
        if not self.training_data:
            return 0.1
        
        # Higher confidence with more training data
        data_confidence = min(1.0, len(self.training_data) / 10.0)
        
        # Lower confidence for extrapolation
        best_match = self._find_best_match(parameters, platform)
        if best_match:
            distance = self._calculate_parameter_distance(
                parameters, best_match['parameters']
            )
            extrapolation_penalty = max(0.1, 1.0 - distance)
            return data_confidence * extrapolation_penalty
        else:
            return 0.1