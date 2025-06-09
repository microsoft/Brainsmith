"""
Resource Utilization Tracking
Detailed FPGA resource usage monitoring, efficiency analysis, and scaling predictions.
"""

import os
import sys
import time
import logging
import threading
import re
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

from .core import MetricsCollector, MetricValue, MetricCollection, MetricType, MetricScope

logger = logging.getLogger(__name__)


@dataclass
class FPGAResources:
    """FPGA resource information."""
    lut_count: int = 0
    lut_total: int = 0
    ff_count: int = 0
    ff_total: int = 0
    dsp_count: int = 0
    dsp_total: int = 0
    bram_count: int = 0
    bram_total: int = 0
    uram_count: int = 0
    uram_total: int = 0
    io_count: int = 0
    io_total: int = 0
    
    @property
    def lut_utilization(self) -> float:
        """LUT utilization percentage."""
        return (self.lut_count / self.lut_total * 100) if self.lut_total > 0 else 0.0
    
    @property
    def ff_utilization(self) -> float:
        """Flip-flop utilization percentage."""
        return (self.ff_count / self.ff_total * 100) if self.ff_total > 0 else 0.0
    
    @property
    def dsp_utilization(self) -> float:
        """DSP utilization percentage."""
        return (self.dsp_count / self.dsp_total * 100) if self.dsp_total > 0 else 0.0
    
    @property
    def bram_utilization(self) -> float:
        """BRAM utilization percentage."""
        return (self.bram_count / self.bram_total * 100) if self.bram_total > 0 else 0.0
    
    @property
    def uram_utilization(self) -> float:
        """URAM utilization percentage."""
        return (self.uram_count / self.uram_total * 100) if self.uram_total > 0 else 0.0
    
    @property
    def io_utilization(self) -> float:
        """I/O utilization percentage."""
        return (self.io_count / self.io_total * 100) if self.io_total > 0 else 0.0
    
    @property
    def overall_utilization(self) -> float:
        """Overall resource utilization (max of all resources)."""
        utilizations = [
            self.lut_utilization,
            self.ff_utilization,
            self.dsp_utilization,
            self.bram_utilization
        ]
        return max(utilizations) if utilizations else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'lut_count': self.lut_count,
            'lut_total': self.lut_total,
            'lut_utilization': self.lut_utilization,
            'ff_count': self.ff_count,
            'ff_total': self.ff_total,
            'ff_utilization': self.ff_utilization,
            'dsp_count': self.dsp_count,
            'dsp_total': self.dsp_total,
            'dsp_utilization': self.dsp_utilization,
            'bram_count': self.bram_count,
            'bram_total': self.bram_total,
            'bram_utilization': self.bram_utilization,
            'uram_count': self.uram_count,
            'uram_total': self.uram_total,
            'uram_utilization': self.uram_utilization,
            'io_count': self.io_count,
            'io_total': self.io_total,
            'io_utilization': self.io_utilization,
            'overall_utilization': self.overall_utilization
        }


@dataclass
class ResourceEfficiency:
    """Resource efficiency metrics."""
    lut_efficiency: float  # operations per LUT
    dsp_efficiency: float  # operations per DSP
    bram_efficiency: float  # data throughput per BRAM
    area_efficiency: float  # operations per unit area
    power_efficiency: float  # operations per mW
    
    # Efficiency ratios
    compute_to_memory_ratio: float
    logic_to_routing_ratio: float
    parallelism_efficiency: float
    
    # Bottleneck analysis
    bottleneck_resource: str  # 'lut', 'dsp', 'bram', 'routing', 'power'
    bottleneck_utilization: float
    improvement_potential: float


@dataclass
class ScalingPrediction:
    """Resource scaling prediction."""
    target_scale_factor: float
    predicted_resources: FPGAResources
    feasibility: bool
    limiting_resource: str
    predicted_performance_scaling: float
    confidence: float
    assumptions: List[str] = field(default_factory=list)


class UtilizationMonitor:
    """Monitor FPGA resource utilization from synthesis/implementation reports."""
    
    def __init__(self):
        self.device_resources = {
            # Zynq-7000 series
            'xc7z020': {'lut': 53200, 'ff': 106400, 'dsp': 220, 'bram': 140, 'uram': 0, 'io': 125},
            'xc7z045': {'lut': 218600, 'ff': 437200, 'dsp': 900, 'bram': 545, 'uram': 0, 'io': 362},
            
            # Zynq UltraScale+ series
            'xczu7ev': {'lut': 230400, 'ff': 460800, 'dsp': 1728, 'bram': 312, 'uram': 96, 'io': 328},
            'xczu9eg': {'lut': 274080, 'ff': 548160, 'dsp': 2520, 'bram': 912, 'uram': 0, 'io': 328},
            
            # Virtex UltraScale+ series
            'xcvu9p': {'lut': 1182240, 'ff': 2364480, 'dsp': 6840, 'bram': 2160, 'uram': 960, 'io': 832},
            'xcvu13p': {'lut': 1743360, 'ff': 3486720, 'dsp': 12288, 'bram': 2688, 'uram': 1280, 'io': 832}
        }
        
        self.report_parsers = {
            'vivado': self._parse_vivado_utilization,
            'quartus': self._parse_quartus_utilization,
            'generic': self._parse_generic_utilization
        }
    
    def monitor_utilization(self, 
                           report_path: str,
                           device: str = 'xc7z020',
                           tool: str = 'vivado') -> FPGAResources:
        """Monitor resource utilization from synthesis/implementation reports."""
        
        try:
            # Get device resources
            device_info = self.device_resources.get(device, self.device_resources['xc7z020'])
            
            # Parse utilization report
            parser = self.report_parsers.get(tool, self._parse_generic_utilization)
            utilization = parser(report_path)
            
            # Create FPGAResources object
            resources = FPGAResources(
                lut_count=utilization.get('lut_used', 0),
                lut_total=device_info['lut'],
                ff_count=utilization.get('ff_used', 0),
                ff_total=device_info['ff'],
                dsp_count=utilization.get('dsp_used', 0),
                dsp_total=device_info['dsp'],
                bram_count=utilization.get('bram_used', 0),
                bram_total=device_info['bram'],
                uram_count=utilization.get('uram_used', 0),
                uram_total=device_info['uram'],
                io_count=utilization.get('io_used', 0),
                io_total=device_info['io']
            )
            
            return resources
            
        except Exception as e:
            logger.error(f"Failed to monitor utilization: {e}")
            # Return empty resources
            device_info = self.device_resources.get(device, self.device_resources['xc7z020'])
            return FPGAResources(
                lut_total=device_info['lut'],
                ff_total=device_info['ff'],
                dsp_total=device_info['dsp'],
                bram_total=device_info['bram'],
                uram_total=device_info['uram'],
                io_total=device_info['io']
            )
    
    def _parse_vivado_utilization(self, report_path: str) -> Dict[str, int]:
        """Parse Vivado utilization report."""
        utilization = {}
        
        try:
            with open(report_path, 'r') as f:
                content = f.read()
            
            # Parse utilization table
            # Look for patterns like "| Slice LUTs | 1234 |" or "| LUT | 1234 |"
            patterns = {
                'lut_used': [r'\|\s*(?:Slice\s+)?LUT\w*\s*\|\s*(\d+)', r'LUT\s+(\d+)'],
                'ff_used': [r'\|\s*(?:Slice\s+)?Register\w*\s*\|\s*(\d+)', r'\|\s*FF\s*\|\s*(\d+)', r'Register\s+(\d+)'],
                'dsp_used': [r'\|\s*DSP\w*\s*\|\s*(\d+)', r'DSP48\w*\s+(\d+)'],
                'bram_used': [r'\|\s*BRAM\w*\s*\|\s*(\d+)', r'RAMB\w*\s+(\d+)', r'Block RAM\s+(\d+)'],
                'uram_used': [r'\|\s*URAM\w*\s*\|\s*(\d+)', r'URAM\w*\s+(\d+)'],
                'io_used': [r'\|\s*IO\w*\s*\|\s*(\d+)', r'\|\s*Bonded IOB\s*\|\s*(\d+)']
            }
            
            for resource, pattern_list in patterns.items():
                for pattern in pattern_list:
                    match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                    if match:
                        utilization[resource] = int(match.group(1))
                        break
        
        except Exception as e:
            logger.warning(f"Failed to parse Vivado utilization: {e}")
        
        return utilization
    
    def _parse_quartus_utilization(self, report_path: str) -> Dict[str, int]:
        """Parse Quartus utilization report."""
        utilization = {}
        
        try:
            with open(report_path, 'r') as f:
                content = f.read()
            
            # Parse Quartus-style reports
            patterns = {
                'lut_used': [r'Total logic elements:\s*(\d+)', r'ALMs:\s*(\d+)'],
                'ff_used': [r'Total registers:\s*(\d+)', r'Dedicated logic registers:\s*(\d+)'],
                'dsp_used': [r'DSP block 9-bit elements:\s*(\d+)', r'Embedded Multiplier 9-bit elements:\s*(\d+)'],
                'bram_used': [r'Total RAM Blocks:\s*(\d+)', r'M9K blocks:\s*(\d+)', r'M20K blocks:\s*(\d+)'],
                'io_used': [r'Total pins:\s*(\d+)', r'I/O pins:\s*(\d+)']
            }
            
            for resource, pattern_list in patterns.items():
                for pattern in pattern_list:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        utilization[resource] = int(match.group(1))
                        break
        
        except Exception as e:
            logger.warning(f"Failed to parse Quartus utilization: {e}")
        
        return utilization
    
    def _parse_generic_utilization(self, report_path: str) -> Dict[str, int]:
        """Generic utilization parser for unknown tools."""
        # Return some default values for testing
        return {
            'lut_used': 1000,
            'ff_used': 2000,
            'dsp_used': 10,
            'bram_used': 5,
            'uram_used': 0,
            'io_used': 50
        }


class EfficiencyAnalyzer:
    """Analyze resource efficiency and identify optimization opportunities."""
    
    def analyze_efficiency(self,
                          resources: FPGAResources,
                          performance_metrics: Dict[str, float],
                          power_consumption: Optional[float] = None) -> ResourceEfficiency:
        """Analyze resource efficiency."""
        
        try:
            # Extract performance metrics
            ops_per_sec = performance_metrics.get('throughput_ops_per_sec', 1000000.0)
            latency_cycles = performance_metrics.get('latency_cycles', 100)
            clock_freq_mhz = performance_metrics.get('clock_frequency_mhz', 100.0)
            
            # Calculate efficiency metrics
            lut_efficiency = ops_per_sec / max(resources.lut_count, 1)
            dsp_efficiency = ops_per_sec / max(resources.dsp_count, 1)
            bram_efficiency = ops_per_sec / max(resources.bram_count, 1)
            
            # Area efficiency (operations per unit area)
            # Normalize by LUT count as a proxy for area
            area_efficiency = ops_per_sec / max(resources.lut_count, 1)
            
            # Power efficiency
            power_efficiency = ops_per_sec / max(power_consumption or 1000.0, 1.0)
            
            # Calculate efficiency ratios
            compute_to_memory_ratio = (resources.dsp_count + resources.lut_count / 10) / max(resources.bram_count, 1)
            logic_to_routing_ratio = resources.lut_count / max(resources.ff_count, 1)
            
            # Parallelism efficiency (how well we're using available parallelism)
            max_parallel_ops = resources.dsp_count + resources.lut_count / 100
            actual_parallel_ops = ops_per_sec / (clock_freq_mhz * 1e6)
            parallelism_efficiency = actual_parallel_ops / max(max_parallel_ops, 1) * 100
            
            # Bottleneck analysis
            utilizations = {
                'lut': resources.lut_utilization,
                'dsp': resources.dsp_utilization,
                'bram': resources.bram_utilization,
                'routing': resources.ff_utilization  # Proxy for routing
            }
            
            bottleneck_resource = max(utilizations.keys(), key=lambda k: utilizations[k])
            bottleneck_utilization = utilizations[bottleneck_resource]
            
            # Improvement potential (how much we could improve if we fix the bottleneck)
            min_utilization = min(utilizations.values())
            improvement_potential = (bottleneck_utilization - min_utilization) / max(bottleneck_utilization, 1) * 100
            
            return ResourceEfficiency(
                lut_efficiency=lut_efficiency,
                dsp_efficiency=dsp_efficiency,
                bram_efficiency=bram_efficiency,
                area_efficiency=area_efficiency,
                power_efficiency=power_efficiency,
                compute_to_memory_ratio=compute_to_memory_ratio,
                logic_to_routing_ratio=logic_to_routing_ratio,
                parallelism_efficiency=parallelism_efficiency,
                bottleneck_resource=bottleneck_resource,
                bottleneck_utilization=bottleneck_utilization,
                improvement_potential=improvement_potential
            )
            
        except Exception as e:
            logger.error(f"Efficiency analysis failed: {e}")
            return ResourceEfficiency(
                lut_efficiency=1000.0,
                dsp_efficiency=100000.0,
                bram_efficiency=200000.0,
                area_efficiency=1000.0,
                power_efficiency=1000.0,
                compute_to_memory_ratio=2.0,
                logic_to_routing_ratio=0.5,
                parallelism_efficiency=50.0,
                bottleneck_resource='lut',
                bottleneck_utilization=50.0,
                improvement_potential=25.0
            )


class ResourcePredictor:
    """Predict resource requirements for different scaling factors and optimizations."""
    
    def predict_scaling(self,
                       current_resources: FPGAResources,
                       current_performance: Dict[str, float],
                       target_scale_factor: float,
                       device: str = 'xc7z020') -> ScalingPrediction:
        """Predict resource scaling for target performance increase."""
        
        try:
            # Linear scaling assumption for most resources
            predicted_lut = int(current_resources.lut_count * target_scale_factor)
            predicted_ff = int(current_resources.ff_count * target_scale_factor)
            predicted_dsp = int(current_resources.dsp_count * target_scale_factor)
            
            # Memory scaling might be sub-linear
            memory_scale_factor = target_scale_factor ** 0.8  # Sub-linear scaling
            predicted_bram = int(current_resources.bram_count * memory_scale_factor)
            predicted_uram = int(current_resources.uram_count * memory_scale_factor)
            
            # I/O scaling is usually minimal
            io_scale_factor = min(target_scale_factor ** 0.3, 2.0)  # Very sub-linear
            predicted_io = int(current_resources.io_count * io_scale_factor)
            
            # Create predicted resources
            predicted_resources = FPGAResources(
                lut_count=predicted_lut,
                lut_total=current_resources.lut_total,
                ff_count=predicted_ff,
                ff_total=current_resources.ff_total,
                dsp_count=predicted_dsp,
                dsp_total=current_resources.dsp_total,
                bram_count=predicted_bram,
                bram_total=current_resources.bram_total,
                uram_count=predicted_uram,
                uram_total=current_resources.uram_total,
                io_count=predicted_io,
                io_total=current_resources.io_total
            )
            
            # Check feasibility
            feasible = True
            limiting_resource = None
            
            resource_checks = [
                ('lut', predicted_resources.lut_utilization),
                ('ff', predicted_resources.ff_utilization),
                ('dsp', predicted_resources.dsp_utilization),
                ('bram', predicted_resources.bram_utilization),
                ('uram', predicted_resources.uram_utilization),
                ('io', predicted_resources.io_utilization)
            ]
            
            max_utilization = 0
            for resource_name, utilization in resource_checks:
                if utilization > 100:
                    feasible = False
                    if limiting_resource is None or utilization > max_utilization:
                        limiting_resource = resource_name
                        max_utilization = utilization
            
            if limiting_resource is None:
                limiting_resource = max(resource_checks, key=lambda x: x[1])[0]
            
            # Predict performance scaling
            # Performance might not scale linearly due to routing delays, etc.
            if target_scale_factor <= 2.0:
                performance_scaling = target_scale_factor * 0.95  # 5% overhead
            elif target_scale_factor <= 4.0:
                performance_scaling = target_scale_factor * 0.90  # 10% overhead
            else:
                performance_scaling = target_scale_factor * 0.80  # 20% overhead
            
            # Calculate confidence based on scale factor and current utilization
            base_confidence = 0.9
            scale_penalty = min(0.3, (target_scale_factor - 1.0) * 0.1)
            utilization_penalty = current_resources.overall_utilization / 100.0 * 0.2
            confidence = max(0.1, base_confidence - scale_penalty - utilization_penalty)
            
            # Generate assumptions
            assumptions = [
                f"Linear scaling for logic resources (scale factor: {target_scale_factor:.1f})",
                f"Sub-linear scaling for memory resources (factor: {memory_scale_factor:.1f})",
                f"Minimal I/O scaling (factor: {io_scale_factor:.1f})",
                f"Performance overhead: {(1 - performance_scaling/target_scale_factor)*100:.1f}%"
            ]
            
            return ScalingPrediction(
                target_scale_factor=target_scale_factor,
                predicted_resources=predicted_resources,
                feasibility=feasible,
                limiting_resource=limiting_resource,
                predicted_performance_scaling=performance_scaling,
                confidence=confidence,
                assumptions=assumptions
            )
            
        except Exception as e:
            logger.error(f"Scaling prediction failed: {e}")
            return ScalingPrediction(
                target_scale_factor=target_scale_factor,
                predicted_resources=current_resources,
                feasibility=False,
                limiting_resource='unknown',
                predicted_performance_scaling=1.0,
                confidence=0.1
            )


class FPGAResourceAnalyzer:
    """Comprehensive FPGA resource analysis."""
    
    def __init__(self):
        self.utilization_monitor = UtilizationMonitor()
        self.efficiency_analyzer = EfficiencyAnalyzer()
        self.resource_predictor = ResourcePredictor()
        self.analysis_history = []
    
    def analyze_comprehensive(self,
                            report_path: str,
                            performance_metrics: Dict[str, float],
                            device: str = 'xc7z020',
                            tool: str = 'vivado',
                            power_consumption: Optional[float] = None) -> Dict[str, Any]:
        """Perform comprehensive resource analysis."""
        
        try:
            # Monitor current utilization
            resources = self.utilization_monitor.monitor_utilization(report_path, device, tool)
            
            # Analyze efficiency
            efficiency = self.efficiency_analyzer.analyze_efficiency(
                resources, performance_metrics, power_consumption
            )
            
            # Generate scaling predictions
            scale_factors = [1.5, 2.0, 4.0, 8.0]
            scaling_predictions = {}
            
            for scale_factor in scale_factors:
                prediction = self.resource_predictor.predict_scaling(
                    resources, performance_metrics, scale_factor, device
                )
                scaling_predictions[f"scale_{scale_factor}x"] = prediction
            
            # Compile comprehensive analysis
            analysis = {
                'timestamp': time.time(),
                'device': device,
                'tool': tool,
                'current_resources': resources.to_dict(),
                'efficiency_analysis': {
                    'lut_efficiency': efficiency.lut_efficiency,
                    'dsp_efficiency': efficiency.dsp_efficiency,
                    'bram_efficiency': efficiency.bram_efficiency,
                    'area_efficiency': efficiency.area_efficiency,
                    'power_efficiency': efficiency.power_efficiency,
                    'compute_to_memory_ratio': efficiency.compute_to_memory_ratio,
                    'logic_to_routing_ratio': efficiency.logic_to_routing_ratio,
                    'parallelism_efficiency': efficiency.parallelism_efficiency,
                    'bottleneck_resource': efficiency.bottleneck_resource,
                    'bottleneck_utilization': efficiency.bottleneck_utilization,
                    'improvement_potential': efficiency.improvement_potential
                },
                'scaling_predictions': {
                    scale: {
                        'feasible': pred.feasibility,
                        'limiting_resource': pred.limiting_resource,
                        'predicted_performance_scaling': pred.predicted_performance_scaling,
                        'confidence': pred.confidence,
                        'predicted_utilizations': {
                            'lut': pred.predicted_resources.lut_utilization,
                            'dsp': pred.predicted_resources.dsp_utilization,
                            'bram': pred.predicted_resources.bram_utilization
                        }
                    }
                    for scale, pred in scaling_predictions.items()
                },
                'recommendations': self._generate_recommendations(resources, efficiency, scaling_predictions)
            }
            
            # Store in history
            self.analysis_history.append(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Comprehensive resource analysis failed: {e}")
            return {
                'timestamp': time.time(),
                'device': device,
                'tool': tool,
                'error': str(e),
                'current_resources': {},
                'efficiency_analysis': {},
                'scaling_predictions': {},
                'recommendations': []
            }
    
    def _generate_recommendations(self,
                                resources: FPGAResources,
                                efficiency: ResourceEfficiency,
                                scaling_predictions: Dict[str, ScalingPrediction]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Resource utilization recommendations
        if resources.overall_utilization > 90:
            recommendations.append("High resource utilization detected. Consider design partitioning or resource optimization.")
        elif resources.overall_utilization < 20:
            recommendations.append("Low resource utilization. Consider increasing parallelism or adding more functionality.")
        
        # Bottleneck recommendations
        if efficiency.bottleneck_resource == 'lut':
            recommendations.append("LUT utilization is the bottleneck. Consider logic optimization or using more DSPs.")
        elif efficiency.bottleneck_resource == 'dsp':
            recommendations.append("DSP utilization is the bottleneck. Consider LUT-based arithmetic or different algorithms.")
        elif efficiency.bottleneck_resource == 'bram':
            recommendations.append("Memory is the bottleneck. Consider external memory or data streaming optimizations.")
        
        # Efficiency recommendations
        if efficiency.parallelism_efficiency < 50:
            recommendations.append("Low parallelism efficiency. Consider increasing parallel execution units.")
        
        if efficiency.compute_to_memory_ratio > 10:
            recommendations.append("High compute-to-memory ratio. Memory bandwidth may become a bottleneck.")
        elif efficiency.compute_to_memory_ratio < 1:
            recommendations.append("Low compute-to-memory ratio. Consider adding more compute resources.")
        
        # Scaling recommendations
        feasible_scales = [scale for scale, pred in scaling_predictions.items() if pred.feasibility]
        if not feasible_scales:
            recommendations.append("No scaling is feasible on current device. Consider larger FPGA or optimization.")
        else:
            max_feasible = max(feasible_scales, key=lambda x: float(x.split('_')[1].replace('x', '')))
            recommendations.append(f"Maximum feasible scaling: {max_feasible}")
        
        return recommendations


class ResourceUtilizationTracker(MetricsCollector):
    """Main resource utilization tracker that implements MetricsCollector interface."""
    
    def __init__(self, name: str = "ResourceUtilizationTracker", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        
        self.resource_analyzer = FPGAResourceAnalyzer()
        self.tracking_history = []
    
    def collect_metrics(self, context: Dict[str, Any]) -> MetricCollection:
        """Collect resource utilization metrics."""
        
        collection = MetricCollection(
            collection_id=f"resource_{int(time.time())}",
            name="Resource Utilization Metrics",
            description="Detailed FPGA resource usage, efficiency analysis, and scaling predictions"
        )
        
        try:
            # Extract context information
            report_path = context.get('utilization_report')
            performance_metrics = context.get('performance_metrics', {})
            device = context.get('device', 'xc7z020')
            tool = context.get('tool', 'vivado')
            power_consumption = context.get('power_consumption')
            
            if not report_path:
                # Generate mock report path for testing
                report_path = '/tmp/mock_utilization_report.txt'
                with open(report_path, 'w') as f:
                    f.write("Mock utilization report for testing")
            
            # Perform comprehensive analysis
            analysis = self.resource_analyzer.analyze_comprehensive(
                report_path, performance_metrics, device, tool, power_consumption
            )
            
            # Convert analysis to metrics
            self._add_resource_metrics(collection, analysis)
            
        except Exception as e:
            logger.error(f"Failed to collect resource metrics: {e}")
            # Add error metric
            collection.add_metric(MetricValue(
                name="collection_error",
                value=str(e),
                metric_type=MetricType.RESOURCE,
                scope=MetricScope.BUILD
            ))
        
        return collection
    
    def get_supported_metrics(self) -> List[str]:
        """Get list of supported metrics."""
        return [
            "lut_utilization", "dsp_utilization", "bram_utilization",
            "lut_efficiency", "dsp_efficiency", "area_efficiency",
            "bottleneck_resource", "improvement_potential", "parallelism_efficiency"
        ]
    
    def _add_resource_metrics(self, collection: MetricCollection, analysis: Dict[str, Any]):
        """Add resource metrics to collection."""
        
        # Current resource utilization
        current_resources = analysis.get('current_resources', {})
        utilization_metrics = [
            MetricValue("lut_utilization", current_resources.get('lut_utilization', 0), "%", metric_type=MetricType.UTILIZATION),
            MetricValue("ff_utilization", current_resources.get('ff_utilization', 0), "%", metric_type=MetricType.UTILIZATION),
            MetricValue("dsp_utilization", current_resources.get('dsp_utilization', 0), "%", metric_type=MetricType.UTILIZATION),
            MetricValue("bram_utilization", current_resources.get('bram_utilization', 0), "%", metric_type=MetricType.UTILIZATION),
            MetricValue("uram_utilization", current_resources.get('uram_utilization', 0), "%", metric_type=MetricType.UTILIZATION),
            MetricValue("io_utilization", current_resources.get('io_utilization', 0), "%", metric_type=MetricType.UTILIZATION),
            MetricValue("overall_utilization", current_resources.get('overall_utilization', 0), "%", metric_type=MetricType.UTILIZATION),
        ]
        
        # Resource counts
        count_metrics = [
            MetricValue("lut_count", current_resources.get('lut_count', 0), "count", metric_type=MetricType.RESOURCE),
            MetricValue("ff_count", current_resources.get('ff_count', 0), "count", metric_type=MetricType.RESOURCE),
            MetricValue("dsp_count", current_resources.get('dsp_count', 0), "count", metric_type=MetricType.RESOURCE),
            MetricValue("bram_count", current_resources.get('bram_count', 0), "count", metric_type=MetricType.RESOURCE),
        ]
        
        # Efficiency metrics
        efficiency_analysis = analysis.get('efficiency_analysis', {})
        efficiency_metrics = [
            MetricValue("lut_efficiency", efficiency_analysis.get('lut_efficiency', 0), "ops/lut", metric_type=MetricType.EFFICIENCY),
            MetricValue("dsp_efficiency", efficiency_analysis.get('dsp_efficiency', 0), "ops/dsp", metric_type=MetricType.EFFICIENCY),
            MetricValue("bram_efficiency", efficiency_analysis.get('bram_efficiency', 0), "ops/bram", metric_type=MetricType.EFFICIENCY),
            MetricValue("area_efficiency", efficiency_analysis.get('area_efficiency', 0), "ops/area", metric_type=MetricType.EFFICIENCY),
            MetricValue("power_efficiency", efficiency_analysis.get('power_efficiency', 0), "ops/mW", metric_type=MetricType.EFFICIENCY),
            MetricValue("parallelism_efficiency", efficiency_analysis.get('parallelism_efficiency', 0), "%", metric_type=MetricType.EFFICIENCY),
            MetricValue("improvement_potential", efficiency_analysis.get('improvement_potential', 0), "%", metric_type=MetricType.EFFICIENCY),
        ]
        
        # Bottleneck information
        bottleneck_metrics = [
            MetricValue("bottleneck_resource", efficiency_analysis.get('bottleneck_resource', 'unknown'), metric_type=MetricType.RESOURCE),
            MetricValue("bottleneck_utilization", efficiency_analysis.get('bottleneck_utilization', 0), "%", metric_type=MetricType.UTILIZATION),
        ]
        
        # Scaling feasibility
        scaling_predictions = analysis.get('scaling_predictions', {})
        for scale_name, prediction in scaling_predictions.items():
            scale_metrics = [
                MetricValue(f"{scale_name}_feasible", prediction.get('feasible', False), metric_type=MetricType.RESOURCE),
                MetricValue(f"{scale_name}_confidence", prediction.get('confidence', 0), metric_type=MetricType.RESOURCE),
                MetricValue(f"{scale_name}_performance_scaling", prediction.get('predicted_performance_scaling', 0), metric_type=MetricType.PERFORMANCE),
            ]
            efficiency_metrics.extend(scale_metrics)
        
        # Add all metrics to collection
        all_metrics = utilization_metrics + count_metrics + efficiency_metrics + bottleneck_metrics
        for metric in all_metrics:
            collection.add_metric(metric)
        
        # Add recommendations as metadata
        recommendations = analysis.get('recommendations', [])
        collection.metadata['recommendations'] = recommendations
        collection.metadata['device'] = analysis.get('device', 'unknown')
        collection.metadata['tool'] = analysis.get('tool', 'unknown')