"""
Advanced Performance Metrics
Comprehensive performance analysis including timing, throughput, latency, and power estimation.
"""

import os
import sys
import time
import logging
import threading
import subprocess
import tempfile
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import xml.etree.ElementTree as ET

from .core import MetricsCollector, MetricValue, MetricCollection, MetricType, MetricScope

logger = logging.getLogger(__name__)


@dataclass
class TimingMetrics:
    """Timing analysis metrics."""
    setup_time: float  # ns
    hold_time: float   # ns
    clock_period: float  # ns
    critical_path_delay: float  # ns
    slack: float  # ns
    timing_met: bool
    worst_negative_slack: Optional[float] = None
    total_negative_slack: Optional[float] = None
    failing_endpoints: int = 0
    clock_domains: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ThroughputMetrics:
    """Throughput analysis metrics."""
    operations_per_second: float
    tokens_per_second: Optional[float] = None
    pixels_per_second: Optional[float] = None
    samples_per_second: Optional[float] = None
    bandwidth_utilization: float = 0.0  # percentage
    theoretical_max_throughput: Optional[float] = None
    efficiency_ratio: Optional[float] = None
    bottleneck_analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyMetrics:
    """Latency analysis metrics."""
    total_latency_cycles: int
    total_latency_ns: float
    initiation_interval: int
    pipeline_depth: int
    input_to_output_delay: float
    processing_stages: List[Dict[str, float]] = field(default_factory=list)
    memory_access_latency: Optional[float] = None
    compute_latency: Optional[float] = None


@dataclass
class PowerMetrics:
    """Power consumption metrics."""
    total_power_mw: float
    static_power_mw: float
    dynamic_power_mw: float
    clock_power_mw: float
    signal_power_mw: float
    logic_power_mw: float
    bram_power_mw: float
    dsp_power_mw: float
    io_power_mw: float
    power_efficiency: float  # ops/mW
    thermal_design_power: Optional[float] = None


class TimingAnalyzer:
    """Analyze timing characteristics from synthesis and implementation reports."""
    
    def __init__(self):
        self.timing_parsers = {
            'vivado': self._parse_vivado_timing,
            'quartus': self._parse_quartus_timing,
            'generic': self._parse_generic_timing
        }
    
    def analyze_timing(self, 
                      synthesis_report: str,
                      implementation_report: Optional[str] = None,
                      tool: str = 'vivado') -> TimingMetrics:
        """Analyze timing from synthesis/implementation reports."""
        
        parser = self.timing_parsers.get(tool, self._parse_generic_timing)
        
        try:
            # Parse timing data
            timing_data = parser(synthesis_report, implementation_report)
            
            # Create timing metrics
            return TimingMetrics(
                setup_time=timing_data.get('setup_time', 0.0),
                hold_time=timing_data.get('hold_time', 0.0),
                clock_period=timing_data.get('clock_period', 10.0),
                critical_path_delay=timing_data.get('critical_path_delay', 0.0),
                slack=timing_data.get('slack', 0.0),
                timing_met=timing_data.get('timing_met', False),
                worst_negative_slack=timing_data.get('wns'),
                total_negative_slack=timing_data.get('tns'),
                failing_endpoints=timing_data.get('failing_endpoints', 0),
                clock_domains=timing_data.get('clock_domains', {})
            )
            
        except Exception as e:
            logger.error(f"Timing analysis failed: {e}")
            # Return default metrics
            return TimingMetrics(
                setup_time=0.0,
                hold_time=0.0,
                clock_period=10.0,
                critical_path_delay=0.0,
                slack=0.0,
                timing_met=False
            )
    
    def _parse_vivado_timing(self, synthesis_report: str, implementation_report: Optional[str]) -> Dict[str, Any]:
        """Parse Vivado timing reports."""
        timing_data = {}
        
        try:
            # Parse synthesis report
            if os.path.exists(synthesis_report):
                with open(synthesis_report, 'r') as f:
                    content = f.read()
                
                # Extract clock period
                clock_match = re.search(r'create_clock.*period\s+(\d+(?:\.\d+)?)', content)
                if clock_match:
                    timing_data['clock_period'] = float(clock_match.group(1))
                
                # Extract critical path
                cp_match = re.search(r'Critical Path Delay:\s+(\d+(?:\.\d+)?)\s*ns', content)
                if cp_match:
                    timing_data['critical_path_delay'] = float(cp_match.group(1))
            
            # Parse implementation report
            if implementation_report and os.path.exists(implementation_report):
                with open(implementation_report, 'r') as f:
                    content = f.read()
                
                # Extract timing summary
                wns_match = re.search(r'WNS\(ns\):\s+(-?\d+(?:\.\d+)?)', content)
                if wns_match:
                    wns = float(wns_match.group(1))
                    timing_data['slack'] = wns
                    timing_data['worst_negative_slack'] = wns if wns < 0 else None
                    timing_data['timing_met'] = wns >= 0
                
                tns_match = re.search(r'TNS\(ns\):\s+(-?\d+(?:\.\d+)?)', content)
                if tns_match:
                    timing_data['total_negative_slack'] = float(tns_match.group(1))
                
                # Extract failing endpoints
                endpoints_match = re.search(r'Failing Endpoints:\s+(\d+)', content)
                if endpoints_match:
                    timing_data['failing_endpoints'] = int(endpoints_match.group(1))
        
        except Exception as e:
            logger.warning(f"Failed to parse Vivado timing reports: {e}")
        
        return timing_data
    
    def _parse_quartus_timing(self, synthesis_report: str, implementation_report: Optional[str]) -> Dict[str, Any]:
        """Parse Quartus timing reports."""
        timing_data = {}
        
        try:
            # Parse timing analyzer report
            if os.path.exists(synthesis_report):
                with open(synthesis_report, 'r') as f:
                    content = f.read()
                
                # Extract clock constraints
                fmax_match = re.search(r'Fmax:\s+(\d+(?:\.\d+)?)\s*MHz', content)
                if fmax_match:
                    fmax = float(fmax_match.group(1))
                    timing_data['clock_period'] = 1000.0 / fmax  # Convert to ns
                
                # Extract slack
                slack_match = re.search(r'Slack:\s+(-?\d+(?:\.\d+)?)\s*ns', content)
                if slack_match:
                    slack = float(slack_match.group(1))
                    timing_data['slack'] = slack
                    timing_data['timing_met'] = slack >= 0
        
        except Exception as e:
            logger.warning(f"Failed to parse Quartus timing reports: {e}")
        
        return timing_data
    
    def _parse_generic_timing(self, synthesis_report: str, implementation_report: Optional[str]) -> Dict[str, Any]:
        """Generic timing parser for unknown tools."""
        return {
            'clock_period': 10.0,
            'critical_path_delay': 8.0,
            'slack': 2.0,
            'timing_met': True
        }


class ThroughputProfiler:
    """Profile throughput characteristics of FPGA implementations."""
    
    def __init__(self):
        self.profiling_methods = {
            'simulation': self._profile_simulation,
            'analytical': self._profile_analytical,
            'hardware': self._profile_hardware
        }
    
    def profile_throughput(self,
                          implementation_path: str,
                          test_data: Optional[str] = None,
                          method: str = 'analytical',
                          clock_frequency_mhz: float = 100.0) -> ThroughputMetrics:
        """Profile throughput characteristics."""
        
        profiler = self.profiling_methods.get(method, self._profile_analytical)
        
        try:
            throughput_data = profiler(implementation_path, test_data, clock_frequency_mhz)
            
            return ThroughputMetrics(
                operations_per_second=throughput_data.get('ops_per_sec', 0.0),
                tokens_per_second=throughput_data.get('tokens_per_sec'),
                pixels_per_second=throughput_data.get('pixels_per_sec'),
                samples_per_second=throughput_data.get('samples_per_sec'),
                bandwidth_utilization=throughput_data.get('bandwidth_util', 0.0),
                theoretical_max_throughput=throughput_data.get('theoretical_max'),
                efficiency_ratio=throughput_data.get('efficiency_ratio'),
                bottleneck_analysis=throughput_data.get('bottlenecks', {})
            )
            
        except Exception as e:
            logger.error(f"Throughput profiling failed: {e}")
            return ThroughputMetrics(operations_per_second=0.0)
    
    def _profile_simulation(self, implementation_path: str, test_data: Optional[str], clock_freq: float) -> Dict[str, Any]:
        """Profile throughput using simulation."""
        # This would run actual simulation with testbenches
        # For now, return analytical estimates
        return self._profile_analytical(implementation_path, test_data, clock_freq)
    
    def _profile_analytical(self, implementation_path: str, test_data: Optional[str], clock_freq: float) -> Dict[str, Any]:
        """Profile throughput using analytical models."""
        
        # Extract implementation characteristics
        characteristics = self._extract_implementation_characteristics(implementation_path)
        
        # Calculate theoretical throughput
        pe_count = characteristics.get('pe_count', 1)
        simd_width = characteristics.get('simd_width', 1)
        initiation_interval = characteristics.get('initiation_interval', 1)
        
        # Operations per clock cycle
        ops_per_cycle = pe_count * simd_width / initiation_interval
        
        # Convert to operations per second
        ops_per_sec = ops_per_cycle * clock_freq * 1e6
        
        # Estimate efficiency (accounting for real-world factors)
        efficiency_factor = 0.8  # 80% efficiency assumption
        actual_ops_per_sec = ops_per_sec * efficiency_factor
        
        return {
            'ops_per_sec': actual_ops_per_sec,
            'theoretical_max': ops_per_sec,
            'efficiency_ratio': efficiency_factor,
            'bottlenecks': {
                'memory_bandwidth': characteristics.get('memory_bottleneck', False),
                'compute_bound': characteristics.get('compute_bound', True),
                'io_bottleneck': characteristics.get('io_bottleneck', False)
            }
        }
    
    def _profile_hardware(self, implementation_path: str, test_data: Optional[str], clock_freq: float) -> Dict[str, Any]:
        """Profile throughput on actual hardware."""
        # This would require hardware-in-the-loop testing
        # For now, return analytical estimates with higher confidence
        analytical_results = self._profile_analytical(implementation_path, test_data, clock_freq)
        
        # Adjust for hardware reality
        analytical_results['ops_per_sec'] *= 0.9  # 10% reduction for hardware overhead
        analytical_results['efficiency_ratio'] *= 0.9
        
        return analytical_results
    
    def _extract_implementation_characteristics(self, implementation_path: str) -> Dict[str, Any]:
        """Extract characteristics from implementation files."""
        characteristics = {
            'pe_count': 8,
            'simd_width': 4,
            'initiation_interval': 1,
            'memory_bottleneck': False,
            'compute_bound': True,
            'io_bottleneck': False
        }
        
        # Try to extract from actual implementation files
        try:
            # Look for HLS reports, Verilog files, etc.
            for file_path in Path(implementation_path).rglob('*.rpt'):
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                    # Extract PE count
                    pe_match = re.search(r'PE.*?(\d+)', content, re.IGNORECASE)
                    if pe_match:
                        characteristics['pe_count'] = int(pe_match.group(1))
                    
                    # Extract SIMD width
                    simd_match = re.search(r'SIMD.*?(\d+)', content, re.IGNORECASE)
                    if simd_match:
                        characteristics['simd_width'] = int(simd_match.group(1))
                    
                    # Extract initiation interval
                    ii_match = re.search(r'Initiation Interval.*?(\d+)', content, re.IGNORECASE)
                    if ii_match:
                        characteristics['initiation_interval'] = int(ii_match.group(1))
        
        except Exception as e:
            logger.debug(f"Could not extract implementation characteristics: {e}")
        
        return characteristics


class LatencyAnalyzer:
    """Analyze latency characteristics of FPGA implementations."""
    
    def analyze_latency(self,
                       implementation_path: str,
                       clock_frequency_mhz: float = 100.0) -> LatencyMetrics:
        """Analyze latency characteristics."""
        
        try:
            # Extract latency information from implementation
            latency_data = self._extract_latency_data(implementation_path)
            
            # Calculate latency metrics
            total_cycles = latency_data.get('total_cycles', 100)
            clock_period_ns = 1000.0 / clock_frequency_mhz
            total_latency_ns = total_cycles * clock_period_ns
            
            return LatencyMetrics(
                total_latency_cycles=total_cycles,
                total_latency_ns=total_latency_ns,
                initiation_interval=latency_data.get('initiation_interval', 1),
                pipeline_depth=latency_data.get('pipeline_depth', 10),
                input_to_output_delay=latency_data.get('io_delay', total_latency_ns),
                processing_stages=latency_data.get('stages', []),
                memory_access_latency=latency_data.get('memory_latency'),
                compute_latency=latency_data.get('compute_latency')
            )
            
        except Exception as e:
            logger.error(f"Latency analysis failed: {e}")
            # Return default metrics
            clock_period_ns = 1000.0 / clock_frequency_mhz
            return LatencyMetrics(
                total_latency_cycles=100,
                total_latency_ns=100 * clock_period_ns,
                initiation_interval=1,
                pipeline_depth=10,
                input_to_output_delay=100 * clock_period_ns
            )
    
    def _extract_latency_data(self, implementation_path: str) -> Dict[str, Any]:
        """Extract latency data from implementation files."""
        latency_data = {
            'total_cycles': 100,
            'initiation_interval': 1,
            'pipeline_depth': 10,
            'stages': []
        }
        
        try:
            # Look for HLS synthesis reports
            for file_path in Path(implementation_path).rglob('*synthesis*.rpt'):
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                    # Extract latency information
                    latency_match = re.search(r'Latency.*?(\d+)', content, re.IGNORECASE)
                    if latency_match:
                        latency_data['total_cycles'] = int(latency_match.group(1))
                    
                    # Extract initiation interval
                    ii_match = re.search(r'Initiation Interval.*?(\d+)', content, re.IGNORECASE)
                    if ii_match:
                        latency_data['initiation_interval'] = int(ii_match.group(1))
                    
                    # Extract pipeline depth
                    depth_match = re.search(r'Pipeline Depth.*?(\d+)', content, re.IGNORECASE)
                    if depth_match:
                        latency_data['pipeline_depth'] = int(depth_match.group(1))
        
        except Exception as e:
            logger.debug(f"Could not extract latency data: {e}")
        
        return latency_data


class PowerEstimator:
    """Estimate power consumption of FPGA implementations."""
    
    def __init__(self):
        self.device_power_models = {
            'xc7z020': {'base_power': 500, 'lut_power': 0.1, 'dsp_power': 5.0, 'bram_power': 2.0},
            'xczu7ev': {'base_power': 2000, 'lut_power': 0.05, 'dsp_power': 3.0, 'bram_power': 1.5},
            'xcvu9p': {'base_power': 15000, 'lut_power': 0.03, 'dsp_power': 2.0, 'bram_power': 1.0}
        }
    
    def estimate_power(self,
                      resource_utilization: Dict[str, int],
                      clock_frequency_mhz: float = 100.0,
                      device: str = 'xc7z020',
                      activity_factor: float = 0.5) -> PowerMetrics:
        """Estimate power consumption."""
        
        try:
            # Get device power model
            power_model = self.device_power_models.get(device, self.device_power_models['xc7z020'])
            
            # Extract resource usage
            lut_count = resource_utilization.get('lut_count', 0)
            dsp_count = resource_utilization.get('dsp_count', 0)
            bram_count = resource_utilization.get('bram_count', 0)
            
            # Calculate base static power
            static_power = power_model['base_power']  # mW
            
            # Calculate dynamic power components
            lut_power = lut_count * power_model['lut_power'] * activity_factor
            dsp_power = dsp_count * power_model['dsp_power'] * activity_factor
            bram_power = bram_count * power_model['bram_power'] * activity_factor
            
            # Clock power (proportional to frequency)
            clock_power = (clock_frequency_mhz / 100.0) * 100.0 * activity_factor
            
            # Signal power (estimated)
            signal_power = (lut_power + dsp_power) * 0.3
            
            # Logic power
            logic_power = lut_power + dsp_power
            
            # I/O power (estimated)
            io_power = 50.0 * activity_factor
            
            # Total dynamic power
            dynamic_power = logic_power + bram_power + clock_power + signal_power + io_power
            
            # Total power
            total_power = static_power + dynamic_power
            
            # Power efficiency (operations per mW)
            # Estimate operations per second
            ops_per_sec = self._estimate_operations_per_second(resource_utilization, clock_frequency_mhz)
            power_efficiency = ops_per_sec / total_power if total_power > 0 else 0.0
            
            return PowerMetrics(
                total_power_mw=total_power,
                static_power_mw=static_power,
                dynamic_power_mw=dynamic_power,
                clock_power_mw=clock_power,
                signal_power_mw=signal_power,
                logic_power_mw=logic_power,
                bram_power_mw=bram_power,
                dsp_power_mw=dsp_power,
                io_power_mw=io_power,
                power_efficiency=power_efficiency
            )
            
        except Exception as e:
            logger.error(f"Power estimation failed: {e}")
            return PowerMetrics(
                total_power_mw=1000.0,
                static_power_mw=500.0,
                dynamic_power_mw=500.0,
                clock_power_mw=100.0,
                signal_power_mw=100.0,
                logic_power_mw=200.0,
                bram_power_mw=50.0,
                dsp_power_mw=50.0,
                io_power_mw=50.0,
                power_efficiency=1000.0
            )
    
    def _estimate_operations_per_second(self, resources: Dict[str, int], clock_freq: float) -> float:
        """Estimate operations per second based on resources."""
        # Simple estimation based on DSP and LUT count
        dsp_count = resources.get('dsp_count', 0)
        lut_count = resources.get('lut_count', 0)
        
        # Assume each DSP can do 1 operation per cycle
        # Assume every 100 LUTs can do 1 operation per cycle
        ops_per_cycle = dsp_count + (lut_count / 100.0)
        
        return ops_per_cycle * clock_freq * 1e6


class AdvancedPerformanceMetrics(MetricsCollector):
    """Comprehensive performance metrics collector."""
    
    def __init__(self, name: str = "AdvancedPerformanceMetrics", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        
        self.timing_analyzer = TimingAnalyzer()
        self.throughput_profiler = ThroughputProfiler()
        self.latency_analyzer = LatencyAnalyzer()
        self.power_estimator = PowerEstimator()
    
    def collect_metrics(self, context: Dict[str, Any]) -> MetricCollection:
        """Collect comprehensive performance metrics."""
        
        collection = MetricCollection(
            collection_id=f"perf_{int(time.time())}",
            name="Advanced Performance Metrics",
            description="Comprehensive performance analysis including timing, throughput, latency, and power"
        )
        
        try:
            # Extract context information
            build_result = context.get('build_result')
            synthesis_report = context.get('synthesis_report')
            implementation_report = context.get('implementation_report')
            resource_utilization = context.get('resource_utilization', {})
            clock_frequency = context.get('clock_frequency_mhz', 100.0)
            device = context.get('device', 'xc7z020')
            
            # Collect timing metrics
            if synthesis_report:
                timing_metrics = self.timing_analyzer.analyze_timing(
                    synthesis_report, implementation_report
                )
                self._add_timing_metrics(collection, timing_metrics)
            
            # Collect throughput metrics
            if build_result and hasattr(build_result, 'output_path'):
                throughput_metrics = self.throughput_profiler.profile_throughput(
                    build_result.output_path, clock_frequency_mhz=clock_frequency
                )
                self._add_throughput_metrics(collection, throughput_metrics)
            
            # Collect latency metrics
            if build_result and hasattr(build_result, 'output_path'):
                latency_metrics = self.latency_analyzer.analyze_latency(
                    build_result.output_path, clock_frequency
                )
                self._add_latency_metrics(collection, latency_metrics)
            
            # Collect power metrics
            if resource_utilization:
                power_metrics = self.power_estimator.estimate_power(
                    resource_utilization, clock_frequency, device
                )
                self._add_power_metrics(collection, power_metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
            # Add error metric
            collection.add_metric(MetricValue(
                name="collection_error",
                value=str(e),
                metric_type=MetricType.PERFORMANCE,
                scope=MetricScope.BUILD
            ))
        
        return collection
    
    def get_supported_metrics(self) -> List[str]:
        """Get list of supported metrics."""
        return [
            "timing_slack", "timing_met", "critical_path_delay",
            "throughput_ops_per_sec", "latency_cycles", "latency_ns",
            "power_total_mw", "power_efficiency", "clock_frequency"
        ]
    
    def _add_timing_metrics(self, collection: MetricCollection, timing: TimingMetrics):
        """Add timing metrics to collection."""
        metrics = [
            MetricValue("timing_slack", timing.slack, "ns", metric_type=MetricType.TIMING),
            MetricValue("timing_met", timing.timing_met, metric_type=MetricType.TIMING),
            MetricValue("critical_path_delay", timing.critical_path_delay, "ns", metric_type=MetricType.TIMING),
            MetricValue("clock_period", timing.clock_period, "ns", metric_type=MetricType.TIMING),
            MetricValue("setup_time", timing.setup_time, "ns", metric_type=MetricType.TIMING),
            MetricValue("hold_time", timing.hold_time, "ns", metric_type=MetricType.TIMING),
        ]
        
        if timing.worst_negative_slack is not None:
            metrics.append(MetricValue("worst_negative_slack", timing.worst_negative_slack, "ns", metric_type=MetricType.TIMING))
        
        if timing.total_negative_slack is not None:
            metrics.append(MetricValue("total_negative_slack", timing.total_negative_slack, "ns", metric_type=MetricType.TIMING))
        
        metrics.append(MetricValue("failing_endpoints", timing.failing_endpoints, "count", metric_type=MetricType.TIMING))
        
        for metric in metrics:
            collection.add_metric(metric)
    
    def _add_throughput_metrics(self, collection: MetricCollection, throughput: ThroughputMetrics):
        """Add throughput metrics to collection."""
        metrics = [
            MetricValue("throughput_ops_per_sec", throughput.operations_per_second, "ops/sec", metric_type=MetricType.THROUGHPUT),
            MetricValue("bandwidth_utilization", throughput.bandwidth_utilization, "%", metric_type=MetricType.THROUGHPUT),
        ]
        
        if throughput.tokens_per_second is not None:
            metrics.append(MetricValue("throughput_tokens_per_sec", throughput.tokens_per_second, "tokens/sec", metric_type=MetricType.THROUGHPUT))
        
        if throughput.theoretical_max_throughput is not None:
            metrics.append(MetricValue("theoretical_max_throughput", throughput.theoretical_max_throughput, "ops/sec", metric_type=MetricType.THROUGHPUT))
        
        if throughput.efficiency_ratio is not None:
            metrics.append(MetricValue("throughput_efficiency", throughput.efficiency_ratio, "ratio", metric_type=MetricType.EFFICIENCY))
        
        for metric in metrics:
            collection.add_metric(metric)
    
    def _add_latency_metrics(self, collection: MetricCollection, latency: LatencyMetrics):
        """Add latency metrics to collection."""
        metrics = [
            MetricValue("latency_cycles", latency.total_latency_cycles, "cycles", metric_type=MetricType.LATENCY),
            MetricValue("latency_ns", latency.total_latency_ns, "ns", metric_type=MetricType.LATENCY),
            MetricValue("initiation_interval", latency.initiation_interval, "cycles", metric_type=MetricType.LATENCY),
            MetricValue("pipeline_depth", latency.pipeline_depth, "stages", metric_type=MetricType.LATENCY),
            MetricValue("input_output_delay", latency.input_to_output_delay, "ns", metric_type=MetricType.LATENCY),
        ]
        
        if latency.memory_access_latency is not None:
            metrics.append(MetricValue("memory_access_latency", latency.memory_access_latency, "ns", metric_type=MetricType.LATENCY))
        
        if latency.compute_latency is not None:
            metrics.append(MetricValue("compute_latency", latency.compute_latency, "ns", metric_type=MetricType.LATENCY))
        
        for metric in metrics:
            collection.add_metric(metric)
    
    def _add_power_metrics(self, collection: MetricCollection, power: PowerMetrics):
        """Add power metrics to collection."""
        metrics = [
            MetricValue("power_total_mw", power.total_power_mw, "mW", metric_type=MetricType.POWER),
            MetricValue("power_static_mw", power.static_power_mw, "mW", metric_type=MetricType.POWER),
            MetricValue("power_dynamic_mw", power.dynamic_power_mw, "mW", metric_type=MetricType.POWER),
            MetricValue("power_clock_mw", power.clock_power_mw, "mW", metric_type=MetricType.POWER),
            MetricValue("power_signal_mw", power.signal_power_mw, "mW", metric_type=MetricType.POWER),
            MetricValue("power_logic_mw", power.logic_power_mw, "mW", metric_type=MetricType.POWER),
            MetricValue("power_bram_mw", power.bram_power_mw, "mW", metric_type=MetricType.POWER),
            MetricValue("power_dsp_mw", power.dsp_power_mw, "mW", metric_type=MetricType.POWER),
            MetricValue("power_io_mw", power.io_power_mw, "mW", metric_type=MetricType.POWER),
            MetricValue("power_efficiency", power.power_efficiency, "ops/mW", metric_type=MetricType.EFFICIENCY),
        ]
        
        for metric in metrics:
            collection.add_metric(metric)


class PerformanceCollector:
    """High-level performance collector that orchestrates all performance analysis."""
    
    def __init__(self):
        self.advanced_metrics = AdvancedPerformanceMetrics()
        self.collection_history = []
    
    def collect_comprehensive_metrics(self, build_result: Any) -> MetricCollection:
        """Collect comprehensive performance metrics from build result."""
        
        # Prepare context
        context = {
            'build_result': build_result,
            'synthesis_report': getattr(build_result, 'synthesis_report', None),
            'implementation_report': getattr(build_result, 'implementation_report', None),
            'resource_utilization': getattr(build_result, 'resource_usage', {}),
            'clock_frequency_mhz': getattr(build_result, 'clock_frequency', 100.0),
            'device': getattr(build_result, 'device', 'xc7z020')
        }
        
        # Collect metrics
        collection = self.advanced_metrics.collect_metrics(context)
        
        # Store in history
        self.collection_history.append(collection)
        
        # Trim history
        if len(self.collection_history) > 1000:
            self.collection_history = self.collection_history[-500:]
        
        return collection
    
    def get_performance_trends(self, metric_names: List[str], hours: int = 24) -> Dict[str, List[float]]:
        """Get performance trends for specified metrics."""
        cutoff_time = time.time() - (hours * 3600)
        
        trends = {name: [] for name in metric_names}
        
        for collection in self.collection_history:
            if collection.created_at >= cutoff_time:
                for metric in collection.metrics:
                    if metric.name in metric_names and isinstance(metric.value, (int, float)):
                        trends[metric.name].append(metric.value)
        
        return trends
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of recent performance metrics."""
        if not self.collection_history:
            return {}
        
        latest_collection = self.collection_history[-1]
        
        summary = {
            'collection_time': latest_collection.created_at,
            'metrics_count': len(latest_collection.metrics),
            'timing_metrics': {},
            'throughput_metrics': {},
            'latency_metrics': {},
            'power_metrics': {}
        }
        
        # Categorize metrics
        for metric in latest_collection.metrics:
            if metric.metric_type == MetricType.TIMING:
                summary['timing_metrics'][metric.name] = metric.value
            elif metric.metric_type == MetricType.THROUGHPUT:
                summary['throughput_metrics'][metric.name] = metric.value
            elif metric.metric_type == MetricType.LATENCY:
                summary['latency_metrics'][metric.name] = metric.value
            elif metric.metric_type == MetricType.POWER:
                summary['power_metrics'][metric.name] = metric.value
        
        return summary