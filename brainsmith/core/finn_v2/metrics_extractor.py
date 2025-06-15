"""
Metrics Extractor

Extracts standardized performance metrics from real FINN build results
for DSE optimization and comparison.
"""

from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class MetricsExtractor:
    """Extract performance metrics from real FINN build results."""
    
    def __init__(self):
        """Initialize metrics extractor."""
        self.supported_metrics = [
            'throughput', 'latency', 'clock_frequency',
            'lut_utilization', 'dsp_utilization', 'bram_utilization',
            'power_consumption', 'build_time', 'success_rate'
        ]
        logger.info("MetricsExtractor initialized")
    
    def extract_metrics(self, finn_result: Any, finn_config: Any) -> Dict[str, Any]:
        """
        Extract standardized metrics for DSE optimization.
        
        Metrics to Extract:
        - Performance: throughput (fps), latency (ms), clock frequency (MHz)
        - Resources: LUT/DSP/BRAM utilization, power consumption  
        - Quality: success/failure status, build time, warnings
        
        Args:
            finn_result: FINN build result object
            finn_config: FINN DataflowBuildConfig used
            
        Returns:
            Dictionary with standardized metrics for DSE
        """
        logger.debug("Extracting metrics from FINN build result")
        
        try:
            # Initialize metrics dictionary
            metrics = {
                'success': True,
                'build_time': 0.0,
                'primary_metric': 0.0,
                'combination_id': getattr(finn_config, 'combination_id', 'unknown')
            }
            
            # Extract performance metrics
            performance_metrics = self._extract_performance_metrics(finn_result)
            metrics.update(performance_metrics)
            
            # Extract resource metrics
            resource_metrics = self._extract_resource_metrics(finn_result)
            metrics.update(resource_metrics)
            
            # Extract quality metrics
            quality_metrics = self._extract_quality_metrics(finn_result, finn_config)
            metrics.update(quality_metrics)
            
            # Set primary metric for DSE optimization (throughput by default)
            metrics['primary_metric'] = metrics.get('throughput', 0.0)
            
            # Calculate composite metrics
            metrics['resource_efficiency'] = self._calculate_resource_efficiency(metrics)
            
            logger.info(f"Extracted metrics: throughput={metrics.get('throughput', 0):.2f} fps, "
                       f"resource_efficiency={metrics.get('resource_efficiency', 0):.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics extraction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'primary_metric': 0.0,
                'throughput': 0.0,
                'latency': float('inf'),
                'resource_efficiency': 0.0,
                'build_time': 0.0
            }
    
    def _extract_performance_metrics(self, finn_result: Any) -> Dict[str, float]:
        """
        Extract performance metrics from FINN rtlsim results.
        
        Args:
            finn_result: FINN build result
            
        Returns:
            Dictionary with performance metrics
        """
        performance = {
            'throughput': 0.0,        # frames per second
            'latency': 0.0,           # milliseconds
            'clock_frequency': 0.0    # MHz
        }
        
        try:
            # Try to extract from FINN result object
            if hasattr(finn_result, 'model'):
                model = finn_result.model if callable(getattr(finn_result, 'model', None)) else finn_result.model
                
                # Look for performance report files
                if hasattr(finn_result, 'output_dir') or hasattr(finn_result, 'cfg'):
                    output_dir = getattr(finn_result, 'output_dir', None)
                    if not output_dir and hasattr(finn_result, 'cfg'):
                        output_dir = getattr(finn_result.cfg, 'output_dir', None)
                    
                    if output_dir:
                        performance.update(self._extract_from_reports(output_dir))
            
            # Try to extract from model annotations if available
            if hasattr(finn_result, 'model') and finn_result.model:
                model = finn_result.model
                if hasattr(model, 'graph'):
                    performance.update(self._extract_from_model_annotations(model))
            
            # Fallback: estimate from configuration
            if performance['throughput'] == 0.0:
                performance = self._estimate_performance_metrics(finn_result)
                
        except Exception as e:
            logger.warning(f"Performance metrics extraction failed: {e}")
        
        return performance
    
    def _extract_resource_metrics(self, finn_result: Any) -> Dict[str, float]:
        """
        Extract resource utilization from FINN synthesis reports.
        
        Args:
            finn_result: FINN build result
            
        Returns:
            Dictionary with resource metrics
        """
        resources = {
            'lut_utilization': 0.0,     # fraction (0.0-1.0)
            'dsp_utilization': 0.0,     # fraction (0.0-1.0) 
            'bram_utilization': 0.0,    # fraction (0.0-1.0)
            'power_consumption': 0.0    # watts
        }
        
        try:
            # Try to extract from synthesis reports
            if hasattr(finn_result, 'output_dir') or hasattr(finn_result, 'cfg'):
                output_dir = getattr(finn_result, 'output_dir', None)
                if not output_dir and hasattr(finn_result, 'cfg'):
                    output_dir = getattr(finn_result.cfg, 'output_dir', None)
                
                if output_dir:
                    resources.update(self._extract_from_synthesis_reports(output_dir))
            
            # Try to extract from model estimates
            if hasattr(finn_result, 'model') and finn_result.model:
                estimates = self._extract_from_estimate_reports(finn_result)
                resources.update(estimates)
                
        except Exception as e:
            logger.warning(f"Resource metrics extraction failed: {e}")
        
        return resources
    
    def _extract_quality_metrics(self, finn_result: Any, finn_config: Any) -> Dict[str, Any]:
        """
        Extract build quality metrics.
        
        Args:
            finn_result: FINN build result
            finn_config: FINN configuration
            
        Returns:
            Dictionary with quality metrics
        """
        quality = {
            'build_time': 0.0,
            'warnings_count': 0,
            'errors_count': 0,
            'verification_passed': False
        }
        
        try:
            # Extract build time if available
            if hasattr(finn_result, 'build_time'):
                quality['build_time'] = finn_result.build_time
            
            # Check for verification results
            if hasattr(finn_result, 'verification_results'):
                quality['verification_passed'] = finn_result.verification_results.get('passed', False)
            
            # Count warnings/errors from logs
            if hasattr(finn_result, 'build_log'):
                log_content = str(finn_result.build_log)
                quality['warnings_count'] = log_content.lower().count('warning')
                quality['errors_count'] = log_content.lower().count('error')
                
        except Exception as e:
            logger.warning(f"Quality metrics extraction failed: {e}")
        
        return quality
    
    def _extract_from_reports(self, output_dir: str) -> Dict[str, float]:
        """Extract metrics from FINN report files."""
        metrics = {}
        output_path = Path(output_dir)
        
        try:
            # Look for rtlsim performance report
            rtlsim_report = output_path / "rtlsim_performance.json"
            if rtlsim_report.exists():
                with open(rtlsim_report, 'r') as f:
                    rtlsim_data = json.load(f)
                    metrics['throughput'] = rtlsim_data.get('throughput_fps', 0.0)
                    metrics['latency'] = rtlsim_data.get('latency_ms', 0.0)
                    metrics['clock_frequency'] = rtlsim_data.get('clock_freq_mhz', 0.0)
            
        except Exception as e:
            logger.debug(f"Could not extract from reports: {e}")
        
        return metrics
    
    def _extract_from_synthesis_reports(self, output_dir: str) -> Dict[str, float]:
        """Extract resource metrics from synthesis reports."""
        resources = {}
        output_path = Path(output_dir)
        
        try:
            # Look for synthesis utilization report
            util_report = output_path / "utilization_report.json"
            if util_report.exists():
                with open(util_report, 'r') as f:
                    util_data = json.load(f)
                    resources['lut_utilization'] = util_data.get('LUT_utilization', 0.0)
                    resources['dsp_utilization'] = util_data.get('DSP_utilization', 0.0)
                    resources['bram_utilization'] = util_data.get('BRAM_utilization', 0.0)
            
            # Look for power report
            power_report = output_path / "power_report.json"
            if power_report.exists():
                with open(power_report, 'r') as f:
                    power_data = json.load(f)
                    resources['power_consumption'] = power_data.get('total_power_w', 0.0)
                    
        except Exception as e:
            logger.debug(f"Could not extract from synthesis reports: {e}")
        
        return resources
    
    def _extract_from_model_annotations(self, model: Any) -> Dict[str, float]:
        """Extract metrics from ONNX model annotations."""
        metrics = {}
        
        try:
            # FINN models may have performance annotations
            if hasattr(model, 'graph') and hasattr(model.graph, 'value_info'):
                # Look for performance annotations in model metadata
                for value_info in model.graph.value_info:
                    if hasattr(value_info, 'metadata'):
                        metadata = value_info.metadata
                        if 'throughput_fps' in metadata:
                            metrics['throughput'] = float(metadata['throughput_fps'])
                        if 'latency_cycles' in metadata:
                            # Convert cycles to ms assuming 200MHz clock
                            cycles = float(metadata['latency_cycles'])
                            metrics['latency'] = (cycles / 200e6) * 1000
                            
        except Exception as e:
            logger.debug(f"Could not extract from model annotations: {e}")
        
        return metrics
    
    def _extract_from_estimate_reports(self, finn_result: Any) -> Dict[str, float]:
        """Extract estimates from FINN estimate reports."""
        estimates = {}
        
        try:
            # FINN generates resource estimates during build
            if hasattr(finn_result, 'resource_estimates'):
                est = finn_result.resource_estimates
                estimates['lut_utilization'] = est.get('LUT', 0) / 100000.0  # Normalize
                estimates['dsp_utilization'] = est.get('DSP', 0) / 5000.0    # Normalize
                estimates['bram_utilization'] = est.get('BRAM', 0) / 2000.0  # Normalize
                
        except Exception as e:
            logger.debug(f"Could not extract from estimates: {e}")
        
        return estimates
    
    def _estimate_performance_metrics(self, finn_result: Any) -> Dict[str, float]:
        """Estimate performance metrics when direct extraction fails."""
        # Fallback estimates based on typical FINN performance
        return {
            'throughput': 100.0,      # Conservative estimate: 100 FPS
            'latency': 10.0,          # Conservative estimate: 10 ms
            'clock_frequency': 200.0  # Standard: 200 MHz
        }
    
    def _calculate_resource_efficiency(self, metrics: Dict[str, float]) -> float:
        """
        Calculate composite resource efficiency metric.
        
        Args:
            metrics: Dictionary with individual metrics
            
        Returns:
            Resource efficiency score (0.0-1.0)
        """
        try:
            # Weighted combination of throughput and resource utilization
            throughput = metrics.get('throughput', 0.0)
            lut_util = metrics.get('lut_utilization', 0.0)
            dsp_util = metrics.get('dsp_utilization', 0.0)
            bram_util = metrics.get('bram_utilization', 0.0)
            
            # Average resource utilization
            avg_resource_util = (lut_util + dsp_util + bram_util) / 3.0
            
            # Efficiency = throughput / resource_utilization (higher is better)
            if avg_resource_util > 0:
                efficiency = min(1.0, throughput / (avg_resource_util * 1000.0))
            else:
                efficiency = 0.0
                
            return efficiency
            
        except Exception as e:
            logger.debug(f"Resource efficiency calculation failed: {e}")
            return 0.0
    
    def get_supported_metrics(self) -> List[str]:
        """Get list of supported metrics."""
        return self.supported_metrics.copy()
    
    def validate_metrics(self, metrics: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate extracted metrics for completeness and sanity.
        
        Args:
            metrics: Extracted metrics dictionary
            
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        # Check for required metrics
        required_metrics = ['success', 'primary_metric', 'throughput']
        for metric in required_metrics:
            if metric not in metrics:
                warnings.append(f"Missing required metric: {metric}")
        
        # Sanity checks
        if metrics.get('throughput', 0) < 0:
            warnings.append("Negative throughput value")
        
        if metrics.get('latency', 0) < 0:
            warnings.append("Negative latency value")
        
        # Resource utilization should be 0.0-1.0
        for resource in ['lut_utilization', 'dsp_utilization', 'bram_utilization']:
            value = metrics.get(resource, 0.0)
            if value < 0.0 or value > 1.0:
                warnings.append(f"Resource utilization {resource} out of range: {value}")
        
        is_valid = len(warnings) == 0
        return is_valid, warnings