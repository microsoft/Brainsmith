"""
Model profiling and roofline analysis tools.

This module provides tools for analyzing model performance characteristics
and generating roofline analysis reports. These tools are supplementary
to the core BrainSmith toolchain and provide additional insights.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Import existing roofline functionality
from .roofline import roofline_analysis as _roofline_analysis
from .model_profiling import RooflineModel


class RooflineProfiler:
    """High-level interface for roofline analysis."""
    
    def __init__(self):
        self.model = RooflineModel()
    
    def profile_model(self, model_config: Dict[str, Any], hardware_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Profile model and generate roofline analysis.
        
        Args:
            model_config: Model configuration dictionary containing:
                - arch: Model architecture ('bert', 'slm_pp', 'slm_tg', 'twin_bert')
                - Model-specific parameters (num_layers, seq_len, etc.)
            hardware_config: Hardware configuration dictionary containing:
                - dsps: Number of DSPs available
                - luts: Number of LUTs available  
                - Resource utilization factors
                - Clock frequencies
                
        Returns:
            Dictionary with roofline analysis results for different data types
        """
        logger.info(f"Profiling model architecture: {model_config.get('arch', 'unknown')}")
        
        # Set up model configuration
        if model_config['arch'] == 'bert':
            self.model.profile_bert(model=model_config)
        elif model_config['arch'] == 'slm_pp':
            self.model.profile_slm_pp(model=model_config)
        elif model_config['arch'] == 'slm_tg':
            self.model.profile_slm_tg(model=model_config)
        elif model_config['arch'] == 'twin_bert':
            # Handle twin BERT profiling
            self.model.profile_bert(model=model_config['model_1'])
            profile_1 = self.model.get_profile()
            self.model.reset_pipeline()
            self.model.profile_bert(model=model_config['model_2'])
            profile_2 = self.model.get_profile()
            # Combine profiles for twin BERT
            combined_profile = self._combine_twin_bert_profiles(profile_1, profile_2)
            return self._analyze_combined_profile(combined_profile, hardware_config, model_config)
        else:
            raise ValueError(f"Unsupported model architecture: {model_config['arch']}")
        
        profile = self.model.get_profile()
        
        # Generate analysis for different data types
        results = {}
        dtypes = [4, 8]  # Default data types to analyze
        if 'dtypes' in model_config:
            dtypes = model_config['dtypes']
            
        for dtype in dtypes:
            results[f'{dtype}bit'] = self._analyze_dtype(profile, hardware_config, model_config, dtype)
        
        logger.info(f"Profiling completed for {len(dtypes)} data types")
        return results
    
    def _combine_twin_bert_profiles(self, profile_1: Dict, profile_2: Dict) -> Dict:
        """Combine profiles from twin BERT models."""
        return {
            'act': profile_1['act'] + profile_2['act'],
            'macs': profile_1['macs'] + profile_2['macs'],
            'w': profile_1['w'] + profile_2['w'],
            'vw': profile_1['vw'] + profile_2['vw'],
            'hbm': profile_1['hbm'] + profile_2['hbm'],
            'cycles': 1  # Offload not supported for twin BERT currently
        }
    
    def _analyze_combined_profile(self, profile: Dict, hw_config: Dict, model_config: Dict) -> Dict:
        """Analyze combined profile for twin BERT."""
        results = {}
        dtypes = [4, 8]
        for dtype in dtypes:
            results[f'{dtype}bit'] = self._analyze_dtype(profile, hw_config, model_config, dtype)
        return results
    
    def _analyze_dtype(self, profile: Dict, hw_config: Dict, model_config: Dict, dtype: int) -> Dict:
        """Analyze profile for specific data type."""
        # This is a simplified analysis - the full implementation would use
        # the complex roofline analysis logic from the existing roofline.py
        
        # Calculate basic metrics
        total_compute = sum(sum(segment) for segment in profile['macs'])
        total_weights = sum(sum(segment) for segment in profile['w'])
        total_activations = sum(sum(segment) for segment in profile['act'])
        
        # Estimate throughput and latency based on hardware config
        available_dsps = int(hw_config['dsps'] * hw_config.get('dsp_util', 0.9))
        dsp_frequency = hw_config.get('dsp_hz', 500e6)
        
        # Simple throughput estimation
        macs_per_dsp = {4: 4, 8: 2}.get(dtype, 1)
        max_throughput = available_dsps * dsp_frequency * macs_per_dsp
        estimated_latency = total_compute / max_throughput
        
        batch_size = model_config.get('batch', 1)
        ips = batch_size / estimated_latency if estimated_latency > 0 else 0
        
        return {
            'throughput_ips': ips,
            'latency_ms': estimated_latency * 1000,
            'total_compute_ops': total_compute,
            'total_weights_bytes': total_weights * dtype,
            'total_activations_bytes': total_activations * dtype,
            'dtype': dtype,
            'hardware_utilization': {
                'dsp_utilization': min(1.0, total_compute / max_throughput),
                'estimated_frequency_mhz': dsp_frequency / 1e6
            }
        }
    
    def generate_report(self, profile_results: Dict[str, Any], output_path: str = None) -> str:
        """
        Generate roofline analysis report.
        
        Args:
            profile_results: Results from profile_model()
            output_path: Optional path to save HTML report
            
        Returns:
            HTML report string
        """
        html_report = self._generate_html_report(profile_results)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html_report)
            logger.info(f"Roofline report saved to: {output_path}")
        
        return html_report
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report from profiling results."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>BrainSmith Roofline Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric { margin: 10px 0; }
                .dtype-section { border: 1px solid #ddd; padding: 15px; margin: 10px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>BrainSmith Roofline Analysis Report</h1>
        """
        
        for dtype_key, data in results.items():
            html += f"""
            <div class="dtype-section">
                <h2>{dtype_key} Analysis</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Throughput (IPS)</td><td>{data.get('throughput_ips', 0):.2f}</td></tr>
                    <tr><td>Latency (ms)</td><td>{data.get('latency_ms', 0):.4f}</td></tr>
                    <tr><td>Total Compute Ops</td><td>{data.get('total_compute_ops', 0):,}</td></tr>
                    <tr><td>Total Weights (bytes)</td><td>{data.get('total_weights_bytes', 0):,}</td></tr>
                    <tr><td>Total Activations (bytes)</td><td>{data.get('total_activations_bytes', 0):,}</td></tr>
                </table>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html


def roofline_analysis(model_config: Dict, hw_config: Dict, dtypes: List[int]) -> Dict[str, Any]:
    """
    Wrapper for existing roofline analysis functionality.
    
    Args:
        model_config: Model configuration dictionary
        hw_config: Hardware configuration dictionary  
        dtypes: List of data type bit widths to analyze
        
    Returns:
        Dictionary with roofline analysis results for each data type
    """
    logger.info(f"Running roofline analysis for model: {model_config.get('arch', 'unknown')}")
    
    try:
        # Call the existing roofline analysis function
        _roofline_analysis(model_config, hw_config, dtypes)
        
        # Since the original function prints results instead of returning them,
        # we provide a structured fallback response
        return {
            'status': 'completed',
            'model_arch': model_config.get('arch', 'unknown'),
            'dtypes_analyzed': dtypes,
            'note': 'Results printed to console - use RooflineProfiler class for structured results'
        }
        
    except Exception as e:
        logger.error(f"Roofline analysis failed: {e}")
        return {
            'error': str(e),
            'status': 'failed'
        }


__all__ = [
    'roofline_analysis',
    'RooflineProfiler'
]