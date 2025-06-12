"""
Essential FINN Interface for BrainSmith Core

Provides clean FINN integration with preparation for 4-hooks transition.
Focuses on practical FINN interfacing without complex abstraction layers.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class FINNHooks:
    """Preparation structure for future 4-hooks FINN interface."""
    
    # Future hook placeholders
    preprocessing_hook: Optional[Any] = None
    transformation_hook: Optional[Any] = None
    optimization_hook: Optional[Any] = None
    generation_hook: Optional[Any] = None
    
    def is_available(self) -> bool:
        """Check if 4-hooks interface is available."""
        return False  # Always False until implemented
    
    def prepare_config(self, design_point: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for future 4-hooks interface."""
        return {
            'preprocessing': design_point.get('preprocessing', {}),
            'transformation': design_point.get('transforms', {}),
            'optimization': design_point.get('hw_optimization', {}),
            'generation': design_point.get('generation', {})
        }


class FINNInterface:
    """Clean FINN integration with 4-hooks preparation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize FINN interface."""
        self.config = config or {}
        self.hooks = FINNHooks()
        
        logger.info("FINNInterface initialized with legacy DataflowBuildConfig support")
    
    def build_accelerator(self, model_path: str, blueprint_config: Dict[str, Any], 
                         output_dir: str = "./output") -> Dict[str, Any]:
        """
        Build FPGA accelerator using FINN.
        
        Args:
            model_path: Path to ONNX model
            blueprint_config: Blueprint configuration
            output_dir: Output directory for results
            
        Returns:
            Build results with performance metrics
        """
        logger.info(f"Building accelerator for model: {model_path}")
        
        try:
            # Create FINN build configuration
            finn_config = self._create_finn_config(blueprint_config, output_dir)
            
            # Execute FINN build
            build_results = self._execute_finn_build(model_path, finn_config)
            
            # Format results
            results = {
                'success': True,
                'output_dir': output_dir,
                'model_path': model_path,
                'rtl_files': build_results.get('rtl_files', []),
                'hls_files': build_results.get('hls_files', []),
                'performance_metrics': self._extract_metrics(build_results),
                'resource_usage': build_results.get('resource_usage', {}),
                'build_config': finn_config
            }
            
            logger.info("FINN build completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"FINN build failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'output_dir': output_dir,
                'model_path': model_path,
                'fallback_results': self._create_fallback_results()
            }
    
    def _create_finn_config(self, blueprint_config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Create FINN DataflowBuildConfig from blueprint."""
        
        # Map blueprint to FINN configuration
        finn_config = {
            # Core FINN parameters
            'output_dir': output_dir,
            'folding_config_file': blueprint_config.get('folding_config'),
            'synth_clk_period_ns': blueprint_config.get('clock_period', 3.33),
            'board': blueprint_config.get('target_device', 'U250'),
            'shell_flow_type': blueprint_config.get('shell_flow', 'vivado_zynq'),
            
            # Build steps configuration
            'steps': [
                'step_qonnx_to_finn',
                'step_tidy_up',
                'step_streamline',
                'step_convert_to_hls',
                'step_create_dataflow_partition',
                'step_target_fps_parallelization',
                'step_apply_folding_config',
                'step_generate_estimate_reports',
                'step_hls_codegen',
                'step_hls_ipgen',
                'step_set_fifo_depths',
                'step_create_stitched_ip',
                'step_synthesize_bitfile'
            ],
            
            # Performance targets
            'target_fps': blueprint_config.get('target_fps', 1000),
            'mvau_wwidth_max': blueprint_config.get('mvau_wwidth_max', 36),
            
            # Additional configuration from blueprint
            **blueprint_config.get('finn_config', {})
        }
        
        return finn_config
    
    def _execute_finn_build(self, model_path: str, finn_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute FINN build process."""
        
        try:
            # Import and use FINN
            from finn.util.fpgadataflow import DataflowBuildConfig
            from finn.builder.build_dataflow import build_dataflow
            
            # Create DataflowBuildConfig
            build_cfg = DataflowBuildConfig(**finn_config)
            
            # Execute build
            build_results = build_dataflow(model=model_path, cfg=build_cfg)
            
            return build_results
        
        except Exception as e:
            logger.error(f"FINN build execution failed: {e}")
            return {
                'error': str(e),
                'rtl_files': [],
                'hls_files': [],
                'performance_metrics': {}
            }
    
    def _extract_metrics(self, build_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance metrics from FINN build results."""
        
        if 'error' in build_results:
            return {'error': 'Build failed'}
        
        return {
            'throughput_fps': build_results.get('throughput_fps', 0),
            'latency_cycles': build_results.get('latency_cycles', 0),
            'clock_frequency_mhz': build_results.get('clock_frequency_mhz', 0),
            'lut_count': build_results.get('lut_count', 0),
            'dsp_count': build_results.get('dsp_count', 0),
            'bram_count': build_results.get('bram_count', 0),
            'estimated_power_w': build_results.get('estimated_power_w', 0)
        }
    
    def _create_mock_results(self, model_path: str, finn_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create mock results when FINN is not available."""
        return {
            'mock_results': True,
            'model_path': model_path,
            'config_summary': str(finn_config),
            'rtl_files': [],
            'hls_files': [],
            'performance_metrics': {
                'throughput_fps': 1000,
                'latency_cycles': 100,
                'clock_frequency_mhz': 300
            },
            'resource_usage': {
                'lut_count': 50000,
                'dsp_count': 1000,
                'bram_count': 200
            }
        }
    
    def _create_fallback_results(self) -> Dict[str, Any]:
        """Create fallback results for error cases."""
        return {
            'fallback': True,
            'message': 'FINN build failed, basic fallback provided',
            'rtl_files': [],
            'hls_files': [],
            'performance_metrics': {}
        }
    
    def validate_config(self, blueprint_config: Dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate blueprint configuration for FINN compatibility."""
        errors = []
        
        # Check required fields
        if 'target_device' not in blueprint_config:
            errors.append("Missing target_device specification")
        
        if 'target_fps' not in blueprint_config:
            errors.append("Missing target_fps specification")
        
        # Validate device support
        supported_devices = ['U250', 'U280', 'ZCU104', 'Alveo-U250', 'Alveo-U280']
        device = blueprint_config.get('target_device', '')
        if device and device not in supported_devices:
            errors.append(f"Unsupported device: {device}")
        
        return len(errors) == 0, errors
    
    def get_supported_devices(self) -> list[str]:
        """Get list of supported FPGA devices."""
        return ['U250', 'U280', 'ZCU104', 'Alveo-U250', 'Alveo-U280']
    
    def prepare_for_4hooks(self, design_point: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for future 4-hooks interface."""
        return self.hooks.prepare_config(design_point)


# Compatibility function for existing API
def create_finn_interface(config: Dict[str, Any] = None) -> FINNInterface:
    """Create FINN interface instance."""
    return FINNInterface(config)