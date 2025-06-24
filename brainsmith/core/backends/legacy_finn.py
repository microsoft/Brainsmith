"""
Legacy FINN Backend Implementation

Implements the legacy FINN interface that expects build steps (collections of transforms)
rather than individual transforms. This backend handles the packing of transforms into
appropriate legacy build steps.
"""

import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import tempfile

from .base import EvaluationBackend, EvaluationRequest, EvaluationResult
from ..finn.legacy_conversion import LegacyConversionLayer
from ..finn.metrics_extractor import MetricsExtractor

logger = logging.getLogger(__name__)


class LegacyFINNBackend(EvaluationBackend):
    """
    Backend for legacy FINN interface using DataflowBuildConfig.
    
    This backend converts modern 6-entrypoint configuration (with transforms)
    to legacy build steps using the LegacyConversionLayer.
    """
    
    def __init__(self, blueprint_config: Dict[str, Any]):
        """
        Initialize legacy FINN backend.
        
        Args:
            blueprint_config: Blueprint V2 configuration
        """
        super().__init__(blueprint_config)
        self.legacy_converter = LegacyConversionLayer()
        self.metrics_extractor = MetricsExtractor()
        
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """
        Execute FINN compilation using legacy DataflowBuildConfig.
        
        Args:
            request: Evaluation request with model and configuration
            
        Returns:
            EvaluationResult with metrics and status
        """
        logger.info(f"Starting legacy FINN evaluation for {request.model_path}")
        
        try:
            # Convert combination to 6-entrypoint config first
            entrypoint_config = self._combination_to_entrypoint_config(request.combination)
            
            # Convert to DataflowBuildConfig via legacy conversion
            dataflow_config = self.legacy_converter.convert_to_dataflow_config(
                entrypoint_config, self.blueprint_config
            )
            
            # Execute in subprocess
            exec_result = self._execute_in_subprocess(
                request.model_path,
                dataflow_config,
                request.work_dir,
                request.timeout
            )
            
            if exec_result['success']:
                # Extract metrics
                metrics = self._extract_metrics(exec_result['output_dir'])
                
                return EvaluationResult(
                    success=True,
                    metrics=metrics,
                    reports=exec_result.get('reports', {}),
                    warnings=exec_result.get('warnings', [])
                )
            else:
                return EvaluationResult(
                    success=False,
                    error=exec_result.get('error', 'Unknown error'),
                    warnings=exec_result.get('warnings', [])
                )
                
        except Exception as e:
            logger.error(f"Legacy FINN evaluation failed: {e}")
            return EvaluationResult(
                success=False,
                error=str(e)
            )
    
    def _combination_to_entrypoint_config(self, combination: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Convert DSE combination to 6-entrypoint configuration.
        
        This matches the logic in FINNEvaluationBridge._combination_to_entrypoint_config()
        """
        # Handle both dict combinations and ComponentCombination objects
        if hasattr(combination, 'canonical_ops'):
            # It's a ComponentCombination object
            entrypoint_config = {
                'entrypoint_1': list(combination.canonical_ops),
                'entrypoint_2': list(combination.model_topology),
                'entrypoint_3': [],  # hw_kernels (kernel names)
                'entrypoint_4': [],  # hw_kernel_specializations  
                'entrypoint_5': list(combination.hw_kernel_transforms),
                'entrypoint_6': list(combination.hw_graph_transforms)
            }
            
            # Process hw_kernels
            for kernel_name, specialization in combination.hw_kernels.items():
                entrypoint_config['entrypoint_3'].append(kernel_name)
                if specialization:
                    entrypoint_config['entrypoint_4'].append(specialization)
        else:
            # It's a dict
            entrypoint_config = {
                'entrypoint_1': combination.get('canonical_ops', []),
                'entrypoint_2': combination.get('model_topology', []),
                'entrypoint_3': combination.get('hw_kernels', []),
                'entrypoint_4': combination.get('hw_kernel_specializations', []),
                'entrypoint_5': combination.get('hw_kernel_transforms', []),
                'entrypoint_6': combination.get('hw_graph_transforms', [])
            }
        
        return entrypoint_config
    
    def _execute_in_subprocess(self, model_path: str, dataflow_config: Any,
                              work_dir: str, timeout: Optional[int]) -> Dict[str, Any]:
        """
        Execute legacy FINN build in subprocess.
        
        Args:
            model_path: Path to ONNX model
            dataflow_config: FINN DataflowBuildConfig object
            work_dir: Working directory
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with execution results
        """
        # Serialize config for subprocess
        config_file = Path(work_dir) / "finn_config.json"
        self._serialize_dataflow_config(dataflow_config, config_file)
        
        # Create execution script
        script_path = Path(work_dir) / "run_legacy_finn.py"
        self._create_execution_script(script_path, model_path, config_file, work_dir)
        
        # Execute in subprocess
        cmd = ["python", str(script_path)]
        logger.info(f"Executing legacy FINN in subprocess: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir
            )
            
            if result.returncode == 0:
                # Parse output
                output_file = Path(work_dir) / "finn_result.json"
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        return json.load(f)
                else:
                    return {
                        'success': False,
                        'error': 'No output file generated',
                        'stdout': result.stdout,
                        'stderr': result.stderr
                    }
            else:
                return {
                    'success': False,
                    'error': f'Process failed with code {result.returncode}',
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Execution timed out after {timeout} seconds'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Subprocess execution failed: {str(e)}'
            }
    
    def _serialize_dataflow_config(self, dataflow_config: Any, config_file: Path):
        """Serialize DataflowBuildConfig for subprocess execution."""
        # Extract configuration parameters
        config_data = {
            'output_dir': dataflow_config.output_dir,
            'synth_clk_period_ns': dataflow_config.synth_clk_period_ns,
            'target_fps': dataflow_config.target_fps,
            'fpga_part': getattr(dataflow_config, 'fpga_part', 'xcu250-figd2104-2L-e'),
            'steps': [step.__name__ for step in dataflow_config.steps],
            # Add other relevant parameters
            'auto_fifo_depths': getattr(dataflow_config, 'auto_fifo_depths', True),
            'save_intermediate_models': getattr(dataflow_config, 'save_intermediate_models', True),
            'folding_config_file': getattr(dataflow_config, 'folding_config_file', None),
            'stop_step': getattr(dataflow_config, 'stop_step', None),
            'start_step': getattr(dataflow_config, 'start_step', None),
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def _create_execution_script(self, script_path: Path, model_path: str,
                               config_file: Path, work_dir: str):
        """Create Python script for subprocess execution."""
        script_content = f'''#!/usr/bin/env python3
"""
Auto-generated script for legacy FINN execution.
"""

import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_legacy_finn():
    """Execute FINN with legacy DataflowBuildConfig."""
    try:
        # Load configuration
        with open("{config_file}", "r") as f:
            config_data = json.load(f)
        
        # Import FINN and step functions
        from finn.builder.build_dataflow import build_dataflow_cfg
        from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType
        from brainsmith.libraries.transforms.steps import *
        from finn.builder.build_dataflow_steps import *
        
        # Map step names back to functions
        step_mapping = {{
            # BrainSmith steps
            'onnx_preprocessing_step': onnx_preprocessing_step,
            'cleanup_step': cleanup_step,
            'cleanup_advanced_step': cleanup_advanced_step,
            'fix_dynamic_dimensions_step': fix_dynamic_dimensions_step,
            'remove_head_step': remove_head_step,
            'remove_tail_step': remove_tail_step,
            'qonnx_to_finn_step': qonnx_to_finn_step,
            'streamlining_step': streamlining_step,
            'infer_hardware_step': infer_hardware_step,
            'generate_reference_io_step': generate_reference_io_step,
            'constrain_folding_and_set_pumped_compute_step': constrain_folding_and_set_pumped_compute_step,
            'shell_metadata_handover_step': shell_metadata_handover_step,
            # FINN steps
            'step_create_dataflow_partition': step_create_dataflow_partition,
            'step_specialize_layers': step_specialize_layers,
            'step_target_fps_parallelization': step_target_fps_parallelization,
            'step_apply_folding_config': step_apply_folding_config,
            'step_minimize_bit_width': step_minimize_bit_width,
            'step_generate_estimate_reports': step_generate_estimate_reports,
            'step_hw_codegen': step_hw_codegen,
            'step_hw_ipgen': step_hw_ipgen,
            'step_set_fifo_depths': step_set_fifo_depths,
            'step_create_stitched_ip': step_create_stitched_ip,
            'step_measure_rtlsim_performance': step_measure_rtlsim_performance
        }}
        
        # Build steps list
        steps = []
        for step_name in config_data['steps']:
            if step_name in step_mapping:
                steps.append(step_mapping[step_name])
            else:
                logger.warning(f"Unknown step: {{step_name}}")
        
        # Create DataflowBuildConfig
        dataflow_config = DataflowBuildConfig(
            steps=steps,
            output_dir=config_data['output_dir'],
            synth_clk_period_ns=config_data['synth_clk_period_ns'],
            target_fps=config_data.get('target_fps'),
            fpga_part=config_data.get('fpga_part', 'xcu250-figd2104-2L-e'),
            auto_fifo_depths=config_data.get('auto_fifo_depths', True),
            save_intermediate_models=config_data.get('save_intermediate_models', True),
            folding_config_file=config_data.get('folding_config_file'),
            stop_step=config_data.get('stop_step'),
            start_step=config_data.get('start_step'),
            generate_outputs=[DataflowOutputType.STITCHED_IP]
        )
        
        # Execute FINN build
        logger.info("Starting legacy FINN build")
        logger.info(f"Model: {model_path}")
        logger.info(f"Steps: {{len(steps)}}")
        
        result = build_dataflow_cfg("{model_path}", dataflow_config)
        
        # Save results
        output_data = {{
            "success": True,
            "output_dir": config_data['output_dir'],
            "warnings": [],
            "reports": {{}}
        }}
        
        with open("{work_dir}/finn_result.json", "w") as f:
            json.dump(output_data, f, indent=2)
            
        logger.info("Legacy FINN build completed successfully")
        
    except Exception as e:
        logger.error(f"Legacy FINN execution failed: {{e}}")
        
        # Save error result
        error_data = {{
            "success": False,
            "error": str(e),
            "warnings": []
        }}
        
        with open("{work_dir}/finn_result.json", "w") as f:
            json.dump(error_data, f, indent=2)

if __name__ == "__main__":
    run_legacy_finn()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        script_path.chmod(0o755)
    
    def _extract_metrics(self, output_dir: str) -> Dict[str, float]:
        """Extract metrics from FINN output directory."""
        # Reuse metrics extraction logic from 6-entrypoint backend
        class FinnResult:
            def __init__(self, output_dir):
                self.output_dir = output_dir
                
        finn_result = FinnResult(output_dir)
        metrics_dict = self.metrics_extractor.extract_metrics(finn_result, None)
        
        # Extract numeric metrics
        return {
            'throughput': metrics_dict.get('throughput', 0.0),
            'latency': metrics_dict.get('latency', 0.0),
            'lut_utilization': metrics_dict.get('lut_utilization', 0.0),
            'dsp_utilization': metrics_dict.get('dsp_utilization', 0.0),
            'bram_utilization': metrics_dict.get('bram_utilization', 0.0),
            'power_consumption': metrics_dict.get('power_consumption', 0.0),
            'resource_efficiency': metrics_dict.get('resource_efficiency', 0.0)
        }
    
    def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate legacy FINN configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check for required blueprint sections
        if 'legacy_preproc' not in self.blueprint_config and 'legacy_postproc' not in self.blueprint_config:
            errors.append("Blueprint must contain 'legacy_preproc' or 'legacy_postproc' for legacy execution")
        
        # Validate build steps if present
        if 'build_steps' in config:
            if not isinstance(config['build_steps'], list):
                errors.append("'build_steps' must be a list")
        
        return errors