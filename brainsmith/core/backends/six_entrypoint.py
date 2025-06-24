"""
Six-Entrypoint Backend Implementation

Implements the modern 6-entrypoint FINN interface where transforms are inserted
at specific points in the compilation flow, not as pre-packaged build steps.
"""

import logging
import subprocess
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base import EvaluationBackend, EvaluationRequest, EvaluationResult
from ..finn.metrics_extractor import MetricsExtractor

logger = logging.getLogger(__name__)


@dataclass
class EntrypointConfig:
    """Configuration for 6-entrypoint execution."""
    entrypoint_1: List[str]  # After model loading
    entrypoint_2: List[str]  # After quantization setup
    entrypoint_3: List[str]  # After hardware kernel mapping
    entrypoint_4: List[str]  # After kernel specialization
    entrypoint_5: List[str]  # After kernel-level optimizations
    entrypoint_6: List[str]  # After graph-level optimizations


class SixEntrypointBackend(EvaluationBackend):
    """
    Backend for modern 6-entrypoint FINN interface.
    
    This backend directly uses FINN's 6-entrypoint API where transforms
    are inserted at specific points in the compilation flow.
    """
    
    def __init__(self, blueprint_config: Dict[str, Any]):
        """
        Initialize six-entrypoint backend.
        
        Args:
            blueprint_config: Blueprint V2 configuration
        """
        super().__init__(blueprint_config)
        self.metrics_extractor = MetricsExtractor()
        self._validate_transforms()
        
    def _validate_transforms(self):
        """Validate that required transforms are available."""
        # Check if FINN transforms are accessible
        try:
            import finn.transformation.general as finn_general
            import finn.transformation.streamline as finn_streamline
            import finn.transformation.fpgadataflow as finn_fpga
            logger.info("FINN transformation modules available")
        except ImportError as e:
            logger.warning(f"Some FINN transformations may not be available: {e}")
    
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """
        Execute FINN compilation using 6-entrypoint interface.
        
        Args:
            request: Evaluation request with model and configuration
            
        Returns:
            EvaluationResult with metrics and status
        """
        logger.info(f"Starting 6-entrypoint evaluation for {request.model_path}")
        
        try:
            # Extract entrypoint configuration
            entrypoint_config = self._extract_entrypoint_config(request.combination)
            
            # Prepare subprocess execution
            exec_result = self._execute_in_subprocess(
                request.model_path,
                entrypoint_config,
                request.work_dir,
                request.timeout
            )
            
            if exec_result['success']:
                # Extract metrics from FINN outputs
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
            logger.error(f"6-entrypoint evaluation failed: {e}")
            return EvaluationResult(
                success=False,
                error=str(e)
            )
    
    def _extract_entrypoint_config(self, combination: Dict[str, Any]) -> EntrypointConfig:
        """
        Extract 6-entrypoint configuration from DSE combination.
        
        Maps combination attributes to transform lists for each entrypoint.
        """
        # Direct mapping from combination to entrypoints
        config = EntrypointConfig(
            entrypoint_1=combination.get('entrypoint_1', []),
            entrypoint_2=combination.get('entrypoint_2', []),
            entrypoint_3=combination.get('entrypoint_3', []),
            entrypoint_4=combination.get('entrypoint_4', []),
            entrypoint_5=combination.get('entrypoint_5', []),
            entrypoint_6=combination.get('entrypoint_6', [])
        )
        
        # If combination uses old attribute names, map them
        if 'canonical_ops' in combination and not config.entrypoint_1:
            config.entrypoint_1 = combination['canonical_ops']
        if 'model_topology' in combination and not config.entrypoint_2:
            config.entrypoint_2 = combination['model_topology']
        if 'hw_kernels' in combination and not config.entrypoint_3:
            # Extract kernel names only
            config.entrypoint_3 = list(combination['hw_kernels'].keys())
        if 'hw_kernel_transforms' in combination and not config.entrypoint_5:
            config.entrypoint_5 = combination['hw_kernel_transforms']
        if 'hw_graph_transforms' in combination and not config.entrypoint_6:
            config.entrypoint_6 = combination['hw_graph_transforms']
            
        return config
    
    def _execute_in_subprocess(self, model_path: str, entrypoint_config: EntrypointConfig,
                              work_dir: str, timeout: Optional[int]) -> Dict[str, Any]:
        """
        Execute FINN 6-entrypoint compilation in subprocess.
        
        Args:
            model_path: Path to ONNX model
            entrypoint_config: 6-entrypoint configuration
            work_dir: Working directory for execution
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with execution results
        """
        # Create execution script
        script_path = Path(work_dir) / "run_six_entrypoint.py"
        self._create_execution_script(script_path, model_path, entrypoint_config, work_dir)
        
        # Execute in subprocess
        cmd = ["python", str(script_path)]
        logger.info(f"Executing 6-entrypoint FINN in subprocess: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir
            )
            
            if result.returncode == 0:
                # Parse output JSON
                output_file = Path(work_dir) / "finn_result.json"
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        return json.load(f)
                else:
                    return {
                        'success': False,
                        'error': 'No output file generated'
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
    
    def _create_execution_script(self, script_path: Path, model_path: str,
                                entrypoint_config: EntrypointConfig, work_dir: str):
        """Create Python script for subprocess execution."""
        script_content = f'''#!/usr/bin/env python3
"""
Auto-generated script for 6-entrypoint FINN execution.
"""

import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_six_entrypoint():
    """Execute FINN with 6-entrypoint interface."""
    try:
        # Import FINN components
        from finn.builder.build_dataflow_v2 import build_dataflow_v2
        from finn.transformation.general import *
        from finn.transformation.streamline import *
        from finn.transformation.fpgadataflow import *
        
        # Model path
        model_path = "{model_path}"
        
        # Create transform lists for each entrypoint
        transforms_ep1 = {json.dumps(entrypoint_config.entrypoint_1)}
        transforms_ep2 = {json.dumps(entrypoint_config.entrypoint_2)}
        transforms_ep3 = {json.dumps(entrypoint_config.entrypoint_3)}
        transforms_ep4 = {json.dumps(entrypoint_config.entrypoint_4)}
        transforms_ep5 = {json.dumps(entrypoint_config.entrypoint_5)}
        transforms_ep6 = {json.dumps(entrypoint_config.entrypoint_6)}
        
        # Map transform names to actual transform objects
        transform_mapping = {{
            # General transforms
            "FoldConstants": FoldConstants(),
            "InferShapes": InferShapes(),
            "RemoveUnusedNodes": RemoveUnusedNodes(),
            "RemoveStaticGraphInputs": RemoveStaticGraphInputs(),
            "InferDataTypes": InferDataTypes(),
            
            # Streamlining transforms
            "Streamline": Streamline(),
            "ConvertBipolarToXnor": ConvertBipolarToXnor(),
            "MoveLinearPastFork": MoveLinearPastFork(),
            "AbsorbSignBiasIntoMultiThreshold": AbsorbSignBiasIntoMultiThreshold(),
            
            # FPGA dataflow transforms
            "AnnotateCycles": AnnotateCycles(),
            "SetFolding": SetFolding(),
            "MinimizeAccumulatorWidth": MinimizeAccumulatorWidth(),
            "MinimizeWeightBitWidth": MinimizeWeightBitWidth(),
        }}
        
        # Convert string names to transform objects
        def get_transforms(transform_names):
            transforms = []
            for name in transform_names:
                if name in transform_mapping:
                    transforms.append(transform_mapping[name])
                else:
                    logger.warning(f"Unknown transform: {{name}}")
            return transforms
        
        # Build entrypoint configuration
        entrypoint_transforms = {{
            1: get_transforms(transforms_ep1),
            2: get_transforms(transforms_ep2),
            3: get_transforms(transforms_ep3),
            4: get_transforms(transforms_ep4),
            5: get_transforms(transforms_ep5),
            6: get_transforms(transforms_ep6),
        }}
        
        # Execute FINN v2 build
        logger.info("Starting FINN 6-entrypoint build")
        result = build_dataflow_v2(
            model_path,
            entrypoint_transforms,
            output_dir="{work_dir}/finn_output",
            synth_clk_period_ns=5.0,
            target_fps=None,
            fpga_part="xcu250-figd2104-2L-e",
            folding_config_file=None,
        )
        
        # Save results
        output_data = {{
            "success": True,
            "output_dir": "{work_dir}/finn_output",
            "warnings": [],
            "reports": {{}}
        }}
        
        with open("{work_dir}/finn_result.json", "w") as f:
            json.dump(output_data, f, indent=2)
            
        logger.info("FINN build completed successfully")
        
    except Exception as e:
        logger.error(f"FINN execution failed: {{e}}")
        
        # Save error result
        error_data = {{
            "success": False,
            "error": str(e),
            "warnings": []
        }}
        
        with open("{work_dir}/finn_result.json", "w") as f:
            json.dump(error_data, f, indent=2)

if __name__ == "__main__":
    run_six_entrypoint()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        script_path.chmod(0o755)
    
    def _extract_metrics(self, output_dir: str) -> Dict[str, float]:
        """
        Extract metrics from FINN output directory.
        
        Args:
            output_dir: FINN output directory path
            
        Returns:
            Dictionary of metrics
        """
        # Use the existing MetricsExtractor
        # Create a mock finn_result object with the output directory
        class FinnResult:
            def __init__(self, output_dir):
                self.output_dir = output_dir
                
        finn_result = FinnResult(output_dir)
        metrics_dict = self.metrics_extractor.extract_metrics(finn_result, None)
        
        # Extract just the numeric metrics
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
        Validate 6-entrypoint configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check for required entrypoints
        for i in range(1, 7):
            key = f'entrypoint_{i}'
            if key in config and not isinstance(config[key], list):
                errors.append(f"{key} must be a list of transform names")
        
        # Validate transform names
        valid_transforms = {
            'FoldConstants', 'InferShapes', 'RemoveUnusedNodes',
            'RemoveStaticGraphInputs', 'InferDataTypes', 'Streamline',
            'ConvertBipolarToXnor', 'MoveLinearPastFork',
            'AbsorbSignBiasIntoMultiThreshold', 'AnnotateCycles',
            'SetFolding', 'MinimizeAccumulatorWidth', 'MinimizeWeightBitWidth'
        }
        
        for i in range(1, 7):
            key = f'entrypoint_{i}'
            if key in config:
                for transform in config[key]:
                    if transform not in valid_transforms:
                        errors.append(f"Unknown transform '{transform}' in {key}")
        
        return errors