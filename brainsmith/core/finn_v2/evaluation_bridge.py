"""
FINN Evaluation Bridge

Main interface that DSE Explorer calls to evaluate component combinations.
Bridges Blueprint V2 design space exploration to real FINN execution.
"""

from typing import Dict, Any, Optional
import logging
from pathlib import Path

from .legacy_conversion import LegacyConversionLayer
from .metrics_extractor import MetricsExtractor
from ..dse_v2.combination_generator import ComponentCombination

logger = logging.getLogger(__name__)


class FINNEvaluationBridge:
    """Bridge from ComponentCombination to FINN execution with real results."""
    
    def __init__(self, blueprint_config: Dict[str, Any]):
        """
        Initialize with blueprint configuration for FINN parameters.
        
        Args:
            blueprint_config: Blueprint V2 configuration dictionary
        """
        self.blueprint_config = blueprint_config
        self.legacy_converter = LegacyConversionLayer()
        self.metrics_extractor = MetricsExtractor()
        
        logger.info("FINNEvaluationBridge initialized with real FINN integration")
    
    def evaluate_combination(self, model_path: str, combination: ComponentCombination) -> Dict[str, Any]:
        """
        Execute real FINN run for given combination.
        
        Flow:
        1. Convert ComponentCombination → 6-entrypoint config
        2. Use LegacyConversionLayer → DataflowBuildConfig  
        3. Execute FINN build_dataflow_cfg() with real FINN
        4. Extract performance metrics from FINN results
        5. Return standardized metrics for DSE
        
        Args:
            model_path: Path to ONNX model file
            combination: ComponentCombination to evaluate
            
        Returns:
            Dictionary with standardized metrics for DSE optimization
        """
        logger.debug(f"Evaluating combination: {combination.combination_id}")
        
        try:
            # Step 1: Convert combination to 6-entrypoint configuration
            entrypoint_config = self._combination_to_entrypoint_config(combination)
            logger.debug(f"Generated entrypoint config: {entrypoint_config}")
            
            # Step 2: Convert to FINN DataflowBuildConfig
            dataflow_config = self.legacy_converter.convert_to_dataflow_config(
                entrypoint_config, self.blueprint_config
            )
            logger.debug("Converted to FINN DataflowBuildConfig")
            
            # Step 3: Execute real FINN build
            finn_result = self._execute_finn_run(model_path, dataflow_config)
            
            # Step 4: Extract standardized metrics
            metrics = self.metrics_extractor.extract_metrics(finn_result, dataflow_config)
            
            logger.info(f"Combination {combination.combination_id} evaluated successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed for combination {combination.combination_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'combination_id': combination.combination_id,
                'primary_metric': 0.0,
                'throughput': 0.0,
                'latency': float('inf'),
                'resource_utilization': 0.0
            }
    
    def _combination_to_entrypoint_config(self, combination: ComponentCombination) -> Dict[str, list]:
        """
        Convert DSE combination to 6-entrypoint configuration.
        
        Maps ComponentCombination attributes to 6-entrypoint structure:
        - canonical_ops → entrypoint_1
        - model_topology → entrypoint_2  
        - hw_kernels → entrypoint_3 (kernels) + entrypoint_4 (specializations)
        - hw_kernel_transforms → entrypoint_5
        - hw_graph_transforms → entrypoint_6
        
        Args:
            combination: ComponentCombination from DSE
            
        Returns:
            6-entrypoint configuration dictionary
        """
        entrypoint_config = {
            'entrypoint_1': list(combination.canonical_ops),
            'entrypoint_2': list(combination.model_topology),
            'entrypoint_3': [],  # hw_kernels (kernel names)
            'entrypoint_4': [],  # hw_kernel_specializations  
            'entrypoint_5': list(combination.hw_kernel_transforms),
            'entrypoint_6': list(combination.hw_graph_transforms)
        }
        
        # Process hw_kernels into entrypoints 3 & 4
        for kernel_name, specialization in combination.hw_kernels.items():
            entrypoint_config['entrypoint_3'].append(kernel_name)
            if specialization:
                entrypoint_config['entrypoint_4'].append(specialization)
        
        logger.debug(f"Converted combination to entrypoint config: {entrypoint_config}")
        return entrypoint_config
    
    def _execute_finn_run(self, model_path: str, dataflow_config) -> Any:
        """
        Execute real FINN build with error handling.
        
        Args:
            model_path: Path to ONNX model
            dataflow_config: FINN DataflowBuildConfig
            
        Returns:
            FINN build result object
        """
        try:
            # Import real FINN components
            from finn.builder.build_dataflow import build_dataflow_cfg
            
            # Validate model path
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Execute real FINN build
            logger.info(f"Executing FINN build for model: {model_path}")
            finn_result = build_dataflow_cfg(model_path, dataflow_config)
            
            logger.info("FINN build completed successfully")
            return finn_result
            
        except ImportError as e:
            logger.error(f"FINN import failed: {e}")
            raise RuntimeError("FINN not available - ensure FINN is installed and accessible")
        except Exception as e:
            logger.error(f"FINN build failed: {e}")
            raise RuntimeError(f"FINN execution failed: {str(e)}")
    
    def get_supported_objectives(self) -> list:
        """Get list of supported optimization objectives."""
        return [
            'throughput',
            'latency', 
            'resource_efficiency',
            'power_consumption',
            'lut_utilization',
            'dsp_utilization',
            'bram_utilization'
        ]
    
    def validate_combination(self, combination: ComponentCombination) -> tuple[bool, list]:
        """
        Validate combination before FINN execution.
        
        Args:
            combination: ComponentCombination to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for empty critical components
        if not combination.canonical_ops and not combination.hw_kernels:
            errors.append("Combination must have at least canonical_ops or hw_kernels")
        
        # Check for conflicting components
        if 'aggressive_streamlining' in combination.model_topology and 'conservative_streamlining' in combination.model_topology:
            errors.append("Cannot use both aggressive and conservative streamlining")
        
        # Validate hw_kernel dependencies
        for kernel, specialization in combination.hw_kernels.items():
            if specialization and kernel not in combination.canonical_ops:
                errors.append(f"Kernel specialization '{specialization}' requires kernel '{kernel}' in canonical_ops")
        
        is_valid = len(errors) == 0
        return is_valid, errors