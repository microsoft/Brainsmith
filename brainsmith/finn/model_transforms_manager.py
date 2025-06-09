"""
Model Transforms Manager for FINN Integration Engine.

Handles configuration of FINN model transforms including:
- Transform sequence selection
- Optimization level configuration
- Quantization settings
- Graph optimizations
"""

import logging
from typing import Dict, Any, List

from .types import ModelTransformsConfig

logger = logging.getLogger(__name__)

class ModelTransformsManager:
    """Manager for FINN model transforms configuration"""
    
    def __init__(self):
        self.transform_sequences = self._load_transform_sequences()
        self.optimization_configs = self._load_optimization_configs()
        self.quantization_presets = self._load_quantization_presets()
    
    def configure(self,
                 optimization_level: str = "standard",
                 target_platform: str = "zynq",
                 performance_targets: Dict[str, float] = None) -> ModelTransformsConfig:
        """Configure model transforms for FINN"""
        
        if performance_targets is None:
            performance_targets = {}
        
        # Select transform sequence based on optimization level
        transforms_sequence = self._select_transform_sequence(optimization_level)
        
        # Configure quantization
        quantization_config = self._configure_quantization(target_platform, performance_targets)
        
        # Select graph optimizations
        graph_optimizations = self._select_graph_optimizations(optimization_level)
        
        return ModelTransformsConfig(
            transforms_sequence=transforms_sequence,
            optimization_level=optimization_level,
            target_platform=target_platform,
            quantization_config=quantization_config,
            graph_optimizations=graph_optimizations
        )
    
    def _load_transform_sequences(self) -> Dict[str, List[str]]:
        """Load predefined transform sequences"""
        return {
            'basic': [
                'InferShapes',
                'FoldConstants',
                'ConvertToHWOps',
                'CreateDataflowPartition'
            ],
            'standard': [
                'InferShapes',
                'FoldConstants',
                'InsertTopK',
                'ConvertToHWOps',
                'CreateDataflowPartition',
                'SpecializeDataflowLayers',
                'ApplyFolding'
            ],
            'aggressive': [
                'InferShapes',
                'FoldConstants',
                'InsertTopK',
                'OptimizeGraph',
                'ConvertToHWOps',
                'CreateDataflowPartition',
                'SpecializeDataflowLayers',
                'ApplyFolding',
                'OptimizeDataflow',
                'InsertFIFO'
            ],
            'minimal': [
                'InferShapes',
                'ConvertToHWOps'
            ]
        }
    
    def _load_optimization_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load optimization configurations for different platforms"""
        return {
            'zynq': {
                'target_clk_ns': 10.0,
                'resource_budget': {'LUT': 0.8, 'DSP': 0.8, 'BRAM': 0.8},
                'optimization_focus': 'balanced',
                'memory_bandwidth': 'medium'
            },
            'ultrascale': {
                'target_clk_ns': 5.0,
                'resource_budget': {'LUT': 0.7, 'DSP': 0.7, 'BRAM': 0.7},
                'optimization_focus': 'performance',
                'memory_bandwidth': 'high'
            },
            'kintex': {
                'target_clk_ns': 8.0,
                'resource_budget': {'LUT': 0.75, 'DSP': 0.75, 'BRAM': 0.75},
                'optimization_focus': 'area',
                'memory_bandwidth': 'medium'
            },
            'virtex': {
                'target_clk_ns': 4.0,
                'resource_budget': {'LUT': 0.6, 'DSP': 0.6, 'BRAM': 0.6},
                'optimization_focus': 'performance',
                'memory_bandwidth': 'very_high'
            }
        }
    
    def _load_quantization_presets(self) -> Dict[str, Dict[str, Any]]:
        """Load quantization presets for different use cases"""
        return {
            'int8_standard': {
                'weights_bitwidth': 8,
                'activations_bitwidth': 8,
                'bias_bitwidth': 32,
                'quantization_type': 'symmetric',
                'calibration_method': 'percentile'
            },
            'int4_aggressive': {
                'weights_bitwidth': 4,
                'activations_bitwidth': 4,
                'bias_bitwidth': 16,
                'quantization_type': 'asymmetric',
                'calibration_method': 'entropy'
            },
            'mixed_precision': {
                'weights_bitwidth': 8,
                'activations_bitwidth': 4,
                'bias_bitwidth': 32,
                'quantization_type': 'symmetric',
                'calibration_method': 'percentile'
            },
            'high_accuracy': {
                'weights_bitwidth': 16,
                'activations_bitwidth': 16,
                'bias_bitwidth': 32,
                'quantization_type': 'symmetric',
                'calibration_method': 'min_max'
            }
        }
    
    def _select_transform_sequence(self, optimization_level: str) -> List[str]:
        """Select transform sequence based on optimization level"""
        if optimization_level in self.transform_sequences:
            sequence = self.transform_sequences[optimization_level].copy()
            logger.debug(f"Selected {optimization_level} transform sequence with {len(sequence)} steps")
            return sequence
        else:
            logger.warning(f"Unknown optimization level: {optimization_level}, using standard")
            return self.transform_sequences['standard'].copy()
    
    def _configure_quantization(self, 
                              target_platform: str,
                              performance_targets: Dict[str, float]) -> Dict[str, Any]:
        """Configure quantization based on platform and targets"""
        
        # Start with platform defaults
        platform_config = self.optimization_configs.get(target_platform, 
                                                        self.optimization_configs['zynq'])
        
        # Select base quantization preset
        if performance_targets.get('throughput', 0) > 1000:
            # High throughput - use aggressive quantization
            config = self.quantization_presets['int4_aggressive'].copy()
            logger.debug("Selected aggressive quantization for high throughput")
        elif performance_targets.get('accuracy', 0) > 0.95:
            # High accuracy requirement - use conservative quantization
            config = self.quantization_presets['high_accuracy'].copy()
            logger.debug("Selected high accuracy quantization")
        else:
            # Balanced approach
            config = self.quantization_presets['int8_standard'].copy()
            logger.debug("Selected standard quantization")
        
        # Platform-specific adjustments
        if target_platform in ['ultrascale', 'virtex']:
            # High-end platforms can handle more complex quantization
            config['enable_per_channel'] = True
            config['enable_bias_correction'] = True
        
        # Add calibration settings
        config['calibration_samples'] = min(1000, 
                                           performance_targets.get('calibration_samples', 500))
        config['calibration_percentile'] = 99.9
        
        return config
    
    def _select_graph_optimizations(self, optimization_level: str) -> List[str]:
        """Select graph optimizations based on level"""
        
        base_optimizations = [
            'FoldConstants',
            'EliminateDeadCode'
        ]
        
        if optimization_level in ['standard', 'aggressive']:
            base_optimizations.extend([
                'CommonSubexpressionElimination',
                'LoopFusion',
                'MemoryOptimization'
            ])
        
        if optimization_level == 'aggressive':
            base_optimizations.extend([
                'PipelineOptimization',
                'AdvancedLoopOptimization',
                'ResourceSharing',
                'DataflowOptimization'
            ])
        
        logger.debug(f"Selected {len(base_optimizations)} graph optimizations for {optimization_level}")
        return base_optimizations
    
    def get_available_optimization_levels(self) -> List[str]:
        """Get list of available optimization levels"""
        return list(self.transform_sequences.keys())
    
    def get_supported_platforms(self) -> List[str]:
        """Get list of supported target platforms"""
        return list(self.optimization_configs.keys())
    
    def get_quantization_presets(self) -> List[str]:
        """Get list of available quantization presets"""
        return list(self.quantization_presets.keys())
    
    def validate_configuration(self, config: ModelTransformsConfig) -> bool:
        """Validate a model transforms configuration"""
        # Check optimization level
        if config.optimization_level not in self.transform_sequences:
            logger.error(f"Invalid optimization level: {config.optimization_level}")
            return False
        
        # Check target platform
        if config.target_platform not in self.optimization_configs:
            logger.error(f"Invalid target platform: {config.target_platform}")
            return False
        
        # Validate quantization config
        if not self._validate_quantization_config(config.quantization_config):
            logger.error("Invalid quantization configuration")
            return False
        
        return True
    
    def _validate_quantization_config(self, quant_config: Dict[str, Any]) -> bool:
        """Validate quantization configuration"""
        required_fields = ['weights_bitwidth', 'activations_bitwidth', 'quantization_type']
        
        # Check required fields
        if not all(field in quant_config for field in required_fields):
            return False
        
        # Validate bitwidths
        weights_bits = quant_config.get('weights_bitwidth', 0)
        activations_bits = quant_config.get('activations_bitwidth', 0)
        
        if not (1 <= weights_bits <= 32) or not (1 <= activations_bits <= 32):
            return False
        
        # Validate quantization type
        valid_types = ['symmetric', 'asymmetric']
        if quant_config.get('quantization_type') not in valid_types:
            return False
        
        return True
    
    def get_recommended_config(self, 
                             model_size: int,
                             target_throughput: float,
                             accuracy_requirement: float) -> Dict[str, Any]:
        """Get recommended configuration based on requirements"""
        
        recommendations = {}
        
        # Optimization level recommendation
        if model_size > 10_000_000:  # Large model
            if target_throughput > 1000:
                recommendations['optimization_level'] = 'aggressive'
            else:
                recommendations['optimization_level'] = 'standard'
        else:  # Smaller model
            recommendations['optimization_level'] = 'standard'
        
        # Quantization recommendation
        if accuracy_requirement > 0.95:
            recommendations['quantization_preset'] = 'high_accuracy'
        elif target_throughput > 1000:
            recommendations['quantization_preset'] = 'int4_aggressive'
        else:
            recommendations['quantization_preset'] = 'int8_standard'
        
        return recommendations