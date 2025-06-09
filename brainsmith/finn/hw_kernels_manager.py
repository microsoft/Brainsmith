"""
Hardware Kernels Manager for FINN Integration Engine.

Handles configuration of FINN hardware kernels including:
- Kernel selection plan validation
- Custom kernels processing
- Folding configuration generation
- Resource sharing configuration
"""

import logging
from typing import Dict, Any, List

from .types import HwKernelsConfig

logger = logging.getLogger(__name__)

class HwKernelsManager:
    """Manager for FINN hardware kernels configuration"""
    
    def __init__(self):
        self.kernel_registry = self._load_kernel_registry()
        self.folding_strategies = self._load_folding_strategies()
        self.memory_configs = self._load_memory_configs()
    
    def configure(self,
                 kernel_selection_plan: Dict[str, str],
                 resource_constraints: Dict[str, float] = None,
                 custom_kernels: Dict[str, Any] = None) -> HwKernelsConfig:
        """Configure hardware kernels for FINN"""
        
        if resource_constraints is None:
            resource_constraints = {}
        if custom_kernels is None:
            custom_kernels = {}
        
        # Validate kernel selection plan
        validated_plan = self._validate_kernel_plan(kernel_selection_plan)
        
        # Process custom kernels
        processed_custom = self._process_custom_kernels(custom_kernels)
        
        # Generate folding configuration
        folding_config = self._generate_folding_config(validated_plan, resource_constraints)
        
        # Configure resource sharing
        resource_sharing = self._configure_resource_sharing(validated_plan)
        
        # Configure memory
        memory_config = self._configure_memory(validated_plan, resource_constraints)
        
        return HwKernelsConfig(
            kernel_selection_plan=validated_plan,
            custom_kernels=processed_custom,
            folding_config=folding_config,
            resource_sharing=resource_sharing,
            memory_config=memory_config
        )
    
    def _load_kernel_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load available FINN kernels registry"""
        return {
            'ConvolutionInputGenerator': {
                'backend': 'rtl',
                'parameters': ['ConvKernelDim', 'IFMChannels', 'Input_precision'],
                'resource_model': {'LUT': 'linear', 'DSP': 'constant'},
                'folding_params': ['SIMD']
            },
            'StreamingFCLayer': {
                'backend': 'hls',
                'parameters': ['PE', 'SIMD', 'MW', 'MH'],
                'resource_model': {'LUT': 'pe_simd', 'DSP': 'pe_dependent'},
                'folding_params': ['PE', 'SIMD']
            },
            'VectorVectorActivation': {
                'backend': 'rtl',
                'parameters': ['PE', 'ActVal', 'NumChannels'],
                'resource_model': {'LUT': 'pe_linear', 'DSP': 'none'},
                'folding_params': ['PE']
            },
            'MatrixVectorActivation': {
                'backend': 'hls',
                'parameters': ['PE', 'SIMD', 'Tiles'],
                'resource_model': {'LUT': 'pe_simd', 'DSP': 'pe_dependent'},
                'folding_params': ['PE', 'SIMD']
            },
            'Thresholding': {
                'backend': 'rtl',
                'parameters': ['NumChannels', 'DataWidth', 'ActVal'],
                'resource_model': {'LUT': 'linear', 'DSP': 'none'},
                'folding_params': ['PE']
            }
        }
    
    def _load_folding_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load folding strategies for different optimization goals"""
        return {
            'throughput': {
                'pe_preference': 'high',
                'simd_preference': 'high',
                'memory_preference': 'internal',
                'pipeline_preference': 'deep'
            },
            'area': {
                'pe_preference': 'low',
                'simd_preference': 'low',
                'memory_preference': 'external',
                'pipeline_preference': 'shallow'
            },
            'balanced': {
                'pe_preference': 'medium',
                'simd_preference': 'medium',
                'memory_preference': 'mixed',
                'pipeline_preference': 'medium'
            },
            'latency': {
                'pe_preference': 'very_high',
                'simd_preference': 'high',
                'memory_preference': 'internal',
                'pipeline_preference': 'very_deep'
            }
        }
    
    def _load_memory_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load memory configuration options"""
        return {
            'internal': {
                'memory_type': 'BRAM',
                'access_pattern': 'sequential',
                'bandwidth': 'high',
                'latency': 'low'
            },
            'external': {
                'memory_type': 'DDR',
                'access_pattern': 'burst',
                'bandwidth': 'medium',
                'latency': 'high'
            },
            'mixed': {
                'memory_type': 'BRAM+DDR',
                'access_pattern': 'adaptive',
                'bandwidth': 'medium',
                'latency': 'medium'
            }
        }
    
    def _validate_kernel_plan(self, kernel_selection_plan: Dict[str, str]) -> Dict[str, str]:
        """Validate kernel selection plan"""
        validated = {}
        
        for layer_name, kernel_name in kernel_selection_plan.items():
            if kernel_name in self.kernel_registry:
                validated[layer_name] = kernel_name
                logger.debug(f"Validated kernel selection: {layer_name} -> {kernel_name}")
            else:
                logger.warning(f"Unknown kernel: {kernel_name} for layer {layer_name}")
                # Try to find a suitable alternative
                alternative = self._find_alternative_kernel(kernel_name)
                if alternative:
                    validated[layer_name] = alternative
                    logger.info(f"Using alternative kernel: {alternative} for layer {layer_name}")
        
        return validated
    
    def _find_alternative_kernel(self, kernel_name: str) -> str:
        """Find alternative kernel based on name similarity or functionality"""
        # Simple fallback mapping
        fallback_mapping = {
            'conv': 'ConvolutionInputGenerator',
            'fc': 'StreamingFCLayer',
            'matmul': 'MatrixVectorActivation',
            'activation': 'VectorVectorActivation',
            'threshold': 'Thresholding'
        }
        
        kernel_lower = kernel_name.lower()
        for pattern, fallback in fallback_mapping.items():
            if pattern in kernel_lower:
                return fallback
        
        # Default fallback
        return 'StreamingFCLayer'
    
    def _process_custom_kernels(self, custom_kernels: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate custom kernels"""
        processed = {}
        
        for kernel_name, kernel_config in custom_kernels.items():
            if self._validate_custom_kernel(kernel_name, kernel_config):
                processed[kernel_name] = kernel_config
                logger.debug(f"Processed custom kernel: {kernel_name}")
            else:
                logger.warning(f"Invalid custom kernel configuration: {kernel_name}")
        
        return processed
    
    def _validate_custom_kernel(self, kernel_name: str, kernel_config: Dict[str, Any]) -> bool:
        """Validate custom kernel configuration"""
        required_fields = ['backend', 'implementation_path']
        
        # Check required fields
        if not all(field in kernel_config for field in required_fields):
            return False
        
        # Validate backend
        valid_backends = ['hls', 'rtl']
        if kernel_config.get('backend') not in valid_backends:
            return False
        
        # Check implementation path exists (basic validation)
        impl_path = kernel_config.get('implementation_path', '')
        if not impl_path or not isinstance(impl_path, str):
            return False
        
        return True
    
    def _generate_folding_config(self, 
                               kernel_plan: Dict[str, str],
                               resource_constraints: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Generate folding configuration for kernels"""
        folding_config = {}
        
        # Determine overall strategy based on constraints
        strategy = self._determine_folding_strategy(resource_constraints)
        strategy_config = self.folding_strategies.get(strategy, self.folding_strategies['balanced'])
        
        for layer_name, kernel_name in kernel_plan.items():
            if kernel_name in self.kernel_registry:
                kernel_info = self.kernel_registry[kernel_name]
                layer_folding = self._generate_layer_folding(
                    kernel_info, strategy_config, resource_constraints
                )
                folding_config[layer_name] = layer_folding
                logger.debug(f"Generated folding config for {layer_name}: {layer_folding}")
        
        return folding_config
    
    def _determine_folding_strategy(self, resource_constraints: Dict[str, float]) -> str:
        """Determine folding strategy based on resource constraints"""
        lut_constraint = resource_constraints.get('luts', 1.0)
        dsp_constraint = resource_constraints.get('dsps', 1.0)
        
        # Very tight constraints -> area optimization
        if lut_constraint < 0.3 or dsp_constraint < 0.3:
            return 'area'
        # Loose constraints -> throughput optimization
        elif lut_constraint > 0.8 and dsp_constraint > 0.8:
            return 'throughput'
        # Medium constraints -> balanced
        else:
            return 'balanced'
    
    def _generate_layer_folding(self, 
                              kernel_info: Dict[str, Any],
                              strategy_config: Dict[str, Any],
                              resource_constraints: Dict[str, float]) -> Dict[str, Any]:
        """Generate folding configuration for a specific layer"""
        config = {}
        
        # Get folding parameters for this kernel
        folding_params = kernel_info.get('folding_params', [])
        
        # Configure PE parallelism
        if 'PE' in folding_params:
            config['PE'] = self._calculate_pe_value(strategy_config, resource_constraints)
        
        # Configure SIMD width
        if 'SIMD' in folding_params:
            config['SIMD'] = self._calculate_simd_value(strategy_config, resource_constraints)
        
        # Configure memory mode
        memory_pref = strategy_config.get('memory_preference', 'mixed')
        config['mem_mode'] = self._select_memory_mode(memory_pref)
        
        # Configure RAM style
        config['ram_style'] = 'auto'  # Let tools decide
        
        return config
    
    def _calculate_pe_value(self, strategy_config: Dict[str, Any], 
                          resource_constraints: Dict[str, float]) -> int:
        """Calculate PE parallelism value"""
        pe_pref = strategy_config.get('pe_preference', 'medium')
        lut_constraint = resource_constraints.get('luts', 1.0)
        
        base_values = {
            'low': 1,
            'medium': 4,
            'high': 16,
            'very_high': 64
        }
        
        base_pe = base_values.get(pe_pref, 4)
        
        # Scale based on resource constraints
        if lut_constraint < 0.5:
            base_pe = max(1, base_pe // 4)
        elif lut_constraint > 0.8:
            base_pe = min(64, base_pe * 2)
        
        return base_pe
    
    def _calculate_simd_value(self, strategy_config: Dict[str, Any],
                            resource_constraints: Dict[str, float]) -> int:
        """Calculate SIMD width value"""
        simd_pref = strategy_config.get('simd_preference', 'medium')
        dsp_constraint = resource_constraints.get('dsps', 1.0)
        
        base_values = {
            'low': 1,
            'medium': 4,
            'high': 8,
            'very_high': 16
        }
        
        base_simd = base_values.get(simd_pref, 4)
        
        # Scale based on DSP constraints
        if dsp_constraint < 0.5:
            base_simd = max(1, base_simd // 2)
        elif dsp_constraint > 0.8:
            base_simd = min(16, base_simd * 2)
        
        return base_simd
    
    def _select_memory_mode(self, memory_preference: str) -> str:
        """Select memory mode based on preference"""
        mode_mapping = {
            'internal': 'internal',
            'external': 'external',
            'mixed': 'internal'  # Default to internal for mixed
        }
        return mode_mapping.get(memory_preference, 'internal')
    
    def _configure_resource_sharing(self, kernel_plan: Dict[str, str]) -> Dict[str, Any]:
        """Configure resource sharing options"""
        config = {
            'enable_sharing': True,
            'sharing_strategies': [],
            'shared_resources': []
        }
        
        # Identify opportunities for resource sharing
        kernel_types = list(set(kernel_plan.values()))
        
        # DSP sharing for arithmetic operations
        arithmetic_kernels = [k for k in kernel_types 
                            if k in ['StreamingFCLayer', 'MatrixVectorActivation']]
        if len(arithmetic_kernels) > 1:
            config['sharing_strategies'].append('dsp_sharing')
            config['shared_resources'].append('DSP')
        
        # Memory sharing for similar access patterns
        memory_kernels = [k for k in kernel_types 
                        if k in ['ConvolutionInputGenerator', 'StreamingFCLayer']]
        if len(memory_kernels) > 1:
            config['sharing_strategies'].append('memory_sharing')
            config['shared_resources'].append('BRAM')
        
        return config
    
    def _configure_memory(self, 
                        kernel_plan: Dict[str, str],
                        resource_constraints: Dict[str, float]) -> Dict[str, Any]:
        """Configure memory settings"""
        bram_constraint = resource_constraints.get('brams', 1.0)
        
        # Determine memory strategy based on constraints
        if bram_constraint < 0.3:
            memory_strategy = 'external'
        elif bram_constraint > 0.8:
            memory_strategy = 'internal'
        else:
            memory_strategy = 'mixed'
        
        config = {
            'default_memory_mode': memory_strategy,
            'memory_mapping': {},
            'buffer_sizes': {},
            'access_patterns': {}
        }
        
        # Configure per-layer memory settings
        for layer_name, kernel_name in kernel_plan.items():
            if kernel_name in self.kernel_registry:
                config['memory_mapping'][layer_name] = memory_strategy
                config['buffer_sizes'][layer_name] = 'auto'
                config['access_patterns'][layer_name] = 'sequential'
        
        return config
    
    def get_available_kernels(self) -> List[str]:
        """Get list of available FINN kernels"""
        return list(self.kernel_registry.keys())
    
    def get_kernel_info(self, kernel_name: str) -> Dict[str, Any]:
        """Get information about a specific kernel"""
        return self.kernel_registry.get(kernel_name, {})
    
    def validate_folding_config(self, folding_config: Dict[str, Dict[str, Any]]) -> bool:
        """Validate folding configuration"""
        for layer_name, layer_config in folding_config.items():
            if not self._validate_layer_folding(layer_config):
                logger.error(f"Invalid folding config for layer {layer_name}")
                return False
        return True
    
    def _validate_layer_folding(self, layer_config: Dict[str, Any]) -> bool:
        """Validate folding configuration for a single layer"""
        # Check PE value
        pe = layer_config.get('PE', 1)
        if not isinstance(pe, int) or pe < 1 or pe > 512:
            return False
        
        # Check SIMD value
        simd = layer_config.get('SIMD', 1)
        if not isinstance(simd, int) or simd < 1 or simd > 128:
            return False
        
        # Check memory mode
        mem_mode = layer_config.get('mem_mode', 'internal')
        if mem_mode not in ['internal', 'external']:
            return False
        
        return True