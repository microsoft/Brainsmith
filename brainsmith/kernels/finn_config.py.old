"""
FINN Configuration Generator
Generate FINN build configurations from kernel selection plans.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

from .selection import SelectionPlan, KernelParameterConfig
from .database import FINNKernelInfo, OperatorType, BackendType

logger = logging.getLogger(__name__)

class MemoryMode(Enum):
    """Memory modes for FINN kernels"""
    INTERNAL = "internal"
    EXTERNAL = "external"
    STREAMING = "streaming"

class RAMStyle(Enum):
    """RAM styles for FPGA implementation"""
    AUTO = "auto"
    BLOCK = "block"
    DISTRIBUTED = "distributed"
    ULTRA = "ultra"

@dataclass
class LayerFoldingConfig:
    """Folding configuration for a specific layer"""
    pe: int
    simd: int
    mem_mode: str
    ram_style: str = "auto"
    folding_factors: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for FINN configuration"""
        config = {
            'PE': self.pe,
            'SIMD': self.simd,
            'mem_mode': self.mem_mode,
            'ram_style': self.ram_style
        }
        
        # Add folding factors
        for factor_name, factor_value in self.folding_factors.items():
            config[factor_name] = factor_value
        
        return config

@dataclass
class FoldingConfig:
    """Complete folding configuration for FINN model"""
    layer_configs: Dict[str, LayerFoldingConfig] = field(default_factory=dict)
    global_settings: Dict[str, Any] = field(default_factory=dict)
    
    def add_layer_config(self, layer_name: str, config: LayerFoldingConfig) -> None:
        """Add folding configuration for a layer"""
        self.layer_configs[layer_name] = config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for FINN"""
        config = {}
        
        # Add layer configurations
        for layer_name, layer_config in self.layer_configs.items():
            config[layer_name] = layer_config.to_dict()
        
        # Add global settings
        config.update(self.global_settings)
        
        return config

@dataclass
class ModelOpsConfig:
    """Configuration for FINN model operations"""
    supported_ops: List[str] = field(default_factory=list)
    custom_ops: Dict[str, str] = field(default_factory=dict)
    frontend_cleanup: List[str] = field(default_factory=list)
    preprocessing_transforms: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to FINN configuration format"""
        return {
            'supported_ops': self.supported_ops,
            'custom_ops': self.custom_ops,
            'frontend_cleanup': self.frontend_cleanup,
            'preprocessing_transforms': self.preprocessing_transforms
        }

@dataclass
class ModelTransformsConfig:
    """Configuration for FINN model transformations"""
    optimization_level: int = 2
    target_platform: str = "zynq"
    performance_mode: str = "balanced"
    transforms_sequence: List[str] = field(default_factory=list)
    transform_options: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to FINN configuration format"""
        return {
            'optimization_level': self.optimization_level,
            'target_platform': self.target_platform,
            'performance_mode': self.performance_mode,
            'transforms_sequence': self.transforms_sequence,
            'transform_options': self.transform_options
        }

@dataclass
class HwKernelsConfig:
    """Configuration for FINN hardware kernels"""
    kernel_selection_plan: Dict[str, str] = field(default_factory=dict)
    resource_constraints: Dict[str, int] = field(default_factory=dict)
    custom_kernels: Dict[str, str] = field(default_factory=dict)
    kernel_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to FINN configuration format"""
        return {
            'kernel_selection': self.kernel_selection_plan,
            'resource_constraints': self.resource_constraints,
            'custom_kernels': self.custom_kernels,
            'kernel_options': self.kernel_options
        }

@dataclass
class HwOptimizationConfig:
    """Configuration for FINN hardware optimization"""
    optimization_strategy: str = "balanced"
    performance_targets: Dict[str, float] = field(default_factory=dict)
    power_constraints: Dict[str, float] = field(default_factory=dict)
    timing_constraints: Dict[str, int] = field(default_factory=dict)
    optimization_directives: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to FINN configuration format"""
        return {
            'optimization_strategy': self.optimization_strategy,
            'performance_targets': self.performance_targets,
            'power_constraints': self.power_constraints,
            'timing_constraints': self.timing_constraints,
            'optimization_directives': self.optimization_directives
        }

@dataclass
class OptimizationDirectives:
    """Optimization directives for FINN"""
    resource_sharing: Dict[str, Any] = field(default_factory=dict)
    memory_optimization: Dict[str, Any] = field(default_factory=dict)
    pipeline_optimization: Dict[str, Any] = field(default_factory=dict)
    dataflow_optimization: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'resource_sharing': self.resource_sharing,
            'memory_optimization': self.memory_optimization,
            'pipeline_optimization': self.pipeline_optimization,
            'dataflow_optimization': self.dataflow_optimization
        }

@dataclass
class FINNBuildConfig:
    """Complete FINN build configuration"""
    model_ops: ModelOpsConfig = field(default_factory=ModelOpsConfig)
    model_transforms: ModelTransformsConfig = field(default_factory=ModelTransformsConfig)
    hw_kernels: HwKernelsConfig = field(default_factory=HwKernelsConfig)
    hw_optimization: HwOptimizationConfig = field(default_factory=HwOptimizationConfig)
    
    # Build settings
    output_dir: str = "./finn_output"
    build_mode: str = "vivado"
    target_device: str = "xc7z020clg400-1"
    clock_frequency: float = 100.0
    
    # Metadata
    config_version: str = "1.0"
    generated_by: str = "BrainSmith"
    
    def is_valid(self) -> bool:
        """Validate configuration completeness"""
        # Check required components
        if not self.model_ops or not self.hw_kernels:
            return False
        
        # Check kernel configurations
        if not self.hw_kernels.kernel_selection_plan:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to complete FINN configuration dictionary"""
        config = {
            'model_ops': self.model_ops.to_dict(),
            'model_transforms': self.model_transforms.to_dict(),
            'hw_kernels': self.hw_kernels.to_dict(),
            'hw_optimization': self.hw_optimization.to_dict(),
            'build_settings': {
                'output_dir': self.output_dir,
                'build_mode': self.build_mode,
                'target_device': self.target_device,
                'clock_frequency': self.clock_frequency
            },
            'metadata': {
                'config_version': self.config_version,
                'generated_by': self.generated_by
            }
        }
        
        return config
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"FINN configuration saved to: {filepath}")

class FINNConfigTemplateLoader:
    """Loads FINN configuration templates"""
    
    def __init__(self):
        self.templates = self._load_default_templates()
    
    def get_template(self, template_name: str) -> Dict[str, Any]:
        """Get configuration template by name"""
        return self.templates.get(template_name, {})
    
    def _load_default_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load default configuration templates"""
        templates = {
            'cnn_quantized': {
                'model_ops': {
                    'supported_ops': ['Conv', 'MatMul', 'Relu', 'MaxPool', 'Add'],
                    'frontend_cleanup': ['RemoveUnusedNodes', 'FoldConstants']
                },
                'model_transforms': {
                    'optimization_level': 2,
                    'transforms_sequence': [
                        'ConvertToHWOps',
                        'CreateDataflowPartition',
                        'OptimizeDataflow'
                    ]
                }
            },
            'lstm_quantized': {
                'model_ops': {
                    'supported_ops': ['MatMul', 'Add', 'Sigmoid', 'Tanh'],
                    'frontend_cleanup': ['RemoveUnusedNodes', 'FoldConstants']
                },
                'model_transforms': {
                    'optimization_level': 1,
                    'transforms_sequence': [
                        'ConvertToHWOps',
                        'CreateDataflowPartition'
                    ]
                }
            },
            'transformer_quantized': {
                'model_ops': {
                    'supported_ops': ['MatMul', 'Add', 'LayerNorm', 'Softmax'],
                    'frontend_cleanup': ['RemoveUnusedNodes', 'FoldConstants']
                },
                'model_transforms': {
                    'optimization_level': 3,
                    'transforms_sequence': [
                        'ConvertToHWOps',
                        'CreateDataflowPartition',
                        'OptimizeDataflow',
                        'OptimizeAttention'
                    ]
                }
            }
        }
        
        return templates

class FINNConfigValidator:
    """Validates FINN configurations"""
    
    def __init__(self):
        self.required_fields = {
            'model_ops': ['supported_ops'],
            'hw_kernels': ['kernel_selection'],
            'build_settings': ['target_device']
        }
        
        self.valid_build_modes = ['vivado', 'vitis', 'simulation']
        self.valid_devices = [
            'xc7z020clg400-1', 'xczu3eg-sbva484-1-e', 'xcvu9p-flga2104-2-i'
        ]
    
    def validate(self, config: FINNBuildConfig) -> 'ValidationResult':
        """Validate FINN build configuration"""
        result = ValidationResult()
        
        # Check required fields
        self._validate_required_fields(config, result)
        
        # Validate build settings
        self._validate_build_settings(config, result)
        
        # Validate kernel configurations
        self._validate_kernel_configs(config, result)
        
        # Validate resource constraints
        self._validate_resource_constraints(config, result)
        
        result.is_valid = len(result.errors) == 0
        
        return result
    
    def _validate_required_fields(self, config: FINNBuildConfig, result: 'ValidationResult') -> None:
        """Validate required configuration fields"""
        
        config_dict = config.to_dict()
        
        for section, required_fields in self.required_fields.items():
            if section not in config_dict:
                result.add_error(f"Missing required section: {section}")
                continue
            
            section_config = config_dict[section]
            for field in required_fields:
                if field not in section_config:
                    result.add_error(f"Missing required field: {section}.{field}")
    
    def _validate_build_settings(self, config: FINNBuildConfig, result: 'ValidationResult') -> None:
        """Validate build settings"""
        
        # Validate build mode
        if config.build_mode not in self.valid_build_modes:
            result.add_error(f"Invalid build mode: {config.build_mode}")
        
        # Validate target device
        if config.target_device not in self.valid_devices:
            result.add_warning(f"Unrecognized target device: {config.target_device}")
        
        # Validate clock frequency
        if config.clock_frequency <= 0 or config.clock_frequency > 1000:
            result.add_error(f"Invalid clock frequency: {config.clock_frequency}")
    
    def _validate_kernel_configs(self, config: FINNBuildConfig, result: 'ValidationResult') -> None:
        """Validate kernel configurations"""
        
        kernel_configs = config.hw_kernels.kernel_options
        
        for layer_name, layer_config in kernel_configs.items():
            # Validate PE and SIMD values
            if 'PE' in layer_config:
                pe_value = layer_config['PE']
                if not isinstance(pe_value, int) or pe_value <= 0:
                    result.add_error(f"Invalid PE value for {layer_name}: {pe_value}")
            
            if 'SIMD' in layer_config:
                simd_value = layer_config['SIMD']
                if not isinstance(simd_value, int) or simd_value <= 0:
                    result.add_error(f"Invalid SIMD value for {layer_name}: {simd_value}")
    
    def _validate_resource_constraints(self, config: FINNBuildConfig, result: 'ValidationResult') -> None:
        """Validate resource constraints"""
        
        constraints = config.hw_kernels.resource_constraints
        
        # Check for reasonable resource limits
        resource_limits = {
            'max_luts': 500000,
            'max_dsps': 10000,
            'max_brams': 2000
        }
        
        for resource, limit in resource_limits.items():
            if resource in constraints:
                if constraints[resource] > limit:
                    result.add_warning(f"Very high {resource} constraint: {constraints[resource]}")

@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        """Add validation error"""
        self.errors.append(error)
        logger.error(f"Config validation error: {error}")
    
    def add_warning(self, warning: str) -> None:
        """Add validation warning"""
        self.warnings.append(warning)
        logger.warning(f"Config validation warning: {warning}")

class FINNConfigGenerator:
    """
    Generate FINN build configurations from selection plans
    
    Main class that converts BrainSmith kernel selection plans into
    complete FINN build configurations ready for execution.
    """
    
    def __init__(self):
        self.template_loader = FINNConfigTemplateLoader()
        self.validator = FINNConfigValidator()
        
        # Default configuration settings
        self.default_settings = {
            'clock_frequency': 100.0,
            'target_device': 'xc7z020clg400-1',
            'build_mode': 'vivado',
            'optimization_level': 2
        }
        
        logger.info("FINN Configuration Generator initialized")
    
    def generate_build_config(self, selection_plan: SelectionPlan) -> FINNBuildConfig:
        """Generate complete FINN build configuration from selection plan"""
        
        logger.info(f"Generating FINN config for {len(selection_plan.selections)} kernels")
        
        config = FINNBuildConfig()
        
        # Generate model operations configuration
        config.model_ops = self._generate_model_ops_config(selection_plan)
        
        # Generate model transforms configuration
        config.model_transforms = self._generate_transforms_config(selection_plan)
        
        # Generate hardware kernels configuration
        config.hw_kernels = self._generate_hw_kernels_config(selection_plan)
        
        # Generate hardware optimization configuration
        config.hw_optimization = self._generate_hw_optimization_config(selection_plan)
        
        # Set build settings from defaults
        config.target_device = self.default_settings['target_device']
        config.clock_frequency = self.default_settings['clock_frequency']
        config.build_mode = self.default_settings['build_mode']
        
        # Validate configuration
        validation_result = self.validator.validate(config)
        if not validation_result.is_valid:
            logger.error(f"Generated configuration is invalid: {validation_result.errors}")
            raise FINNConfigurationError(f"Invalid configuration: {validation_result.errors}")
        
        if validation_result.warnings:
            logger.warning(f"Configuration warnings: {validation_result.warnings}")
        
        logger.info("FINN configuration generated successfully")
        return config
    
    def _generate_model_ops_config(self, selection_plan: SelectionPlan) -> ModelOpsConfig:
        """Generate model operations configuration"""
        
        # Extract supported operators from selections
        supported_ops = set()
        custom_ops = {}
        
        for layer_id, selection in selection_plan.selections.items():
            kernel = selection.kernel
            
            # Map FINN operator types to supported ops
            op_mapping = {
                OperatorType.CONVOLUTION: 'Conv',
                OperatorType.MATMUL: 'MatMul',
                OperatorType.THRESHOLDING: 'Threshold',
                OperatorType.LAYERNORM: 'LayerNorm',
                OperatorType.POOL: 'Pool',
                OperatorType.ELEMENTWISE: 'Add',
                OperatorType.RESHAPE: 'Reshape',
                OperatorType.CONCAT: 'Concat'
            }
            
            if kernel.operator_type in op_mapping:
                supported_ops.add(op_mapping[kernel.operator_type])
            
            # Add custom operators if any
            if kernel.backend_type == BackendType.PYTHON:
                custom_ops[layer_id] = kernel.implementation_files.get('python', '')
        
        return ModelOpsConfig(
            supported_ops=list(supported_ops),
            custom_ops=custom_ops,
            frontend_cleanup=[
                'RemoveUnusedNodes',
                'FoldConstants',
                'ConvertBipolarToXNOR',
                'ConvertToHWOps'
            ],
            preprocessing_transforms=[
                'NormalizeInput',
                'QuantizeWeights'
            ]
        )
    
    def _generate_transforms_config(self, selection_plan: SelectionPlan) -> ModelTransformsConfig:
        """Generate model transforms configuration"""
        
        # Determine optimization level based on selections
        has_complex_ops = any(
            sel.kernel.operator_type in [OperatorType.CONVOLUTION, OperatorType.MATMUL]
            for sel in selection_plan.selections.values()
        )
        
        optimization_level = 3 if has_complex_ops else 2
        
        # Standard transformation sequence
        transforms_sequence = [
            'InferShapes',
            'FoldConstants',
            'GiveUniqueNodeNames',
            'ConvertToHWOps',
            'CreateDataflowPartition',
            'SpecializeLayers',
            'MinimizeAccumulatorWidth',
            'OptimizeDataflow'
        ]
        
        # Add advanced transforms for high optimization
        if optimization_level >= 3:
            transforms_sequence.extend([
                'OptimizeMemoryBandwidth',
                'OptimizePipeline',
                'BalanceResources'
            ])
        
        return ModelTransformsConfig(
            optimization_level=optimization_level,
            target_platform="zynq",
            performance_mode="balanced",
            transforms_sequence=transforms_sequence,
            transform_options={
                'dataflow_optimization': True,
                'resource_optimization': True,
                'timing_optimization': True
            }
        )
    
    def _generate_hw_kernels_config(self, selection_plan: SelectionPlan) -> HwKernelsConfig:
        """Generate hardware kernels configuration"""
        
        kernel_selection_plan = {}
        kernel_options = {}
        resource_constraints = {}
        
        for layer_id, selection in selection_plan.selections.items():
            kernel = selection.kernel
            params = selection.parameters
            
            # Add kernel selection
            kernel_selection_plan[layer_id] = kernel.name
            
            # Add kernel options
            kernel_options[layer_id] = {
                'PE': params.pe_parallelism,
                'SIMD': params.simd_width,
                'mem_mode': params.memory_mode,
                'ram_style': params.ram_style,
                'pipeline_depth': params.pipeline_depth
            }
            
            # Add folding factors
            kernel_options[layer_id].update(params.folding_factors)
            
            # Add custom directives if any
            if params.custom_directives:
                kernel_options[layer_id]['custom_directives'] = params.custom_directives
        
        # Set global resource constraints
        total_resources = selection_plan.estimated_total_resources
        resource_constraints = {
            'max_luts': total_resources.get('lut', 0) * 2,  # Allow some headroom
            'max_dsps': total_resources.get('dsp', 0) * 2,
            'max_brams': total_resources.get('bram', 0) * 2
        }
        
        return HwKernelsConfig(
            kernel_selection_plan=kernel_selection_plan,
            resource_constraints=resource_constraints,
            kernel_options=kernel_options
        )
    
    def _generate_hw_optimization_config(self, selection_plan: SelectionPlan) -> HwOptimizationConfig:
        """Generate hardware optimization configuration"""
        
        # Extract performance targets from global configuration
        global_config = selection_plan.global_configuration
        performance_targets = global_config.global_settings.get('performance_targets', {})
        
        # Set default targets if not specified
        if not performance_targets:
            total_perf = selection_plan.estimated_total_performance
            performance_targets = {
                'target_throughput': total_perf.get('throughput', 1000),
                'max_latency': total_perf.get('latency', 100),
                'target_frequency': 100.0
            }
        
        # Extract optimization directives
        optimization_directives = global_config.optimization_directives
        
        return HwOptimizationConfig(
            optimization_strategy="balanced",
            performance_targets=performance_targets,
            timing_constraints={'max_latency': 1000},
            optimization_directives=optimization_directives
        )
    
    def create_folding_config(self, parameters: 'ParameterConfiguration') -> FoldingConfig:
        """Create FINN folding configuration from parameter configuration"""
        
        folding_config = FoldingConfig()
        
        for kernel_name, params in parameters.kernel_configs.items():
            layer_folding = LayerFoldingConfig(
                pe=params.pe_parallelism,
                simd=params.simd_width,
                mem_mode=params.memory_mode,
                ram_style=params.ram_style,
                folding_factors=params.folding_factors
            )
            
            folding_config.add_layer_config(kernel_name, layer_folding)
        
        # Add global folding settings
        folding_config.global_settings = parameters.global_settings
        
        return folding_config
    
    def generate_optimization_directives(self, selection_plan: SelectionPlan) -> OptimizationDirectives:
        """Generate FINN optimization directives for enhanced performance"""
        
        directives = OptimizationDirectives()
        
        # Resource sharing directives
        directives.resource_sharing = {
            'enable_dsp_sharing': True,
            'enable_memory_sharing': True,
            'sharing_threshold': 0.8
        }
        
        # Memory optimization directives
        directives.memory_optimization = {
            'enable_memory_folding': True,
            'enable_memory_coalescing': True,
            'buffer_depth_optimization': True
        }
        
        # Pipeline optimization directives
        directives.pipeline_optimization = {
            'enable_pipeline_balancing': True,
            'target_ii': 1,  # Initiation interval
            'enable_loop_pipelining': True
        }
        
        # Dataflow optimization directives
        directives.dataflow_optimization = {
            'enable_dataflow_optimization': True,
            'buffer_size_optimization': True,
            'enable_streaming': True
        }
        
        return directives
    
    def apply_template(self, template_name: str, selection_plan: SelectionPlan) -> FINNBuildConfig:
        """Apply configuration template to selection plan"""
        
        # Get base template
        template = self.template_loader.get_template(template_name)
        
        if not template:
            logger.warning(f"Template '{template_name}' not found, using default generation")
            return self.generate_build_config(selection_plan)
        
        # Generate base configuration
        config = self.generate_build_config(selection_plan)
        
        # Apply template overrides
        if 'model_ops' in template:
            for key, value in template['model_ops'].items():
                setattr(config.model_ops, key, value)
        
        if 'model_transforms' in template:
            for key, value in template['model_transforms'].items():
                setattr(config.model_transforms, key, value)
        
        logger.info(f"Applied template '{template_name}' to configuration")
        return config
    
    def export_config(self, config: FINNBuildConfig, filepath: str, format: str = 'json') -> None:
        """Export configuration to file"""
        
        if format == 'json':
            config.save_to_file(filepath)
        else:
            raise ValueError(f"Unsupported export format: {format}")

class FINNConfigurationError(Exception):
    """Exception raised for FINN configuration errors"""
    pass