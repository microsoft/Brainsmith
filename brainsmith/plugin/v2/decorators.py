"""
Plugin Decorators V2

BREAKING CHANGE: Complete redesign of decorator system.
- Eliminates kernel_inference type confusion
- Enforces plugin contracts
- Provides better validation
- Simplified semantics
"""

import logging
from typing import Optional, List, Dict, Any, Type, Union
from functools import wraps

from .registry import get_registry, PluginSpec, PluginState
from .contracts import (
    PluginContract, TransformContract, KernelContract, BackendContract,
    PluginDependency, SystemDependency, ValidationResult
)

logger = logging.getLogger(__name__)


def transform(
    name: str,
    stage: Optional[str] = None,
    target_kernel: Optional[str] = None,
    version: str = "0.0.0",
    author: Optional[str] = None,
    description: Optional[str] = None,
    framework: str = "brainsmith",
    dependencies: List[Union[PluginDependency, SystemDependency]] = None,
    tags: List[str] = None,
    config_schema: Optional[Any] = None,
    **custom_metadata
):
    """
    Transform plugin decorator.
    
    BREAKING CHANGE: 
    - No more kernel_inference type - everything is type="transform"
    - Must specify either stage OR target_kernel (mutually exclusive)
    - All transforms must implement TransformContract
    
    Args:
        name: Unique transform name
        stage: Compilation stage (cleanup, topology_opt, kernel_opt, dataflow_opt)
        target_kernel: Target kernel for inference transforms (mutually exclusive with stage)
        version: Semantic version string
        author: Author name or organization
        description: Human-readable description
        framework: Framework name (defaults to "brainsmith")
        dependencies: List of plugin or system dependencies
        tags: List of tags for categorization
        config_schema: Configuration schema for runtime config
        **custom_metadata: Additional metadata fields
    
    Example:
        # Regular transform
        @transform(name="ExpandNorms", stage="topology_opt")
        class ExpandNorms(TransformContract):
            pass
        
        # Kernel inference transform  
        @transform(name="InferLayerNorm", target_kernel="LayerNorm")
        class InferLayerNorm(TransformContract):
            pass
    """
    def decorator(cls: Type) -> Type:
        # Validate mutual exclusion
        if stage and target_kernel:
            raise ValueError(
                f"Transform '{name}' cannot specify both 'stage' and 'target_kernel'. "
                f"Use 'stage' for regular transforms or 'target_kernel' for kernel inference."
            )
        
        if not stage and not target_kernel:
            raise ValueError(
                f"Transform '{name}' must specify either 'stage' or 'target_kernel'. "
                f"Use 'stage' for regular transforms or 'target_kernel' for kernel inference."
            )
        
        # Validate contract compliance
        if not issubclass(cls, TransformContract):
            raise TypeError(f"Transform '{name}' must inherit from TransformContract")
        
        # Validate stage if provided
        if stage:
            valid_stages = ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt"]
            if stage not in valid_stages:
                logger.warning(f"Transform '{name}' uses non-standard stage '{stage}'. "
                             f"Standard stages: {valid_stages}")
        
        # Create plugin specification
        plugin_spec = PluginSpec(
            plugin_id="",  # Will be assigned during registration
            name=name,
            type="transform",  # Single type, no more kernel_inference confusion
            plugin_class=cls,
            stage=stage,
            target_kernel=target_kernel,
            version=version,
            author=author,
            description=description,
            framework=framework,
            tags=tags or [],
            dependencies=dependencies or [],
            config_schema=config_schema,
            custom_metadata=custom_metadata
        )
        
        # Store metadata on class for introspection
        cls._plugin_spec = plugin_spec
        cls._plugin_metadata = plugin_spec.to_dict()
        
        # Store classification info on class
        cls._stage = stage
        cls._target_kernel = target_kernel
        cls._framework = framework
        
        # Register with registry
        registry = get_registry()
        result = registry.register(plugin_spec)
        
        if result.success:
            logger.debug(f"Registered transform: {name}")
            if result.warnings:
                for warning in result.warnings:
                    logger.warning(f"Transform {name}: {warning}")
        else:
            logger.error(f"Failed to register transform {name}: {result.errors}")
            # Don't raise exception - allow class to be defined even if registration fails
        
        return cls
    
    return decorator


def kernel(
    name: str,
    op_type: Optional[str] = None,
    domain: Optional[str] = None,
    version: str = "0.0.0",
    author: Optional[str] = None,
    description: Optional[str] = None,
    framework: str = "brainsmith",
    dependencies: List[Union[PluginDependency, SystemDependency]] = None,
    tags: List[str] = None,
    config_schema: Optional[Any] = None,
    **custom_metadata
):
    """
    Kernel plugin decorator.
    
    BREAKING CHANGE: Must implement KernelContract.
    
    Args:
        name: Unique kernel name
        op_type: ONNX operation type (defaults to name)
        domain: ONNX domain (defaults to brainsmith.kernels.{name.lower()})
        version: Semantic version string
        author: Author name or organization
        description: Human-readable description
        framework: Framework name (defaults to "brainsmith")
        dependencies: List of plugin or system dependencies
        tags: List of tags for categorization
        config_schema: Configuration schema for runtime config
        **custom_metadata: Additional metadata fields
    
    Example:
        @kernel(name="LayerNorm", op_type="LayerNorm")
        class LayerNorm(KernelContract):
            pass
    """
    def decorator(cls: Type) -> Type:
        # Validate contract compliance
        if not issubclass(cls, KernelContract):
            raise TypeError(f"Kernel '{name}' must inherit from KernelContract")
        
        # Set defaults
        actual_op_type = op_type or name
        actual_domain = domain or f"brainsmith.kernels.{name.lower()}"
        
        # Create plugin specification
        plugin_spec = PluginSpec(
            plugin_id="",  # Will be assigned during registration
            name=name,
            type="kernel",
            plugin_class=cls,
            version=version,
            author=author,
            description=description,
            framework=framework,
            tags=tags or [],
            dependencies=dependencies or [],
            config_schema=config_schema,
            custom_metadata={
                'op_type': actual_op_type,
                'domain': actual_domain,
                **custom_metadata
            }
        )
        
        # Store metadata on class
        cls._plugin_spec = plugin_spec
        cls._plugin_metadata = plugin_spec.to_dict()
        cls._op_type = actual_op_type
        cls._domain = actual_domain
        cls._framework = framework
        
        # Register with registry
        registry = get_registry()
        result = registry.register(plugin_spec)
        
        if result.success:
            logger.debug(f"Registered kernel: {name}")
            if result.warnings:
                for warning in result.warnings:
                    logger.warning(f"Kernel {name}: {warning}")
        else:
            logger.error(f"Failed to register kernel {name}: {result.errors}")
        
        return cls
    
    return decorator


def backend(
    name: str,
    target_kernel: str,
    backend_type: str,
    version: str = "0.0.0",
    author: Optional[str] = None,
    description: Optional[str] = None,
    framework: str = "brainsmith",
    dependencies: List[Union[PluginDependency, SystemDependency]] = None,
    tags: List[str] = None,
    config_schema: Optional[Any] = None,
    **custom_metadata
):
    """
    Backend plugin decorator.
    
    BREAKING CHANGE: Must implement BackendContract.
    
    Args:
        name: Unique backend name
        target_kernel: Name of kernel this backend implements (required)
        backend_type: Type of backend ("hls" or "rtl") (required)
        version: Semantic version string
        author: Author name or organization
        description: Human-readable description
        framework: Framework name (defaults to "brainsmith")
        dependencies: List of plugin or system dependencies
        tags: List of tags for categorization
        config_schema: Configuration schema for runtime config
        **custom_metadata: Additional metadata fields
    
    Example:
        @backend(name="LayerNormHLS", target_kernel="LayerNorm", backend_type="hls")
        class LayerNormHLS(BackendContract):
            pass
    """
    def decorator(cls: Type) -> Type:
        # Validate required parameters
        if not target_kernel:
            raise ValueError(f"Backend '{name}' must specify 'target_kernel'")
        
        if not backend_type:
            raise ValueError(f"Backend '{name}' must specify 'backend_type'")
        
        # Validate backend type
        valid_types = ["hls", "rtl"]
        if backend_type not in valid_types:
            raise ValueError(f"Invalid backend_type '{backend_type}'. Must be one of: {valid_types}")
        
        # Validate contract compliance
        if not issubclass(cls, BackendContract):
            raise TypeError(f"Backend '{name}' must inherit from BackendContract")
        
        # Create plugin specification
        plugin_spec = PluginSpec(
            plugin_id="",  # Will be assigned during registration
            name=name,
            type="backend",
            plugin_class=cls,
            target_kernel=target_kernel,
            backend_type=backend_type,
            version=version,
            author=author,
            description=description,
            framework=framework,
            tags=tags or [],
            dependencies=dependencies or [],
            config_schema=config_schema,
            custom_metadata=custom_metadata
        )
        
        # Store metadata on class
        cls._plugin_spec = plugin_spec
        cls._plugin_metadata = plugin_spec.to_dict()
        cls._target_kernel = target_kernel
        cls._backend_type = backend_type
        cls._framework = framework
        
        # Register with registry
        registry = get_registry()
        result = registry.register(plugin_spec)
        
        if result.success:
            logger.debug(f"Registered backend: {name}")
            if result.warnings:
                for warning in result.warnings:
                    logger.warning(f"Backend {name}: {warning}")
        else:
            logger.error(f"Failed to register backend {name}: {result.errors}")
        
        return cls
    
    return decorator


def validate_plugin_class(cls: Type, expected_contract: Type) -> ValidationResult:
    """
    Validate that a plugin class meets contract requirements.
    
    Args:
        cls: Plugin class to validate
        expected_contract: Expected contract interface
        
    Returns:
        ValidationResult with validation outcome
    """
    errors = []
    warnings = []
    
    # Check inheritance
    if not issubclass(cls, expected_contract):
        errors.append(f"Class {cls.__name__} does not inherit from {expected_contract.__name__}")
        return ValidationResult(False, errors, warnings)
    
    # Check for required plugin metadata
    if not hasattr(cls, '_plugin_spec'):
        warnings.append(f"Class {cls.__name__} missing _plugin_spec metadata")
    
    # Check abstract methods
    abstract_methods = getattr(cls, '__abstractmethods__', set())
    if abstract_methods:
        errors.append(f"Class {cls.__name__} has unimplemented abstract methods: {abstract_methods}")
    
    # Try to instantiate (basic smoke test)
    try:
        instance = cls()
        # Try calling contract methods
        if hasattr(instance, 'validate_environment'):
            instance.validate_environment()
        if hasattr(instance, 'get_dependencies'):
            instance.get_dependencies()
        if hasattr(instance, 'get_metadata'):
            instance.get_metadata()
    except Exception as e:
        errors.append(f"Class {cls.__name__} failed instantiation test: {e}")
    
    return ValidationResult(len(errors) == 0, errors, warnings)