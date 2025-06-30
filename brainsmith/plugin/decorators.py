"""
Unified Plugin Decorator System

This module provides a single, unified decorator system for all plugin types,
replacing the scattered decorator systems across the codebase.

The unified system:
- Uses one @plugin decorator for all types (transforms, kernels, backends, steps)
- Integrates with the Phase 1 registry system
- Provides automatic registration with both BrainSmith and framework-native systems
- Supports type-specific validation and metadata
- Eliminates decorator code duplication
"""

import logging
import warnings
from typing import Optional, List, Dict, Any, Callable, Type, Union
from enum import Enum

from .core.data_models import PluginType, FrameworkType, DiscoveryMethod
from .core.registry import get_plugin_registry
from .core.data_models import PluginInfo

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when plugin metadata validation fails."""
    pass


class PluginDecoratorConfig:
    """Configuration for plugin decorator behavior."""
    
    def __init__(self):
        self.auto_register_qonnx = True
        self.auto_register_finn = True
        self.strict_validation = True
        self.warn_on_override = True
    
    def disable_auto_registration(self) -> None:
        """Disable automatic registration with external frameworks."""
        self.auto_register_qonnx = False
        self.auto_register_finn = False
    
    def enable_permissive_mode(self) -> None:
        """Enable permissive validation mode."""
        self.strict_validation = False
        self.warn_on_override = False


# Global configuration instance
_config = PluginDecoratorConfig()


def configure_plugin_decorator(**kwargs) -> None:
    """
    Configure global plugin decorator behavior.
    
    Args:
        auto_register_qonnx: Enable QONNX registration (default: True)
        auto_register_finn: Enable FINN registration (default: True)
        strict_validation: Enable strict metadata validation (default: True)
        warn_on_override: Warn when overriding existing plugins (default: True)
    """
    global _config
    
    for key, value in kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
        else:
            raise ValueError(f"Unknown configuration option: {key}")


class PluginMetadataValidator:
    """Validates plugin metadata based on type-specific requirements."""
    
    REQUIRED_FIELDS = {
        PluginType.TRANSFORM: ["name", "stage"],
        PluginType.KERNEL: ["name"],
        PluginType.BACKEND: ["name", "kernel", "backend_type"],
        PluginType.STEP: ["name", "category"],
        PluginType.KERNEL_INFERENCE: ["name", "kernel"]
    }
    
    MUTUALLY_EXCLUSIVE = {
        PluginType.TRANSFORM: [("stage", "kernel")]  # Can't specify both
    }
    
    VALID_VALUES = {
        "stage": ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt"],
        "backend_type": ["hls", "rtl"],
        "framework": ["brainsmith", "qonnx", "finn"]
    }
    
    @classmethod
    def validate(cls, plugin_type: PluginType, metadata: Dict[str, Any]) -> None:
        """
        Validate plugin metadata for given type.
        
        Args:
            plugin_type: Type of plugin being validated
            metadata: Metadata dictionary to validate
            
        Raises:
            ValidationError: If validation fails
        """
        # Check required fields
        required = cls.REQUIRED_FIELDS.get(plugin_type, [])
        for field in required:
            if field not in metadata or metadata[field] is None:
                raise ValidationError(
                    f"{plugin_type.value} plugins require '{field}' field"
                )
        
        # Check mutually exclusive fields
        exclusive = cls.MUTUALLY_EXCLUSIVE.get(plugin_type, [])
        for field_set in exclusive:
            present = [f for f in field_set if f in metadata and metadata[f] is not None]
            if len(present) > 1:
                raise ValidationError(
                    f"{plugin_type.value} plugins cannot specify both {' and '.join(present)}"
                )
        
        # Check valid values (skip None values and enum types)
        for field, valid_values in cls.VALID_VALUES.items():
            value = metadata.get(field)
            if value is not None:
                # Convert enum values to string for validation
                if hasattr(value, 'value'):
                    value = value.value
                if value not in valid_values:
                    raise ValidationError(
                        f"Invalid {field} '{value}'. "
                        f"Valid values: {valid_values}"
                    )
        
        # Type-specific validation
        if plugin_type == PluginType.TRANSFORM:
            cls._validate_transform(metadata)
        elif plugin_type == PluginType.BACKEND:
            cls._validate_backend(metadata)
        elif plugin_type == PluginType.STEP:
            cls._validate_step(metadata)
    
    @classmethod
    def _validate_transform(cls, metadata: Dict[str, Any]) -> None:
        """Validate transform-specific metadata."""
        # Must have either stage or kernel, but not both
        has_stage = "stage" in metadata and metadata["stage"] is not None
        has_kernel = "kernel" in metadata and metadata["kernel"] is not None
        
        if has_stage and has_kernel:
            raise ValidationError(
                "Transform cannot specify both 'stage' and 'kernel'. "
                "Use 'stage' for transforms or 'kernel' for inference transforms."
            )
        
        if not has_stage and not has_kernel:
            raise ValidationError(
                "Transform must specify either 'stage' or 'kernel'. "
                "Use 'stage' for transforms or 'kernel' for inference transforms."
            )
    
    @classmethod
    def _validate_backend(cls, metadata: Dict[str, Any]) -> None:
        """Validate backend-specific metadata."""
        if "backend_type" in metadata:
            backend_type = metadata["backend_type"]
            if backend_type not in ["hls", "rtl"]:
                raise ValidationError(
                    f"Invalid backend_type '{backend_type}'. Must be 'hls' or 'rtl'"
                )
    
    @classmethod
    def _validate_step(cls, metadata: Dict[str, Any]) -> None:
        """Validate step-specific metadata."""
        # Dependencies must be a list if present
        if "dependencies" in metadata:
            deps = metadata["dependencies"]
            if not isinstance(deps, list):
                raise ValidationError(
                    f"Step dependencies must be a list, got {type(deps)}"
                )


class PluginRegistrationHandler:
    """Handles registration with external framework systems."""
    
    @staticmethod
    def register_with_qonnx(plugin_type: PluginType, cls: Type, metadata: Dict[str, Any]) -> None:
        """Register plugin with QONNX systems."""
        if not _config.auto_register_qonnx:
            return
        
        try:
            if plugin_type in [PluginType.TRANSFORM, PluginType.KERNEL_INFERENCE]:
                PluginRegistrationHandler._register_qonnx_transform(cls, metadata)
            elif plugin_type in [PluginType.KERNEL, PluginType.BACKEND]:
                PluginRegistrationHandler._register_qonnx_custom_op(cls, metadata)
        except ImportError:
            logger.warning("QONNX not available, skipping QONNX registration")
        except Exception as e:
            logger.warning(f"Failed to register with QONNX: {e}")
    
    @staticmethod
    def _register_qonnx_transform(cls: Type, metadata: Dict[str, Any]) -> None:
        """Register with QONNX transformation registry."""
        from qonnx.transformation.registry import register_transformation
        
        # Filter out BrainSmith-specific metadata and fields passed separately
        qonnx_kwargs = {
            k: v for k, v in metadata.items() 
            if k not in ['stage', 'kernel', 'framework', 'backend_type', 'category', 'name', 'description', 'author', 'version', 'tags']
        }
        
        # Apply QONNX decorator
        qonnx_decorator = register_transformation(
            name=metadata['name'],
            description=metadata.get('description'),
            tags=metadata.get('tags', []),
            author=metadata.get('author'),
            version=metadata.get('version'),
            **qonnx_kwargs
        )
        
        # Apply to class
        decorated_cls = qonnx_decorator(cls)
        
        # Copy QONNX metadata back to original class
        if hasattr(decorated_cls, '_qonnx_metadata'):
            cls._qonnx_metadata = decorated_cls._qonnx_metadata
        
        logger.debug(f"Registered transform with QONNX: {metadata['name']}")
    
    @staticmethod
    def _register_qonnx_custom_op(cls: Type, metadata: Dict[str, Any]) -> None:
        """Register with QONNX custom op registry."""
        from qonnx.custom_op.registry import register_op
        
        # Determine domain and op_type
        if metadata.get('plugin_type') == PluginType.BACKEND:
            # Backends use their kernel's domain/op_type
            kernel_name = metadata['kernel']
            op_domain = metadata.get('domain', f"brainsmith.kernels.{kernel_name.lower()}")
            op_type_name = metadata.get('op_type', kernel_name)
        else:
            # Kernels use their own domain/op_type
            name = metadata['name']
            op_domain = metadata.get('domain', f"brainsmith.kernels.{name.lower()}")
            op_type_name = metadata.get('op_type', name)
        
        # Apply QONNX decorator
        decorated_cls = register_op(op_domain, op_type_name)(cls)
        
        # Copy QONNX metadata back to original class
        if hasattr(decorated_cls, '_qonnx_metadata'):
            cls._qonnx_metadata = decorated_cls._qonnx_metadata
        
        logger.debug(f"Registered custom op with QONNX: {op_type_name} in domain {op_domain}")
    
    @staticmethod
    def register_with_finn(plugin_type: PluginType, cls: Type, metadata: Dict[str, Any]) -> None:
        """Register plugin with FINN systems."""
        if not _config.auto_register_finn:
            return
        
        try:
            if plugin_type == PluginType.STEP:
                PluginRegistrationHandler._register_finn_step(cls, metadata)
        except ImportError:
            logger.warning("FINN not available, skipping FINN registration")
        except Exception as e:
            logger.warning(f"Failed to register with FINN: {e}")
    
    @staticmethod
    def _register_finn_step(cls: Type, metadata: Dict[str, Any]) -> None:
        """Register with FINN step registry."""
        try:
            from brainsmith.steps.registry import FinnStepRegistry
            
            registry = FinnStepRegistry()
            registry.register(
                name=metadata['name'],
                func=cls,
                category=metadata.get('category', 'unknown'),
                dependencies=metadata.get('dependencies', []),
                description=metadata.get('description', '')
            )
            
            logger.debug(f"Registered step with FINN: {metadata['name']}")
        except ImportError:
            logger.warning("FINN step registry not available")


def plugin(
    type: Union[str, PluginType],
    name: str,
    # Transform-specific
    stage: Optional[str] = None,
    kernel: Optional[str] = None,
    # Backend-specific  
    backend_type: Optional[str] = None,
    # Step-specific
    category: Optional[str] = None,
    dependencies: Optional[List[str]] = None,
    # Common metadata
    description: Optional[str] = None,
    author: Optional[str] = None,
    version: Optional[str] = None,
    framework: Optional[Union[str, FrameworkType]] = None,
    # QONNX-specific
    op_type: Optional[str] = None,
    domain: Optional[str] = None,
    tags: Optional[List[str]] = None,
    # Additional metadata
    **kwargs
) -> Callable[[Type], Type]:
    """
    Unified plugin decorator for all plugin types.
    
    This decorator replaces @transform, @kernel, @backend, and @finn_step
    with a single, consistent interface.
    
    Args:
        type: Plugin type ("transform", "kernel", "backend", "step")
        name: Plugin name (required)
        stage: Transform stage (transforms only)
        kernel: Target kernel name (inference transforms and backends)
        backend_type: Backend type "hls" or "rtl" (backends only)
        category: Step category (steps only)
        dependencies: Step dependencies (steps only)
        description: Human-readable description
        author: Author name or organization
        version: Version string
        framework: Framework attribution
        op_type: ONNX operation type (for QONNX registration)
        domain: ONNX domain (for QONNX registration)
        tags: Tags for categorization
        **kwargs: Additional metadata
    
    Returns:
        Decorated class with plugin metadata and registrations
    
    Raises:
        ValidationError: If metadata validation fails
        
    Examples:
        # Transform
        @plugin(type="transform", name="ExpandNorms", stage="topology_opt")
        class ExpandNorms(Transformation):
            pass
        
        # Kernel inference transform
        @plugin(type="transform", name="InferLayerNorm", kernel="LayerNorm")
        class InferLayerNorm(Transformation):
            pass
        
        # Kernel
        @plugin(type="kernel", name="LayerNorm", op_type="LayerNorm")
        class LayerNorm(HWCustomOp):
            pass
        
        # Backend
        @plugin(type="backend", name="LayerNormHLS", kernel="LayerNorm", backend_type="hls")
        class LayerNormHLS(LayerNorm, HLSBackend):
            pass
        
        # Step
        @plugin(type="step", name="shell_metadata_handover", category="metadata")
        def shell_metadata_handover(model, cfg):
            return model
    """
    def decorator(cls: Type) -> Type:
        # Convert type to enum if needed
        if isinstance(type, str):
            try:
                plugin_type = PluginType(type)
            except ValueError:
                raise ValidationError(f"Invalid plugin type '{type}'. Valid types: {[t.value for t in PluginType]}")
        else:
            plugin_type = type
        
        # Build metadata dictionary
        metadata = {
            "name": name,
            "framework": FrameworkType(framework) if framework else FrameworkType.BRAINSMITH,
            "description": description,
            "author": author,
            "version": version,
            "discovery_method": DiscoveryMethod.DECORATOR,
            **kwargs
        }
        
        # Add type-specific metadata
        if plugin_type == PluginType.TRANSFORM:
            if kernel:
                # This is actually a kernel inference transform
                plugin_type = PluginType.KERNEL_INFERENCE
                metadata["kernel"] = kernel
                metadata["stage"] = None
            else:
                metadata["stage"] = stage
        elif plugin_type == PluginType.BACKEND:
            metadata["kernel"] = kernel
            metadata["backend_type"] = backend_type
        elif plugin_type == PluginType.STEP:
            metadata["category"] = category or "unknown"
            metadata["dependencies"] = dependencies or []
        
        # Add QONNX-specific metadata
        if op_type:
            metadata["op_type"] = op_type
        if domain:
            metadata["domain"] = domain
        if tags:
            metadata["tags"] = tags
        
        # Validate metadata
        if _config.strict_validation:
            try:
                PluginMetadataValidator.validate(plugin_type, metadata)
            except ValidationError as e:
                raise ValidationError(f"Plugin '{name}' validation failed: {e}")
        
        # Create plugin info
        plugin_info = PluginInfo(
            name=name,
            plugin_class=cls,
            plugin_type=plugin_type.value,
            framework=metadata["framework"].value,
            discovery_method=DiscoveryMethod.DECORATOR.value,
            metadata=metadata
        )
        
        # Register with BrainSmith registry
        registry = get_plugin_registry()
        
        # Check for existing plugin
        existing = registry.get_plugin(name)
        if existing and _config.warn_on_override:
            logger.warning(f"Overriding existing plugin '{name}' from {existing.framework}")
        
        # Register plugin
        registry.register_plugin(plugin_info)
        logger.debug(f"Registered {plugin_type.value} plugin: {name}")
        
        # Store metadata on class for introspection
        cls._plugin_metadata = {
            "type": plugin_type.value,
            "name": name,
            **metadata
        }
        
        # Register with external frameworks
        PluginRegistrationHandler.register_with_qonnx(plugin_type, cls, metadata)
        PluginRegistrationHandler.register_with_finn(plugin_type, cls, metadata)
        
        return cls
    
    return decorator


# Convenience decorators for backward compatibility and ease of use
def transform(name: str, stage: Optional[str] = None, kernel: Optional[str] = None, **kwargs):
    """
    Convenience decorator for transforms.
    
    This is a thin wrapper around @plugin for transforms.
    """
    return plugin(type="transform", name=name, stage=stage, kernel=kernel, **kwargs)


def kernel(name: str, op_type: Optional[str] = None, domain: Optional[str] = None, **kwargs):
    """
    Convenience decorator for kernels.
    
    This is a thin wrapper around @plugin for kernels.
    """
    return plugin(type="kernel", name=name, op_type=op_type, domain=domain, **kwargs)


def backend(name: str, kernel: str, backend_type: str, **kwargs):
    """
    Convenience decorator for backends.
    
    This is a thin wrapper around @plugin for backends.
    """
    return plugin(type="backend", name=name, kernel=kernel, backend_type=backend_type, **kwargs)


def step(name: str, category: str = "unknown", dependencies: Optional[List[str]] = None, **kwargs):
    """
    Convenience decorator for steps.
    
    This is a thin wrapper around @plugin for steps.
    """
    return plugin(type="step", name=name, category=category, dependencies=dependencies, **kwargs)


# Deprecation decorators for legacy imports
def _create_deprecation_decorator(old_name: str, new_decorator: Callable):
    """Create a deprecation wrapper for old decorators."""
    def deprecated_decorator(*args, **kwargs):
        warnings.warn(
            f"@{old_name} is deprecated. Use @plugin or the convenience decorators instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return new_decorator(*args, **kwargs)
    
    deprecated_decorator.__name__ = old_name
    deprecated_decorator.__doc__ = f"DEPRECATED: Use @plugin or convenience decorators instead."
    return deprecated_decorator


# Create deprecation wrappers for old decorator names
finn_step = _create_deprecation_decorator("finn_step", step)


# Export configuration function and validators for advanced usage
__all__ = [
    # Main decorators
    "plugin",
    "transform", 
    "kernel",
    "backend", 
    "step",
    # Legacy support
    "finn_step",
    # Configuration
    "configure_plugin_decorator",
    "PluginDecoratorConfig",
    # Validation
    "PluginMetadataValidator",
    "ValidationError",
    # Registration
    "PluginRegistrationHandler"
]