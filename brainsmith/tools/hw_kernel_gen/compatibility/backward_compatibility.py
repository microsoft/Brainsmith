"""
Backward Compatibility Layer for Legacy API Support.

This module provides a comprehensive backward compatibility layer that allows
existing code to continue working with minimal changes while gradually
migrating to the enhanced architecture.
"""

import warnings
from typing import Dict, Any, Optional, Union, List, Callable
from pathlib import Path
from functools import wraps
from dataclasses import asdict

from ..enhanced_config import PipelineConfig, GeneratorType
from ..enhanced_generator_base import GenerationResult
from ..enhanced_data_structures import RTLModule
from ..orchestration.generator_factory import GeneratorFactory
from ..orchestration.integration_orchestrator import IntegrationOrchestrator
from .legacy_adapter import create_legacy_adapter
from ..migration.migration_utilities import ConfigurationMigrator, DataStructureMigrator


def deprecated(reason: str = None, version: str = None):
    """
    Decorator to mark functions as deprecated.
    
    Args:
        reason: Reason for deprecation
        version: Version when deprecated
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            message = f"Function {func.__name__} is deprecated"
            if version:
                message += f" since version {version}"
            if reason:
                message += f": {reason}"
            message += ". Please migrate to the enhanced architecture."
            
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator


class LegacyAPILayer:
    """
    Backward compatibility layer for legacy API functions.
    
    This class provides legacy function interfaces while internally using
    the enhanced architecture.
    """
    
    def __init__(self):
        self._config = None
        self._factory = None
        self._orchestrator = None
        self._data_migrator = DataStructureMigrator()
    
    def _ensure_initialized(self, **kwargs):
        """Ensure compatibility layer is initialized."""
        if self._config is None:
            # Create default configuration
            self._config = PipelineConfig()
            
            # Apply any legacy configuration if provided
            if "template_dir" in kwargs:
                template_dir = kwargs["template_dir"]
                if template_dir:
                    self._config.template.template_dirs = [Path(template_dir)]
            
            if "output_dir" in kwargs:
                output_dir = kwargs["output_dir"]
                if output_dir:
                    self._config.generation.output_dir = Path(output_dir)
        
        if self._factory is None:
            self._factory = GeneratorFactory(self._config)
            
            # Register legacy adapters
            self._register_legacy_adapters()
        
        if self._orchestrator is None:
            self._orchestrator = IntegrationOrchestrator(self._config, generator_factory=self._factory)
    
    def _register_legacy_adapters(self):
        """Register legacy adapters with the factory."""
        from ..orchestration.generator_factory import GeneratorCapability, GeneratorPriority
        
        # Register HW Custom Op adapter
        hw_adapter = create_legacy_adapter(GeneratorType.HW_CUSTOM_OP, self._config)
        if hw_adapter:
            self._factory.registry.register_generator(
                name="legacy_hw_custom_op",
                generator_class=type(hw_adapter),
                capabilities={GeneratorCapability.HW_CUSTOM_OP},
                priority=GeneratorPriority.LOW,  # Lower priority than enhanced generators
                version="legacy",
                description="Legacy HW Custom Op adapter"
            )
        
        # Register RTL adapter
        rtl_adapter = create_legacy_adapter(GeneratorType.RTL_BACKEND, self._config)
        if rtl_adapter:
            self._factory.registry.register_generator(
                name="legacy_rtl_backend",
                generator_class=type(rtl_adapter),
                capabilities={GeneratorCapability.RTL_BACKEND},
                priority=GeneratorPriority.LOW,
                version="legacy",
                description="Legacy RTL backend adapter"
            )


# Global instance for legacy API compatibility
_legacy_api = LegacyAPILayer()


# Legacy function wrappers
@deprecated("Use EnhancedHWCustomOpGenerator instead", "2.0.0")
def generate_hwcustomop(
    hw_kernel: Any,
    output_path: Optional[Path] = None,
    class_name: Optional[str] = None,
    source_file: str = "unknown.sv",
    template_dir: Optional[Path] = None,
    **kwargs
) -> str:
    """
    Legacy function for HW Custom Op generation.
    
    This function provides backward compatibility for existing code while
    internally using the enhanced architecture.
    """
    _legacy_api._ensure_initialized(template_dir=template_dir, output_dir=output_path)
    
    try:
        # Convert hw_kernel to RTLModule if needed
        if not isinstance(hw_kernel, RTLModule):
            rtl_module = _legacy_api._data_migrator.migrate_hw_kernel_to_rtl_module(hw_kernel)
        else:
            rtl_module = hw_kernel
        
        # Use legacy adapter
        adapter = create_legacy_adapter(GeneratorType.HW_CUSTOM_OP, _legacy_api._config)
        
        inputs = {
            "rtl_module": rtl_module,
            "output_path": output_path or Path("hw_custom_op.py"),
            "class_name": class_name,
            "source_file": source_file,
            **kwargs
        }
        
        result = adapter.generate(inputs)
        
        if result.success and result.artifacts:
            return result.artifacts[0].content
        else:
            raise RuntimeError(f"Generation failed: {result.errors}")
            
    except Exception as e:
        # Fallback to error message for legacy compatibility
        raise RuntimeError(f"Legacy HW Custom Op generation failed: {e}")


@deprecated("Use EnhancedRTLBackendGenerator instead", "2.0.0") 
def generate_rtl_backend(
    rtl_module: Any,
    output_dir: Optional[Path] = None,
    backend_class_name: Optional[str] = None,
    **kwargs
) -> Path:
    """
    Legacy function for RTL Backend generation.
    """
    _legacy_api._ensure_initialized(output_dir=output_dir)
    
    try:
        # Ensure rtl_module is RTLModule
        if not isinstance(rtl_module, RTLModule):
            rtl_module = _legacy_api._data_migrator.migrate_hw_kernel_to_rtl_module(rtl_module)
        
        # Use legacy adapter or enhanced generator
        adapter = create_legacy_adapter(GeneratorType.RTL_BACKEND, _legacy_api._config)
        
        inputs = {
            "rtl_module": rtl_module,
            "output_dir": output_dir or Path("rtl_backend"),
            "backend_class_name": backend_class_name,
            **kwargs
        }
        
        result = adapter.generate(inputs)
        
        if result.success and result.artifacts:
            # Return path to first generated file for legacy compatibility
            output_file = (output_dir or Path("rtl_backend")) / result.artifacts[0].file_name
            return output_file
        else:
            raise RuntimeError(f"Generation failed: {result.errors}")
            
    except Exception as e:
        raise RuntimeError(f"Legacy RTL Backend generation failed: {e}")


@deprecated("Use generate_rtl_template from RTL template generator", "2.0.0")
def generate_rtl_template(hw_kernel_data: Any, output_dir: Path) -> Path:
    """
    Legacy function for RTL template generation.
    
    This maintains compatibility with the existing RTL template function.
    """
    try:
        # Try to import and use the legacy function directly if available
        from ..generators.rtl_template_generator import generate_rtl_template as legacy_func
        return legacy_func(hw_kernel_data, output_dir)
    except ImportError:
        # Fallback to adapter approach
        _legacy_api._ensure_initialized(output_dir=output_dir)
        
        rtl_module = _legacy_api._data_migrator.migrate_hw_kernel_to_rtl_module(hw_kernel_data)
        adapter = create_legacy_adapter(GeneratorType.RTL_BACKEND, _legacy_api._config)
        
        inputs = {
            "rtl_module": rtl_module,
            "output_dir": output_dir
        }
        
        result = adapter.generate(inputs)
        
        if result.success and result.artifacts:
            output_file = output_dir / result.artifacts[0].file_name
            # Write the content to file for legacy compatibility
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(result.artifacts[0].content)
            return output_file
        else:
            raise RuntimeError(f"RTL template generation failed: {result.errors}")


class LegacyHardwareKernelGenerator:
    """
    Legacy Hardware Kernel Generator class for backward compatibility.
    
    This class provides the same interface as the original HardwareKernelGenerator
    while internally using the enhanced architecture.
    """
    
    @deprecated("Use IntegrationOrchestrator instead", "2.0.0")
    def __init__(
        self,
        template_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        **kwargs
    ):
        """Initialize legacy HKG with backward compatibility."""
        self.template_dir = template_dir
        self.output_dir = output_dir
        
        # Initialize enhanced components
        self.config = PipelineConfig()
        
        if template_dir:
            self.config.template.template_dirs = [template_dir]
        if output_dir:
            self.config.generation.output_dir = output_dir
        
        # Apply additional configuration
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.orchestrator = IntegrationOrchestrator(self.config)
        self.data_migrator = DataStructureMigrator()
    
    @deprecated("Use IntegrationOrchestrator.orchestrate_complete_generation", "2.0.0")
    def generate_all(
        self, 
        hw_kernel: Any, 
        generate_hw_custom_op: bool = True,
        generate_rtl_backend: bool = True,
        generate_documentation: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Legacy generate_all method with backward compatibility.
        
        Returns a dictionary with generated content for compatibility.
        """
        try:
            # Convert hw_kernel to RTLModule
            if not isinstance(hw_kernel, RTLModule):
                rtl_module = self.data_migrator.migrate_hw_kernel_to_rtl_module(hw_kernel)
            else:
                rtl_module = hw_kernel
            
            # Configure what to generate based on flags
            if generate_hw_custom_op:
                self.config.generator_type = GeneratorType.AUTO_HW_CUSTOM_OP
            
            # Use orchestrator for generation
            result = self.orchestrator.orchestrate_complete_generation(rtl_module)
            
            # Convert result to legacy format
            legacy_result = {
                "success": result.success,
                "errors": result.errors,
                "warnings": result.warnings,
                "artifacts": {}
            }
            
            # Organize artifacts by type for legacy compatibility
            for artifact in result.all_artifacts:
                artifact_type = artifact.artifact_type
                if artifact_type not in legacy_result["artifacts"]:
                    legacy_result["artifacts"][artifact_type] = []
                
                legacy_result["artifacts"][artifact_type].append({
                    "file_name": artifact.file_name,
                    "content": artifact.content,
                    "metadata": artifact.metadata
                })
            
            return legacy_result
            
        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)],
                "warnings": [],
                "artifacts": {}
            }
    
    @deprecated("Use enhanced configuration system", "2.0.0")
    def set_template_dir(self, template_dir: Path):
        """Legacy method to set template directory."""
        self.template_dir = template_dir
        self.config.template.template_dirs = [template_dir]
    
    @deprecated("Use enhanced configuration system", "2.0.0")
    def set_output_dir(self, output_dir: Path):
        """Legacy method to set output directory."""
        self.output_dir = output_dir
        self.config.generation.output_dir = output_dir


class LegacyConfigurationWrapper:
    """
    Wrapper for legacy configuration formats.
    
    This class allows legacy code to continue using old configuration
    patterns while internally converting to the new format.
    """
    
    def __init__(self, legacy_config: Optional[Dict[str, Any]] = None):
        """Initialize with legacy configuration."""
        self.migrator = ConfigurationMigrator()
        
        if legacy_config:
            self.enhanced_config, self.migration_report = self.migrator.migrate_legacy_config(legacy_config)
        else:
            self.enhanced_config = PipelineConfig()
            self.migration_report = None
    
    @deprecated("Use PipelineConfig directly", "2.0.0")
    def get_template_dir(self) -> Optional[Path]:
        """Get template directory in legacy format."""
        if self.enhanced_config.template.template_dirs:
            return self.enhanced_config.template.template_dirs[0]
        return None
    
    @deprecated("Use PipelineConfig directly", "2.0.0")
    def set_template_dir(self, template_dir: Path):
        """Set template directory in legacy format."""
        self.enhanced_config.template.template_dirs = [template_dir]
    
    @deprecated("Use PipelineConfig directly", "2.0.0")
    def get_output_dir(self) -> Path:
        """Get output directory in legacy format."""
        return self.enhanced_config.generation.output_dir
    
    @deprecated("Use PipelineConfig directly", "2.0.0")
    def set_output_dir(self, output_dir: Path):
        """Set output directory in legacy format."""
        self.enhanced_config.generation.output_dir = output_dir
    
    def to_enhanced_config(self) -> PipelineConfig:
        """Get the enhanced configuration."""
        return self.enhanced_config
    
    def get_migration_report(self):
        """Get migration report if available."""
        return self.migration_report


# Backward compatibility aliases
HardwareKernelGenerator = LegacyHardwareKernelGenerator
HKG = LegacyHardwareKernelGenerator  # Common abbreviation


def enable_legacy_warnings(enabled: bool = True):
    """
    Enable or disable legacy deprecation warnings.
    
    Args:
        enabled: Whether to show deprecation warnings
    """
    if enabled:
        warnings.filterwarnings("default", category=DeprecationWarning)
    else:
        warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_migration_status() -> Dict[str, Any]:
    """
    Get status of migration-related components.
    
    Returns:
        Dictionary with migration status information
    """
    return {
        "legacy_api_active": _legacy_api._config is not None,
        "adapters_registered": _legacy_api._factory is not None,
        "enhanced_architecture_available": True,
        "recommendations": [
            "Migrate to EnhancedHWCustomOpGenerator for new projects",
            "Use IntegrationOrchestrator for complex workflows", 
            "Update configuration to use PipelineConfig",
            "Plan gradual migration using adapters",
            "Review migration utilities for assistance"
        ]
    }


# Legacy import compatibility
def setup_legacy_imports():
    """Setup legacy import compatibility."""
    import sys
    from types import ModuleType
    
    # Create legacy module aliases
    legacy_module = ModuleType("brainsmith.tools.hw_kernel_gen.legacy")
    legacy_module.HardwareKernelGenerator = LegacyHardwareKernelGenerator
    legacy_module.generate_hwcustomop = generate_hwcustomop
    legacy_module.generate_rtl_backend = generate_rtl_backend
    legacy_module.generate_rtl_template = generate_rtl_template
    
    sys.modules["brainsmith.tools.hw_kernel_gen.legacy"] = legacy_module


# Initialize legacy import compatibility
setup_legacy_imports()