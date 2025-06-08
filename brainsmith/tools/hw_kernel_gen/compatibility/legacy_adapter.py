"""
Legacy Generator Adapter for Backward Compatibility.

This module provides compatibility adapters that allow existing generator
implementations to work with the new Week 3 orchestration architecture
while maintaining their original API interfaces.
"""

import warnings
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from abc import ABC, abstractmethod

from ..enhanced_config import PipelineConfig, GeneratorType
from ..enhanced_generator_base import GeneratorBase, GenerationResult, GeneratedArtifact
from ..enhanced_data_structures import RTLModule
from ..errors import BrainsmithError, ConfigurationError


class LegacyGeneratorAdapter(GeneratorBase):
    """
    Base adapter class for wrapping legacy generators.
    
    This adapter allows legacy generators to work with the new orchestration
    system while preserving their original interfaces.
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._legacy_generator = None
        self._initialized = False
    
    @abstractmethod
    def _create_legacy_generator(self) -> Any:
        """Create the legacy generator instance."""
        pass
    
    @abstractmethod
    def _convert_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert new framework inputs to legacy format."""
        pass
    
    @abstractmethod
    def _convert_outputs(self, legacy_result: Any) -> GenerationResult:
        """Convert legacy output to new framework format."""
        pass
    
    def _get_legacy_generator(self) -> Any:
        """Get or create the legacy generator instance."""
        if not self._initialized:
            self._legacy_generator = self._create_legacy_generator()
            self._initialized = True
        return self._legacy_generator
    
    def generate(self, inputs: Dict[str, Any]) -> GenerationResult:
        """Generate using legacy generator with compatibility layer."""
        try:
            # Convert inputs to legacy format
            legacy_inputs = self._convert_inputs(inputs)
            
            # Get legacy generator
            legacy_gen = self._get_legacy_generator()
            
            # Call legacy generator (method varies by generator)
            legacy_result = self._call_legacy_generator(legacy_gen, legacy_inputs)
            
            # Convert result to new format
            return self._convert_outputs(legacy_result)
            
        except Exception as e:
            result = GenerationResult(success=False)
            result.add_error(f"Legacy generator failed: {e}")
            return result
    
    @abstractmethod
    def _call_legacy_generator(self, generator: Any, inputs: Dict[str, Any]) -> Any:
        """Call the legacy generator with converted inputs."""
        pass


class HWCustomOpLegacyAdapter(LegacyGeneratorAdapter):
    """Adapter for legacy HWCustomOpGenerator."""
    
    def get_template_name(self) -> str:
        return "hw_custom_op_slim.py.j2"
    
    def get_artifact_type(self) -> str:
        return "hwcustomop"
    
    def _create_legacy_generator(self) -> Any:
        """Create legacy HWCustomOpGenerator."""
        try:
            from ..generators.hw_custom_op_generator import HWCustomOpGenerator
            template_dir = self.config.template.template_dirs[0] if self.config.template.template_dirs else None
            return HWCustomOpGenerator(template_dir=template_dir)
        except ImportError as e:
            raise ConfigurationError(
                f"Legacy HWCustomOpGenerator not available: {e}",
                suggestion="Check if the legacy generator module exists"
            )
    
    def _convert_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert new inputs to legacy format."""
        legacy_inputs = {}
        
        # Extract HWKernel from RTLModule or direct input
        if "hw_kernel" in inputs:
            legacy_inputs["hw_kernel"] = inputs["hw_kernel"]
        elif "rtl_module" in inputs:
            # Convert RTLModule to HWKernel if needed
            rtl_module = inputs["rtl_module"]
            if isinstance(rtl_module, RTLModule):
                # Create HWKernel from RTLModule
                legacy_inputs["hw_kernel"] = self._rtl_module_to_hw_kernel(rtl_module)
            else:
                legacy_inputs["hw_kernel"] = rtl_module
        
        # Extract output path
        legacy_inputs["output_path"] = inputs.get(
            "output_path", 
            self.config.generation.output_dir / "hw_custom_op.py"
        )
        
        # Extract class name
        legacy_inputs["class_name"] = inputs.get("class_name")
        
        # Extract source file
        legacy_inputs["source_file"] = inputs.get("source_file", "unknown.sv")
        
        return legacy_inputs
    
    def _rtl_module_to_hw_kernel(self, rtl_module: RTLModule) -> Any:
        """Convert RTLModule to HWKernel for legacy compatibility."""
        # Create a minimal HWKernel with required fields
        # This is a simplified conversion - in practice, you'd need
        # more sophisticated mapping based on your data structures
        try:
            # Try to import HWKernel if it exists
            from ..enhanced_data_structures import HWKernel
            return HWKernel(
                module_name=rtl_module.name,
                interfaces=rtl_module.interfaces,
                parameters=rtl_module.parameters
            )
        except (ImportError, AttributeError):
            # If HWKernel doesn't exist or conversion fails,
            # return the RTLModule itself and let the legacy generator handle it
            return rtl_module
    
    def _call_legacy_generator(self, generator: Any, inputs: Dict[str, Any]) -> Any:
        """Call legacy HWCustomOpGenerator."""
        return generator.generate_hwcustomop(
            hw_kernel=inputs["hw_kernel"],
            output_path=inputs["output_path"],
            class_name=inputs.get("class_name"),
            source_file=inputs.get("source_file", "unknown.sv")
        )
    
    def _convert_outputs(self, legacy_result: str) -> GenerationResult:
        """Convert legacy string result to GenerationResult."""
        result = GenerationResult(success=True)
        
        # Create artifact from legacy string result
        artifact = GeneratedArtifact(
            file_name="hw_custom_op.py",
            content=legacy_result,
            artifact_type="hwcustomop",
            metadata={
                "generator": "legacy_hw_custom_op",
                "adapter_used": True
            }
        )
        
        result.add_artifact(artifact)
        return result


class RTLTemplateLegacyAdapter(LegacyGeneratorAdapter):
    """Adapter for legacy RTL template generator."""
    
    def get_template_name(self) -> str:
        return "rtl_wrapper.v.j2"
    
    def get_artifact_type(self) -> str:
        return "wrapper"
    
    def _create_legacy_generator(self) -> Any:
        """Legacy RTL generator is a function, so return a wrapper."""
        try:
            from ..generators.rtl_template_generator import generate_rtl_template
            return generate_rtl_template
        except ImportError as e:
            raise ConfigurationError(
                f"Legacy RTL template generator not available: {e}",
                suggestion="Check if the legacy generator module exists"
            )
    
    def _convert_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert new inputs to legacy format."""
        legacy_inputs = {}
        
        # Extract HWKernel data
        if "hw_kernel_data" in inputs:
            legacy_inputs["hw_kernel_data"] = inputs["hw_kernel_data"]
        elif "rtl_module" in inputs:
            # Convert RTLModule to HWKernel format
            rtl_module = inputs["rtl_module"]
            legacy_inputs["hw_kernel_data"] = self._rtl_module_to_hw_kernel(rtl_module)
        
        # Extract output directory
        legacy_inputs["output_dir"] = inputs.get(
            "output_dir", 
            self.config.generation.output_dir
        )
        
        return legacy_inputs
    
    def _rtl_module_to_hw_kernel(self, rtl_module: RTLModule) -> Any:
        """Convert RTLModule to HWKernel for RTL template."""
        # For RTL template, we might need to create a different format
        # This depends on what the legacy RTL generator expects
        return rtl_module
    
    def _call_legacy_generator(self, generator_func: Any, inputs: Dict[str, Any]) -> Any:
        """Call legacy RTL template function."""
        return generator_func(
            hw_kernel_data=inputs["hw_kernel_data"],
            output_dir=inputs["output_dir"]
        )
    
    def _convert_outputs(self, legacy_result: Path) -> GenerationResult:
        """Convert legacy Path result to GenerationResult."""
        result = GenerationResult(success=True)
        
        try:
            # Read the generated file content
            if legacy_result.exists():
                content = legacy_result.read_text()
                
                artifact = GeneratedArtifact(
                    file_name=legacy_result.name,
                    content=content,
                    artifact_type="wrapper",
                    metadata={
                        "generator": "legacy_rtl_template",
                        "adapter_used": True,
                        "output_path": str(legacy_result)
                    }
                )
                
                result.add_artifact(artifact)
            else:
                result.add_error(f"Legacy generator output file not found: {legacy_result}")
                result.success = False
                
        except Exception as e:
            result.add_error(f"Failed to read legacy generator output: {e}")
            result.success = False
        
        return result


class LegacyGeneratorFactory:
    """Factory for creating legacy generator adapters."""
    
    @staticmethod
    def create_adapter(
        generator_type: GeneratorType, 
        config: PipelineConfig
    ) -> Optional[LegacyGeneratorAdapter]:
        """Create appropriate legacy adapter for generator type."""
        
        adapters = {
            GeneratorType.HW_CUSTOM_OP: HWCustomOpLegacyAdapter,
            GeneratorType.AUTO_HW_CUSTOM_OP: HWCustomOpLegacyAdapter,
            # RTL Backend can use RTL template adapter as fallback
            GeneratorType.RTL_BACKEND: RTLTemplateLegacyAdapter,
            GeneratorType.AUTO_RTL_BACKEND: RTLTemplateLegacyAdapter,
        }
        
        adapter_class = adapters.get(generator_type)
        if adapter_class:
            return adapter_class(config)
        
        return None
    
    @staticmethod
    def get_available_legacy_adapters() -> List[GeneratorType]:
        """Get list of generator types that have legacy adapters."""
        return [
            GeneratorType.HW_CUSTOM_OP,
            GeneratorType.AUTO_HW_CUSTOM_OP,
            GeneratorType.RTL_BACKEND,
            GeneratorType.AUTO_RTL_BACKEND,
        ]


def create_legacy_adapter(
    generator_type: Union[GeneratorType, str], 
    config: PipelineConfig
) -> Optional[GeneratorBase]:
    """
    Convenience function to create legacy adapters.
    
    Args:
        generator_type: Type of generator to create adapter for
        config: Pipeline configuration
        
    Returns:
        Legacy adapter instance or None if not available
    """
    if isinstance(generator_type, str):
        try:
            generator_type = GeneratorType(generator_type)
        except ValueError:
            warnings.warn(f"Unknown generator type: {generator_type}")
            return None
    
    adapter = LegacyGeneratorFactory.create_adapter(generator_type, config)
    
    if adapter:
        warnings.warn(
            f"Using legacy adapter for {generator_type.value}. "
            f"Consider migrating to enhanced generator implementation.",
            DeprecationWarning,
            stacklevel=2
        )
    
    return adapter


# Backward compatibility aliases
HWCustomOpAdapter = HWCustomOpLegacyAdapter
RTLTemplateAdapter = RTLTemplateLegacyAdapter