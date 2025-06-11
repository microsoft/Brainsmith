"""
Unified Hardware Kernel Generator.

This module implements the main UnifiedHWKGGenerator class that combines
RTL parsing with Interface-Wise Dataflow Modeling to generate complete
HWCustomOp and RTLBackend implementations using DataflowModel-powered
instantiation instead of complex template generation.

This replaces the old template-heavy approach with a cleaner instantiation
model where templates simply instantiate AutoHWCustomOp with DataflowModel
instances rather than generating all implementation code.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ...dataflow.rtl_integration import RTLDataflowConverter
from ..hw_kernel_gen.rtl_parser import parse_rtl_file
from ..hw_kernel_gen.data import GenerationResult
from .template_system import create_template_system

logger = logging.getLogger(__name__)


@dataclass
class UnifiedGenerationResult:
    """Result of unified generation process."""
    success: bool
    generated_files: List[Path] = None
    dataflow_model: Optional[Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    performance_analysis: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.generated_files is None:
            self.generated_files = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class UnifiedHWKGGenerator:
    """
    Generate AutoHWCustomOp/AutoRTLBackend instances from DataflowModel.
    
    This class implements the unified HWKG approach where:
    1. RTL is parsed to RTLParsingResult
    2. RTLParsingResult is converted to DataflowModel  
    3. Templates instantiate AutoHWCustomOp/AutoRTLBackend with DataflowModel
    4. Generated code uses mathematical foundation instead of placeholders
    
    This eliminates the need for complex template generation while providing
    the same functionality through the inheritance-based approach.
    """
    
    def __init__(self):
        """Initialize unified HWKG generator."""
        self.rtl_converter = RTLDataflowConverter()
        self.template_loader, self.context_builder = create_template_system()
        
    def generate_from_rtl(self, rtl_file: Path, compiler_data: Dict[str, Any], 
                         output_dir: Path, **kwargs) -> UnifiedGenerationResult:
        """
        Complete generation pipeline from RTL file.
        
        Args:
            rtl_file: Path to SystemVerilog RTL file
            compiler_data: Compiler configuration data
            output_dir: Output directory for generated files
            **kwargs: Additional generation options
            
        Returns:
            UnifiedGenerationResult: Complete generation results
        """
        try:
            logger.info(f"Starting unified generation for RTL file: {rtl_file}")
            
            # Step 1: Parse RTL file using existing RTL parser
            rtl_result = parse_rtl_file(rtl_file)
            if not rtl_result:
                return UnifiedGenerationResult(
                    success=False,
                    errors=[f"Failed to parse RTL file: {rtl_file}"]
                )
            
            # Step 2: Convert RTLParsingResult to DataflowModel
            conversion_result = self.rtl_converter.convert(rtl_result)
            if not conversion_result.success:
                return UnifiedGenerationResult(
                    success=False,
                    errors=[f"RTL to DataflowModel conversion failed"] + conversion_result.errors,
                    warnings=conversion_result.warnings
                )
            
            dataflow_model = conversion_result.dataflow_model
            logger.info(f"Successfully created DataflowModel for kernel: {rtl_result.name}")
            
            # Step 3: Generate files using DataflowModel
            generation_result = self._generate_files(
                rtl_result, dataflow_model, output_dir, **kwargs
            )
            
            return generation_result
            
        except Exception as e:
            error_msg = f"Unexpected error during unified generation: {str(e)}"
            logger.error(error_msg)
            return UnifiedGenerationResult(
                success=False,
                errors=[error_msg]
            )
    
    def generate_hwcustomop(self, dataflow_model, kernel_name: str, 
                           output_dir: Path) -> Optional[Path]:
        """
        Generate HWCustomOp instantiation file.
        
        Args:
            dataflow_model: DataflowModel instance
            kernel_name: Name of the kernel
            output_dir: Output directory
            
        Returns:
            Path to generated HWCustomOp file or None if failed
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate class name
            from ...dataflow.core.class_naming import generate_class_name
            class_name = generate_class_name(kernel_name)
            
            # Build template context using context builder
            context = self.context_builder.build_hwcustomop_context(dataflow_model, kernel_name)
            
            # Load and render template
            rendered_content = self.template_loader.render_template(
                "hwcustomop_instantiation.py.j2", context
            )
            
            # Write generated file
            output_file = output_dir / f"{kernel_name}_hwcustomop.py"
            with open(output_file, 'w') as f:
                f.write(rendered_content)
            
            logger.info(f"Generated HWCustomOp file: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to generate HWCustomOp: {str(e)}")
            return None
    
    def generate_rtlbackend(self, dataflow_model, kernel_name: str, 
                           output_dir: Path) -> Optional[Path]:
        """
        Generate RTLBackend instantiation file.
        
        Args:
            dataflow_model: DataflowModel instance
            kernel_name: Name of the kernel
            output_dir: Output directory
            
        Returns:
            Path to generated RTLBackend file or None if failed
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate class name
            from ...dataflow.core.class_naming import generate_backend_class_name
            class_name = generate_backend_class_name(kernel_name)
            
            # Build template context using context builder
            context = self.context_builder.build_rtlbackend_context(dataflow_model, kernel_name)
            
            # Load and render template
            rendered_content = self.template_loader.render_template(
                "rtlbackend_instantiation.py.j2", context
            )
            
            # Write generated file
            output_file = output_dir / f"{kernel_name}_rtlbackend.py"
            with open(output_file, 'w') as f:
                f.write(rendered_content)
            
            logger.info(f"Generated RTLBackend file: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to generate RTLBackend: {str(e)}")
            return None
    
    def generate_test_suite(self, dataflow_model, kernel_name: str, 
                           output_dir: Path) -> Optional[Path]:
        """
        Generate test suite for the kernel.
        
        Args:
            dataflow_model: DataflowModel instance
            kernel_name: Name of the kernel
            output_dir: Output directory
            
        Returns:
            Path to generated test file or None if failed
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Build template context using context builder
            context = self.context_builder.build_test_context(dataflow_model, kernel_name)
            
            # Load and render template
            rendered_content = self.template_loader.render_template(
                "test_suite.py.j2", context
            )
            
            # Write generated file
            output_file = output_dir / f"test_{kernel_name}.py"
            with open(output_file, 'w') as f:
                f.write(rendered_content)
            
            logger.info(f"Generated test file: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to generate test suite: {str(e)}")
            return None
    
    def _generate_files(self, rtl_result, dataflow_model, output_dir: Path, 
                       **kwargs) -> UnifiedGenerationResult:
        """
        Generate all files for the kernel.
        
        Args:
            rtl_result: RTLParsingResult from RTL parsing
            dataflow_model: DataflowModel instance
            output_dir: Output directory
            **kwargs: Additional options
            
        Returns:
            UnifiedGenerationResult: Generation results
        """
        generated_files = []
        errors = []
        warnings = []
        
        try:
            kernel_name = rtl_result.name
            
            # Generate HWCustomOp
            hwcustomop_file = self.generate_hwcustomop(dataflow_model, kernel_name, output_dir)
            if hwcustomop_file:
                generated_files.append(hwcustomop_file)
            else:
                errors.append("Failed to generate HWCustomOp file")
            
            # Generate RTLBackend
            rtlbackend_file = self.generate_rtlbackend(dataflow_model, kernel_name, output_dir)
            if rtlbackend_file:
                generated_files.append(rtlbackend_file)
            else:
                errors.append("Failed to generate RTLBackend file")
            
            # Generate test suite
            test_file = self.generate_test_suite(dataflow_model, kernel_name, output_dir)
            if test_file:
                generated_files.append(test_file)
            else:
                warnings.append("Failed to generate test suite (non-critical)")
            
            # Generate documentation if requested
            if kwargs.get('generate_docs', False):
                doc_file = self._generate_documentation(dataflow_model, kernel_name, output_dir)
                if doc_file:
                    generated_files.append(doc_file)
                else:
                    warnings.append("Failed to generate documentation (non-critical)")
            
            success = len(generated_files) >= 2  # At least HWCustomOp and RTLBackend
            
            return UnifiedGenerationResult(
                success=success,
                generated_files=generated_files,
                dataflow_model=dataflow_model,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            error_msg = f"Error during file generation: {str(e)}"
            logger.error(error_msg)
            return UnifiedGenerationResult(
                success=False,
                generated_files=generated_files,
                dataflow_model=dataflow_model,
                errors=errors + [error_msg],
                warnings=warnings
            )
    
    # Context building is now handled by DataflowContextBuilder in template_system.py
    
    # Template system is now handled by UnifiedTemplateLoader and DataflowContextBuilder
    
    # Helper methods are now handled by DataflowContextBuilder
    
    def _generate_documentation(self, dataflow_model, kernel_name: str, 
                               output_dir: Path) -> Optional[Path]:
        """Generate documentation file."""
        # Placeholder - will be implemented
        return None


def create_unified_generator() -> UnifiedHWKGGenerator:
    """
    Factory function for creating UnifiedHWKGGenerator instances.
    
    Returns:
        UnifiedHWKGGenerator: Configured generator instance
    """
    return UnifiedHWKGGenerator()