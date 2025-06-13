"""
Unified generator that replaces all legacy generator classes.

This module provides the UnifiedGenerator class that uses Phase 2 template
context generation exclusively to generate all artifacts for RTL kernels.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from brainsmith.dataflow.core.kernel_metadata import KernelMetadata
from .templates.context_generator import TemplateContextGenerator
from .templates.template_context import TemplateContext

logger = logging.getLogger(__name__)


class UnifiedGeneratorError(Exception):
    """Custom exception for UnifiedGenerator errors."""
    pass


class UnifiedGenerator:
    """
    Unified generator that replaces all legacy generator classes.
    
    Uses Phase 2 template context generation exclusively to create:
    - AutoHWCustomOp subclasses with runtime parameter extraction
    - Enhanced RTL wrappers with parameter validation
    - Test suites with Phase 2 parameter handling
    
    Features:
    - Single interface for all generation tasks
    - Phase 2 template system with validated symbolic BDIM
    - Runtime parameter extraction from ONNX nodes
    - Parameter whitelist validation
    - Enhanced error handling and logging
    """
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize UnifiedGenerator with Phase 2 template system.
        
        Args:
            template_dir: Directory containing Jinja2 templates.
                         If None, uses default template directory.
        """
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        
        self.template_dir = template_dir
        self.template_context_generator = TemplateContextGenerator()
        
        # Initialize Jinja2 environment
        try:
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                trim_blocks=True,
                lstrip_blocks=True
            )
            
            # Add custom filters
            self.jinja_env.filters['repr'] = repr
            
            # Add custom tests
            self.jinja_env.tests['none'] = lambda x: x is None
            logger.info(f"Initialized UnifiedGenerator with templates from {template_dir}")
        except Exception as e:
            raise UnifiedGeneratorError(f"Failed to initialize Jinja2 environment: {e}")
    
    def generate_hw_custom_op(self, kernel_metadata: KernelMetadata) -> str:
        """
        Generate AutoHWCustomOp subclass using Phase 2 template.
        
        Args:
            kernel_metadata: Parsed and validated kernel metadata
            
        Returns:
            Generated Python code for AutoHWCustomOp subclass
            
        Raises:
            UnifiedGeneratorError: If generation fails
        """
        try:
            logger.info(f"Generating HWCustomOp for kernel: {kernel_metadata.name}")
            
            # Generate Phase 2 template context
            template_ctx = self.template_context_generator.generate_template_context(kernel_metadata)
            
            # Validate template context
            validation_errors = template_ctx.validate()
            if validation_errors:
                raise UnifiedGeneratorError(f"Template context validation failed: {validation_errors}")
            
            # Load and render Phase 2 template
            try:
                template = self.jinja_env.get_template("hw_custom_op_phase2.py.j2")
            except TemplateNotFound:
                raise UnifiedGeneratorError(
                    f"Phase 2 template not found: hw_custom_op_phase2.py.j2 in {self.template_dir}"
                )
            
            # Convert template context to dictionary for rendering
            context_dict = self._template_context_to_dict(template_ctx)
            
            # Render template
            generated_code = template.render(**context_dict)
            
            logger.info(f"Successfully generated HWCustomOp for {kernel_metadata.name}")
            return generated_code
            
        except Exception as e:
            logger.error(f"Failed to generate HWCustomOp for {kernel_metadata.name}: {e}")
            raise UnifiedGeneratorError(f"HWCustomOp generation failed: {e}")
    
    def generate_rtl_wrapper(self, kernel_metadata: KernelMetadata) -> str:
        """
        Generate enhanced RTL wrapper with Phase 2 parameter validation.
        
        Args:
            kernel_metadata: Parsed and validated kernel metadata
            
        Returns:
            Generated SystemVerilog RTL wrapper code
            
        Raises:
            UnifiedGeneratorError: If generation fails
        """
        try:
            logger.info(f"Generating RTL wrapper for kernel: {kernel_metadata.name}")
            
            # Generate Phase 2 template context
            template_ctx = self.template_context_generator.generate_template_context(kernel_metadata)
            
            # Load and render enhanced RTL wrapper template
            try:
                template = self.jinja_env.get_template("rtl_wrapper_v2.v.j2")
            except TemplateNotFound:
                # Fallback to legacy template if v2 not available yet
                try:
                    template = self.jinja_env.get_template("rtl_wrapper.v.j2")
                    logger.warning("Using legacy RTL wrapper template - v2 not found")
                except TemplateNotFound:
                    raise UnifiedGeneratorError(
                        f"No RTL wrapper template found in {self.template_dir}"
                    )
            
            # Convert template context to dictionary for rendering
            context_dict = self._template_context_to_dict(template_ctx)
            
            # Render template
            generated_code = template.render(**context_dict)
            
            logger.info(f"Successfully generated RTL wrapper for {kernel_metadata.name}")
            return generated_code
            
        except Exception as e:
            logger.error(f"Failed to generate RTL wrapper for {kernel_metadata.name}: {e}")
            raise UnifiedGeneratorError(f"RTL wrapper generation failed: {e}")
    
    def generate_test_suite(self, kernel_metadata: KernelMetadata) -> str:
        """
        Generate test suite with Phase 2 parameter handling.
        
        Args:
            kernel_metadata: Parsed and validated kernel metadata
            
        Returns:
            Generated Python test suite code
            
        Raises:
            UnifiedGeneratorError: If generation fails
        """
        try:
            logger.info(f"Generating test suite for kernel: {kernel_metadata.name}")
            
            # Generate Phase 2 template context
            template_ctx = self.template_context_generator.generate_template_context(kernel_metadata)
            
            # Load and render enhanced test suite template
            try:
                template = self.jinja_env.get_template("test_suite_v2.py.j2")
            except TemplateNotFound:
                # Fallback to legacy template if v2 not available yet
                try:
                    template = self.jinja_env.get_template("test_suite.py.j2")
                    logger.warning("Using legacy test suite template - v2 not found")
                except TemplateNotFound:
                    raise UnifiedGeneratorError(
                        f"No test suite template found in {self.template_dir}"
                    )
            
            # Convert template context to dictionary for rendering
            context_dict = self._template_context_to_dict(template_ctx)
            
            # Render template
            generated_code = template.render(**context_dict)
            
            logger.info(f"Successfully generated test suite for {kernel_metadata.name}")
            return generated_code
            
        except Exception as e:
            logger.error(f"Failed to generate test suite for {kernel_metadata.name}: {e}")
            raise UnifiedGeneratorError(f"Test suite generation failed: {e}")
    
    def generate_all(self, kernel_metadata: KernelMetadata) -> Dict[str, str]:
        """
        Generate all artifacts for a kernel.
        
        Args:
            kernel_metadata: Parsed and validated kernel metadata
            
        Returns:
            Dictionary mapping filenames to generated content
            
        Raises:
            UnifiedGeneratorError: If any generation fails
        """
        try:
            logger.info(f"Generating all artifacts for kernel: {kernel_metadata.name}")
            
            results = {}
            
            # Generate HWCustomOp (required)
            results[f"{kernel_metadata.name}_hw_custom_op.py"] = self.generate_hw_custom_op(kernel_metadata)
            
            # Generate RTL wrapper (optional - may fail if template missing)
            try:
                results[f"{kernel_metadata.name}_wrapper.v"] = self.generate_rtl_wrapper(kernel_metadata)
            except UnifiedGeneratorError as e:
                logger.warning(f"RTL wrapper generation failed: {e}")
            
            # Generate test suite (optional - may fail if template missing)
            try:
                results[f"test_{kernel_metadata.name}.py"] = self.generate_test_suite(kernel_metadata)
            except UnifiedGeneratorError as e:
                logger.warning(f"Test suite generation failed: {e}")
            
            logger.info(f"Successfully generated {len(results)} artifacts for {kernel_metadata.name}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate artifacts for {kernel_metadata.name}: {e}")
            raise UnifiedGeneratorError(f"Artifact generation failed: {e}")
    
    def _template_context_to_dict(self, template_ctx: TemplateContext) -> Dict[str, any]:
        """
        Convert TemplateContext to dictionary for Jinja2 rendering.
        
        Args:
            template_ctx: Template context object
            
        Returns:
            Dictionary suitable for Jinja2 template rendering
        """
        # Use the existing conversion method from TemplateContextGenerator
        return self.template_context_generator._template_context_to_dict(template_ctx)
