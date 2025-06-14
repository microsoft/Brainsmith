"""
Unified generator that replaces all legacy generator classes.

This module provides the UnifiedGenerator class that uses Phase 2 template
context generation exclusively to generate all artifacts for RTL kernels.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from brainsmith.dataflow.core.kernel_metadata import KernelMetadata
from .data import GenerationResult
from .templates.context_generator import TemplateContextGenerator
from .templates.template_context import TemplateContext

logger = logging.getLogger(__name__)


class UnifiedGeneratorError(Exception):
    """Custom exception for UnifiedGenerator errors."""
    pass


class UnifiedGenerator:
    """
    Clean, single-workflow generator for RTL kernel artifacts.
    
    Uses Phase 2 template context generation exclusively to create:
    - AutoHWCustomOp subclasses with runtime parameter extraction
    - Enhanced RTL wrappers with parameter validation
    - Test suites with Phase 2 parameter handling
    
    Key Features:
    - Single clean workflow: generate_and_write()
    - Integrated file system management
    - Phase 2 template system with validated symbolic BDIM
    - Runtime parameter extraction from ONNX nodes
    - Parameter whitelist validation
    - Enhanced error handling and logging
    
    Primary Interface:
    - generate_and_write(): Complete generation and file writing in one call
    """
    
    def __init__(self, template_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
        """
        Initialize UnifiedGenerator with Phase 2 template system and optional file output.
        
        Args:
            template_dir: Directory containing Jinja2 templates.
                         If None, uses default template directory.
            output_dir: Base directory for generated files.
                       If None, file writing capabilities are disabled until set.
        """
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        
        self.template_dir = template_dir
        self.output_dir = output_dir
        self.template_context_generator = TemplateContextGenerator()
        
        # Initialize output directory if provided
        if self.output_dir is not None:
            self._ensure_output_directory()
        
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
            if self.output_dir:
                logger.info(f"Output directory configured: {self.output_dir}")
        except Exception as e:
            raise UnifiedGeneratorError(f"Failed to initialize Jinja2 environment: {e}")
    
    def _generate_hw_custom_op(self, kernel_metadata: KernelMetadata, template_ctx: TemplateContext) -> str:
        """
        Internal method to generate AutoHWCustomOp subclass using Phase 2 template.
        
        Args:
            kernel_metadata: Parsed and validated kernel metadata
            template_ctx: Pre-validated template context
            
        Returns:
            Generated Python code for AutoHWCustomOp subclass
            
        Raises:
            UnifiedGeneratorError: If generation fails
        """
        try:
            logger.debug(f"Generating HWCustomOp for kernel: {kernel_metadata.name}")
            
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
            
            logger.debug(f"Successfully generated HWCustomOp for {kernel_metadata.name}")
            return generated_code
            
        except Exception as e:
            logger.error(f"Failed to generate HWCustomOp for {kernel_metadata.name}: {e}")
            raise UnifiedGeneratorError(f"HWCustomOp generation failed: {e}")
    
    def _generate_rtl_wrapper(self, kernel_metadata: KernelMetadata, template_ctx: TemplateContext) -> str:
        """
        Internal method to generate enhanced RTL wrapper with Phase 2 parameter validation.
        
        Args:
            kernel_metadata: Parsed and validated kernel metadata
            template_ctx: Pre-validated template context
            
        Returns:
            Generated SystemVerilog RTL wrapper code
            
        Raises:
            UnifiedGeneratorError: If generation fails
        """
        try:
            logger.debug(f"Generating RTL wrapper for kernel: {kernel_metadata.name}")
            
            # Load and render minimal RTL wrapper template
            try:
                template = self.jinja_env.get_template("rtl_wrapper_minimal.v.j2")
            except TemplateNotFound:
                # Fallback to v2 template
                try:
                    template = self.jinja_env.get_template("rtl_wrapper_v2.v.j2")
                    logger.warning("Using v2 RTL wrapper template - minimal not found")
                except TemplateNotFound:
                    # Fallback to legacy template
                    try:
                        template = self.jinja_env.get_template("rtl_wrapper.v.j2")
                        logger.warning("Using legacy RTL wrapper template - v2 and minimal not found")
                    except TemplateNotFound:
                        raise UnifiedGeneratorError(
                            f"No RTL wrapper template found in {self.template_dir}"
                        )
            
            # Convert template context to dictionary for rendering
            context_dict = self._template_context_to_dict(template_ctx)
            
            # Render template
            generated_code = template.render(**context_dict)
            
            logger.debug(f"Successfully generated RTL wrapper for {kernel_metadata.name}")
            return generated_code
            
        except Exception as e:
            logger.error(f"Failed to generate RTL wrapper for {kernel_metadata.name}: {e}")
            raise UnifiedGeneratorError(f"RTL wrapper generation failed: {e}")
    
    def _generate_test_suite(self, kernel_metadata: KernelMetadata, template_ctx: TemplateContext) -> str:
        """
        Internal method to generate test suite with Phase 2 parameter handling.
        
        Args:
            kernel_metadata: Parsed and validated kernel metadata
            template_ctx: Pre-validated template context
            
        Returns:
            Generated Python test suite code
            
        Raises:
            UnifiedGeneratorError: If generation fails
        """
        try:
            logger.debug(f"Generating test suite for kernel: {kernel_metadata.name}")
            
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
            
            logger.debug(f"Successfully generated test suite for {kernel_metadata.name}")
            return generated_code
            
        except Exception as e:
            logger.error(f"Failed to generate test suite for {kernel_metadata.name}: {e}")
            raise UnifiedGeneratorError(f"Test suite generation failed: {e}")
    
    
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
    
    # ===== File System Management (Phase 3/4 Integration) =====
    
    def _ensure_output_directory(self) -> None:
        """Ensure output directory exists and is writable."""
        if not self.output_dir:
            return
            
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = self.output_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            
        except PermissionError:
            raise UnifiedGeneratorError(f"No write permission for output directory: {self.output_dir}")
        except Exception as e:
            raise UnifiedGeneratorError(f"Failed to create output directory {self.output_dir}: {e}")
    
    def _create_kernel_directory(self, kernel_name: str) -> Path:
        """
        Create kernel-specific directory.
        
        Args:
            kernel_name: Name of the kernel for directory naming
            
        Returns:
            Path to the created kernel directory
            
        Raises:
            UnifiedGeneratorError: If output_dir not set or directory creation fails
        """
        if not self.output_dir:
            raise UnifiedGeneratorError("Output directory not configured. Set output_dir in constructor or call set_output_dir()")
        
        kernel_dir = self.output_dir / kernel_name
        try:
            kernel_dir.mkdir(exist_ok=True)
            return kernel_dir
        except Exception as e:
            raise UnifiedGeneratorError(f"Failed to create kernel directory {kernel_dir}: {e}")
    
    def set_output_dir(self, output_dir: Path) -> None:
        """
        Set output directory after initialization.
        
        Args:
            output_dir: Path to output directory
        """
        self.output_dir = output_dir
        self._ensure_output_directory()
        logger.info(f"Output directory set to: {self.output_dir}")
    
    # ===== Primary Integration Methods (Phase 3/4 Integration) =====
    
    def generate_and_write(
        self, 
        kernel_metadata: KernelMetadata,
        write_files: bool = True,
        include_templates: Optional[List[str]] = None,
        output_structure: str = "hierarchical"
    ) -> GenerationResult:
        """
        Generate all artifacts and optionally write to filesystem.
        
        This is the primary method for Phase 3/4 integration, combining
        code generation and file writing in a single operation.
        
        Args:
            kernel_metadata: Parsed and validated kernel metadata
            write_files: Whether to write files to filesystem (default: True)
            include_templates: List of template types to generate (default: all)
            output_structure: Directory structure ("hierarchical" or "flat")
            
        Returns:
            GenerationResult with complete file tracking
            
        Raises:
            UnifiedGeneratorError: If generation or file writing fails
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting integrated generation for kernel: {kernel_metadata.name}")
            
            # Create GenerationResult to track everything
            result = GenerationResult(
                kernel_name=kernel_metadata.name,
                source_file=kernel_metadata.source_file,
                kernel_metadata=kernel_metadata
            )
            
            # Generate template context
            template_ctx = self.template_context_generator.generate_template_context(kernel_metadata)
            result.template_context = template_ctx
            
            # Validate template context
            validation_errors = template_ctx.validate()
            if validation_errors:
                for error in validation_errors:
                    result.add_error(f"Template context validation: {error}")
                return result
            
            # Generate selected templates
            generated_files = self._generate_selected_templates(
                kernel_metadata, template_ctx, include_templates
            )
            
            # Add generated files to result
            for filename, content in generated_files.items():
                result.add_generated_file(filename, content)
            
            # Write files if requested
            if write_files:
                if not self.output_dir:
                    result.add_error("Output directory not configured - cannot write files")
                else:
                    try:
                        output_dir = self._determine_output_directory(kernel_metadata.name, output_structure)
                        written_files = result.write_all_files(output_dir)
                        logger.info(f"Successfully wrote {len(written_files)} files to {output_dir}")
                    except Exception as e:
                        result.add_error(f"File writing failed: {e}")
            
            # Record timing
            generation_time = (time.time() - start_time) * 1000
            result.generation_time_ms = generation_time
            
            if result.is_success():
                logger.info(f"Successfully completed integrated generation for {kernel_metadata.name} in {generation_time:.1f}ms")
            else:
                logger.error(f"Generation failed for {kernel_metadata.name}: {result.errors}")
            
            return result
            
        except Exception as e:
            logger.error(f"Integrated generation failed for {kernel_metadata.name}: {e}")
            raise UnifiedGeneratorError(f"Integrated generation failed: {e}")
    
    
    def _generate_selected_templates(
        self, 
        kernel_metadata: KernelMetadata, 
        template_ctx: TemplateContext,
        include_templates: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Generate selected templates based on include_templates filter.
        
        Args:
            kernel_metadata: Kernel metadata
            template_ctx: Template context
            include_templates: List of template names to include (None = all)
            
        Returns:
            Dictionary of generated files
        """
        generated_files = {}
        
        # Define available templates
        available_templates = {
            "hw_custom_op": lambda: self._generate_hw_custom_op(kernel_metadata, template_ctx),
            "rtl_wrapper": lambda: self._generate_rtl_wrapper(kernel_metadata, template_ctx),
            "test_suite": lambda: self._generate_test_suite(kernel_metadata, template_ctx)
        }
        
        # Define corresponding filenames
        filenames = {
            "hw_custom_op": f"{kernel_metadata.name}_hw_custom_op.py",
            "rtl_wrapper": f"{kernel_metadata.name}_wrapper.v",
            "test_suite": f"test_{kernel_metadata.name}.py"
        }
        
        # Generate requested templates
        templates_to_generate = include_templates or list(available_templates.keys())
        
        for template_name in templates_to_generate:
            if template_name not in available_templates:
                logger.warning(f"Unknown template type: {template_name}")
                continue
                
            try:
                content = available_templates[template_name]()
                filename = filenames[template_name]
                generated_files[filename] = content
                logger.debug(f"Generated {template_name}: {filename}")
            except Exception as e:
                logger.warning(f"Failed to generate {template_name}: {e}")
                # Continue with other templates
        
        return generated_files
    
    def _determine_output_directory(self, kernel_name: str, output_structure: str) -> Path:
        """
        Determine output directory based on structure preference.
        
        Args:
            kernel_name: Name of the kernel
            output_structure: "hierarchical" or "flat"
            
        Returns:
            Path to output directory for this kernel
        """
        if output_structure == "flat":
            return self.output_dir
        else:  # hierarchical (default)
            return self._create_kernel_directory(kernel_name)
    
    # ===== Template Management (Enhanced) =====
    
    def get_available_templates(self) -> List[str]:
        """
        Get list of available template files.
        
        Returns:
            List of template filenames found in template directory
        """
        try:
            template_files = []
            for template_file in self.template_dir.glob("*.j2"):
                template_files.append(template_file.name)
            return sorted(template_files)
        except Exception as e:
            logger.warning(f"Failed to list template files: {e}")
            return []
    
    def validate_templates(self) -> Dict[str, bool]:
        """
        Validate that required templates are available.
        
        Returns:
            Dictionary mapping template names to availability status
        """
        required_templates = [
            "hw_custom_op_phase2.py.j2",
            "rtl_wrapper_v2.v.j2", 
            "test_suite_v2.py.j2"
        ]
        
        fallback_templates = [
            "rtl_wrapper.v.j2",
            "test_suite.py.j2"
        ]
        
        status = {}
        
        # Check required templates
        for template in required_templates:
            template_path = self.template_dir / template
            status[template] = template_path.exists()
        
        # Check fallback templates
        for template in fallback_templates:
            template_path = self.template_dir / template
            status[template] = template_path.exists()
        
        return status
