"""
Kernel integrator for KI using the modular generator system.

This module provides the KernelIntegrator class that orchestrates the generation
workflow using the new GeneratorManager and discoverable generator system.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from .types.metadata import KernelMetadata
from .types.generation import GenerationResult, GeneratedFile
from .generators import GeneratorManager
from .templates.context_generator import TemplateContextGenerator
from .templates.template_context import TemplateContext

logger = logging.getLogger(__name__)


class KernelIntegratorError(Exception):
    """Custom exception for KernelIntegrator errors."""
    pass


class KernelIntegrator:
    """
    Orchestrates RTL kernel artifact generation using modular generator system.
    
    Uses the new GeneratorManager for discoverable, extensible generation:
    - AutoHWCustomOp subclasses with runtime parameter extraction
    - Enhanced RTL wrappers with parameter validation
    - RTL backend with Phase 2 parameter handling
    - Extensible via additional generators
    
    Key Features:
    - Modular generator system with auto-discovery
    - Single clean workflow: generate_and_write()
    - Integrated file system management
    - Phase 2 template system with validated symbolic BDIM
    - Enhanced error handling and logging
    
    Primary Interface:
    - generate_and_write(): Complete generation and file writing in one call
    """
    
    def __init__(
        self, 
        generator_dir: Optional[Path] = None,
        template_dir: Optional[Path] = None, 
        output_dir: Optional[Path] = None,
        use_direct_generators: bool = False
    ):
        """
        Initialize KernelIntegrator with modular generator system.
        
        Args:
            generator_dir: Directory containing generator Python files.
                          If None, uses default generators directory.
            template_dir: Directory containing Jinja2 templates.
                         If None, uses default template directory.
            output_dir: Base directory for generated files.
                       If None, file writing capabilities are disabled until set.
            use_direct_generators: Use new direct generators (V2) that bypass TemplateContext.
        """
        # Set up default directories
        if generator_dir is None:
            generator_dir = Path(__file__).parent / "generators"
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        
        self.generator_dir = generator_dir
        self.template_dir = template_dir
        self.output_dir = output_dir
        self.use_direct_generators = use_direct_generators
        
        # Initialize components
        try:
            if use_direct_generators:
                # Initialize direct generators (V2)
                self.direct_generators = self._load_direct_generators()
                logger.info(f"Initialized KernelIntegrator with {len(self.direct_generators)} direct generators")
            else:
                # Initialize legacy system
                self.generator_manager = GeneratorManager(generator_dir, template_dir)
                self.context_generator = TemplateContextGenerator()
                logger.info(f"Initialized KernelIntegrator with {len(self.generator_manager.list_generators())} generators")
            
            # Initialize output directory if provided
            if self.output_dir is not None:
                self._ensure_output_directory()
                
        except Exception as e:
            raise KernelIntegratorError(f"Failed to initialize KernelIntegrator: {e}")
    
    def _load_direct_generators(self) -> Dict[str, 'GeneratorBase']:
        """Load direct generators (V2) that bypass TemplateContext."""
        from .generators.hw_custom_op_v2 import HWCustomOpGeneratorV2
        from .generators.rtl_backend_v2 import RTLBackendGeneratorV2
        from .generators.rtl_wrapper_v2 import RTLWrapperGeneratorV2
        
        generators = {}
        
        # Instantiate generators with template directory
        hw_gen = HWCustomOpGeneratorV2(self.template_dir)
        rtl_gen = RTLBackendGeneratorV2(self.template_dir)
        wrapper_gen = RTLWrapperGeneratorV2(self.template_dir)
        
        # Register by name
        generators[hw_gen.name] = hw_gen
        generators[rtl_gen.name] = rtl_gen
        generators[wrapper_gen.name] = wrapper_gen
        
        return generators
    
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
            raise KernelIntegratorError(f"No write permission for output directory: {self.output_dir}")
        except Exception as e:
            raise KernelIntegratorError(f"Failed to create output directory {self.output_dir}: {e}")
    
    def _create_kernel_directory(self, kernel_name: str) -> Path:
        """
        Create kernel-specific directory.
        
        Args:
            kernel_name: Name of the kernel for directory naming
            
        Returns:
            Path to the created kernel directory
            
        Raises:
            KernelIntegratorError: If output_dir not set or directory creation fails
        """
        if not self.output_dir:
            raise KernelIntegratorError("Output directory not configured. Set output_dir in constructor or call set_output_dir()")
        
        kernel_dir = self.output_dir / kernel_name
        try:
            kernel_dir.mkdir(exist_ok=True)
            return kernel_dir
        except Exception as e:
            raise KernelIntegratorError(f"Failed to create kernel directory {kernel_dir}: {e}")
    
    def set_output_dir(self, output_dir: Path) -> None:
        """
        Set output directory after initialization.
        
        Args:
            output_dir: Path to output directory
        """
        self.output_dir = output_dir
        self._ensure_output_directory()
        logger.info(f"Output directory set to: {self.output_dir}")
    
    def generate_and_write(
        self, 
        kernel_metadata: KernelMetadata,
        write_files: bool = True,
        include_generators: Optional[List[str]] = None,
        output_structure: str = "hierarchical"
    ) -> GenerationResult:
        """
        Generate all artifacts and optionally write to filesystem.
        
        This is the primary method for kernel integration, combining
        code generation and file writing in a single operation using
        the modular generator system.
        
        Args:
            kernel_metadata: Parsed and validated kernel metadata
            write_files: Whether to write files to filesystem (default: True)
            include_generators: List of generator names to use (default: all)
            output_structure: Directory structure ("hierarchical" or "flat")
            
        Returns:
            GenerationResult with complete file tracking
            
        Raises:
            KernelIntegratorError: If generation or file writing fails
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting kernel integration for: {kernel_metadata.name}")
            
            # Create GenerationResult to track everything
            result = GenerationResult()
            
            if self.use_direct_generators:
                # Direct path - no TemplateContext
                generated_files = self._generate_direct(
                    kernel_metadata, include_generators
                )
            else:
                # Legacy path with TemplateContext
                # Generate template context
                template_ctx = self.context_generator.generate_template_context(kernel_metadata)
                
                # Validate template context
                validation_errors = template_ctx.validate()
                if validation_errors:
                    for error in validation_errors:
                        result.add_error(f"Template context validation: {error}")
                    return result
                
                # Generate selected artifacts using GeneratorManager
                generated_files = self._generate_selected_artifacts(
                    kernel_metadata, template_ctx, include_generators
                )
            
            # Add generated files to result
            for filename, content in generated_files.items():
                generated_file = GeneratedFile(
                    path=Path(filename),
                    content=content,
                    description=f"Generated for {kernel_metadata.name}"
                )
                result.add_file(generated_file)
            
            # Write files if requested
            if write_files:
                if not self.output_dir:
                    result.add_error("Output directory not configured - cannot write files")
                else:
                    try:
                        output_dir = self._determine_output_directory(kernel_metadata.name, output_structure)
                        # Write files to output directory
                        for generated_file in result.generated_files:
                            output_path = output_dir / generated_file.path
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            output_path.write_text(generated_file.content)
                        written_files = [output_dir / f.path for f in result.generated_files]
                        logger.info(f"Successfully wrote {len(written_files)} files to {output_dir}")
                    except Exception as e:
                        result.add_error(f"File writing failed: {e}")
            
            # Record timing
            generation_time = (time.time() - start_time) * 1000
            result.generation_time_ms = generation_time
            
            if result.is_success:
                logger.info(f"Successfully completed kernel integration for {kernel_metadata.name} in {generation_time:.1f}ms")
            else:
                logger.error(f"Kernel integration failed for {kernel_metadata.name}: {result.errors}")
            
            return result
            
        except Exception as e:
            logger.error(f"Kernel integration failed for {kernel_metadata.name}: {e}")
            raise KernelIntegratorError(f"Kernel integration failed: {e}")
    
    def _generate_selected_artifacts(
        self, 
        kernel_metadata: KernelMetadata, 
        template_ctx: TemplateContext,
        include_generators: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Generate selected artifacts using GeneratorManager.
        
        Args:
            kernel_metadata: Kernel metadata
            template_ctx: Template context
            include_generators: List of generator names to include (None = all)
            
        Returns:
            Dictionary of generated files
        """
        generated_files = {}
        
        # Get generators to process
        available_generators = self.generator_manager.list_generators()
        generators_to_use = include_generators or available_generators
        
        for generator_name in generators_to_use:
            if generator_name not in available_generators:
                logger.warning(f"Unknown generator: {generator_name}")
                continue
                
            try:
                # Render using GeneratorManager
                content = self.generator_manager.render_generator(generator_name, template_ctx)
                filename = self.generator_manager.get_output_filename(generator_name, kernel_metadata.name)
                
                generated_files[filename] = content
                logger.debug(f"Generated {generator_name}: {filename}")
                
            except Exception as e:
                logger.warning(f"Failed to generate {generator_name}: {e}")
                # Continue with other generators
        
        return generated_files
    
    def _generate_direct(
        self,
        kernel_metadata: KernelMetadata,
        include_generators: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Generate artifacts using direct generators (V2).
        
        Args:
            kernel_metadata: Kernel metadata
            include_generators: List of generator names to include (None = all)
            
        Returns:
            Dictionary of generated files
        """
        generated_files = {}
        
        # Get generators to process
        available_generators = list(self.direct_generators.keys())
        generators_to_use = include_generators or available_generators
        
        for generator_name in generators_to_use:
            if generator_name not in available_generators:
                logger.warning(f"Unknown generator: {generator_name}")
                continue
                
            try:
                # Get generator
                generator = self.direct_generators[generator_name]
                
                # Generate content directly from metadata
                content = generator.generate(kernel_metadata)
                filename = generator.get_output_filename(kernel_metadata.name)
                
                generated_files[filename] = content
                logger.debug(f"Generated {generator_name}: {filename}")
                
            except Exception as e:
                logger.warning(f"Failed to generate {generator_name}: {e}")
                # Continue with other generators
        
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
        # Always use flat structure - output directly to specified directory
        return self.output_dir
    
    def list_generators(self) -> List[str]:
        """
        Get list of available generator names.
        
        Returns:
            List of generator names
        """
        if self.use_direct_generators:
            return list(self.direct_generators.keys())
        else:
            return self.generator_manager.list_generators()
    
    def validate_generators(self) -> Dict[str, bool]:
        """
        Validate that all generator templates exist.
        
        Returns:
            Dictionary mapping generator names to template availability status
        """
        return self.generator_manager.validate_templates()
    
    def get_generator_info(self) -> Dict[str, Dict]:
        """
        Get information about all available generators.
        
        Returns:
            Dictionary mapping generator names to their info
        """
        info = {}
        for name in self.generator_manager.list_generators():
            generator = self.generator_manager.get_generator(name)
            if generator:
                info[name] = {
                    "template_file": generator.template_file,
                    "output_pattern": generator.output_pattern,
                    "class_name": generator.__class__.__name__
                }
        return info