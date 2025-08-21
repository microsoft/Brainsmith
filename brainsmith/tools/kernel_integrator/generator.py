"""
Code generator for Brainsmith kernels.

This module provides a generator that works with KernelMetadata
objects to generate FINN-compatible kernel implementations.
"""

from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

from brainsmith.tools.kernel_integrator.metadata import KernelMetadata


class KernelGenerator:
    """
    Code generator for Brainsmith kernels.
    
    Generates AutoHWCustomOp, RTLBackend, and RTL wrapper code
    from KernelMetadata objects.
    """
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize generator with template directory.
        
        Args:
            template_dir: Directory containing Jinja2 templates. 
                         Defaults to kernel_integrator/templates/
        """
        if template_dir is None:
            # Default to templates directory relative to this file
            template_dir = Path(__file__).parent / "templates"
        
        self.template_dir = Path(template_dir)
        
        # Setup Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
    
    def generate_autohwcustomop(self, kernel_metadata: KernelMetadata) -> str:
        """
        Generate AutoHWCustomOp code from KernelMetadata.
        
        Args:
            kernel_metadata: Complete kernel metadata object
            
        Returns:
            Generated Python code as string
            
        Raises:
            FileNotFoundError: If template file not found
            Exception: If template rendering fails
        """
        try:
            template = self.jinja_env.get_template('auto_hw_custom_op.py.j2')
            return template.render(kernel_metadata=kernel_metadata)
        except Exception as e:
            raise RuntimeError(f"Failed to generate AutoHWCustomOp code: {e}") from e
    
    def generate_autohwcustomop_to_file(self, kernel_metadata: KernelMetadata, output_path: Path) -> None:
        """
        Generate AutoHWCustomOp code and write to file.
        
        Args:
            kernel_metadata: Complete kernel metadata object
            output_path: Path where generated code should be written
        """
        code = self.generate_autohwcustomop(kernel_metadata)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write generated code
        output_path.write_text(code, encoding='utf-8')
    
    # Backward compatibility aliases
    def generate(self, kernel_metadata: KernelMetadata) -> str:
        """Backward compatibility alias for generate_autohwcustomop."""
        return self.generate_autohwcustomop(kernel_metadata)
    
    def generate_to_file(self, kernel_metadata: KernelMetadata, output_path: Path) -> None:
        """Backward compatibility alias for generate_autohwcustomop_to_file."""
        self.generate_autohwcustomop_to_file(kernel_metadata, output_path)
    
    def generate_rtl_wrapper(self, kernel_metadata: KernelMetadata) -> str:
        """
        Generate RTL wrapper from KernelMetadata.
        
        Args:
            kernel_metadata: Complete kernel metadata object
            
        Returns:
            Generated SystemVerilog wrapper code as string
            
        Raises:
            FileNotFoundError: If template file not found
            Exception: If template rendering fails
        """
        try:
            template = self.jinja_env.get_template('rtl_wrapper.v.j2')
            return template.render(
                kernel_metadata=kernel_metadata,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate RTL wrapper: {e}") from e
    
    def generate_rtl_wrapper_to_file(self, kernel_metadata: KernelMetadata, output_path: Path) -> None:
        """
        Generate RTL wrapper and write to file.
        
        Args:
            kernel_metadata: Complete kernel metadata object
            output_path: Path where generated wrapper should be written
        """
        code = self.generate_rtl_wrapper(kernel_metadata)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write generated code
        output_path.write_text(code, encoding='utf-8')
    
    def generate_rtl_backend(self, kernel_metadata: KernelMetadata) -> str:
        """
        Generate RTLBackend code from KernelMetadata.
        
        Args:
            kernel_metadata: Complete kernel metadata object
            
        Returns:
            Generated RTLBackend Python code as string
            
        Raises:
            FileNotFoundError: If template file not found
            Exception: If template rendering fails
        """
        try:
            template = self.jinja_env.get_template('auto_rtl_backend.py.j2')
            return template.render(
                kernel_metadata=kernel_metadata,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate RTLBackend code: {e}") from e
    
    def generate_rtl_backend_to_file(self, kernel_metadata: KernelMetadata, output_path: Path) -> None:
        """
        Generate RTLBackend code and write to file.
        
        Args:
            kernel_metadata: Complete kernel metadata object
            output_path: Path where generated code should be written
        """
        code = self.generate_rtl_backend(kernel_metadata)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write generated code
        output_path.write_text(code, encoding='utf-8')
    
    def generate_all(self, kernel_metadata: KernelMetadata, output_dir: Path) -> Dict[str, Path]:
        """
        Generate all artifacts for a kernel.
        
        Args:
            kernel_metadata: Complete kernel metadata object
            output_dir: Directory where all generated files should be written
            
        Returns:
            Dictionary mapping artifact type to generated file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        outputs = {}
        
        # Generate AutoHWCustomOp
        hw_path = output_dir / f"{kernel_metadata.name}_hw_custom_op.py"
        self.generate_autohwcustomop_to_file(kernel_metadata, hw_path)
        outputs['autohwcustomop'] = hw_path
        
        # Generate RTLBackend
        rtl_path = output_dir / f"{kernel_metadata.name}_rtl.py"
        self.generate_rtl_backend_to_file(kernel_metadata, rtl_path)
        outputs['rtlbackend'] = rtl_path
        
        # Generate RTL wrapper
        wrapper_path = output_dir / f"{kernel_metadata.name}_wrapper.v"
        self.generate_rtl_wrapper_to_file(kernel_metadata, wrapper_path)
        outputs['wrapper'] = wrapper_path
        
        return outputs