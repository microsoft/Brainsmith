"""
Direct AutoHWCustomOp generator.

This module provides a minimal generator that works directly with KernelMetadata
objects without any intermediate transformations or abstraction layers.
"""

from pathlib import Path
from typing import Optional
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata


class DirectAutoHWCustomOpGenerator:
    """
    Direct AutoHWCustomOp code generator.
    
    Works directly with KernelMetadata structure using simple Jinja2 template
    rendering with no intermediate processing or abstraction layers.
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
            template_dir = Path(__file__).parent.parent / "templates"
        
        self.template_dir = Path(template_dir)
        
        # Setup Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
    
    def generate(self, kernel_metadata: KernelMetadata) -> str:
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
            template = self.jinja_env.get_template('autohwcustomop_direct.py.j2')
            return template.render(kernel_metadata=kernel_metadata)
        except Exception as e:
            raise RuntimeError(f"Failed to generate AutoHWCustomOp code: {e}") from e
    
    def generate_to_file(self, kernel_metadata: KernelMetadata, output_path: Path) -> None:
        """
        Generate AutoHWCustomOp code and write to file.
        
        Args:
            kernel_metadata: Complete kernel metadata object
            output_path: Path where generated code should be written
        """
        code = self.generate(kernel_metadata)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write generated code
        output_path.write_text(code, encoding='utf-8')
    
    def generate_rtl_wrapper(self, kernel_metadata: KernelMetadata) -> str:
        """
        Generate RTL wrapper directly from KernelMetadata.
        
        Args:
            kernel_metadata: Complete kernel metadata object
            
        Returns:
            Generated SystemVerilog wrapper code as string
            
        Raises:
            FileNotFoundError: If template file not found
            Exception: If template rendering fails
        """
        try:
            template = self.jinja_env.get_template('rtl_wrapper_direct.v.j2')
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