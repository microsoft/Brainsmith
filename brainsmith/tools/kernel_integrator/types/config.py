"""
Configuration types for kernel integrator.

This module contains the configuration class used to control
the kernel integration process.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re


@dataclass
class Config:
    """Kernel integrator configuration.
    
    Controls the behavior of the kernel integration process including
    input/output paths, naming, and debug options.
    """
    rtl_file: Path
    output_dir: Path
    module_name: Optional[str] = None
    class_name: Optional[str] = None
    template_dir: Optional[Path] = None
    debug: bool = False
    
    # Advanced options
    auto_link: bool = True
    validate_protocols: bool = True
    generate_tests: bool = False
    
    def __post_init__(self):
        """Validate configuration and set defaults."""
        # Validate RTL file exists
        if not self.rtl_file.exists():
            raise ValueError(f"RTL file not found: {self.rtl_file}")
        
        # Ensure output directory
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default module name from filename
        if self.module_name is None:
            self.module_name = self.rtl_file.stem
        
        # Set default class name from module name
        if self.class_name is None:
            self.class_name = self._to_camel_case(self.module_name)
        
        # Set default template directory
        if self.template_dir is None:
            # Use templates from package
            import brainsmith.tools.kernel_integrator.templates as templates
            self.template_dir = Path(templates.__file__).parent
    
    def _to_camel_case(self, name: str) -> str:
        """Convert module_name to ClassName.
        
        Examples:
            matrix_multiply -> MatrixMultiply
            my_kernel_v2 -> MyKernelV2
        """
        # Handle special cases for version numbers
        name = re.sub(r'_v(\d+)$', r'V\1', name)
        
        # Split by underscore and capitalize
        parts = name.split('_')
        return ''.join(part.capitalize() for part in parts)
    
    @property
    def hw_custom_op_path(self) -> Path:
        """Path for generated HWCustomOp file."""
        return self.output_dir / f"{self.class_name}.py"
    
    @property 
    def rtl_backend_path(self) -> Path:
        """Path for generated RTL backend file."""
        return self.output_dir / f"{self.module_name}_rtl_backend.py"
    
    @property
    def rtl_wrapper_path(self) -> Path:
        """Path for generated RTL wrapper file."""
        return self.output_dir / f"{self.module_name}_wrapper.v"
    
    @property
    def metadata_path(self) -> Path:
        """Path for generation metadata file."""
        return self.output_dir / "generation_metadata.json"
    
    def get_template_path(self, template_name: str) -> Path:
        """Get full path to a template file.
        
        Args:
            template_name: Name of template file (e.g., "hw_custom_op.py.j2")
            
        Returns:
            Full path to template file
            
        Raises:
            ValueError: If template not found
        """
        template_path = self.template_dir / template_name
        if not template_path.exists():
            raise ValueError(f"Template not found: {template_path}")
        return template_path
    
    def validate_templates(self) -> bool:
        """Check that all required templates exist.
        
        Returns:
            True if all templates found
        """
        required_templates = [
            "hw_custom_op.py.j2",
            "rtl_backend.py.j2", 
            "rtl_wrapper_minimal.v.j2"
        ]
        
        for template in required_templates:
            try:
                self.get_template_path(template)
            except ValueError:
                return False
                
        return True
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "rtl_file": str(self.rtl_file),
            "output_dir": str(self.output_dir),
            "module_name": self.module_name,
            "class_name": self.class_name,
            "template_dir": str(self.template_dir) if self.template_dir else None,
            "debug": self.debug,
            "auto_link": self.auto_link,
            "validate_protocols": self.validate_protocols,
            "generate_tests": self.generate_tests,
        }