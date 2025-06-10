"""
Simple configuration for HWKG.

Provides clean configuration without complex validation frameworks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .errors import ConfigurationError


@dataclass
class Config:
    """Simple configuration for Hardware Kernel Generator."""
    rtl_file: Path
    compiler_data_file: Path
    output_dir: Path
    template_dir: Optional[Path] = None
    debug: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.rtl_file.exists():
            raise ConfigurationError(f"RTL file not found: {self.rtl_file}")
        if not self.compiler_data_file.exists():
            raise ConfigurationError(f"Compiler data file not found: {self.compiler_data_file}")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate template directory if provided
        if self.template_dir and not self.template_dir.exists():
            raise ConfigurationError(f"Template directory not found: {self.template_dir}")
    
    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments."""
        return cls(
            rtl_file=Path(args.rtl_file),
            compiler_data_file=Path(args.compiler_data),
            output_dir=Path(args.output),
            template_dir=Path(args.template_dir) if hasattr(args, 'template_dir') and args.template_dir else None,
            debug=args.debug if hasattr(args, 'debug') else False
        )