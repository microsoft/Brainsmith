"""
Configuration for Hardware Kernel Generator.

Simplified configuration that handles only the essentials needed for the
unified generation pipeline: RTL file, output directory, and debug settings.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """
    Simplified configuration for RTL-to-template generation.
    
    Only contains essential parameters needed for the unified generation pipeline.
    Removed complex configuration options and compiler data requirements.
    """
    rtl_file: Path
    output_dir: Path
    debug: bool = False
    
    def __post_init__(self):
        """Validate inputs and prepare output directory."""
        # Validate RTL file exists
        if not self.rtl_file.exists():
            raise ValueError(f"RTL file not found: {self.rtl_file}")
        
        if not self.rtl_file.suffix.lower() in ['.sv', '.v']:
            raise ValueError(f"RTL file must be SystemVerilog (.sv) or Verilog (.v), got: {self.rtl_file.suffix}")
        
        # Create output directory if it doesn't exist
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise ValueError(f"No permission to create output directory: {self.output_dir}")
        except Exception as e:
            raise ValueError(f"Failed to create output directory {self.output_dir}: {e}")
    
    @classmethod
    def from_args(cls, args) -> 'Config':
        """
        Create config from CLI arguments.
        
        Args:
            args: Parsed command line arguments from argparse
            
        Returns:
            Config instance with validated parameters
        """
        return cls(
            rtl_file=Path(args.rtl_file),
            output_dir=Path(args.output),
            debug=getattr(args, 'debug', False),
            template_version=getattr(args, 'template_version', 'phase2')
        )

    def get_summary(self) -> dict:
        """Get configuration summary for logging/debugging."""
        return {
            "rtl_file": str(self.rtl_file),
            "rtl_file_size": self.rtl_file.stat().st_size if self.rtl_file.exists() else 0,
            "output_dir": str(self.output_dir),
            "debug": self.debug,
            "template_version": self.template_version
        }
