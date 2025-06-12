"""
Simple configuration for HWKG.

Radically simplified from the over-engineered 461-line version.
Does exactly what's needed: validate files and create output directory.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Simple configuration for RTL-to-template generation."""
    rtl_file: Path
    compiler_data_file: Path  
    output_dir: Path
    debug: bool = False
    
    def __post_init__(self):
        """Basic validation - files exist, create output dir."""
        if not self.rtl_file.exists():
            raise ValueError(f"RTL file not found: {self.rtl_file}")
        if not self.compiler_data_file.exists():
            raise ValueError(f"Compiler data file not found: {self.compiler_data_file}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_args(cls, args):
        """Create config from CLI arguments."""
        return cls(
            rtl_file=Path(args.rtl_file),
            compiler_data_file=Path(args.compiler_data),
            output_dir=Path(args.output),
            debug=getattr(args, 'debug', False)
        )