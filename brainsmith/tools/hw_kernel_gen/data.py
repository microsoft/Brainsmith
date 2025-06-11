"""
Enhanced data structures for HWKG.

Based on HWKernel from RTL parser with enhanced GenerationResult
for tracking generation status and results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass  
class GenerationResult:
    """
    Result of code generation.
    
    Identical to hw_kernel_gen_simple GenerationResult with additional
    tracking for advanced pragma processing.
    """
    generated_files: List[Path]
    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    bdim_processing_used: bool = False
    complexity_level: str = "simple"
    
    def add_error(self, error: str):
        """Add an error to the result."""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str):
        """Add a warning to the result."""
        self.warnings.append(warning)
    
    def add_generated_file(self, file_path: Path):
        """Add a successfully generated file."""
        self.generated_files.append(file_path)
    
    def set_bdim_processing(self, used: bool):
        """Track whether advanced BDIM processing was used."""
        self.bdim_processing_used = used