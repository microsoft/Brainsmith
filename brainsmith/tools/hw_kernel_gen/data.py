"""
Enhanced data structures for Phase 3 HWKG.

Provides unified data structures for the Phase 3 clean-break refactor,
including enhanced GenerationResult with Phase 2 template integration.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from brainsmith.dataflow.core.kernel_metadata import KernelMetadata
    from .templates.template_context import TemplateContext

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """
    Enhanced generation result for Phase 3.
    
    Contains all information about a successful (or failed) generation,
    including generated files, metadata, and any errors or warnings.
    
    Features:
    - Phase 2 template system integration (TemplateContext)
    - Rich kernel metadata tracking (KernelMetadata)
    - Performance monitoring (generation_time_ms)
    - Enhanced file tracking (filename -> content mapping)
    - Backward compatibility with legacy methods
    """
    kernel_name: str
    source_file: Path
    generated_files: Dict[str, str] = field(default_factory=dict)
    template_context: Optional["TemplateContext"] = None
    kernel_metadata: Optional["KernelMetadata"] = None
    validation_passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    generation_time_ms: Optional[float] = None
    
    def add_error(self, error: str) -> None:
        """Add an error message and mark validation as failed."""
        self.errors.append(error)
        self.validation_passed = False
        logger.error(f"Generation error for {self.kernel_name}: {error}")
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        logger.warning(f"Generation warning for {self.kernel_name}: {warning}")
    
    def is_success(self) -> bool:
        """Check if generation was successful (no errors and at least one file generated)."""
        return len(self.errors) == 0 and self.validation_passed and len(self.generated_files) > 0
    
    def add_generated_file(self, filename: str, content: str) -> None:
        """Add a generated file with its content."""
        self.generated_files[filename] = content
        logger.debug(f"Added generated file for {self.kernel_name}: {filename} ({len(content)} chars)")
    
    def get_file_count(self) -> int:
        """Get the number of generated files."""
        return len(self.generated_files)
    
    def get_summary(self) -> Dict[str, any]:
        """Get a summary of the generation result."""
        return {
            "kernel_name": self.kernel_name,
            "source_file": str(self.source_file),
            "success": self.is_success(),
            "files_generated": self.get_file_count(),
            "validation_passed": self.validation_passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "generation_time_ms": self.generation_time_ms
        }
    
    # ===== Backward Compatibility Methods =====
    

# ===== Additional Phase 3 Data Structures =====

@dataclass
class ValidationResult:
    """
    Result of Phase 3 validation checks.
    
    Used for validating generated code, templates, and configurations
    before finalizing generation results.
    """
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks_performed: List[str] = field(default_factory=list)
    
    def add_error(self, error: str, check_name: str = "unknown") -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.passed = False
        self.checks_performed.append(f"{check_name} (FAILED)")
        logger.error(f"Validation error in {check_name}: {error}")
    
    def add_warning(self, warning: str, check_name: str = "unknown") -> None:
        """Add a validation warning."""
        self.warnings.append(warning)
        self.checks_performed.append(f"{check_name} (WARNING)")
        logger.warning(f"Validation warning in {check_name}: {warning}")
    
    def add_check(self, check_name: str) -> None:
        """Mark a check as passed."""
        self.checks_performed.append(f"{check_name} (PASSED)")
        logger.debug(f"Validation check passed: {check_name}")


@dataclass  
class PerformanceMetrics:
    """
    Performance tracking for Phase 3 generation.
    
    Tracks timing and resource usage during the generation process
    for optimization and monitoring purposes.
    """
    parsing_time_ms: Optional[float] = None
    template_generation_time_ms: Optional[float] = None
    file_writing_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    files_generated: int = 0
    total_lines_generated: int = 0
    
    def get_summary(self) -> Dict[str, any]:
        """Get a summary of performance metrics."""
        return {
            "parsing_time_ms": self.parsing_time_ms,
            "template_generation_time_ms": self.template_generation_time_ms,
            "file_writing_time_ms": self.file_writing_time_ms,
            "total_time_ms": self.total_time_ms,
            "memory_peak_mb": self.memory_peak_mb,
            "files_generated": self.files_generated,
            "total_lines_generated": self.total_lines_generated,
            "avg_lines_per_file": self.total_lines_generated / max(1, self.files_generated)
        }


# ===== Utility Functions =====

def create_generation_result(
    kernel_name: str,
    source_file: Path,
    template_context: Optional["TemplateContext"] = None,
    kernel_metadata: Optional["KernelMetadata"] = None
) -> GenerationResult:
    """
    Create a new GenerationResult with proper initialization.
    
    Convenience function for creating GenerationResult instances
    with consistent defaults and validation.
    """
    return GenerationResult(
        kernel_name=kernel_name,
        source_file=source_file,
        template_context=template_context,
        kernel_metadata=kernel_metadata
    )

# ===== Module Exports =====

__all__ = [
    # Main data structures
    "GenerationResult",
    "ValidationResult", 
    "PerformanceMetrics",
    
    # Utility functions
    "create_generation_result",
    "merge_generation_results",
]