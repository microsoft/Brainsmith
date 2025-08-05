"""
Generation types for code generation process.

This module contains types used during the code generation phase,
including file generation, contexts, and results.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from .metadata import KernelMetadata
from .rtl import ValidationResult


@dataclass
class GeneratedFile:
    """Single generated file.
    
    Represents a file to be written with its content and metadata.
    """
    path: Path
    content: str
    description: Optional[str] = None
    
    def write(self) -> None:
        """Write content to file.
        
        Creates parent directories if needed.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.content, encoding='utf-8')
    
    @property
    def size(self) -> int:
        """Size of content in bytes."""
        return len(self.content.encode('utf-8'))
    
    @property
    def line_count(self) -> int:
        """Number of lines in content."""
        return len(self.content.splitlines())


@dataclass
class GenerationContext:
    """Context for template generation.
    
    Contains all information needed by templates to generate code.
    """
    kernel_metadata: KernelMetadata
    output_dir: Path
    class_name: str
    
    # Template-specific data
    template_data: Dict[str, Any] = field(default_factory=dict)
    
    # Generation options
    debug: bool = False
    dry_run: bool = False
    
    def add_template_data(self, key: str, value: Any) -> None:
        """Add data for template use."""
        self.template_data[key] = value
    
    def get_template_data(self, key: str, default: Any = None) -> Any:
        """Get template data with optional default."""
        return self.template_data.get(key, default)
    
    @property
    def module_name(self) -> str:
        """Convenience accessor for module name."""
        return self.kernel_metadata.module_name
    
    @property
    def output_path(self) -> Path:
        """Full output path for main generated file."""
        return self.output_dir / f"{self.class_name}.py"


@dataclass  
class PerformanceMetrics:
    """Performance tracking for generation process.
    
    Tracks timing and resource usage during generation.
    """
    parsing_time_ms: Optional[float] = None
    template_generation_time_ms: Optional[float] = None
    file_writing_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    files_generated: int = 0
    total_lines_generated: int = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        return {
            "parsing_time_ms": self.parsing_time_ms,
            "template_generation_time_ms": self.template_generation_time_ms,
            "file_writing_time_ms": self.file_writing_time_ms,
            "total_time_ms": self.total_time_ms,
            "memory_peak_mb": self.memory_peak_mb,
            "files_generated": self.files_generated,
            "total_lines_generated": self.total_lines_generated,
            "avg_lines_per_file": (
                self.total_lines_generated / max(1, self.files_generated)
                if self.files_generated > 0 else 0
            )
        }


@dataclass
class GenerationResult:
    """Result of generation process.
    
    Contains generated files, validation results, and metrics.
    """
    generated_files: List[GeneratedFile] = field(default_factory=list)
    validation_result: ValidationResult = field(default_factory=ValidationResult)
    performance_metrics: Optional[PerformanceMetrics] = None
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_success(self) -> bool:
        """Check if generation succeeded."""
        return self.validation_result.is_valid and len(self.errors) == 0
    
    def add_file(self, file: GeneratedFile) -> None:
        """Add a generated file."""
        self.generated_files.append(file)
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.validation_result.add_error(message)
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        self.validation_result.add_warning(message)
    
    def write_all(self) -> None:
        """Write all generated files.
        
        Only writes if generation was successful.
        """
        if not self.is_success:
            raise RuntimeError("Cannot write files - generation failed")
            
        for file in self.generated_files:
            file.write()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of generation result."""
        summary = {
            "success": self.is_success,
            "files_generated": len(self.generated_files),
            "total_size": sum(f.size for f in self.generated_files),
            "total_lines": sum(f.line_count for f in self.generated_files),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
        }
        
        if self.performance_metrics:
            summary["performance"] = self.performance_metrics.get_summary()
            
        return summary
    
    def write_summary(self, path: Path) -> None:
        """Write generation summary to JSON file."""
        summary = self.get_summary()
        summary["files"] = [
            {
                "path": str(f.path),
                "size": f.size,
                "lines": f.line_count,
                "description": f.description
            }
            for f in self.generated_files
        ]
        
        if self.errors:
            summary["error_messages"] = self.errors
        if self.warnings:
            summary["warning_messages"] = self.warnings
            
        path.write_text(json.dumps(summary, indent=2))


@dataclass
class GenerationValidationResult:
    """
    Result of comprehensive generation validation checks.
    
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
    
    def add_warning(self, warning: str, check_name: str = "unknown") -> None:
        """Add a validation warning."""
        self.warnings.append(warning)
        self.checks_performed.append(f"{check_name} (WARNING)")
    
    def add_check(self, check_name: str) -> None:
        """Mark a check as passed."""
        self.checks_performed.append(f"{check_name} (PASSED)")