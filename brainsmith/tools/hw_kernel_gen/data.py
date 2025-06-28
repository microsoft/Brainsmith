############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Shared data structures for Hardware Kernel Generator.

This module contains data structures that are shared across the hw_kernel_gen
package, including common enums, type system, generation results, and validation.

Organization:
- Enums: Direction, InterfaceType
- Type System: DatatypeConstraintGroup and validation functions  
- Generation Results: GenerationResult, PerformanceMetrics
- Validation: GenerationValidationResult
- Utility Functions

RTL parser specific classes are in rtl_parser/rtl_data.py
Metadata classes are in metadata.py
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

# Import QONNX datatypes - required dependency
from qonnx.core.datatype import DataType, BaseDataType

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from .templates.template_context import TemplateContext
    from .metadata import KernelMetadata

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class InterfaceType(Enum):
    """Unified interface types with inherent protocol-role relationships"""
    
    # AXI-Stream interfaces (dataflow)
    INPUT = "input"      # AXI-Stream input for activation data
    OUTPUT = "output"    # AXI-Stream output for result data  
    WEIGHT = "weight"    # AXI-Stream input for weight/parameter data
    
    # AXI-Lite interfaces (configuration)
    CONFIG = "config"    # AXI-Lite for runtime configuration
    
    # Global control signals
    CONTROL = "control"  # Global control signals (clk, rst, etc.)
    
    # Unknown/fallback
    UNKNOWN = "unknown"  # Unrecognized interfaces
    
    @property
    def protocol(self) -> str:
        """Get the hardware protocol for this interface type"""
        protocol_map = {
            InterfaceType.INPUT: "axi_stream",
            InterfaceType.OUTPUT: "axi_stream", 
            InterfaceType.WEIGHT: "axi_stream",
            InterfaceType.CONFIG: "axi_lite",
            InterfaceType.CONTROL: "global_control",
            InterfaceType.UNKNOWN: "unknown"
        }
        return protocol_map[self]
    
    @property
    def is_dataflow(self) -> bool:
        """Check if this interface participates in dataflow"""
        return self in [InterfaceType.INPUT, InterfaceType.OUTPUT, InterfaceType.WEIGHT]
    
    @property
    def is_axi_stream(self) -> bool:
        """Check if this interface uses AXI-Stream protocol"""
        return self.protocol == "axi_stream"
    
    @property
    def is_axi_lite(self) -> bool:
        """Check if this interface uses AXI-Lite protocol"""
        return self.protocol == "axi_lite"
    
    @property
    def is_configuration(self) -> bool:
        """Check if this interface is for configuration"""
        return self in [InterfaceType.CONFIG, InterfaceType.CONTROL]
    
    @property
    def direction(self) -> str:
        """Get the expected direction for this interface type"""
        direction_map = {
            InterfaceType.INPUT: "input",
            InterfaceType.WEIGHT: "input", 
            InterfaceType.OUTPUT: "output",
            InterfaceType.CONFIG: "bidirectional",
            InterfaceType.CONTROL: "input",
            InterfaceType.UNKNOWN: "unknown"
        }
        return direction_map[self]
    
    def __str__(self) -> str:
        """String representation"""
        return f"InterfaceType.{self.name}"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"InterfaceType.{self.name}('{self.value}', protocol='{self.protocol}')"


# ============================================================================
# TYPE SYSTEM
# ============================================================================

@dataclass
class DatatypeConstraintGroup:
    """
    Simple constraint group: [DTYPE, MIN_WIDTH, MAX_WIDTH]
    
    Examples:
        DatatypeConstraintGroup("INT", 4, 8)    # INT4, INT5, INT6, INT7, INT8
        DatatypeConstraintGroup("UINT", 8, 16)  # UINT8, UINT16
        DatatypeConstraintGroup("FIXED", 8, 16) # FIXED<8,N>, FIXED<16,N>
    """
    base_type: str      # "INT", "UINT", "FIXED", "FLOAT", "BIPOLAR", "TERNARY"
    min_width: int      # Minimum bit width (inclusive)
    max_width: int      # Maximum bit width (inclusive)
    
    def __post_init__(self):
        """Validate constraint group parameters."""
        if self.min_width <= 0:
            raise ValueError(f"min_width must be positive, got {self.min_width}")
        if self.max_width < self.min_width:
            raise ValueError(f"max_width ({self.max_width}) must be >= min_width ({self.min_width})")
        
        valid_base_types = ["INT", "UINT", "FIXED", "FLOAT", "BIPOLAR", "TERNARY", "BINARY"]
        if self.base_type not in valid_base_types:
            raise ValueError(f"Invalid base_type '{self.base_type}'. Must be one of {valid_base_types}")


def validate_datatype_against_constraints(
    datatype: BaseDataType, 
    constraint_groups: List[DatatypeConstraintGroup]
) -> bool:
    """
    Check if a QONNX datatype satisfies any constraint group.
    
    Args:
        datatype: QONNX BaseDataType instance to validate
        constraint_groups: List of constraint groups to check against
        
    Returns:
        True if datatype satisfies at least one constraint group
    """
    if not constraint_groups:
        return True  # No constraints = allow anything
        
    for group in constraint_groups:
        if _matches_constraint_group(datatype, group):
            return True
    return False


def _matches_constraint_group(datatype: BaseDataType, group: DatatypeConstraintGroup) -> bool:
    """Check if datatype matches a single constraint group."""
    # Extract base type from QONNX canonical name
    canonical_name = datatype.get_canonical_name()
    
    # Check base type
    if group.base_type == "INT" and not (canonical_name.startswith("INT") and datatype.signed()):
        return False
    elif group.base_type == "UINT" and not (canonical_name.startswith("UINT") or canonical_name == "BINARY"):
        return False
    elif group.base_type == "FIXED" and not canonical_name.startswith("FIXED<"):
        return False
    elif group.base_type == "FLOAT" and not canonical_name.startswith("FLOAT"):
        return False
    elif group.base_type == "BIPOLAR" and canonical_name != "BIPOLAR":
        return False
    elif group.base_type == "TERNARY" and canonical_name != "TERNARY":
        return False
    elif group.base_type == "BINARY" and canonical_name != "BINARY":
        return False
    
    # Check bitwidth range
    bitwidth = datatype.bitwidth()
    return group.min_width <= bitwidth <= group.max_width


# ============================================================================
# GENERATION RESULTS
# ============================================================================

@dataclass
class GenerationResult:
    """
    Enhanced generation result for Phase 3/4 integration.
    
    Contains all information about a successful (or failed) generation,
    including generated files, metadata, and any errors or warnings.
    
    Features:
    - Phase 2 template system integration (TemplateContext)
    - Rich kernel metadata tracking (KernelMetadata)
    - Performance monitoring (generation_time_ms)
    - Enhanced file tracking (filename -> content mapping)
    - Integrated file writing capabilities (Phase 3/4 integration)
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
    
    # File writing tracking (Phase 3/4 integration)
    output_directory: Optional[Path] = None
    files_written: List[Path] = field(default_factory=list)
    metadata_files: List[Path] = field(default_factory=list)
    
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
        summary = {
            "kernel_name": self.kernel_name,
            "source_file": str(self.source_file),
            "success": self.is_success(),
            "files_generated": self.get_file_count(),
            "validation_passed": self.validation_passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "generation_time_ms": self.generation_time_ms
        }
        
        # Add file writing info if available
        if self.output_directory:
            summary.update({
                "output_directory": str(self.output_directory),
                "files_written": len(self.files_written),
                "metadata_files": len(self.metadata_files)
            })
        
        return summary
    
    # ===== File Writing Methods (Phase 3/4 Integration) =====
    
    def write_file(self, filename: str, content: str, output_dir: Path) -> Path:
        """
        Write a single generated file to the output directory.
        
        Args:
            filename: Name of the file to write
            content: Content to write to the file
            output_dir: Directory to write the file to
            
        Returns:
            Path to the written file
            
        Raises:
            RuntimeError: If file writing fails
        """
        try:
            file_path = output_dir / filename
            file_path.write_text(content, encoding='utf-8')
            self.files_written.append(file_path)
            logger.debug(f"Wrote file for {self.kernel_name}: {file_path}")
            return file_path
        except Exception as e:
            error_msg = f"Failed to write file {filename}: {e}"
            self.add_error(error_msg)
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def write_all_files(self, output_dir: Path) -> List[Path]:
        """
        Write all generated files to the output directory.
        
        Args:
            output_dir: Directory to write files to
            
        Returns:
            List of paths to written files
            
        Raises:
            RuntimeError: If any file writing fails
        """
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            self.output_directory = output_dir
            
            written_files = []
            
            # Write all generated files
            for filename, content in self.generated_files.items():
                file_path = self.write_file(filename, content, output_dir)
                written_files.append(file_path)
            
            # Create metadata files
            metadata_files = self.create_metadata_files(output_dir)
            written_files.extend(metadata_files)
            
            logger.info(f"Successfully wrote {len(written_files)} files for {self.kernel_name}")
            return written_files
            
        except Exception as e:
            error_msg = f"Failed to write files for {self.kernel_name}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def create_metadata_files(self, output_dir: Path) -> List[Path]:
        """
        Create generation metadata files.
        
        Args:
            output_dir: Directory to write metadata files to
            
        Returns:
            List of paths to created metadata files
        """
        metadata_files = []
        
        try:
            # Create generation_metadata.json
            metadata = {
                "kernel_name": self.kernel_name,
                "source_file": str(self.source_file),
                "validation_passed": self.validation_passed,
                "success": self.is_success(),
                "errors": self.errors,
                "warnings": self.warnings,
                "generated_files": list(self.generated_files.keys()),
                "generation_time_ms": self.generation_time_ms,
                "summary": self.get_summary()
            }
            
            # Add template context info if available
            if self.template_context:
                metadata["template_context"] = {
                    "parameter_count": len(self.template_context.parameter_definitions),
                    "interface_count": len(self.template_context.interface_metadata),
                    "required_parameters": self.template_context.required_attributes,
                    "whitelisted_defaults": self.template_context.whitelisted_defaults
                }
            
            # Add kernel metadata info if available
            if self.kernel_metadata:
                metadata["kernel_metadata"] = {
                    "parameter_count": len(self.kernel_metadata.parameters),
                    "interface_count": len(self.kernel_metadata.interfaces),
                    "parameter_names": [p.name for p in self.kernel_metadata.parameters]
                }
            
            metadata_file = output_dir / "generation_metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2, default=str))
            self.metadata_files.append(metadata_file)
            metadata_files.append(metadata_file)
            
            # Create generation_summary.txt
            summary_file = self._create_summary_file(output_dir)
            metadata_files.append(summary_file)
            
        except Exception as e:
            logger.warning(f"Failed to create metadata files for {self.kernel_name}: {e}")
        
        return metadata_files
    
    def _create_summary_file(self, output_dir: Path) -> Path:
        """Create human-readable summary log file."""
        summary_lines = [
            f"Generation Summary for {self.kernel_name}",
            "=" * 50,
            f"Source File: {self.source_file}",
            f"Output Directory: {output_dir}",
            f"Success: {self.is_success()}",
            f"Validation Passed: {self.validation_passed}",
            f"Files Generated: {len(self.generated_files)}",
            ""
        ]
        
        if self.files_written:
            summary_lines.append("Generated Files:")
            for file_path in self.files_written:
                summary_lines.append(f"  - {file_path}")
            summary_lines.append("")
        
        if self.errors:
            summary_lines.append("Errors:")
            for error in self.errors:
                summary_lines.append(f"  - {error}")
            summary_lines.append("")
        
        if self.warnings:
            summary_lines.append("Warnings:")
            for warning in self.warnings:
                summary_lines.append(f"  - {warning}")
            summary_lines.append("")
        
        if self.generation_time_ms:
            summary_lines.append(f"Generation Time: {self.generation_time_ms:.2f} ms")
        
        summary_file = output_dir / "generation_summary.txt"
        summary_file.write_text("\n".join(summary_lines))
        self.metadata_files.append(summary_file)
        return summary_file


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


# ============================================================================
# VALIDATION RESULTS
# ============================================================================

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


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "InterfaceType",
    
    # Type System
    "DatatypeConstraintGroup",
    "validate_datatype_against_constraints",
    "DataType",
    "BaseDataType",
    
    # Generation Results
    "GenerationResult",
    "PerformanceMetrics",
    
    # Validation
    "GenerationValidationResult",
    
    # Utility functions
    "create_generation_result",
]