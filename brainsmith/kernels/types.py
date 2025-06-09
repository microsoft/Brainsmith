"""
FINN Kernel Type Definitions
Common types and data structures for the kernel management system.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class KernelInfo:
    """Basic kernel information structure."""
    name: str
    operator_type: str
    backend_type: str
    implementation_path: str


@dataclass
class KernelMetadata:
    """Extended kernel metadata from discovery."""
    name: str
    operator_type: str
    backend_type: str
    implementation_files: Dict[str, str]
    parameterization: Dict[str, Any]
    finn_version_compatibility: List[str] = field(default_factory=list)


@dataclass
class ParameterSchema:
    """Schema for kernel parameterization."""
    pe_range: Optional[tuple] = None
    simd_range: Optional[tuple] = None
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)