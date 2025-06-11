"""
Simple data types for kernel package management.
North Star-aligned: Pure data structures without complex abstractions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class OperatorType(Enum):
    """Supported FINN operator types"""
    CONVOLUTION = "Convolution"
    MATMUL = "MatMul"
    THRESHOLDING = "Thresholding"
    LAYERNORM = "LayerNorm"
    POOL = "Pool"
    ELEMENTWISE = "ElementWise"
    RESHAPE = "Reshape"
    CONCAT = "Concat"
    ATTENTION = "Attention"
    CUSTOM = "Custom"


class BackendType(Enum):
    """Implementation backend types"""
    RTL = "RTL"
    HLS = "HLS"
    PYTHON = "Python"


@dataclass
class KernelPackage:
    """Simple kernel package representation"""
    name: str
    operator_type: str
    backend: str
    version: str
    author: str = ""
    license: str = ""
    description: str = ""
    
    # Parameter specifications
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # File mappings (logical name -> actual file path)
    files: Dict[str, str] = field(default_factory=dict)
    
    # Performance characteristics
    performance: Dict[str, Any] = field(default_factory=dict)
    
    # Validation and testing info
    validation: Dict[str, Any] = field(default_factory=dict)
    
    # Repository information (for community kernels)
    repository: Dict[str, str] = field(default_factory=dict)
    
    # Package directory path
    package_path: str = ""
    
    def get_file_path(self, logical_name: str) -> Optional[str]:
        """Get absolute path for a logical file name"""
        if logical_name not in self.files:
            return None
        
        relative_path = self.files[logical_name]
        if self.package_path:
            import os
            return os.path.join(self.package_path, relative_path)
        return relative_path
    
    def get_pe_range(self) -> tuple[int, int]:
        """Get PE parallelism range"""
        return tuple(self.parameters.get('pe_range', [1, 64]))
    
    def get_simd_range(self) -> tuple[int, int]:
        """Get SIMD width range"""
        return tuple(self.parameters.get('simd_range', [1, 32]))
    
    def get_supported_datatypes(self) -> List[str]:
        """Get supported data types"""
        return self.parameters.get('supported_datatypes', ['int8'])
    
    def is_compatible_with(self, requirements: Dict[str, Any]) -> bool:
        """Check if kernel is compatible with requirements"""
        # Check operator type
        if 'operator_type' in requirements:
            if self.operator_type != requirements['operator_type']:
                return False
        
        # Check data type support
        if 'datatype' in requirements:
            if requirements['datatype'] not in self.get_supported_datatypes():
                return False
        
        # Check PE requirements
        if 'min_pe' in requirements or 'max_pe' in requirements:
            pe_min, pe_max = self.get_pe_range()
            if 'min_pe' in requirements and requirements['min_pe'] is not None and pe_max < requirements['min_pe']:
                return False
            if 'max_pe' in requirements and requirements['max_pe'] is not None and pe_min > requirements['max_pe']:
                return False
        
        # Check SIMD requirements
        if 'min_simd' in requirements or 'max_simd' in requirements:
            simd_min, simd_max = self.get_simd_range()
            if 'min_simd' in requirements and requirements['min_simd'] is not None and simd_max < requirements['min_simd']:
                return False
            if 'max_simd' in requirements and requirements['max_simd'] is not None and simd_min > requirements['max_simd']:
                return False
        
        return True


@dataclass
class ValidationResult:
    """Result of kernel package validation"""
    is_valid: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        """Add validation error"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add validation warning"""
        self.warnings.append(warning)


@dataclass
class KernelRequirements:
    """Requirements for kernel selection"""
    operator_type: str
    datatype: str = "int8"
    min_pe: Optional[int] = None
    max_pe: Optional[int] = None
    min_simd: Optional[int] = None
    max_simd: Optional[int] = None
    backend_preference: Optional[str] = None
    performance_requirements: Dict[str, float] = field(default_factory=dict)
    resource_constraints: Dict[str, int] = field(default_factory=dict)


@dataclass
class KernelSelection:
    """Selected kernel with optimized parameters"""
    kernel: KernelPackage
    pe_parallelism: int
    simd_width: int
    memory_mode: str = "internal"
    folding_factors: Dict[str, int] = field(default_factory=dict)
    custom_options: Dict[str, Any] = field(default_factory=dict)
    
    def to_finn_config(self) -> Dict[str, Any]:
        """Convert to FINN configuration format"""
        config = {
            'PE': self.pe_parallelism,
            'SIMD': self.simd_width,
            'mem_mode': self.memory_mode
        }
        
        # Add folding factors
        config.update(self.folding_factors)
        
        # Add custom options
        config.update(self.custom_options)
        
        return config