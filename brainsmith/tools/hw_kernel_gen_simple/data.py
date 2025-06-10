"""
Simple data structures for HWKG.

Provides clean data representation without complex validation frameworks.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class HWKernel:
    """Simple hardware kernel representation."""
    name: str
    class_name: str
    interfaces: List[Dict[str, Any]]
    rtl_parameters: List[Dict[str, Any]]
    source_file: Path
    compiler_data: Dict[str, Any]
    
    @property
    def kernel_name(self) -> str:
        """Get kernel name for templates."""
        return self.name
    
    @property
    def generation_timestamp(self) -> str:
        """Get timestamp for templates."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    @property
    def resource_estimation_required(self) -> bool:
        """Check if resource estimation is needed."""
        return self.compiler_data.get('enable_resource_estimation', False)
    
    @property
    def verification_required(self) -> bool:
        """Check if verification is needed."""
        return self.compiler_data.get('enable_verification', False)
    
    @property
    def weight_interfaces_count(self) -> int:
        """Count weight interfaces for resource estimation."""
        return len([iface for iface in self.interfaces 
                   if iface.get('dataflow_type', '').upper() == 'WEIGHT'])
    
    @property
    def kernel_complexity(self) -> str:
        """Estimate kernel complexity for resource calculations."""
        interface_count = len(self.interfaces)
        param_count = len(self.rtl_parameters)
        
        if interface_count <= 2 and param_count <= 3:
            return 'low'
        elif interface_count <= 4 and param_count <= 6:
            return 'medium'
        else:
            return 'high'
    
    @property
    def kernel_type(self) -> str:
        """Infer kernel type from name for resource estimation."""
        name_lower = self.name.lower()
        if any(term in name_lower for term in ['matmul', 'gemm', 'dot']):
            return 'matmul'
        elif any(term in name_lower for term in ['conv', 'convolution']):
            return 'conv'
        elif any(term in name_lower for term in ['threshold', 'compare']):
            return 'threshold'
        elif any(term in name_lower for term in ['norm', 'batch', 'layer']):
            return 'norm'
        else:
            return 'generic'


@dataclass  
class GenerationResult:
    """Result of code generation."""
    generated_files: List[Path]
    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
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