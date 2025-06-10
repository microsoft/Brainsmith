"""
Enhanced data structures for unified HWKG.

Based on hw_kernel_gen_simple foundation with optional BDIM pragma support,
following Interface-Wise Dataflow Modeling axioms.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class UnifiedHWKernel:
    """
    Enhanced hardware kernel representation with optional BDIM pragma support.
    
    Based on hw_kernel_gen_simple HWKernel class foundation with enhancements
    for advanced pragma processing while maintaining all smart properties.
    Follows HWKG Axiom 6: Metadata-Driven Generation.
    """
    name: str
    class_name: str
    interfaces: List[Dict[str, Any]]
    rtl_parameters: List[Dict[str, Any]]
    source_file: Path
    compiler_data: Dict[str, Any]
    
    # Enhanced fields for optional BDIM pragma support
    bdim_metadata: Optional[Dict[str, Any]] = None
    pragma_sophistication_level: str = "simple"  # simple | advanced
    parsing_warnings: List[str] = field(default_factory=list)
    
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
    
    @property
    def has_enhanced_bdim(self) -> bool:
        """
        Check if kernel has enhanced BDIM pragma information.
        
        Following Interface-Wise Dataflow Axiom 4: Pragma-to-Chunking Conversion.
        """
        return (self.pragma_sophistication_level == "advanced" 
                and self.bdim_metadata is not None)
    
    @property
    def dataflow_interfaces(self) -> List[Dict[str, Any]]:
        """
        Get interfaces with dataflow type classification.
        
        Following Interface-Wise Dataflow Axiom 3: Interface Types.
        Classifies into Input, Output, Weight, Config/Control categories.
        """
        dataflow_interfaces = []
        for iface in self.interfaces:
            # Enhanced dataflow classification - handle both dict and object types
            interface_type = iface.get('type', {})
            if hasattr(interface_type, 'name'):
                type_name = interface_type.name
            elif isinstance(interface_type, dict):
                type_name = interface_type.get('name', '')
            else:
                type_name = str(interface_type)
            
            if type_name == 'AXI_STREAM':
                dataflow_interfaces.append(iface)
        return dataflow_interfaces
    
    @property
    def chunking_strategies(self) -> Dict[str, Any]:
        """
        Get chunking strategies from BDIM pragmas if available.
        
        Following Interface-Wise Dataflow Axiom 2: Core Relationship
        tensor_dims → chunked into → num_blocks pieces of shape block_dims.
        """
        if self.has_enhanced_bdim:
            return self.bdim_metadata.get('chunking_strategies', {})
        return {}
    
    @property
    def tensor_dims_metadata(self) -> Dict[str, Any]:
        """
        Get tensor dimension metadata from BDIM pragmas.
        
        Following Interface-Wise Dataflow Axiom 1: Data Hierarchy
        Tensor → Block → Stream → Element
        """
        if self.has_enhanced_bdim:
            return self.bdim_metadata.get('tensor_dims', {})
        return {}
    
    @property
    def block_dims_metadata(self) -> Dict[str, Any]:
        """
        Get block dimension metadata from BDIM pragmas.
        
        Following Interface-Wise Dataflow Axiom 2: tensor_dims → block_dims.
        """
        if self.has_enhanced_bdim:
            return self.bdim_metadata.get('block_dims', {})
        return {}
    
    @property
    def stream_dims_metadata(self) -> Dict[str, Any]:
        """
        Get stream dimension metadata from BDIM pragmas.
        
        Following Interface-Wise Dataflow Axiom 2: block_dims → stream_dims.
        """
        if self.has_enhanced_bdim:
            return self.bdim_metadata.get('stream_dims', {})
        return {}
    
    def add_parsing_warning(self, warning: str):
        """Add a parsing warning to track issues during generation."""
        self.parsing_warnings.append(warning)


@dataclass  
class GenerationResult:
    """
    Result of unified code generation.
    
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