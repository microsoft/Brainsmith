"""
Enhanced interface analyzer for extracting operation-specific optimization hints.

This module enhances RTL analysis to extract operation-specific formatting hints
that enable automatic tensor formatting optimization.
"""

from typing import Dict, Any, List, Optional
from ..parsers.rtl_parser import EnhancedRTLParsingResult
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.interface_types import InterfaceType


class EnhancedInterfaceAnalyzer:
    """Enhanced analyzer to extract operation-specific formatting hints from RTL."""
    
    def analyze_rtl_for_dataflow_interfaces(self, rtl_result: EnhancedRTLParsingResult) -> List[InterfaceMetadata]:
        """
        Extract interfaces with operation-specific optimization hints.
        
        This method enhances standard interface extraction with detailed analysis
        of RTL patterns to determine optimal tensor formatting strategies.
        
        Args:
            rtl_result: Results from enhanced RTL parsing
            
        Returns:
            List of interface metadata with optimization hints
        """
        interfaces = []
        
        for interface_info in rtl_result.interfaces:
            # Create base interface metadata
            interface_metadata = self._create_base_interface_metadata(interface_info)
            
            # Extract operation-specific hints from RTL analysis
            optimization_hints = self._extract_optimization_hints(interface_info, rtl_result)
            operation_type = self._infer_operation_type(interface_info, rtl_result)
            memory_pattern = self._analyze_memory_access_pattern(interface_info, rtl_result)
            
            # Enhance metadata with optimization information
            interface_metadata.metadata = {
                'optimization_flags': optimization_hints,
                'operation_type': operation_type,
                'memory_pattern': memory_pattern,
                'tensor_formatting_hints': self._generate_tensor_formatting_hints(
                    interface_info, operation_type, optimization_hints
                )
            }
            
            interfaces.append(interface_metadata)
        
        return interfaces
    
    def _create_base_interface_metadata(self, interface_info) -> InterfaceMetadata:
        """Create base interface metadata from RTL interface information."""
        # This would be implemented based on the existing interface creation logic
        # For now, create a basic interface metadata structure
        return InterfaceMetadata(
            name=interface_info.name,
            interface_type=interface_info.interface_type,
            datatype_constraints=getattr(interface_info, 'datatype_constraints', []),
            chunking_strategy=getattr(interface_info, 'chunking_strategy', None)
        )
    
    def _extract_optimization_hints(self, interface_info, rtl_result: EnhancedRTLParsingResult) -> Dict[str, Any]:
        """Extract formatting optimization hints from RTL pragmas and analysis."""
        
        hints = {}
        
        # Check for SIMD optimization pragmas
        if self._has_simd_optimization_pragma(interface_info, rtl_result):
            hints["needs_simd_flip"] = True
        
        # Check for transpose requirements (matrix operations)
        if self._requires_matrix_transpose(interface_info, rtl_result):
            hints["needs_transpose"] = True
        
        # Check for PE distribution requirements
        if self._requires_pe_distribution(interface_info, rtl_result):
            hints["needs_pe_interleaving"] = True
        
        # Check for memory layout optimizations
        if self._has_memory_optimization_patterns(interface_info, rtl_result):
            hints["memory_optimization"] = True
        
        # Check for convolution-specific patterns
        if self._has_convolution_patterns(interface_info, rtl_result):
            hints["spatial_flattening"] = True
        
        return hints
    
    def _infer_operation_type(self, interface_info, rtl_result: EnhancedRTLParsingResult) -> str:
        """Infer operation type from RTL characteristics."""
        
        # Matrix multiplication: 2D weight interfaces with MAC patterns
        if (len(getattr(interface_info, 'tensor_dims', [])) == 2 and 
            interface_info.interface_type == InterfaceType.WEIGHT and
            self._has_mac_pattern(rtl_result)):
            return "matrix_multiplication"
        
        # Convolution: 3D+ weight interfaces with spatial patterns
        if (len(getattr(interface_info, 'tensor_dims', [])) >= 3 and
            interface_info.interface_type == InterfaceType.WEIGHT and
            self._has_spatial_pattern(rtl_result)):
            return "convolution"
        
        # Threshold operations: output interfaces with comparison patterns
        if (interface_info.interface_type == InterfaceType.OUTPUT and
            self._has_threshold_pattern(rtl_result)):
            return "threshold"
        
        # Element-wise operations: simple input/output without complex patterns
        if self._has_elementwise_pattern(rtl_result):
            return "elementwise"
        
        return "generic"
    
    def _analyze_memory_access_pattern(self, interface_info, rtl_result: EnhancedRTLParsingResult) -> Dict[str, Any]:
        """Analyze memory access patterns for optimization."""
        
        pattern = {
            "access_type": "unknown",
            "parallelism_factor": 1,
            "memory_depth": 0,
            "burst_support": False
        }
        
        # Analyze RTL for memory access patterns
        if hasattr(rtl_result, 'memory_analysis'):
            memory_info = rtl_result.memory_analysis.get(interface_info.name, {})
            
            pattern.update({
                "access_type": memory_info.get("access_type", "sequential"),
                "parallelism_factor": memory_info.get("parallel_ports", 1),
                "memory_depth": memory_info.get("depth", 0),
                "burst_support": memory_info.get("burst_capable", False)
            })
        
        return pattern
    
    def _generate_tensor_formatting_hints(self, 
                                        interface_info, 
                                        operation_type: str,
                                        optimization_hints: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific tensor formatting hints based on analysis."""
        
        hints = {
            "operation_type": operation_type,
            "preferred_layout": "default"
        }
        
        # Matrix multiplication specific hints
        if operation_type == "matrix_multiplication":
            hints.update({
                "preferred_layout": "weight_major",
                "transpose_required": True,
                "pe_distribution": "outer_dim",
                "simd_optimization": optimization_hints.get("needs_simd_flip", True)
            })
        
        # Convolution specific hints
        elif operation_type == "convolution":
            hints.update({
                "preferred_layout": "channel_major",
                "spatial_flattening": True,
                "pe_distribution": "channel_wise",
                "simd_optimization": optimization_hints.get("needs_simd_flip", True)
            })
        
        # Threshold specific hints
        elif operation_type == "threshold":
            hints.update({
                "preferred_layout": "channel_major",
                "pe_distribution": "channel_wise",
                "simd_optimization": False  # Thresholds typically don't need SIMD flip
            })
        
        return hints
    
    # RTL Pattern Detection Methods
    
    def _has_simd_optimization_pragma(self, interface_info, rtl_result: EnhancedRTLParsingResult) -> bool:
        """Check for SIMD optimization pragmas in RTL."""
        # Look for SIMD-related pragmas or patterns
        if hasattr(rtl_result, 'pragmas'):
            simd_pragmas = [p for p in rtl_result.pragmas 
                          if 'SIMD' in p.content or 'simd' in p.content.lower()]
            return len(simd_pragmas) > 0
        return True  # Default to True for most operations
    
    def _requires_matrix_transpose(self, interface_info, rtl_result: EnhancedRTLParsingResult) -> bool:
        """Check if interface requires matrix transpose for optimal access."""
        # Matrix operations typically require transpose for hardware efficiency
        return (interface_info.interface_type == InterfaceType.WEIGHT and
                len(getattr(interface_info, 'tensor_dims', [])) == 2)
    
    def _requires_pe_distribution(self, interface_info, rtl_result: EnhancedRTLParsingResult) -> bool:
        """Check if interface requires PE distribution."""
        # Most weight and output interfaces benefit from PE distribution
        return interface_info.interface_type in [InterfaceType.WEIGHT, InterfaceType.OUTPUT]
    
    def _has_memory_optimization_patterns(self, interface_info, rtl_result: EnhancedRTLParsingResult) -> bool:
        """Check for memory optimization patterns in RTL."""
        # Look for memory optimization indicators
        if hasattr(rtl_result, 'memory_analysis'):
            return interface_info.name in rtl_result.memory_analysis
        return False
    
    def _has_convolution_patterns(self, interface_info, rtl_result: EnhancedRTLParsingResult) -> bool:
        """Check for convolution-specific patterns."""
        # Look for spatial dimension handling or kernel patterns
        if hasattr(rtl_result, 'operation_patterns'):
            conv_patterns = ['kernel', 'spatial', 'conv', 'filter']
            return any(pattern in str(rtl_result.operation_patterns).lower() 
                      for pattern in conv_patterns)
        return False
    
    def _has_mac_pattern(self, rtl_result: EnhancedRTLParsingResult) -> bool:
        """Check for multiply-accumulate patterns (matrix operations)."""
        # Look for MAC units or matrix multiplication patterns
        if hasattr(rtl_result, 'computation_patterns'):
            mac_patterns = ['mac', 'multiply', 'accumulate', 'dot_product', 'matmul']
            return any(pattern in str(rtl_result.computation_patterns).lower()
                      for pattern in mac_patterns)
        return True  # Default assumption for weight interfaces
    
    def _has_spatial_pattern(self, rtl_result: EnhancedRTLParsingResult) -> bool:
        """Check for spatial processing patterns (convolution operations)."""
        # Look for spatial dimension processing
        if hasattr(rtl_result, 'spatial_analysis'):
            return len(rtl_result.spatial_analysis) > 0
        return False
    
    def _has_threshold_pattern(self, rtl_result: EnhancedRTLParsingResult) -> bool:
        """Check for threshold/comparison patterns."""
        # Look for comparison or threshold operations
        if hasattr(rtl_result, 'operation_patterns'):
            threshold_patterns = ['threshold', 'compare', 'relu', 'activation']
            return any(pattern in str(rtl_result.operation_patterns).lower()
                      for pattern in threshold_patterns)
        return False
    
    def _has_elementwise_pattern(self, rtl_result: EnhancedRTLParsingResult) -> bool:
        """Check for element-wise operation patterns."""
        # Look for simple element-wise operations
        if hasattr(rtl_result, 'operation_patterns'):
            elementwise_patterns = ['add', 'sub', 'mul', 'div', 'elementwise']
            return any(pattern in str(rtl_result.operation_patterns).lower()
                      for pattern in elementwise_patterns)
        return False
    
    def extract_template_context_enhancements(self, 
                                            interfaces: List[InterfaceMetadata]) -> Dict[str, Any]:
        """
        Extract template context enhancements for automatic tensor formatting.
        
        This method provides additional context variables that the enhanced
        template can use to generate appropriate automatic tensor formatting methods.
        
        Args:
            interfaces: List of analyzed interface metadata
            
        Returns:
            Template context enhancements
        """
        context = {
            "has_weight_interfaces": False,
            "has_threshold_interfaces": False,
            "has_memory_calculations": False,
            "operation_types": set(),
            "optimization_features": set()
        }
        
        for interface in interfaces:
            # Check for weight interfaces
            if interface.interface_type == InterfaceType.WEIGHT:
                context["has_weight_interfaces"] = True
                context["has_memory_calculations"] = True
            
            # Check for threshold interfaces (output interfaces with threshold patterns)
            if (interface.interface_type == InterfaceType.OUTPUT and
                interface.metadata and
                interface.metadata.get("operation_type") == "threshold"):
                context["has_threshold_interfaces"] = True
                context["has_memory_calculations"] = True
            
            # Collect operation types
            if interface.metadata and "operation_type" in interface.metadata:
                context["operation_types"].add(interface.metadata["operation_type"])
            
            # Collect optimization features
            if interface.metadata and "optimization_flags" in interface.metadata:
                for flag, value in interface.metadata["optimization_flags"].items():
                    if value:
                        context["optimization_features"].add(flag)
        
        # Convert sets to lists for template compatibility
        context["operation_types"] = list(context["operation_types"])
        context["optimization_features"] = list(context["optimization_features"])
        
        return context


def create_enhanced_interface_analyzer() -> EnhancedInterfaceAnalyzer:
    """Factory function to create enhanced interface analyzer."""
    return EnhancedInterfaceAnalyzer()