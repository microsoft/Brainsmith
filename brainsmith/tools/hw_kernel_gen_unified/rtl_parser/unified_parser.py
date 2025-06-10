"""
Unified RTL parser with optional BDIM sophistication.

Based on hw_kernel_gen_simple safe extraction approach with optional
enhanced BDIM pragma processing for advanced users.
Follows RTL Parser Axioms and Interface-Wise Dataflow Modeling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..errors import RTLParsingError, BDimProcessingError
from ..data import UnifiedHWKernel


def parse_rtl_file(rtl_file: Path, advanced_pragmas: bool = False) -> UnifiedHWKernel:
    """
    Parse RTL file and return unified hardware kernel representation.
    
    Args:
        rtl_file: Path to SystemVerilog file
        advanced_pragmas: Enable enhanced BDIM pragma processing
        
    Returns:
        UnifiedHWKernel: Enhanced kernel representation
        
    Raises:
        RTLParsingError: If parsing fails
        
    Following RTL Parser Axiom 1: Parser Pipeline
    SystemVerilog → AST → Interfaces → Templates
    """
    try:
        # Always use simple parser as foundation (error resilience)
        from brainsmith.tools.hw_kernel_gen_simple.rtl_parser import parse_rtl_file as simple_parse
        
        # Parse with simple system (proven robust)
        rtl_data = simple_parse(rtl_file)
        
        # Convert to unified format
        unified_kernel = _convert_to_unified(rtl_data, rtl_file)
        
        # Enhance with BDIM processing if enabled
        if advanced_pragmas:
            unified_kernel = _enhance_with_bdim(unified_kernel, rtl_file)
        
        return unified_kernel
        
    except Exception as e:
        raise RTLParsingError(f"Failed to parse RTL file {rtl_file}: {e}") from e


def _convert_to_unified(rtl_data, rtl_file: Path) -> UnifiedHWKernel:
    """
    Convert simple RTL data to unified format.
    
    Maintains all smart properties and error resilience from simple system
    while preparing for optional BDIM enhancement.
    """
    # Generate class name using same logic as simple system
    class_name = _generate_class_name(rtl_data.module_name)
    
    # Extract compiler data safely (should be empty dict from simple parser)
    compiler_data = {}
    
    unified_kernel = UnifiedHWKernel(
        name=rtl_data.module_name,
        class_name=class_name,
        interfaces=rtl_data.interfaces,
        rtl_parameters=rtl_data.parameters,
        source_file=rtl_file,
        compiler_data=compiler_data,
        pragma_sophistication_level="simple"
    )
    
    return unified_kernel


def _enhance_with_bdim(kernel: UnifiedHWKernel, rtl_file: Path) -> UnifiedHWKernel:
    """
    Add enhanced BDIM pragma processing using sophisticated RTL parser.
    
    Graceful degradation - log warnings but continue if enhancement fails.
    Following Interface-Wise Dataflow Axiom 4: Pragma-to-Chunking Conversion.
    """
    try:
        # Import sophisticated RTL parser for BDIM extraction
        from brainsmith.tools.hw_kernel_gen.rtl_parser import RTLParser
        from ..pragma_integration import BDimProcessor
        
        # Use sophisticated parser for BDIM metadata
        full_parser = RTLParser(debug=False)
        hw_kernel_full = full_parser.parse_file(str(rtl_file))
        
        # Extract BDIM metadata using processor
        bdim_processor = BDimProcessor()
        bdim_metadata = bdim_processor.extract_bdim_metadata(hw_kernel_full)
        
        if bdim_metadata:
            kernel.bdim_metadata = bdim_metadata
            kernel.pragma_sophistication_level = "advanced"
            
            # Enhance interfaces with BDIM information
            kernel.interfaces = _enhance_interfaces_with_bdim(kernel.interfaces, bdim_metadata)
        
    except ImportError as e:
        warning = f"Advanced BDIM processing unavailable: {e}"
        kernel.add_parsing_warning(warning)
        print(f"Warning: {warning}")
        
    except Exception as e:
        # Graceful degradation - log warning but continue
        warning = f"Advanced BDIM processing failed: {e}"
        kernel.add_parsing_warning(warning)
        print(f"Warning: {warning}")
        print("Continuing with simple pragma processing...")
    
    return kernel


def _enhance_interfaces_with_bdim(interfaces: List[Dict[str, Any]], bdim_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Enhance interface data with BDIM metadata.
    
    Following Interface-Wise Dataflow Axioms for tensor_dims, block_dims, stream_dims.
    """
    enhanced_interfaces = []
    
    for iface in interfaces:
        enhanced_iface = iface.copy()
        
        interface_name = iface.get('name', '')
        
        # Add BDIM chunking information if available
        if interface_name in bdim_metadata.get('chunking_strategies', {}):
            chunking_info = bdim_metadata['chunking_strategies'][interface_name]
            enhanced_iface.update({
                'tensor_dims': chunking_info.get('tensor_dims', []),
                'block_dims': chunking_info.get('block_dims', []),
                'stream_dims': chunking_info.get('stream_dims', []),
                'chunking_strategy': chunking_info.get('strategy_type', 'default')
            })
        
        # Add dataflow type classification following Axiom 3: Interface Types
        if iface.get('type', {}).get('name') == 'AXI_STREAM':
            enhanced_iface['dataflow_category'] = _classify_dataflow_interface(interface_name)
        
        enhanced_interfaces.append(enhanced_iface)
    
    return enhanced_interfaces


def _classify_dataflow_interface(interface_name: str) -> str:
    """
    Classify AXI-Stream interface into dataflow categories.
    
    Following Interface-Wise Dataflow Axiom 3: Interface Types
    - Input: AXI-Stream activation data in
    - Output: AXI-Stream activation data out  
    - Weight: AXI-Stream weight data in
    """
    name_lower = interface_name.lower()
    
    # Weight patterns
    if any(pattern in name_lower for pattern in ['weight', 'w_axis', 'param']):
        return "WEIGHT"
    # Input patterns (s_axis = slave = input)
    elif any(pattern in name_lower for pattern in ['s_axis', 'input', 'in_']) or name_lower.startswith('s_'):
        return "INPUT"
    # Output patterns (m_axis = master = output)
    elif any(pattern in name_lower for pattern in ['m_axis', 'output', 'out_']) or name_lower.startswith('m_'):
        return "OUTPUT"
    else:
        # Default based on AXI naming convention
        return "INPUT"


def _generate_class_name(module_name: str) -> str:
    """Generate Python class name from module name (same as simple system)."""
    # Convert snake_case or kebab-case to PascalCase
    parts = module_name.replace('-', '_').split('_')
    return ''.join(word.capitalize() for word in parts)


class UnifiedRTLParser:
    """
    Unified RTL parser class with optional BDIM sophistication.
    
    Provides object-oriented interface for advanced users while maintaining
    the simple parse_rtl_file function for basic use cases.
    """
    
    def __init__(self, advanced_pragmas: bool = False, debug: bool = False):
        """
        Initialize unified RTL parser.
        
        Args:
            advanced_pragmas: Enable enhanced BDIM pragma processing
            debug: Enable debug output
        """
        self.advanced_pragmas = advanced_pragmas
        self.debug = debug
        
        if advanced_pragmas:
            try:
                from ..pragma_integration import BDimProcessor
                self.bdim_processor = BDimProcessor()
            except ImportError:
                if debug:
                    print("Warning: BDIM processor not available, advanced pragmas disabled")
                self.advanced_pragmas = False
    
    def parse_file(self, rtl_file: Path) -> UnifiedHWKernel:
        """Parse RTL file and return unified kernel representation."""
        return parse_rtl_file(rtl_file, advanced_pragmas=self.advanced_pragmas)
    
    def parse_string(self, rtl_content: str, module_name: str = "unknown") -> UnifiedHWKernel:
        """
        Parse RTL content from string.
        
        Following RTL Parser Axiom 7: Dual Input Support.
        """
        # Write to temporary file and parse
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sv', delete=False) as tmp_file:
            tmp_file.write(rtl_content)
            tmp_path = Path(tmp_file.name)
        
        try:
            return self.parse_file(tmp_path)
        finally:
            # Clean up temporary file
            tmp_path.unlink(missing_ok=True)