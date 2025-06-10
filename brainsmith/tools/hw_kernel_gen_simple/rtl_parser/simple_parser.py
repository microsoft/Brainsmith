"""
Simple RTL parser implementation.

Wraps the existing RTL parser with a clean, simple interface.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from ..errors import RTLParsingError


@dataclass
class RTLData:
    """Simple representation of parsed RTL data."""
    module_name: str
    interfaces: List[Dict[str, Any]]
    parameters: List[Dict[str, Any]]
    pragmas: List[Dict[str, Any]]


def parse_rtl_file(rtl_file: Path) -> RTLData:
    """
    Parse RTL file and return simple data structure.
    
    Args:
        rtl_file: Path to SystemVerilog file
        
    Returns:
        RTLData: Parsed RTL information
        
    Raises:
        RTLParsingError: If parsing fails
    """
    try:
        # Import the existing RTL parser
        from brainsmith.tools.hw_kernel_gen.rtl_parser import RTLParser
        
        # Create parser and parse file
        parser = RTLParser(debug=False)
        hw_kernel = parser.parse_file(str(rtl_file))
        
        # Convert to simple format
        interfaces = []
        for interface_name, interface_obj in hw_kernel.interfaces.items():
            try:
                # Create a type object with name attribute for template compatibility
                type_obj = type('InterfaceType', (), {'name': _safe_get_type(interface_obj)})()
                
                interface_dict = {
                    'name': interface_name,
                    'type': type_obj,
                    'dataflow_type': _map_to_dataflow_type(interface_obj),
                    'ports': _safe_get_ports(interface_obj),
                    'enhanced_tdim': _extract_enhanced_tdim(interface_obj),
                    'datatype_constraints': _extract_datatype_constraints(interface_obj)
                }
                interfaces.append(interface_dict)
            except Exception as e:
                print(f"Warning: Failed to process interface {interface_name}: {e}")
                continue
        
        parameters = []
        for param in hw_kernel.parameters:
            param_dict = {
                'name': param.name,
                'default_value': getattr(param, 'default_value', None),
                'type': getattr(param, 'param_type', 'integer'),
                'description': getattr(param, 'description', '')
            }
            parameters.append(param_dict)
        
        pragmas = []
        for pragma in hw_kernel.pragmas:
            pragma_dict = {
                'type': pragma.type.name if hasattr(pragma.type, 'name') else str(pragma.type),
                'content': getattr(pragma, 'content', ''),
                'line_number': getattr(pragma, 'line_number', 0)
            }
            pragmas.append(pragma_dict)
        
        return RTLData(
            module_name=hw_kernel.name,
            interfaces=interfaces,
            parameters=parameters,
            pragmas=pragmas
        )
        
    except Exception as e:
        raise RTLParsingError(f"Failed to parse RTL file {rtl_file}: {e}") from e


def _safe_get_type(interface_obj) -> str:
    """Safely get interface type as string."""
    if hasattr(interface_obj, 'type'):
        interface_type = interface_obj.type
        if hasattr(interface_type, 'name'):
            return interface_type.name
        else:
            return str(interface_type)
    return 'UNKNOWN'


def _safe_get_direction(port) -> str:
    """Safely get port direction as string."""
    if hasattr(port, 'direction'):
        direction = port.direction
        if hasattr(direction, 'name'):
            return direction.name
        else:
            return str(direction)
    return 'UNKNOWN'


def _safe_get_ports(interface_obj) -> List[Dict[str, Any]]:
    """Safely extract ports from interface object."""
    ports = []
    
    if hasattr(interface_obj, 'ports'):
        ports_attr = interface_obj.ports
        
        # Handle both dict and list formats
        if isinstance(ports_attr, dict):
            for port_name, port_obj in ports_attr.items():
                try:
                    port_dict = {
                        'name': getattr(port_obj, 'name', port_name),
                        'direction': _safe_get_direction(port_obj),
                        'width': getattr(port_obj, 'width', '1'),
                        'is_signed': getattr(port_obj, 'is_signed', False)
                    }
                    ports.append(port_dict)
                except Exception as e:
                    print(f"Warning: Failed to process port {port_name}: {e}")
                    continue
        elif isinstance(ports_attr, list):
            for port_obj in ports_attr:
                try:
                    port_dict = {
                        'name': getattr(port_obj, 'name', 'unknown'),
                        'direction': _safe_get_direction(port_obj),
                        'width': getattr(port_obj, 'width', '1'),
                        'is_signed': getattr(port_obj, 'is_signed', False)
                    }
                    ports.append(port_dict)
                except Exception as e:
                    print(f"Warning: Failed to process port: {e}")
                    continue
    
    return ports


def _map_to_dataflow_type(interface_obj) -> str:
    """Map interface object to dataflow type string."""
    # Use metadata direction if available
    if hasattr(interface_obj, 'metadata') and 'direction' in interface_obj.metadata:
        direction = interface_obj.metadata['direction']
        if hasattr(direction, 'name'):
            direction_name = direction.name
        else:
            direction_name = str(direction)
            
        if 'INPUT' in direction_name:
            return 'INPUT'
        elif 'OUTPUT' in direction_name:
            return 'OUTPUT'
    
    # Fallback to interface type and name patterns
    name = interface_obj.name.lower()
    
    if hasattr(interface_obj.type, 'name'):
        type_name = interface_obj.type.name
    else:
        type_name = str(interface_obj.type)
        
    if 'AXI_STREAM' in type_name:
        # Use naming convention as fallback
        if 's_axis' in name or 'input' in name:
            return 'INPUT'
        elif 'm_axis' in name or 'output' in name:
            return 'OUTPUT'
    elif 'AXI_LITE' in type_name:
        return 'CONFIG'
    elif 'GLOBAL_CONTROL' in type_name:
        return 'CONTROL'
    
    # Name-based fallback
    if 'weight' in name or 'param' in name:
        return 'WEIGHT'
    elif 'config' in name or 'ctrl' in name:
        return 'CONFIG'
    
    return 'INPUT'  # Safe default


def _extract_enhanced_tdim(interface_obj) -> Dict[str, Any]:
    """Extract enhanced TDIM information from interface."""
    # Look for TDIM pragma information
    if hasattr(interface_obj, 'pragmas'):
        for pragma in interface_obj.pragmas:
            if hasattr(pragma, 'type') and 'TDIM' in str(pragma.type):
                # Parse TDIM pragma
                content = pragma.content
                if isinstance(content, str) and 'TDIM' in content:
                    parts = content.strip().split()
                    if len(parts) >= 4:
                        try:
                            chunk_index = int(parts[3])
                            chunk_sizes = parts[4] if len(parts) > 4 else "[16]"
                            return {
                                'chunk_index': chunk_index,
                                'chunk_sizes': chunk_sizes
                            }
                        except (ValueError, IndexError):
                            pass
    
    # Default: no enhanced TDIM
    return None


def _extract_datatype_constraints(interface_obj) -> List[Dict[str, Any]]:
    """Extract datatype constraints from interface."""
    # Default constraints based on interface type
    name = interface_obj.name.lower()
    
    if 'weight' in name or 'param' in name:
        # Weight interfaces typically use signed types
        return [
            {'finn_type': 'INT8', 'bit_width': 8, 'signed': True},
            {'finn_type': 'INT16', 'bit_width': 16, 'signed': True}
        ]
    elif 'config' in name or 'ctrl' in name:
        # Config interfaces use integer types
        return [
            {'finn_type': 'UINT32', 'bit_width': 32, 'signed': False}
        ]
    else:
        # Default to unsigned 8-bit for activation streams
        return [
            {'finn_type': 'UINT8', 'bit_width': 8, 'signed': False}
        ]