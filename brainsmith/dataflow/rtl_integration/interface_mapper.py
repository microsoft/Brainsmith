"""
Interface Mapper for RTL to DataflowInterface conversion.

This module maps RTL interfaces (from HWKG RTL parser) to DataflowInterface
objects, handling interface type mapping, default tensor shape inference,
and integration with chunking strategies and datatype constraints.

This supports HWKG Axiom 1: Interface-Wise Dataflow Foundation by providing
the interface mapping component of the RTL → DataflowInterface → FINN pipeline.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple

from ..core.dataflow_interface import DataflowInterface, DataflowInterfaceType, DataflowDataType
from ..core.interface_metadata import DataTypeConstraint
from ..core.block_chunking import ChunkingStrategy, DefaultChunkingStrategy

logger = logging.getLogger(__name__)


class InterfaceMapper:
    """
    Map RTL interfaces to DataflowInterface objects.
    
    This class handles the conversion from RTL parser interface representations
    to DataflowInterface objects, including interface type mapping, tensor shape
    inference, and integration with pragma-derived chunking strategies.
    
    Responsibilities:
    - Map RTL interface types to DataflowInterfaceType enum
    - Infer default tensor shapes from RTL port information
    - Integrate chunking strategies from pragma conversion
    - Apply datatype constraints from pragma conversion
    - Create complete DataflowInterface objects
    """
    
    def __init__(self):
        """Initialize interface mapper."""
        # RTL interface type to DataflowInterfaceType mapping
        self.interface_type_mapping = {
            'AXI_STREAM': DataflowInterfaceType.INPUT,    # Default AXI-Stream to INPUT
            'GLOBAL_CONTROL': DataflowInterfaceType.CONFIG,
            'AXI_LITE': DataflowInterfaceType.CONFIG,
            'UNKNOWN': DataflowInterfaceType.INPUT        # Default unknown to INPUT
        }
        
        # Port pattern mapping for interface type inference
        self.port_patterns = {
            'input': ['tdata', 'tvalid', 'tready', 'data', 'valid', 'ready'],
            'output': ['tdata', 'tvalid', 'tready', 'data', 'valid', 'ready'],
            'weight': ['weight', 'weights', 'param', 'parameters'],
            'config': ['config', 'ctrl', 'control', 'enable', 'start', 'ap_']
        }
    
    def map_interface_type(self, rtl_interface) -> DataflowInterfaceType:
        """
        Map RTL interface type to DataflowInterfaceType.
        
        Args:
            rtl_interface: RTL Interface object from parser
            
        Returns:
            DataflowInterfaceType: Mapped interface type
        """
        try:
            # Get RTL interface type
            rtl_type = rtl_interface.type.name if hasattr(rtl_interface.type, 'name') else str(rtl_interface.type)
            
            # Check for weight interface hints in metadata or name
            if self._is_weight_interface(rtl_interface):
                return DataflowInterfaceType.WEIGHT
            
            # Check for output interface hints
            if self._is_output_interface(rtl_interface):
                return DataflowInterfaceType.OUTPUT
                
            # Use default mapping
            mapped_type = self.interface_type_mapping.get(rtl_type, DataflowInterfaceType.INPUT)
            
            logger.debug(f"Mapped RTL interface type '{rtl_type}' to DataflowInterfaceType.{mapped_type.name}")
            return mapped_type
            
        except Exception as e:
            logger.warning(f"Failed to map interface type for {rtl_interface.name}: {str(e)}")
            return DataflowInterfaceType.INPUT  # Safe default
    
    def _is_weight_interface(self, rtl_interface) -> bool:
        """Check if RTL interface should be mapped to WEIGHT type."""
        interface_name = rtl_interface.name.lower()
        
        # Check name patterns
        for pattern in self.port_patterns['weight']:
            if pattern in interface_name:
                return True
        
        # Check metadata for weight hints
        if hasattr(rtl_interface, 'metadata') and rtl_interface.metadata:
            if rtl_interface.metadata.get('is_weight', False):
                return True
            if 'weight' in str(rtl_interface.metadata).lower():
                return True
                
        return False
    
    def _is_output_interface(self, rtl_interface) -> bool:
        """Check if RTL interface should be mapped to OUTPUT type."""
        interface_name = rtl_interface.name.lower()
        
        # Check for output patterns
        output_patterns = ['out', 'output', 'result', 'y', 'res']
        for pattern in output_patterns:
            if interface_name.startswith(pattern):
                return True
                
        # Check port directions
        if hasattr(rtl_interface, 'ports') and rtl_interface.ports:
            output_ports = 0
            total_ports = 0
            
            for port in rtl_interface.ports.values():
                if hasattr(port, 'direction'):
                    total_ports += 1
                    if port.direction.name == 'OUTPUT':
                        output_ports += 1
            
            # If majority of ports are outputs, consider it an output interface
            if total_ports > 0 and output_ports / total_ports > 0.5:
                return True
                
        return False
    
    def infer_tensor_shape(self, rtl_interface, hw_kernel=None) -> List[int]:
        """
        Infer default tensor shape from RTL interface information.
        
        Args:
            rtl_interface: RTL Interface object
            hw_kernel: Parent HWKernel for context (optional)
            
        Returns:
            List[int]: Inferred tensor shape
        """
        try:
            # Try to extract width information from ports
            data_width = self._extract_data_width(rtl_interface)
            
            if data_width and data_width > 1:
                # For multi-bit interfaces, use width as tensor dimension
                return [data_width]
            else:
                # Default fallback shape
                return [128]  # 1D tensor with 128 elements
                
        except Exception as e:
            logger.warning(f"Failed to infer tensor shape for {rtl_interface.name}: {str(e)}")
            return [128]  # Safe fallback
    
    def _extract_data_width(self, rtl_interface) -> Optional[int]:
        """
        Extract data width from RTL interface ports.
        
        Args:
            rtl_interface: RTL Interface object
            
        Returns:
            Data width in bits or None if not found
        """
        if not hasattr(rtl_interface, 'ports') or not rtl_interface.ports:
            return None
            
        # Look for data ports (TDATA, data, etc.)
        data_ports = []
        for port_name, port in rtl_interface.ports.items():
            port_name_lower = port_name.lower()
            if any(pattern in port_name_lower for pattern in ['tdata', 'data', 'value']):
                data_ports.append(port)
        
        if not data_ports:
            # No specific data ports found, use first port
            data_ports = list(rtl_interface.ports.values())[:1]
        
        # Extract width from first data port
        if data_ports:
            port = data_ports[0]
            if hasattr(port, 'width') and port.width:
                try:
                    # Try to parse width as integer
                    if port.width.isdigit():
                        return int(port.width)
                    else:
                        # Handle width expressions like "[7:0]"
                        import re
                        match = re.search(r'\[(\d+):0\]', port.width)
                        if match:
                            return int(match.group(1)) + 1
                        # Handle simple expressions like "8"
                        match = re.search(r'(\d+)', port.width)
                        if match:
                            return int(match.group(1))
                except (ValueError, AttributeError):
                    pass
        
        return None
    
    def create_dataflow_interface(self, rtl_interface, interface_type: DataflowInterfaceType,
                                 chunking_strategy: Optional[ChunkingStrategy] = None,
                                 dtype_constraint: Optional[DataTypeConstraint] = None,
                                 hw_kernel=None) -> Optional[DataflowInterface]:
        """
        Create DataflowInterface from RTL interface with pragma integration.
        
        Args:
            rtl_interface: RTL Interface object
            interface_type: Mapped DataflowInterfaceType
            chunking_strategy: ChunkingStrategy from pragma conversion (optional)
            dtype_constraint: DataTypeConstraint from pragma conversion (optional)
            hw_kernel: Parent HWKernel for context (optional)
            
        Returns:
            DataflowInterface instance or None if creation fails
        """
        try:
            # Step 1: Infer tensor shape
            tensor_dims = self.infer_tensor_shape(rtl_interface, hw_kernel)
            
            # Step 2: Apply chunking strategy to get block_dims
            if chunking_strategy:
                try:
                    _, block_dims = chunking_strategy.compute_chunking(tensor_dims, rtl_interface.name)
                except Exception as e:
                    logger.warning(f"Failed to apply chunking strategy: {str(e)}")
                    block_dims = tensor_dims.copy()  # Fallback to full tensor
            else:
                block_dims = tensor_dims.copy()  # Default: no chunking
            
            # Step 3: Create datatype from constraint or infer default
            if dtype_constraint:
                dataflow_dtype = DataflowDataType(
                    base_type=dtype_constraint.allowed_types[0] if dtype_constraint.allowed_types else "UINT",
                    bitwidth=dtype_constraint.max_bits,
                    signed=dtype_constraint.signed,
                    finn_type=dtype_constraint.finn_type
                )
            else:
                # Infer default datatype
                data_width = self._extract_data_width(rtl_interface)
                bitwidth = data_width if data_width and data_width <= 32 else 8
                
                dataflow_dtype = DataflowDataType(
                    base_type="UINT",
                    bitwidth=bitwidth,
                    signed=False,
                    finn_type=f"UINT{bitwidth}"
                )
            
            # Step 4: Initialize stream_dims with minimal parallelism
            stream_dims = [1] * len(block_dims)
            
            # Step 5: Create DataflowInterface
            dataflow_interface = DataflowInterface(
                name=rtl_interface.name,
                interface_type=interface_type,
                tensor_dims=tensor_dims,
                block_dims=block_dims,
                stream_dims=stream_dims,
                dtype=dataflow_dtype
            )
            
            logger.debug(f"Created DataflowInterface for {rtl_interface.name}: "
                        f"type={interface_type.name}, tensor_dims={tensor_dims}, "
                        f"block_dims={block_dims}, dtype={dataflow_dtype.finn_type}")
            
            return dataflow_interface
            
        except Exception as e:
            logger.error(f"Failed to create DataflowInterface for {rtl_interface.name}: {str(e)}")
            return None
    
    def create_interface_with_defaults(self, interface_name: str, 
                                     interface_type: DataflowInterfaceType) -> DataflowInterface:
        """
        Create DataflowInterface with sensible defaults.
        
        Args:
            interface_name: Name of the interface
            interface_type: DataflowInterfaceType
            
        Returns:
            DataflowInterface with default configuration
        """
        # Default configuration
        tensor_dims = [128]  # 1D tensor, 128 elements
        block_dims = [128]   # Process full tensor by default
        stream_dims = [1]    # Minimal parallelism
        
        dataflow_dtype = DataflowDataType(
            base_type="UINT",
            bitwidth=8,
            signed=False,
            finn_type="UINT8"
        )
        
        return DataflowInterface(
            name=interface_name,
            interface_type=interface_type,
            tensor_dims=tensor_dims,
            block_dims=block_dims,
            stream_dims=stream_dims,
            dtype=dataflow_dtype
        )
    
    def validate_interface_compatibility(self, rtl_interface, dataflow_interface: DataflowInterface) -> bool:
        """
        Validate compatibility between RTL and DataflowInterface.
        
        Args:
            rtl_interface: Original RTL Interface
            dataflow_interface: Created DataflowInterface
            
        Returns:
            bool: True if compatible, False otherwise
        """
        try:
            # Check name consistency
            if rtl_interface.name != dataflow_interface.name:
                logger.warning(f"Interface name mismatch: RTL={rtl_interface.name}, "
                             f"Dataflow={dataflow_interface.name}")
                return False
            
            # Check tensor dimensions make sense
            if not dataflow_interface.tensor_dims or any(d <= 0 for d in dataflow_interface.tensor_dims):
                logger.warning(f"Invalid tensor dimensions: {dataflow_interface.tensor_dims}")
                return False
            
            # Check block dimensions consistency
            if len(dataflow_interface.tensor_dims) != len(dataflow_interface.block_dims):
                logger.warning(f"Tensor/block dimension mismatch: "
                             f"tensor={dataflow_interface.tensor_dims}, "
                             f"block={dataflow_interface.block_dims}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating interface compatibility: {str(e)}")
            return False


def create_interface_mapper() -> InterfaceMapper:
    """
    Factory function for creating InterfaceMapper instances.
    
    Returns:
        InterfaceMapper: Configured mapper instance
    """
    return InterfaceMapper()