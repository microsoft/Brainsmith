"""
RTL Integration Module for Interface-Wise Dataflow Modeling.

This module provides the bridge between RTL parsing (from HWKG) and 
DataflowModel creation, enabling automated conversion of SystemVerilog
RTL with @brainsmith pragmas into complete DataflowModel instances.

Core Components:
- RTLDataflowConverter: Convert HWKernel to DataflowModel 
- PragmaToStrategyConverter: Convert @brainsmith pragmas to ChunkingStrategy
- InterfaceMapper: Map RTL interfaces to DataflowInterface objects

This module implements HWKG Axiom 1: Interface-Wise Dataflow Foundation
by providing the missing RTL → DataflowInterface → FINN pipeline.
"""

from .rtl_converter import RTLDataflowConverter
from .pragma_converter import PragmaToStrategyConverter  
from .interface_mapper import InterfaceMapper

__all__ = [
    'RTLDataflowConverter',
    'PragmaToStrategyConverter', 
    'InterfaceMapper'
]


def create_interface_metadata(name: str, interface_type: str, 
                            chunking_strategy, dtype_constraints, 
                            axi_metadata) -> 'InterfaceMetadata':
    """
    Factory function for creating InterfaceMetadata instances.
    
    This is the main entry point for template generation to create
    interface metadata from RTL parsing results.
    
    Args:
        name: Interface name
        interface_type: Interface type (INPUT/OUTPUT/WEIGHT/CONFIG)
        chunking_strategy: ChunkingStrategy instance
        dtype_constraints: DataTypeConstraint instance
        axi_metadata: AXI interface metadata dict
        
    Returns:
        InterfaceMetadata: Complete interface metadata instance
    """
    from ..core.interface_metadata import InterfaceMetadata, DataTypeConstraint
    from ..core.dataflow_interface import DataflowInterfaceType
    
    # Convert string interface type to enum
    if isinstance(interface_type, str):
        interface_type = DataflowInterfaceType(interface_type)
    
    return InterfaceMetadata(
        name=name,
        interface_type=interface_type,
        chunking_strategy=chunking_strategy,
        dtype_constraint=dtype_constraints,
        axi_metadata=axi_metadata
    )