"""
RTL to DataflowModel Converter.

This module implements the core conversion pipeline from HWKernel (RTL parsing result)
to DataflowModel instances, bridging the gap between HWKG's RTL parsing and the
Interface-Wise Dataflow Modeling system.

This implements HWKG Axiom 1: Interface-Wise Dataflow Foundation by providing
the missing RTL → DataflowInterface → FINN pipeline.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..core.dataflow_model import DataflowModel
from ..core.dataflow_interface import DataflowInterface, DataflowInterfaceType, DataflowDataType
from ..core.interface_metadata import InterfaceMetadata, DataTypeConstraint
from .pragma_converter import PragmaToStrategyConverter
from .interface_mapper import InterfaceMapper

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Result of RTL to DataflowModel conversion."""
    dataflow_model: Optional[DataflowModel]
    success: bool
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class RTLDataflowConverter:
    """
    Convert HWKernel to DataflowModel with complete pragma processing.
    
    This class implements the core integration between HWKG's RTL parsing
    and the Interface-Wise Dataflow Modeling system, enabling automatic
    conversion of SystemVerilog RTL with @brainsmith pragmas into complete
    DataflowModel instances.
    
    Conversion Pipeline:
    1. Extract interfaces from HWKernel
    2. Apply pragma-based chunking strategies  
    3. Build DataflowInterface objects
    4. Create unified DataflowModel
    """
    
    def __init__(self):
        """Initialize RTL to DataflowModel converter."""
        self.pragma_converter = PragmaToStrategyConverter()
        self.interface_mapper = InterfaceMapper()
        
    def convert(self, hw_kernel) -> ConversionResult:
        """
        Complete conversion pipeline from HWKernel to DataflowModel.
        
        Args:
            hw_kernel: HWKernel instance from RTL parser
            
        Returns:
            ConversionResult: Success/failure with DataflowModel or errors
        """
        try:
            logger.info(f"Starting RTL to DataflowModel conversion for kernel: {hw_kernel.name}")
            
            # Step 1: Validate HWKernel input
            validation_result = self._validate_hw_kernel(hw_kernel)
            if not validation_result.success:
                return validation_result
                
            # Step 2: Convert RTL interfaces to DataflowInterface objects
            dataflow_interfaces = []
            conversion_errors = []
            
            for interface_name, rtl_interface in hw_kernel.interfaces.items():
                logger.debug(f"Converting interface: {interface_name}")
                
                try:
                    # Find relevant pragmas for this interface
                    interface_pragmas = self._find_interface_pragmas(rtl_interface, hw_kernel.pragmas)
                    
                    # Convert RTL interface to DataflowInterface
                    dataflow_interface = self._convert_interface(
                        rtl_interface, interface_pragmas, hw_kernel
                    )
                    
                    if dataflow_interface:
                        dataflow_interfaces.append(dataflow_interface)
                        logger.debug(f"Successfully converted interface: {interface_name}")
                    else:
                        conversion_errors.append(f"Failed to convert interface: {interface_name}")
                        
                except Exception as e:
                    error_msg = f"Error converting interface {interface_name}: {str(e)}"
                    logger.error(error_msg)
                    conversion_errors.append(error_msg)
            
            if not dataflow_interfaces:
                return ConversionResult(
                    dataflow_model=None,
                    success=False,
                    errors=["No interfaces could be converted"] + conversion_errors
                )
            
            # Step 3: Create DataflowModel from converted interfaces
            try:
                dataflow_model = DataflowModel(
                    interfaces=dataflow_interfaces,
                    parameters={
                        "kernel_name": hw_kernel.name,
                        "source_file": str(hw_kernel.source_file) if hw_kernel.source_file else None,
                        "pragma_level": hw_kernel.pragma_sophistication_level,
                        "conversion_warnings": hw_kernel.parsing_warnings
                    }
                )
                
                logger.info(f"Successfully created DataflowModel for kernel: {hw_kernel.name}")
                
                return ConversionResult(
                    dataflow_model=dataflow_model,
                    success=True,
                    warnings=conversion_errors  # Non-fatal conversion issues become warnings
                )
                
            except Exception as e:
                error_msg = f"Failed to create DataflowModel: {str(e)}"
                logger.error(error_msg)
                return ConversionResult(
                    dataflow_model=None,
                    success=False,
                    errors=[error_msg] + conversion_errors
                )
                
        except Exception as e:
            error_msg = f"Unexpected error during conversion: {str(e)}"
            logger.error(error_msg)
            return ConversionResult(
                dataflow_model=None,
                success=False,
                errors=[error_msg]
            )
    
    def _validate_hw_kernel(self, hw_kernel) -> ConversionResult:
        """
        Validate HWKernel input for conversion compatibility.
        
        Args:
            hw_kernel: HWKernel instance to validate
            
        Returns:
            ConversionResult: Validation result
        """
        errors = []
        warnings = []
        
        # Check required attributes
        if not hasattr(hw_kernel, 'name') or not hw_kernel.name:
            errors.append("HWKernel missing required 'name' attribute")
            
        if not hasattr(hw_kernel, 'interfaces'):
            errors.append("HWKernel missing required 'interfaces' attribute")
        elif not hw_kernel.interfaces:
            warnings.append("HWKernel has no interfaces defined")
            
        if not hasattr(hw_kernel, 'pragmas'):
            warnings.append("HWKernel has no 'pragmas' attribute - using empty list")
            hw_kernel.pragmas = []
            
        # Validate interface structure
        if hasattr(hw_kernel, 'interfaces') and hw_kernel.interfaces:
            for iface_name, iface in hw_kernel.interfaces.items():
                if not hasattr(iface, 'type'):
                    errors.append(f"Interface {iface_name} missing 'type' attribute")
                if not hasattr(iface, 'ports'):
                    errors.append(f"Interface {iface_name} missing 'ports' attribute")
        
        if errors:
            return ConversionResult(
                dataflow_model=None,
                success=False,
                errors=errors,
                warnings=warnings
            )
        
        return ConversionResult(
            dataflow_model=None,
            success=True,
            warnings=warnings
        )
    
    def _find_interface_pragmas(self, rtl_interface, all_pragmas: List) -> List:
        """
        Find pragmas relevant to a specific RTL interface.
        
        Args:
            rtl_interface: RTL Interface object
            all_pragmas: List of all pragmas from HWKernel
            
        Returns:
            List of pragmas relevant to this interface
        """
        relevant_pragmas = []
        interface_name = rtl_interface.name
        
        for pragma in all_pragmas:
            # Check if pragma targets this interface
            if hasattr(pragma, 'parsed_data') and pragma.parsed_data:
                pragma_interface_name = pragma.parsed_data.get('interface_name')
                if pragma_interface_name and (
                    pragma_interface_name == interface_name or
                    interface_name.startswith(pragma_interface_name) or
                    pragma_interface_name.startswith(interface_name)
                ):
                    relevant_pragmas.append(pragma)
        
        return relevant_pragmas
    
    def _convert_interface(self, rtl_interface, pragmas: List, hw_kernel) -> Optional[DataflowInterface]:
        """
        Convert single RTL interface to DataflowInterface.
        
        Args:
            rtl_interface: RTL Interface object  
            pragmas: List of pragmas relevant to this interface
            hw_kernel: Parent HWKernel for context
            
        Returns:
            DataflowInterface instance or None if conversion fails
        """
        try:
            # Step 1: Map RTL interface type to DataflowInterfaceType
            dataflow_interface_type = self.interface_mapper.map_interface_type(rtl_interface)
            
            # Step 2: Apply pragma-based chunking strategies
            chunking_strategy = None
            dtype_constraint = None
            
            for pragma in pragmas:
                if hasattr(pragma, 'type'):
                    if pragma.type.name == 'BDIM':
                        # Convert BDIM pragma to chunking strategy
                        chunking_strategy = self.pragma_converter.convert_bdim_pragma(pragma)
                    elif pragma.type.name == 'DATATYPE':
                        # Convert DATATYPE pragma to constraint
                        dtype_constraint = self.pragma_converter.convert_datatype_pragma(pragma)
            
            # Step 3: Use interface mapper to create DataflowInterface
            dataflow_interface = self.interface_mapper.create_dataflow_interface(
                rtl_interface=rtl_interface,
                interface_type=dataflow_interface_type,
                chunking_strategy=chunking_strategy,
                dtype_constraint=dtype_constraint,
                hw_kernel=hw_kernel
            )
            
            return dataflow_interface
            
        except Exception as e:
            logger.error(f"Failed to convert interface {rtl_interface.name}: {str(e)}")
            return None
    
    def _apply_pragma_strategies(self, interface: DataflowInterface, pragmas: List) -> DataflowInterface:
        """
        Apply BDIM/DATATYPE pragmas to interface.
        
        Args:
            interface: DataflowInterface to modify
            pragmas: List of pragmas to apply
            
        Returns:
            Modified DataflowInterface
        """
        # This method may be used for post-processing if needed
        # Currently, pragma application is handled in _convert_interface
        return interface


def create_rtl_dataflow_converter() -> RTLDataflowConverter:
    """
    Factory function for creating RTLDataflowConverter instances.
    
    Returns:
        RTLDataflowConverter: Configured converter instance
    """
    return RTLDataflowConverter()