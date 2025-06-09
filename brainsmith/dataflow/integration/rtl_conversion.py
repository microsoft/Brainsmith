"""
RTL to Dataflow Interface Conversion Pipeline.

This module provides the conversion system that transforms RTL Parser Interface objects
into DataflowInterface objects with proper dimension mapping, datatype constraints,
and metadata integration for the Interface-Wise Dataflow Modeling Framework.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Interface as RTLInterface, InterfaceType as RTLInterfaceType
from brainsmith.dataflow.core.dataflow_interface import (
    DataflowInterface, DataflowInterfaceType, DataflowDataType, DataTypeConstraint
)
from brainsmith.dataflow.core.tensor_chunking import TensorChunking
from brainsmith.dataflow.core.validation import ValidationError, ValidationSeverity

logger = logging.getLogger(__name__)


class RTLInterfaceConverter:
    """
    Converts RTL Parser Interface objects to DataflowInterface objects.
    
    Handles:
    - Interface type mapping (AXI-Stream -> INPUT/OUTPUT/WEIGHT)
    - Dimension extraction from ONNX metadata and TDIM pragmas
    - Datatype constraint conversion from DATATYPE pragmas
    - Default constraint assignment for unconstrained interfaces
    """
    
    def __init__(self, onnx_metadata: Optional[Dict] = None):
        """
        Initialize converter with optional ONNX metadata.
        
        Args:
            onnx_metadata: Optional ONNX model metadata for tensor shape inference
        """
        self.onnx_metadata = onnx_metadata or {}
        self.tensor_chunking = TensorChunking()
        
    def convert_interfaces(self, rtl_interfaces: Dict[str, RTLInterface], 
                         parameters: Optional[Dict[str, Any]] = None) -> List[DataflowInterface]:
        """
        Convert RTL Parser interfaces to DataflowInterface objects.
        
        Args:
            rtl_interfaces: Dictionary of RTL Parser Interface objects
            parameters: Module parameters for TDIM pragma evaluation
            
        Returns:
            List of DataflowInterface objects
            
        Raises:
            ValueError: If conversion fails for critical interfaces
        """
        logger.info(f"Converting {len(rtl_interfaces)} RTL interfaces to DataflowInterface objects")
        
        dataflow_interfaces = []
        conversion_errors = []
        
        for rtl_name, rtl_interface in rtl_interfaces.items():
            try:
                dataflow_interface = self._convert_single_interface(rtl_interface, parameters or {})
                if dataflow_interface:
                    dataflow_interfaces.append(dataflow_interface)
                    logger.debug(f"Successfully converted interface '{rtl_name}' to DataflowInterface")
            except Exception as e:
                error_msg = f"Failed to convert interface '{rtl_name}': {e}"
                logger.error(error_msg)
                conversion_errors.append(error_msg)
        
        if conversion_errors:
            logger.warning(f"Conversion completed with {len(conversion_errors)} errors: {conversion_errors}")
        
        logger.info(f"Successfully converted {len(dataflow_interfaces)} interfaces")
        return dataflow_interfaces
    
    def _convert_single_interface(self, rtl_interface: RTLInterface, 
                                parameters: Dict[str, Any]) -> Optional[DataflowInterface]:
        """
        Convert a single RTL interface to DataflowInterface.
        
        Args:
            rtl_interface: RTL Parser Interface object
            parameters: Module parameters for dimension evaluation
            
        Returns:
            DataflowInterface object or None if conversion not applicable
        """
        # Step 1: Determine DataflowInterfaceType
        dataflow_type = self._map_interface_type(rtl_interface)
        if dataflow_type is None:
            logger.debug(f"Skipping interface '{rtl_interface.name}' - not a data interface")
            return None
        
        # Step 2: Extract dimensions (qDim/tDim) from metadata and pragmas
        qDim, tDim = self._extract_dimensions(rtl_interface, parameters)
        
        # Step 3: Initialize sDim (will be updated during parallelism optimization)
        sDim = self._initialize_stream_dimensions(tDim)
        
        # Step 4: Extract datatype information
        dtype = self._extract_datatype(rtl_interface)
        
        # Step 5: Extract datatype constraints
        allowed_datatypes = self._extract_datatype_constraints(rtl_interface)
        
        # Step 6: Extract AXI metadata
        axi_metadata = self._extract_axi_metadata(rtl_interface)
        
        # Step 7: Create DataflowInterface
        try:
            dataflow_interface = DataflowInterface(
                name=rtl_interface.name,
                interface_type=dataflow_type,
                qDim=qDim,
                tDim=tDim,
                sDim=sDim,
                dtype=dtype,
                allowed_datatypes=allowed_datatypes,
                axi_metadata=axi_metadata,
                constraints=[],  # Will be populated by validation framework
                pragma_metadata=self._extract_pragma_metadata(rtl_interface)
            )
            
            logger.debug(f"Created DataflowInterface: {rtl_interface.name} -> {dataflow_type.value}")
            return dataflow_interface
            
        except Exception as e:
            logger.error(f"Failed to create DataflowInterface for '{rtl_interface.name}': {e}")
            raise
    
    def _map_interface_type(self, rtl_interface: RTLInterface) -> Optional[DataflowInterfaceType]:
        """
        Map RTL interface type to DataflowInterfaceType.
        
        Args:
            rtl_interface: RTL Parser Interface object
            
        Returns:
            DataflowInterfaceType or None for non-data interfaces
        """
        if rtl_interface.type == RTLInterfaceType.GLOBAL_CONTROL:
            return DataflowInterfaceType.CONTROL
        elif rtl_interface.type == RTLInterfaceType.AXI_LITE:
            return DataflowInterfaceType.CONFIG
        elif rtl_interface.type == RTLInterfaceType.AXI_STREAM:
            # Determine if INPUT, OUTPUT, or WEIGHT based on metadata
            if rtl_interface.metadata.get("is_weight"):
                return DataflowInterfaceType.WEIGHT
            elif "in" in rtl_interface.name.lower() or rtl_interface.name.startswith("s_"):
                return DataflowInterfaceType.INPUT
            elif "out" in rtl_interface.name.lower() or rtl_interface.name.startswith("m_"):
                return DataflowInterfaceType.OUTPUT
            else:
                # Default to INPUT for ambiguous cases
                logger.warning(f"Ambiguous AXI-Stream interface '{rtl_interface.name}', defaulting to INPUT")
                return DataflowInterfaceType.INPUT
        else:
            return None
    
    def _extract_dimensions(self, rtl_interface: RTLInterface, 
                          parameters: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        """
        Extract qDim and tDim from interface metadata and pragmas.
        
        Args:
            rtl_interface: RTL Parser Interface object
            parameters: Module parameters for dimension evaluation
            
        Returns:
            Tuple of (qDim, tDim) lists
        """
        # Check for TDIM pragma override first
        if "tdim_override" in rtl_interface.metadata:
            tDim = rtl_interface.metadata["tdim_override"]
            logger.debug(f"Using TDIM pragma override for '{rtl_interface.name}': tDim = {tDim}")
            
            # If qDim is provided in metadata, use it; otherwise infer from ONNX
            if "qdim_override" in rtl_interface.metadata:
                qDim = rtl_interface.metadata["qdim_override"]
            else:
                qDim = self._infer_qDim_from_onnx(rtl_interface, tDim)
            
            return qDim, tDim
        
        # Use ONNX metadata for standard dimension inference
        onnx_layout = self.onnx_metadata.get(f"{rtl_interface.name}_layout")
        onnx_shape = self.onnx_metadata.get(f"{rtl_interface.name}_shape")
        
        if onnx_layout and onnx_shape:
            # In new architecture: qDim = original ONNX shape, tDim = chunking strategy result
            qDim = list(onnx_shape)  # Preserve original tensor shape
            tDim = self.tensor_chunking.get_layout_aware_chunking(onnx_shape, onnx_layout)[1]  # Get default chunking
            logger.debug(f"Inferred dimensions from ONNX for '{rtl_interface.name}': qDim={qDim}, tDim={tDim}")
            return qDim, tDim
        
        # Default dimensions for interfaces without ONNX metadata
        default_qDim, default_tDim = self._get_default_dimensions(rtl_interface)
        logger.debug(f"Using default dimensions for '{rtl_interface.name}': qDim={default_qDim}, tDim={default_tDim}")
        return default_qDim, default_tDim
    
    def _infer_qDim_from_onnx(self, rtl_interface: RTLInterface, tDim: List[int]) -> List[int]:
        """
        Infer qDim from ONNX metadata given tDim from TDIM pragma.
        
        Args:
            rtl_interface: RTL Parser Interface object
            tDim: Tensor dimensions from TDIM pragma
            
        Returns:
            Inferred qDim list
        """
        onnx_shape = self.onnx_metadata.get(f"{rtl_interface.name}_shape")
        if onnx_shape:
            try:
                # Compute qDim such that qDim * tDim = original_shape (with broadcasting)
                qDim = TensorChunking._compute_qDim_from_chunking(onnx_shape, tDim)
                return qDim
            except Exception as e:
                logger.warning(f"Failed to infer qDim from ONNX shape for '{rtl_interface.name}': {e}")
        
        # Fallback: qDim should be compatible with tDim (preserve divisibility)
        # In the new architecture, qDim should represent original tensor shape
        # Use tDim as a reasonable default when ONNX shape is not available
        return list(tDim)
    
    def _get_default_dimensions(self, rtl_interface: RTLInterface) -> Tuple[List[int], List[int]]:
        """
        Get default dimensions for interfaces without ONNX metadata.
        
        Args:
            rtl_interface: RTL Parser Interface object
            
        Returns:
            Tuple of default (qDim, tDim)
        """
        if rtl_interface.type == RTLInterfaceType.AXI_STREAM:
            # Default stream dimensions based on interface type
            if rtl_interface.metadata.get("is_weight"):
                return [32], [16]  # Default weight dimensions
            else:
                return [16], [8]   # Default activation dimensions
        elif rtl_interface.type == RTLInterfaceType.AXI_LITE:
            return [1], [1]        # Config interfaces are scalar
        else:
            return [1], [1]        # Control interfaces are scalar
    
    def _initialize_stream_dimensions(self, tDim: List[int]) -> List[int]:
        """
        Initialize stream dimensions (sDim) based on tensor dimensions.
        
        Args:
            tDim: Tensor dimensions
            
        Returns:
            Initial stream dimensions (will be updated during parallelism optimization)
        """
        # Default to single element per cycle
        return [1] * len(tDim)
    
    def _extract_datatype(self, rtl_interface: RTLInterface) -> DataflowDataType:
        """
        Extract datatype information from interface metadata.
        
        Args:
            rtl_interface: RTL Parser Interface object
            
        Returns:
            DataflowDataType object
        """
        # Check for explicit datatype specification
        if "explicit_datatype" in rtl_interface.metadata:
            datatype_info = rtl_interface.metadata["explicit_datatype"]
            base_type = datatype_info.get("base_type", "INT")
            bitwidth = datatype_info.get("bitwidth", 8)
            signed = datatype_info.get("signed", True)
            finn_type = datatype_info.get("finn_type")
            
            # Generate finn_type if not provided
            if not finn_type:
                if signed:
                    finn_type = f"{base_type}{bitwidth}"
                else:
                    finn_type = f"U{base_type}{bitwidth}"
            
            return DataflowDataType(
                base_type=base_type,
                bitwidth=bitwidth,
                signed=signed,
                finn_type=finn_type
            )
        
        # Check if there are datatype constraints that should influence the default
        if "datatype_constraints" in rtl_interface.metadata:
            constraint_info = rtl_interface.metadata["datatype_constraints"]
            base_types = constraint_info["base_types"]
            min_bitwidth = constraint_info["min_bitwidth"]
            max_bitwidth = constraint_info["max_bitwidth"]
            
            # Use the first allowed base type and minimum bitwidth as default
            if base_types:
                base_type = base_types[0]
                bitwidth = min_bitwidth
                signed = base_type == "INT"
                finn_type = f"{base_type}{bitwidth}"
                
                return DataflowDataType(
                    base_type=base_type,
                    bitwidth=bitwidth,
                    signed=signed,
                    finn_type=finn_type
                )
        
        # Default datatype based on interface type
        if rtl_interface.type == RTLInterfaceType.AXI_STREAM:
            if rtl_interface.metadata.get("is_weight"):
                # Default weight datatype
                return DataflowDataType(base_type="INT", bitwidth=8, signed=True, finn_type="INT8")
            else:
                # Default activation datatype
                return DataflowDataType(base_type="UINT", bitwidth=8, signed=False, finn_type="UINT8")
        else:
            # Default for config/control interfaces
            return DataflowDataType(base_type="UINT", bitwidth=32, signed=False, finn_type="UINT32")
    
    def _extract_datatype_constraints(self, rtl_interface: RTLInterface) -> List[DataTypeConstraint]:
        """
        Extract datatype constraints from DATATYPE pragma metadata.
        
        Args:
            rtl_interface: RTL Parser Interface object
            
        Returns:
            List of DataTypeConstraint objects
        """
        constraints = []
        
        # Check for DATATYPE pragma constraints
        if "datatype_constraints" in rtl_interface.metadata:
            constraint_info = rtl_interface.metadata["datatype_constraints"]
            
            constraint = DataTypeConstraint(
                base_types=constraint_info["base_types"],
                min_bitwidth=constraint_info["min_bitwidth"],
                max_bitwidth=constraint_info["max_bitwidth"],
                signed_allowed=True,
                unsigned_allowed=True
            )
            constraints.append(constraint)
            
            logger.debug(f"Extracted datatype constraints for '{rtl_interface.name}': {constraint}")
        
        # Add default constraint if no explicit constraints
        if not constraints:
            default_constraint = self._get_default_datatype_constraint(rtl_interface)
            constraints.append(default_constraint)
        
        return constraints
    
    def _get_default_datatype_constraint(self, rtl_interface: RTLInterface) -> DataTypeConstraint:
        """
        Get default datatype constraint for interfaces without DATATYPE pragma.
        
        Args:
            rtl_interface: RTL Parser Interface object
            
        Returns:
            Default DataTypeConstraint
        """
        if rtl_interface.type == RTLInterfaceType.AXI_STREAM:
            # Flexible constraints for data interfaces
            return DataTypeConstraint(
                base_types=["INT", "UINT", "FIXED"],
                min_bitwidth=1,
                max_bitwidth=32,
                signed_allowed=True,
                unsigned_allowed=True
            )
        else:
            # More restrictive for config/control
            return DataTypeConstraint(
                base_types=["UINT"],
                min_bitwidth=8,
                max_bitwidth=32,
                signed_allowed=False,
                unsigned_allowed=True
            )
    
    def _extract_axi_metadata(self, rtl_interface: RTLInterface) -> Dict[str, Any]:
        """
        Extract AXI-specific metadata from RTL interface.
        
        Args:
            rtl_interface: RTL Parser Interface object
            
        Returns:
            Dictionary of AXI metadata
        """
        axi_metadata = {}
        
        # Copy relevant metadata from RTL interface
        if rtl_interface.type in [RTLInterfaceType.AXI_STREAM, RTLInterfaceType.AXI_LITE]:
            # Data width information
            if "data_width" in rtl_interface.metadata:
                axi_metadata["data_width"] = rtl_interface.metadata["data_width"]
            
            # Address width for AXI-Lite
            if rtl_interface.type == RTLInterfaceType.AXI_LITE and "address_width" in rtl_interface.metadata:
                axi_metadata["address_width"] = rtl_interface.metadata["address_width"]
            
            # Protocol-specific features
            if "has_tlast" in rtl_interface.metadata:
                axi_metadata["has_tlast"] = rtl_interface.metadata["has_tlast"]
            
            if "has_tkeep" in rtl_interface.metadata:
                axi_metadata["has_tkeep"] = rtl_interface.metadata["has_tkeep"]
        
        return axi_metadata
    
    def _extract_pragma_metadata(self, rtl_interface: RTLInterface) -> Dict[str, Any]:
        """
        Extract pragma-related metadata for debugging and validation.
        
        Args:
            rtl_interface: RTL Parser Interface object
            
        Returns:
            Dictionary of pragma metadata
        """
        pragma_metadata = {}
        
        # TDIM pragma information
        if "tdim_expressions" in rtl_interface.metadata:
            pragma_metadata["tdim_expressions"] = rtl_interface.metadata["tdim_expressions"]
        
        # DATATYPE pragma information
        if "datatype_constraints" in rtl_interface.metadata:
            pragma_metadata["datatype_pragma_applied"] = True
        
        # WEIGHT pragma information
        if "is_weight" in rtl_interface.metadata:
            pragma_metadata["weight_pragma_applied"] = rtl_interface.metadata["is_weight"]
            if "weight_type" in rtl_interface.metadata:
                pragma_metadata["weight_type"] = rtl_interface.metadata["weight_type"]
        
        return pragma_metadata


class ConversionValidationError(Exception):
    """Exception raised when interface conversion validation fails."""
    pass


def validate_conversion_result(dataflow_interfaces: List[DataflowInterface]) -> List[ValidationError]:
    """
    Validate the result of RTL to Dataflow interface conversion.
    
    Args:
        dataflow_interfaces: List of converted DataflowInterface objects
        
    Returns:
        List of validation errors (empty if all valid)
    """
    errors = []
    
    # Check for required interface types
    input_interfaces = [iface for iface in dataflow_interfaces if iface.interface_type == DataflowInterfaceType.INPUT]
    output_interfaces = [iface for iface in dataflow_interfaces if iface.interface_type == DataflowInterfaceType.OUTPUT]
    
    if not input_interfaces:
        errors.append(ValidationError(
            component="conversion",
            error_type="missing_required_interface",
            message="No INPUT interfaces found after conversion",
            severity=ValidationSeverity.WARNING,
            context={"required_type": "INPUT"}
        ))
    
    if not output_interfaces:
        errors.append(ValidationError(
            component="conversion", 
            error_type="missing_required_interface",
            message="No OUTPUT interfaces found after conversion",
            severity=ValidationSeverity.WARNING,
            context={"required_type": "OUTPUT"}
        ))
    
    # Validate individual interfaces
    for iface in dataflow_interfaces:
        validation_result = iface.validate_constraints()
        
        # Handle both ValidationResult objects and legacy list returns
        if hasattr(validation_result, 'errors'):
            # New ValidationResult object with separate error lists
            errors.extend(validation_result.errors)
            errors.extend(validation_result.warnings)
            # Note: ValidationResult doesn't have 'info' attribute, only errors and warnings
        else:
            # Legacy behavior - plain list of errors
            errors.extend(validation_result)
    
    return errors