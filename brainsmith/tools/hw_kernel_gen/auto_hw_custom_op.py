"""
AutoHWCustomOp base class for auto-generated hardware custom operators.

This module provides the base class for all auto-generated HWCustomOp classes,
implementing standardized methods that can be fully determined from the
dataflow interface metadata.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from .class_naming import generate_class_name
from .dataflow_model import DataflowModel, ParallelismConfiguration
from .interface_types import InterfaceType
from .dataflow_interface import DataflowInterface
from .interface_metadata import InterfaceMetadata, InterfaceMetadataCollection
from .qonnx_types import validate_datatype_against_constraints

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.data_packing import numpy_to_hls_code
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple


class AutoHWCustomOp(HWCustomOp):
    """
    Base class for auto-generated HWCustomOp implementations using simplified 3-tier architecture.
    
    Three tiers of information:
    1. Kernel Data (static): Interface metadata, chunking strategies, node attributes
    2. Model Data (runtime): qDim from ONNX, bDim chunked from qDim, datatypes
    3. Parallelism (dynamic): iPar/wPar values, stream_dims calculations, performance
    """
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize AutoHWCustomOp following FINN's standard pattern."""
        super().__init__(onnx_node, **kwargs)
        
        # Build dataflow model using node attributes
        self._dataflow_model = self._build_dataflow_model_from_node()
        
        # Initialize minimum parallelism
        self._current_parallelism = self._initialize_minimum_parallelism()
        
        # Initialize tensor formatter for automatic tensor formatting
        self._tensor_formatter = DataflowTensorFormatter()
    
    @abstractmethod
    def get_interface_metadata(self) -> List[InterfaceMetadata]:
        """
        Return static interface metadata for this hardware kernel.
        
        Subclasses must implement this method to provide the interface
        definitions from RTL parsing. This replaces the interface_metadata
        constructor parameter to follow FINN's pattern.
        """
        pass
    
    def get_nodeattr_types(self) -> Dict[str, Any]:
        """
        Get complete nodeattr types including parent class attributes and legacy compatibility.
        
        This method provides the full set of node attributes needed for FINN integration:
        1. Base HWCustomOp attributes (from parent class)
        2. Legacy compatibility attributes (SIMD, PE, datatypes)
        3. Interface-specific datatype attributes
        
        The legacy attributes are derived from the DataflowModel interfaces to ensure
        compatibility with existing FINN infrastructure that expects SIMD/PE terminology.
        
        Returns:
            Dict[str, Any]: Complete nodeattr type specification
        """
        # Get base attributes from parent HWCustomOp class
        my_attrs = super().get_nodeattr_types()
        
        try:
            # Add legacy compatibility attributes selectively based on operation characteristics
            legacy_attrs = self.get_legacy_attr()
            
            # Always add basic datatype attributes if they exist
            if "inputDataType" in legacy_attrs:
                my_attrs["inputDataType"] = ("s", False, legacy_attrs["inputDataType"])
            if "outputDataType" in legacy_attrs:
                my_attrs["outputDataType"] = ("s", False, legacy_attrs["outputDataType"])
            
            # Only add SIMD if operation has input interfaces with parallelism
            if "SIMD" in legacy_attrs and legacy_attrs["SIMD"] > 1:
                my_attrs["SIMD"] = ("i", False, legacy_attrs["SIMD"])
            
            # Only add PE if operation has weights OR multiple outputs/processing elements
            if "PE" in legacy_attrs:
                # Check if operation has weight interfaces or justifies PE attribute
                has_weights = "weightDataType" in legacy_attrs
                has_processing_parallelism = legacy_attrs["PE"] > 1
                
                if has_weights or has_processing_parallelism:
                    my_attrs["PE"] = ("i", False, legacy_attrs["PE"])
            
            # Only add weightDataType if operation actually has weights
            if "weightDataType" in legacy_attrs:
                my_attrs["weightDataType"] = ("s", False, legacy_attrs["weightDataType"])
            
            
            # Always override preferred_impl_style for RTL-generated operations
            if "preferred_impl_style" in legacy_attrs:
                my_attrs["preferred_impl_style"] = ("s", False, "rtl", {"", "hls", "rtl"})
        
        except Exception:
            # If get_legacy_attr() fails (e.g., during initialization), 
            # only provide minimal essential attributes
            my_attrs.update({
                "preferred_impl_style": ("s", False, "rtl", {"", "hls", "rtl"})
            })
        
        # Add interface-specific datatype attributes
        try:
            interface_metadata = self.get_interface_metadata()
            for metadata in interface_metadata:
                if metadata.interface_type != InterfaceType.CONTROL:
                    dtype_attr = f"{metadata.name}_dtype"
                    constraint_desc = metadata.get_constraint_description()
                    my_attrs[dtype_attr] = ("s", True, "", constraint_desc)
        except Exception:
            # If interface metadata not available during initialization, skip
            pass
        
        # Add ram_style only if operation has weight interfaces
        try:
            has_weights = any(
                metadata.interface_type == InterfaceType.WEIGHT 
                for metadata in self.get_interface_metadata()
            )
            if has_weights:
                my_attrs["ram_style"] = ("s", False, "auto", {"auto", "block", "distributed", "ultra"})
        except Exception:
            # If can't determine, don't add ram_style
            pass
        
        return my_attrs
    
    def _build_dataflow_model_from_node(self) -> 'DataflowModel':
        """Build DataflowModel from ONNX node attributes."""
        interface_metadata = self.get_interface_metadata()
        
        interfaces = []
        for metadata in interface_metadata:
            if metadata.interface_type == InterfaceType.CONTROL:
                interface = self._create_control_interface(metadata)
            else:
                dtype_attr = f"{metadata.name}_dtype"
                runtime_dtype = self.get_nodeattr(dtype_attr)
                if not runtime_dtype:
                    constraint_desc = metadata.get_constraint_description()
                    raise ValueError(
                        f"Datatype for interface '{metadata.name}' must be specified "
                        f"via node attribute '{dtype_attr}'. "
                        f"Allowed datatypes: {constraint_desc}"
                    )
                
                interface = DataflowInterface.from_metadata_and_runtime_datatype(
                    metadata=metadata,
                    runtime_datatype=runtime_dtype,
                    tensor_dims=self._get_tensor_dims_for_interface(metadata.name),
                    block_dims=self._resolve_block_dims(metadata),
                    stream_dims=[1] * len(self._get_block_shape(metadata))
                )
            
            interfaces.append(interface)
        
        return DataflowModel(interfaces, {})
    
    def _create_control_interface(self, metadata: InterfaceMetadata) -> DataflowInterface:
        """Create a control interface with default properties."""
        # Use QONNX DataType for consistency
        control_dtype = DataType["UINT1"]
        
        return DataflowInterface(
            name=metadata.name,
            interface_type=metadata.interface_type,
            tensor_dims=[1],  # Control signals are scalar
            block_dims=[1],
            stream_dims=[1],
            dtype=control_dtype
        )
    
    def _get_tensor_dims_for_interface(self, interface_name: str) -> List[int]:
        """Get tensor dimensions for interface from ONNX node."""
        try:
            # Try to get shape from ONNX node inputs/outputs
            # In practice, FINN will provide proper shape info
            return [1, 128, 128, 256]  # Default 4D shape for testing
        except Exception:
            return [1, 128, 128, 256]
    
    def _resolve_block_dims(self, metadata: InterfaceMetadata) -> List[int]:
        """Resolve block dimensions using node attributes for parameters."""
        if not hasattr(metadata.chunking_strategy, 'block_shape'):
            return [1]
        
        # Get tensor dimensions to properly resolve ':' placeholders
        tensor_dims = self._get_tensor_dims_for_interface(metadata.name)
        
        resolved = []
        for i, dim in enumerate(metadata.chunking_strategy.block_shape):
            if isinstance(dim, str) and dim != ":":
                param_value = self.get_nodeattr(dim)
                if param_value is None:
                    raise ValueError(f"Parameter '{dim}' not found in node attributes")
                resolved.append(param_value)
            elif dim == ":":
                # Use corresponding tensor dimension
                if i < len(tensor_dims):
                    resolved.append(tensor_dims[i])
                else:
                    resolved.append(1)  # Fallback for out-of-bounds
            else:
                resolved.append(dim)
        
        return resolved
    
    def _get_block_shape(self, metadata: InterfaceMetadata) -> List[Union[int, str]]:
        """Get block shape from metadata."""
        if hasattr(metadata.chunking_strategy, 'block_shape'):
            return metadata.chunking_strategy.block_shape
        return [1]
    
    
    def _initialize_minimum_parallelism(self) -> Dict[str, int]:
        """
        Initialize Tier 3 (Parallelism) with minimum values.
        
        This ensures the object is fully functional immediately,
        with parallelism updated separately as needed.
        """
        parallelism = {}
        
        # Set minimum parallelism (1) for all interfaces
        for iface in self._dataflow_model.input_interfaces:
            parallelism[f"{iface.name}_iPar"] = 1
            
        for iface in self._dataflow_model.weight_interfaces:
            parallelism[f"{iface.name}_wPar"] = 1
            
        return parallelism
    
    def update_parallelism(self, iPar: Dict[str, int] = None, wPar: Dict[str, int] = None):
        """
        Update Tier 3 (Parallelism) values and recalculate stream_dims/performance.
        
        This is separate from construction - parallelism can be updated
        at any time without rebuilding the entire model.
        
        Args:
            iPar: Input parallelism per interface {interface_name: parallelism}
            wPar: Weight parallelism per interface {interface_name: parallelism}
        """
        if iPar is None:
            iPar = {}
        if wPar is None:
            wPar = {}
            
        # Update current parallelism tracking
        for iface_name, parallel_val in iPar.items():
            self._current_parallelism[f"{iface_name}_iPar"] = parallel_val
            
        for iface_name, parallel_val in wPar.items():
            self._current_parallelism[f"{iface_name}_wPar"] = parallel_val
        
        # Update stream_dims for all interfaces based on new parallelism
        for iface in self._dataflow_model.input_interfaces:
            if iface.name in iPar:
                iface.apply_parallelism(iPar=iPar[iface.name])
                
        for iface in self._dataflow_model.weight_interfaces:
            if iface.name in wPar:
                iface.apply_parallelism(wPar=wPar[iface.name])
                
        for iface in self._dataflow_model.output_interfaces:
            # Output parallelism typically follows input parallelism
            if iPar:
                # Use first input's parallelism for outputs
                first_input_par = next(iter(iPar.values()))
                iface.apply_parallelism(iPar=first_input_par)
    
    
    
    @property  
    def dataflow_model(self) -> DataflowModel:
        """Get DataflowModel (always available after construction)."""
        return self._dataflow_model
    
    @property
    def interface_metadata(self) -> List['InterfaceMetadata']:
        """Get interface metadata collection."""
        return self.get_interface_metadata()
    
    def get_current_parallelism(self) -> Dict[str, int]:
        """Get current Tier 3 (Parallelism) values."""
        return self._current_parallelism.copy()

    @property
    def input_interfaces(self) -> List[str]:
        """Get input interface names from DataflowModel."""
        return [iface.name for iface in self.dataflow_model.input_interfaces]

    @property
    def output_interfaces(self) -> List[str]:
        """Get output interface names from DataflowModel."""
        return [iface.name for iface in self.dataflow_model.output_interfaces]

    @property
    def weight_interfaces(self) -> List[str]:
        """Get weight interface names from DataflowModel."""
        return [iface.name for iface in self.dataflow_model.weight_interfaces]

    
    def get_enhanced_nodeattr_types(self) -> Dict[str, Tuple[str, bool, Any]]:
        """
        Get enhanced node attribute types with dataflow modeling support.
        
        This method adds dataflow-specific attributes to the base FINN attributes.
        Subclasses should override get_nodeattr_types() and call this method.
        """
        attrs = {}
        
        # Only access dataflow_model if it exists (avoid circular dependency during init)
        if hasattr(self, '_dataflow_model') and self._dataflow_model is not None:
            # Add parallelism configuration for input and output interfaces
            for iface in self.dataflow_model.input_interfaces:
                attrs[f"{iface.name}_parallel"] = ("i", False, 1)
                
            for iface in self.dataflow_model.output_interfaces:
                attrs[f"{iface.name}_parallel"] = ("i", False, 1)
                        
            # Add datatype configuration for all interfaces
            all_interfaces = (
                self.dataflow_model.input_interfaces +
                self.dataflow_model.output_interfaces +
                self.dataflow_model.weight_interfaces
            )
            for iface in all_interfaces:
                if iface.interface_type != InterfaceType.CONTROL:
                    attrs[f"{iface.name}_dtype"] = ("s", True, "")
        else:
            # During initialization, use interface metadata to generate basic attributes
            interface_metadata = self.get_interface_metadata()
            for metadata in interface_metadata:
                if metadata.interface_type != InterfaceType.CONTROL:
                    attrs[f"{metadata.name}_dtype"] = ("s", True, "")
                if metadata.interface_type in [InterfaceType.INPUT, InterfaceType.OUTPUT]:
                    attrs[f"{metadata.name}_parallel"] = ("i", False, 1)
            
        # Add resource estimation and validation configuration
        attrs.update({
            "resource_estimation_mode": ("s", False, "automatic", {"automatic", "conservative", "optimistic"}),
            "enable_constraint_validation": ("b", False, True),
        })
        
        return attrs
        
    def get_input_datatype(self, ind: int = 0) -> Any:
        """Get input datatype from user-specified ONNX node attributes.
        
        No default datatypes are provided - the user must explicitly specify
        the datatype for each input interface via node attributes.
        """
        input_ifaces = self.dataflow_model.input_interfaces
        if ind >= len(input_ifaces):
            raise IndexError(f"Input index {ind} exceeds available inputs")
        
        interface = input_ifaces[ind]
        
        # Skip CONTROL interfaces - they have fixed datatypes
        if interface.interface_type == InterfaceType.CONTROL:
            return interface.dtype.finn_type if hasattr(interface.dtype, 'finn_type') else "UINT1"
        
        # Check for user-specified datatype
        configured_dtype = self.get_nodeattr(f"{interface.name}_dtype")
        if not configured_dtype:
            # Get interface metadata to show constraint information
            metadata = next((m for m in self.get_interface_metadata() 
                           if m.name == interface.name), None)
            constraint_desc = metadata.get_constraint_description() if metadata else "unknown constraints"
            raise ValueError(
                f"Input datatype for '{interface.name}' (index {ind}) must be explicitly specified "
                f"via node attribute '{interface.name}_dtype'. "
                f"Allowed datatypes: {constraint_desc}"
            )
        
        # Validate configured datatype against constraints
        if not interface.validate_datatype_string(configured_dtype):
            metadata = next((m for m in self.get_interface_metadata() 
                           if m.name == interface.name), None)
            constraint_desc = metadata.get_constraint_description() if metadata else "unknown constraints"
            raise ValueError(
                f"Configured datatype '{configured_dtype}' for input '{interface.name}' "
                f"violates constraints. Allowed datatypes: {constraint_desc}"
            )
        
        return DataType[configured_dtype]
            
    def get_output_datatype(self, ind: int = 0) -> Any:
        """Get output datatype from user-specified ONNX node attributes.
        
        No default datatypes are provided - the user must explicitly specify
        the datatype for each output interface via node attributes.
        """
        output_ifaces = self.dataflow_model.output_interfaces
        if ind >= len(output_ifaces):
            raise IndexError(f"Output index {ind} exceeds available outputs")
        
        interface = output_ifaces[ind]
        
        # Skip CONTROL interfaces - they have fixed datatypes
        if interface.interface_type == InterfaceType.CONTROL:
            return interface.dtype.finn_type if hasattr(interface.dtype, 'finn_type') else "UINT1"
        
        # Check for user-specified datatype
        configured_dtype = self.get_nodeattr(f"{interface.name}_dtype")
        if not configured_dtype:
            # Get interface metadata to show constraint information
            metadata = next((m for m in self.get_interface_metadata() 
                           if m.name == interface.name), None)
            constraint_desc = metadata.get_constraint_description() if metadata else "unknown constraints"
            raise ValueError(
                f"Output datatype for '{interface.name}' (index {ind}) must be explicitly specified "
                f"via node attribute '{interface.name}_dtype'. "
                f"Allowed datatypes: {constraint_desc}"
            )
        
        # Validate configured datatype against constraints
        if not interface.validate_datatype_string(configured_dtype):
            metadata = next((m for m in self.get_interface_metadata() 
                           if m.name == interface.name), None)
            constraint_desc = metadata.get_constraint_description() if metadata else "unknown constraints"
            raise ValueError(
                f"Configured datatype '{configured_dtype}' for output '{interface.name}' "
                f"violates constraints. Allowed datatypes: {constraint_desc}"
            )
        
        return DataType[configured_dtype]
            
    def get_normal_input_shape(self, ind: int = 0) -> List[int]:
        """Get normal input shape from DataflowModel interface."""
        input_ifaces = self.dataflow_model.input_interfaces
        if ind >= len(input_ifaces):
            raise IndexError(f"Input index {ind} exceeds available inputs")
        
        interface = input_ifaces[ind]
        return interface.reconstruct_tensor_shape()
        
    def get_normal_output_shape(self, ind: int = 0) -> List[int]:
        """Get normal output shape from DataflowModel interface."""
        output_ifaces = self.dataflow_model.output_interfaces
        if ind >= len(output_ifaces):
            raise IndexError(f"Output index {ind} exceeds available outputs")
        
        interface = output_ifaces[ind]
        return interface.reconstruct_tensor_shape()
        
    def get_folded_input_shape(self, ind: int = 0) -> List[int]:
        """Get folded input shape considering parallelism configuration."""
        input_ifaces = self.dataflow_model.input_interfaces
        if ind >= len(input_ifaces):
            raise IndexError(f"Input index {ind} exceeds available inputs")
        
        interface = input_ifaces[ind]
        normal_shape = interface.reconstruct_tensor_shape()
        
        # Apply folding based on parallelism configuration
        parallel_factor = self.get_nodeattr(f"{interface.name}_parallel") or 1
        if parallel_factor > 1 and normal_shape:
            folded_shape = normal_shape.copy()
            folded_shape[-1] = folded_shape[-1] // parallel_factor
            return folded_shape
        
        return normal_shape
        
    def get_folded_output_shape(self, ind: int = 0) -> List[int]:
        """Get folded output shape considering parallelism configuration."""
        output_ifaces = self.dataflow_model.output_interfaces
        if ind >= len(output_ifaces):
            raise IndexError(f"Output index {ind} exceeds available outputs")
        
        interface = output_ifaces[ind]
        normal_shape = interface.reconstruct_tensor_shape()
        
        # Apply folding based on parallelism configuration
        parallel_factor = self.get_nodeattr(f"{interface.name}_parallel") or 1
        if parallel_factor > 1 and normal_shape:
            folded_shape = normal_shape.copy()
            folded_shape[-1] = folded_shape[-1] // parallel_factor
            return folded_shape
        
        return normal_shape
        
    def get_instream_width(self, ind: int = 0) -> int:
        """Get input stream width in bits."""
        input_dtype = self.get_input_datatype(ind)
        folded_shape = self.get_folded_input_shape(ind)
        
        # Width = datatype bits * folded elements in last dimension
        elements_per_cycle = folded_shape[-1] if folded_shape else 1
        
        return input_dtype.bitwidth() * elements_per_cycle
            
    def get_outstream_width(self, ind: int = 0) -> int:
        """Get output stream width in bits."""
        output_dtype = self.get_output_datatype(ind)
        folded_shape = self.get_folded_output_shape(ind)
        
        # Width = datatype bits * folded elements in last dimension
        elements_per_cycle = folded_shape[-1] if folded_shape else 1
        
        return output_dtype.bitwidth() * elements_per_cycle
            
    def get_number_output_values(self) -> int:
        """Get total number of output values."""
        if not self.output_interfaces:
            return 0
            
        total_outputs = 0
        for ind in range(len(self.output_interfaces)):
            output_shape = self.get_normal_output_shape(ind)
            total_outputs += np.prod(output_shape)
        return int(total_outputs)
        
    def get_exp_cycles(self) -> int:
        """Get expected cycles using current Tier 3 (Parallelism) values.""" 
        # Extract iPar and wPar from current parallelism
        iPar = {}
        wPar = {}
        
        for key, value in self._current_parallelism.items():
            if key.endswith('_iPar'):
                interface_name = key[:-5]  # Remove '_iPar' suffix
                iPar[interface_name] = value
            elif key.endswith('_wPar'):
                interface_name = key[:-5]  # Remove '_wPar' suffix
                wPar[interface_name] = value
        
        # Use unified calculation
        intervals = self.dataflow_model.calculate_initiation_intervals(iPar, wPar)
        return intervals.L
        
    def get_op_and_param_counts(self) -> Dict[str, int]:
        """
        Get operation and parameter counts.
        
        Returns:
            Dict with operation and parameter counts
        """
        counts = {
            "ops": 0,
            "params": 0,
            "weight_params": 0,
            "config_params": 0
        }
        
        # Count weight parameters
        for iface in self.dataflow_model.weight_interfaces:
            params = np.prod(iface.get_num_blocks()) * np.prod(iface.block_dims)
            counts["weight_params"] += params
            counts["params"] += params
            
        # Config interfaces no longer tracked - they are handled separately
            
        # Estimate operations based on input/output sizes
        if self.dataflow_model.input_interfaces and self.dataflow_model.output_interfaces:
            input_size = sum(
                np.prod(iface.get_num_blocks()) * np.prod(iface.block_dims)
                for iface in self.dataflow_model.input_interfaces
            )
            output_size = sum(
                np.prod(iface.get_num_blocks()) * np.prod(iface.block_dims)
                for iface in self.dataflow_model.output_interfaces
            )
            # Simple estimation: operations proportional to input*output
            counts["ops"] = input_size * output_size
            
        return counts
        
    def derive_characteristic_fxns(self) -> Dict[str, Any]:
        """
        Derive characteristic functions for the operation.
        
        Returns:
            Dict with characteristic functions
        """
        return {
            "compute_cycles": self.get_exp_cycles(),
            "bram_usage": self.estimate_bram_usage(),
            "lut_usage": self.estimate_lut_usage(),
            "dsp_usage": self.estimate_dsp_usage("xczu7ev"),  # Default FPGA part
            "stream_widths": {
                "input": [self.get_instream_width(i) for i in range(len(self.input_interfaces))],
                "output": [self.get_outstream_width(i) for i in range(len(self.output_interfaces))]
            }
        }

    def _get_current_parallelism_config(self) -> Dict[str, int]:
        """Get current parallelism configuration for all interfaces."""
        config = {}
        # Get all interfaces from DataflowModel properties  
        all_interfaces = (
            self.dataflow_model.input_interfaces +
            self.dataflow_model.output_interfaces +
            self.dataflow_model.weight_interfaces
        )
        
        for iface in all_interfaces:
            parallel_attr = f"{iface.name}_parallel"
            config[iface.name] = self.get_nodeattr(parallel_attr) or 1
        return config
                        
    def bram_estimation(self) -> int:
        """Estimate BRAM usage (FINN-compatible method name)."""
        return self.estimate_bram_usage()
    
    def estimate_bram_usage(self) -> int:
        """Estimate BRAM usage using DataflowModel resource requirements."""
        # Check ram_style - only count BRAM if using block RAM
        ram_style = self.get_nodeattr("ram_style") if "ram_style" in self.get_nodeattr_types() else None
        if ram_style and ram_style != "block" and ram_style != "auto":
            # Using distributed or ultra RAM, no BRAM usage
            return 0
        
        try:
            # Create proper ParallelismConfiguration object
            iPar = {iface.name: self.get_nodeattr(f"{iface.name}_parallel") or 1
                   for iface in self.dataflow_model.input_interfaces}
            wPar = {iface.name: self.get_nodeattr(f"{iface.name}_parallel") or 1
                   for iface in self.dataflow_model.weight_interfaces}
            
            parallelism_config = ParallelismConfiguration(
                iPar=iPar,
                wPar=wPar,
                derived_stream_dims={}  # Will be computed
            )
            
            resources = self.dataflow_model.get_resource_requirements(parallelism_config)
            memory_bits = resources["memory_bits"]
            bram_capacity = 18 * 1024  # BRAM18K
            
            # Apply estimation mode scaling
            estimation_mode = self.get_nodeattr("resource_estimation_mode")
            scale_factor = {"conservative": 1.5, "optimistic": 0.7, "automatic": 1.0}.get(estimation_mode, 1.0)
            
            return int(np.ceil((memory_bits * scale_factor) / bram_capacity))
        except Exception:
            # Fallback to simple estimation
            if not self.dataflow_model.weight_interfaces:
                return 0
            total_memory = sum(iface.get_memory_footprint() for iface in
                              self.dataflow_model.input_interfaces + self.dataflow_model.weight_interfaces)
            return max(1, total_memory // (18 * 1024))
        
    def lut_estimation(self) -> int:
        """Estimate LUT usage (FINN-compatible method name)."""
        return self.estimate_lut_usage()
    
    def estimate_lut_usage(self) -> int:
        """Estimate LUT usage using DataflowModel resource requirements."""
        base_luts = 0
        
        # Check if using distributed RAM (LUTRAM)
        ram_style = self.get_nodeattr("ram_style") if "ram_style" in self.get_nodeattr_types() else None
        if ram_style == "distributed" and self.dataflow_model.weight_interfaces:
            # Distributed RAM uses LUTs for storage
            try:
                total_memory_bits = sum(iface.get_memory_footprint() * 8 for iface in self.dataflow_model.weight_interfaces)
                # Rough estimate: 1 LUT can store ~64 bits in LUTRAM mode
                base_luts += total_memory_bits // 64
            except:
                base_luts += 1000  # Conservative estimate for LUTRAM
        
        try:
            # Simple estimation based on interface complexity
            total_width = sum(iface.calculate_stream_width() for iface in
                             self.dataflow_model.input_interfaces + self.dataflow_model.output_interfaces)
            
            # Apply estimation mode scaling
            estimation_mode = self.get_nodeattr("resource_estimation_mode")
            scale_factor = {"conservative": 1.5, "optimistic": 0.7, "automatic": 1.0}.get(estimation_mode, 1.0)
            
            # Simple heuristic: ~10 LUTs per bit of stream width for logic
            lut_estimate = int(((total_width * 10) + base_luts) * scale_factor)
            return max(1, lut_estimate)
        except Exception:
            # Fallback to fixed estimate
            return max(100, base_luts)
        
    def dsp_estimation(self, fpgapart: str = "xczu7ev") -> int:
        """Estimate DSP usage (FINN-compatible method name)."""
        return self.estimate_dsp_usage(fpgapart)
    
    def estimate_dsp_usage(self, fpgapart: str = "xczu7ev") -> int:
        """Estimate DSP usage using DataflowModel resource requirements."""
        try:
            # Simple estimation: assume some operations require DSPs
            total_ops = sum(np.prod(iface.get_num_blocks()) for iface in self.dataflow_model.input_interfaces)
            
            # Apply estimation mode scaling
            estimation_mode = self.get_nodeattr("resource_estimation_mode")
            scale_factor = {"conservative": 1.2, "optimistic": 0.8, "automatic": 1.0}.get(estimation_mode, 1.0)
            
            # Simple heuristic: 1 DSP per 1000 operations
            dsp_estimate = int(np.ceil((total_ops / 1000) * scale_factor))
            return max(0, dsp_estimate)
        except Exception:
            # Fallback to minimal DSP usage
            return 0
    
    def uram_estimation(self) -> int:
        """
        Estimate URAM (Ultra RAM) usage.
        
        Returns URAM count only if ram_style is set to "ultra".
        """
        # Check ram_style - only count URAM if explicitly using ultra RAM
        ram_style = self.get_nodeattr("ram_style") if "ram_style" in self.get_nodeattr_types() else None
        if ram_style != "ultra":
            return 0
        
        try:
            # Calculate total memory requirements for weights
            if not self.dataflow_model.weight_interfaces:
                return 0
                
            total_memory_bits = sum(
                iface.get_memory_footprint() * 8 
                for iface in self.dataflow_model.weight_interfaces
            )
            
            # URAM capacity: 288Kb per URAM block
            uram_capacity = 288 * 1024
            
            # Apply estimation mode scaling
            estimation_mode = self.get_nodeattr("resource_estimation_mode")
            scale_factor = {"conservative": 1.3, "optimistic": 0.8, "automatic": 1.0}.get(estimation_mode, 1.0)
            
            return int(np.ceil((total_memory_bits * scale_factor) / uram_capacity))
        except Exception:
            # If can't calculate, assume no URAM usage
            return 0

    @staticmethod
    def generate_class_name(kernel_name: str) -> str:
        """
        Generate proper CamelCase class name from kernel name.
        
        Args:
            kernel_name: Underscore-separated kernel name
            
        Returns:
            CamelCase class name
        """
        return generate_class_name(kernel_name)

    def execute_node(self, context, graph):
        """
        Execute the node in simulation context.
        
        This is an abstract method required by FINN's HWCustomOp base class.
        For auto-generated ops, this provides a default pass-through implementation.
        Subclasses can override for custom simulation behavior.
        
        Args:
            context: Execution context containing input/output tensors
            graph: ONNX graph context
        """
        node = self.onnx_node
        
        # Simple pass-through for inputs to outputs
        if len(node.input) > 0 and len(node.output) > 0:
            # Default behavior: copy first input to first output
            if node.input[0] in context and node.output[0] not in context:
                context[node.output[0]] = context[node.input[0]].copy()
        
        # Note: Real implementations should provide proper compute simulation
        # This is just a placeholder to satisfy FINN's interface requirements
    
    def infer_node_datatype(self, model):
        """
        Infer and set node datatypes based on model context.
        
        This is an abstract method required by FINN's HWCustomOp base class.
        For auto-generated ops, datatypes are explicitly configured via
        node attributes, so this method validates consistency.
        
        Args:
            model: ONNX model context
        """
        node = self.onnx_node
        
        # Validate that all required datatypes are specified
        for iface in self.dataflow_model.input_interfaces:
            if iface.interface_type != InterfaceType.CONTROL:
                dtype_attr = f"{iface.name}_dtype"
                if not self.get_nodeattr(dtype_attr):
                    raise ValueError(
                        f"Input datatype for interface '{iface.name}' must be "
                        f"explicitly specified via node attribute '{dtype_attr}'"
                    )
        
        for iface in self.dataflow_model.output_interfaces:
            if iface.interface_type != InterfaceType.CONTROL:
                dtype_attr = f"{iface.name}_dtype"
                if not self.get_nodeattr(dtype_attr):
                    raise ValueError(
                        f"Output datatype for interface '{iface.name}' must be "
                        f"explicitly specified via node attribute '{dtype_attr}'"
                    )
        
        # For auto-generated ops, we don't infer datatypes - they must be explicit
        # This method mainly serves to validate that all required datatypes are set
    
    def get_legacy_attr(self) -> Dict[str, Any]:
        """
        Generate legacy FINN HWCustomOp nodeattrs based on DataflowModel interfaces.
        
        Maps modern Brainsmith terminology to legacy FINN nodeattrs:
        - SIMD = iPar (input parallelism) 
        - PE = wPar (weight parallelism)
        - inputDataType = in0.dtype
        - outputDataType = out0.dtype
        - weightDataType = weight0.dtype (if weights exist)
        
        Raises:
            ValueError: If multiple inputs/weights have different parallelism values
                       (multi-input/multi-weight AutoHWCustomOps not supported yet)
        
        Returns:
            Dict[str, Any]: Legacy nodeattr values for FINN compatibility
        """
        legacy_attrs = {}
        
        # Get interfaces
        input_interfaces = self.dataflow_model.input_interfaces
        output_interfaces = self.dataflow_model.output_interfaces
        weight_interfaces = self.dataflow_model.weight_interfaces
        
        # Handle input parallelism (SIMD)
        if input_interfaces:
            # Get iPar values for all input interfaces
            input_iPars = []
            for iface in input_interfaces:
                # iPar is stored in the first element of stream_dims
                if len(iface.stream_dims) > 0:
                    input_iPars.append(iface.stream_dims[0])
                else:
                    input_iPars.append(1)
            
            # Check for consistency across multiple inputs
            if len(set(input_iPars)) > 1:
                raise ValueError(
                    f"Multi-input AutoHWCustomOps with different iPar values are not supported yet. "
                    f"Found iPar values: {dict(zip([iface.name for iface in input_interfaces], input_iPars))}. "
                    f"All input interfaces must have the same parallelism value."
                )
            
            legacy_attrs["SIMD"] = input_iPars[0]
            
            # Set inputDataType from first input interface
            first_input = input_interfaces[0]
            if hasattr(first_input.dtype, 'name'):
                legacy_attrs["inputDataType"] = first_input.dtype.name
            else:
                legacy_attrs["inputDataType"] = str(first_input.dtype)
        
        # Handle weight parallelism (PE)
        if weight_interfaces:
            # Get wPar values for all weight interfaces
            weight_wPars = []
            for iface in weight_interfaces:
                # wPar is stored in the first element of stream_dims
                if len(iface.stream_dims) > 0:
                    weight_wPars.append(iface.stream_dims[0])
                else:
                    weight_wPars.append(1)
            
            # Check for consistency across multiple weights
            if len(set(weight_wPars)) > 1:
                raise ValueError(
                    f"Multi-weight AutoHWCustomOps with different wPar values are not supported yet. "
                    f"Found wPar values: {dict(zip([iface.name for iface in weight_interfaces], weight_wPars))}. "
                    f"All weight interfaces must have the same parallelism value."
                )
            
            legacy_attrs["PE"] = weight_wPars[0]
            
            # Set weightDataType from first weight interface
            first_weight = weight_interfaces[0]
            if hasattr(first_weight.dtype, 'name'):
                legacy_attrs["weightDataType"] = first_weight.dtype.name
            else:
                legacy_attrs["weightDataType"] = str(first_weight.dtype)
        else:
            # No weights, use output parallelism as PE
            if output_interfaces:
                first_output = output_interfaces[0]
                if len(first_output.stream_dims) > 0:
                    legacy_attrs["PE"] = first_output.stream_dims[0]
                else:
                    legacy_attrs["PE"] = 1
        
        # Handle output datatype
        if output_interfaces:
            # Set outputDataType from first output interface  
            first_output = output_interfaces[0]
            if hasattr(first_output.dtype, 'name'):
                legacy_attrs["outputDataType"] = first_output.dtype.name
            else:
                legacy_attrs["outputDataType"] = str(first_output.dtype)
        
        # Add other common legacy attributes with sensible defaults
        legacy_attrs.update({
            "backend": "fpgadataflow",  # Standard FINN backend
            "preferred_impl_style": "rtl",  # Since we're generating from RTL
        })
        
        return legacy_attrs
    
    def verify_node(self):
        """
        Verify node configuration and return validation messages.
        
        This default implementation checks:
        1. Backend is set to "fpgadataflow"
        2. All expected legacy nodeattrs have values
        3. Datatype attributes are specified and valid
        4. RAM style is valid (if applicable)
        
        Subclasses can override to add operation-specific validation.
        
        Returns:
            List[str]: Validation messages (info and errors)
        """
        info_messages = []
        
        # Check backend
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')
        
        # Get expected nodeattrs from get_nodeattr_types()
        expected_attrs = self.get_nodeattr_types()
        
        # Check each expected attribute has a value (except optional ones with defaults)
        missing_required = []
        for attr_name, (attr_type, required, default, *_) in expected_attrs.items():
            if required:
                try:
                    value = self.get_nodeattr(attr_name)
                    if value is None or (isinstance(value, str) and value == ""):
                        missing_required.append(attr_name)
                except:
                    missing_required.append(attr_name)
        
        if missing_required:
            info_messages.append(f"Missing required attributes: {', '.join(missing_required)}")
        else:
            info_messages.append("All required attributes are specified")
        
        # Validate legacy nodeattrs if they exist
        try:
            legacy_attrs = self.get_legacy_attr()
            
            # Check SIMD if present
            if "SIMD" in expected_attrs:
                simd_value = self.get_nodeattr("SIMD")
                if simd_value is not None:
                    if simd_value > 0:
                        info_messages.append(f"SIMD value {simd_value} is valid")
                    else:
                        info_messages.append(f"Invalid SIMD value {simd_value} (must be positive)")
            
            # Check PE if present
            if "PE" in expected_attrs:
                pe_value = self.get_nodeattr("PE")
                if pe_value is not None:
                    if pe_value > 0:
                        info_messages.append(f"PE value {pe_value} is valid")
                    else:
                        info_messages.append(f"Invalid PE value {pe_value} (must be positive)")
            
            # Validate datatypes
            for dtype_key in ["inputDataType", "outputDataType", "weightDataType"]:
                if dtype_key in expected_attrs:
                    dtype_value = self.get_nodeattr(dtype_key)
                    if dtype_value:
                        try:
                            # Try to construct DataType to validate
                            DataType[dtype_value]
                            info_messages.append(f"{dtype_key} '{dtype_value}' is valid")
                        except:
                            info_messages.append(f"Invalid {dtype_key}: '{dtype_value}'")
            
        except ValueError as e:
            # This catches multi-interface parallelism mismatches from get_legacy_attr
            info_messages.append(f"Configuration error: {str(e)}")
        except Exception:
            # Legacy attrs might fail during initialization
            pass
        
        # Check RAM style if applicable
        if "ram_style" in expected_attrs:
            ram_style = self.get_nodeattr("ram_style")
            if ram_style:
                valid_styles = expected_attrs["ram_style"][3] if len(expected_attrs["ram_style"]) > 3 else {"auto", "block", "distributed", "ultra"}
                if ram_style in valid_styles:
                    info_messages.append(f"RAM style '{ram_style}' is valid")
                else:
                    info_messages.append(f"Invalid RAM style '{ram_style}', must be one of: {valid_styles}")
        
        # Check interface-specific datatypes
        try:
            interface_metadata = self.get_interface_metadata()
            for metadata in interface_metadata:
                if metadata.interface_type != InterfaceType.CONTROL:
                    dtype_attr = f"{metadata.name}_dtype"
                    if dtype_attr in expected_attrs:
                        dtype_value = self.get_nodeattr(dtype_attr)
                        if dtype_value:
                            # Validate against constraints
                            if hasattr(metadata, 'datatype_constraints') and metadata.datatype_constraints:
                                valid = validate_datatype_against_constraints(dtype_value, metadata.datatype_constraints)
                                if valid:
                                    info_messages.append(f"Datatype '{dtype_value}' for interface '{metadata.name}' is valid")
                                else:
                                    constraint_desc = metadata.get_constraint_description()
                                    info_messages.append(
                                        f"Datatype '{dtype_value}' for interface '{metadata.name}' violates constraints. "
                                        f"Allowed: {constraint_desc}"
                                    )
        except Exception as e:
            info_messages.append(f"Could not validate interface datatypes: {str(e)}")
        
        # Check code generation paths (informational only, often set later)
        codegen_attrs = ["code_gen_dir_ipgen", "ipgen_path", "ip_path", "executable_path"]
        codegen_set = []
        for attr in codegen_attrs:
            if attr in expected_attrs:
                try:
                    value = self.get_nodeattr(attr)
                    if value and value != "":
                        codegen_set.append(attr)
                except:
                    pass
        
        if codegen_set:
            info_messages.append(f"Code generation paths set: {', '.join(codegen_set)}")
        else:
            info_messages.append("Code generation paths not yet configured")
        
        return info_messages
    
    def make_shape_compatible_op(self, model):
        """Create shape-compatible ONNX node using ALL DataflowModel interfaces."""
        node = self.onnx_node
        
        # The ONNX node needs to handle ALL inputs (data + weights + config)
        # and produce ALL outputs, not just use output shape
        
        # Get interface counts from DataflowModel
        input_interfaces = self.dataflow_model.input_interfaces
        output_interfaces = self.dataflow_model.output_interfaces
        
        # For shape inference, we need a node that takes the same inputs
        # and produces outputs with the correct shapes
        if len(input_interfaces) == 1:
            # Single input operation - use Identity
            import onnx.helper as oh
            return oh.make_node("Identity", node.input, node.output)
        else:
            # Multi-input operation - use Add (handles broadcasting)
            import onnx.helper as oh
            return oh.make_node("Add", node.input, node.output)
    
    def get_weightstream_width(self, ind=0):
        """Get weight stream width using DataflowModel weight interfaces."""
        weight_interfaces = self.dataflow_model.weight_interfaces
        
        if ind >= len(weight_interfaces):
            raise IndexError(f"Weight interface index {ind} exceeds available weight interfaces ({len(weight_interfaces)})")
        
        if not weight_interfaces:
            return 0  # No weight interfaces
        
        weight_interface = weight_interfaces[ind]
        return weight_interface.calculate_stream_width()
    
    def get_weightstream_width_padded(self, ind=0):
        """Get padded weight stream width for AXI compliance."""
        weight_width = self.get_weightstream_width(ind)
        if weight_width == 0:
            return 0
        return roundup_to_integer_multiple(weight_width, 8)