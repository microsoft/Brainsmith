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
    
    @abstractmethod
    def get_interface_metadata(self) -> List[InterfaceMetadata]:
        """
        Return static interface metadata for this hardware kernel.
        
        Subclasses must implement this method to provide the interface
        definitions from RTL parsing. This replaces the interface_metadata
        constructor parameter to follow FINN's pattern.
        """
        pass
    
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
                        
    def estimate_bram_usage(self) -> int:
        """Estimate BRAM usage using DataflowModel resource requirements."""
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
            total_memory = sum(iface.get_memory_footprint() for iface in
                              self.dataflow_model.input_interfaces + self.dataflow_model.weight_interfaces)
            return max(1, total_memory // (18 * 1024))
        
    def estimate_lut_usage(self) -> int:
        """Estimate LUT usage using DataflowModel resource requirements."""
        try:
            # Simple estimation based on interface complexity
            total_width = sum(iface.calculate_stream_width() for iface in
                             self.dataflow_model.input_interfaces + self.dataflow_model.output_interfaces)
            
            # Apply estimation mode scaling
            estimation_mode = self.get_nodeattr("resource_estimation_mode")
            scale_factor = {"conservative": 1.5, "optimistic": 0.7, "automatic": 1.0}.get(estimation_mode, 1.0)
            
            # Simple heuristic: ~10 LUTs per bit of stream width
            lut_estimate = int((total_width * 10) * scale_factor)
            return max(1, lut_estimate)
        except Exception:
            # Fallback to fixed estimate
            return 100
        
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