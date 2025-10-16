############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
AutoHWCustomOp base class using Kernel Modeling system.

This module provides a modern implementation of AutoHWCustomOp that leverages
the Kernel Modeling system for all shape, datatype, and performance calculations.
It serves as a base class for auto-generated HWCustomOp implementations,
providing automatic implementation of all FINN-required methods through
delegation to KernelModel interfaces.

Key Features:
- Clean integration with Kernel Modeling system
- Automatic shape and stream width calculations
- SDIM-based parallelism configuration
- Legacy SIMD/PE compatibility
- Resource estimation from performance metrics
"""

import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union

# FINN imports
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple

# Kernel Modeling imports
from brainsmith.dataflow import (
    KernelDefinition,
    KernelModel,
    InputInterface,
    OutputInterface
)
from brainsmith.dataflow.base import ParameterBinding


class AutoHWCustomOp(HWCustomOp, ABC):
    """
    Base class for HWCustomOp implementations using Kernel Modeling.
    
    This class bridges FINN's HWCustomOp interface with the Kernel Modeling
    system, providing automatic implementation of all required methods through
    delegation to KernelModel interfaces.
    
    The class implements a three-tier architecture:
    1. Static: KernelDefinition defines interfaces and constraints
    2. Runtime: KernelModel instantiated with concrete types and shapes
    3. Dynamic: SDIM configuration for parallelism control
    
    Initialization Flow:
        The KernelModel is automatically initialized when InferShapes is run
        on the model. This happens via the make_shape_compatible_op() hook,
        ensuring the model is ready before any shape-dependent operations.
    
    Attributes:
        _kernel_def: KernelDefinition providing static schema
        _kernel_model: KernelModel with runtime instances (initialized during InferShapes)
        _sdim_config: Cached SDIM configuration
    """
    
    def __init__(self, onnx_node, kernel_definition: KernelDefinition, **kwargs):
        """
        Initialize AutoHWCustomOp with a KernelDefinition.
        
        Args:
            onnx_node: ONNX node containing attributes and connections
            kernel_definition: Static kernel schema with interface definitions
            **kwargs: Additional arguments passed to HWCustomOp
        """
        super().__init__(onnx_node, **kwargs)
        
        # Store the kernel definition
        self._kernel_def = kernel_definition
        
        # KernelModel created lazily when runtime info available
        self._kernel_model = None
        
        # Cache for SDIM configuration
        self._sdim_config = {}
        
        # Try to initialize KernelModel immediately if ModelWrapper is available
        if hasattr(self, 'model') and self.model is not None:
            try:
                self._analyze_and_create_model(self.model)
                self._build_kernel_model()
            except Exception as e:
                print(f"DEBUG: Failed to initialize KernelModel in constructor: {e}")
                # Don't fail construction, will try later
    
    def _ensure_kernel_model(self):
        """
        Ensure KernelModel exists.
        
        Raises:
            RuntimeError: If KernelModel is not initialized
        """
        if self._kernel_model is None:
            raise RuntimeError(
                f"KernelModel not initialized for {self.__class__.__name__}. "
                f"The node must be properly initialized with tensor shapes from the ONNX graph. "
                f"This typically happens during shape inference (InferShapes transformation) "
                f"or when make_shape_compatible_op() is called. "
                f"Ensure the node has been added to a ModelWrapper and shapes have been inferred."
            )
    
    def update_node_model(self, model):
        """
        Update KernelModel with shapes from ONNX graph.
        This is called when ModelWrapper is available.
        
        Args:
            model: ModelWrapper with access to ONNX graph
        """
        # Create/update our KernelModel with full context
        self._analyze_and_create_model(model)
        self._build_kernel_model()
    
    def set_model_context(self, model):
        """
        Set the model context and initialize KernelModel.
        
        This should be called when the node is added to a model or
        after transformations that recreate nodes.
        
        Args:
            model: ModelWrapper containing this node
        """
        self.model = model
        if self._kernel_model is None:
            try:
                self._analyze_and_create_model(model)
                self._build_kernel_model()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize KernelModel for {self.__class__.__name__}: {e}"
                )
    
    def _analyze_and_create_model(self, model):
        """
        Analyze ONNX tensors and create KernelModel with proper specs.
        
        Args:
            model: ModelWrapper with access to ONNX graph
        """
        # Extract interface specifications
        input_specs = self._extract_input_specs_from_onnx(model)
        output_specs = self._extract_output_specs_from_onnx(model)
        
        # Get parameter binding
        param_binding = self._extract_parameter_binding()
        
        # Create KernelModel with analyzed specs
        self._kernel_model = self._kernel_def.create_model(
            input_specs=input_specs,
            output_specs=output_specs,
            parameter_binding=param_binding
        )
        
        # Apply any legacy attribute mappings
        self._apply_legacy_attributes()
    
    def _extract_input_specs_from_onnx(self, model) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
        """
        Extract input specifications by analyzing ONNX tensors.
        
        Assumes ONNX inputs are in the same order as kernel input definitions.
        Validates that weights have initializers and non-weights don't.
        
        Args:
            model: ModelWrapper with access to ONNX graph
            
        Returns:
            Dictionary mapping input names to (shape, datatype) tuples
        """
        specs = {}
        
        # Process each input in order
        for i, inp_def in enumerate(self._kernel_def.input_definitions):
            if i >= len(self.onnx_node.input):
                if not inp_def.optional:
                    raise ValueError(f"Missing required input '{inp_def.name}' at position {i}")
                continue
            
            tensor_name = self.onnx_node.input[i]
            if not tensor_name:
                if not inp_def.optional:
                    raise ValueError(f"Missing required input '{inp_def.name}' at position {i}")
                continue
            
            # Get tensor info
            shape = model.get_tensor_shape(tensor_name)
            dtype = model.get_tensor_datatype(tensor_name)
            has_initializer = model.get_initializer(tensor_name) is not None
            
            # Validate weight expectations
            if inp_def.is_weight and not has_initializer:
                raise ValueError(
                    f"Input '{inp_def.name}' at position {i} is defined as a weight "
                    f"but ONNX tensor '{tensor_name}' has no initializer"
                )
            elif not inp_def.is_weight and has_initializer:
                raise ValueError(
                    f"Input '{inp_def.name}' at position {i} is not defined as a weight "
                    f"but ONNX tensor '{tensor_name}' has an initializer"
                )
            
            # Store the spec
            specs[inp_def.name] = (tuple(shape), dtype)
        
        return specs
    
    def _extract_output_specs_from_onnx(self, model) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
        """
        Extract output specifications by analyzing ONNX tensors.
        
        Assumes ONNX outputs are in the same order as kernel output definitions.
        
        Args:
            model: ModelWrapper with access to ONNX graph
            
        Returns:
            Dictionary mapping output names to (shape, datatype) tuples
        """
        specs = {}
        
        # Process each output in order
        for i, out_def in enumerate(self._kernel_def.output_definitions):
            if i >= len(self.onnx_node.output):
                raise ValueError(f"Missing output '{out_def.name}' at position {i}")
            
            tensor_name = self.onnx_node.output[i]
            if not tensor_name:
                raise ValueError(f"Missing output '{out_def.name}' at position {i}")
            
            # Get tensor info
            shape = model.get_tensor_shape(tensor_name)
            dtype = model.get_tensor_datatype(tensor_name)
            
            specs[out_def.name] = (tuple(shape), dtype)
        
        return specs
    
    def _extract_input_specs(self) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
        """
        Extract input specifications from KernelModel.
        
        This method is provided for compatibility with generated code.
        It simply returns the specs from the already-created KernelModel.
        
        Returns:
            Dictionary mapping input names to (shape, datatype) tuples
        """
        self._ensure_kernel_model()
        
        specs = {}
        for inp_model in self._kernel_model.input_models:
            name = inp_model.definition.name
            specs[name] = (inp_model.tensor_dims, inp_model.datatype)
        return specs
    
    def _extract_output_specs(self) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
        """
        Extract output specifications from KernelModel.
        
        This method is provided for compatibility with generated code.
        It simply returns the specs from the already-created KernelModel.
        
        Returns:
            Dictionary mapping output names to (shape, datatype) tuples
        """
        self._ensure_kernel_model()
        
        specs = {}
        for out_model in self._kernel_model.output_models:
            name = out_model.definition.name
            specs[name] = (out_model.tensor_dims, out_model.datatype)
        return specs
    
    def _extract_parameter_binding(self) -> ParameterBinding:
        """
        Extract kernel parameters from ONNX node attributes.
        
        Only extracts parameters that are defined as node attributes.
        Shape parameters (BDIM/SDIM) are handled by KernelModel configuration,
        not through parameter binding.
        
        Returns:
            ParameterBinding with parameter name->value mappings
        """
        params = {}
        
        # Get all defined node attributes for this kernel
        nodeattr_types = self.get_nodeattr_types()
        
        # Extract only the parameters that are actually node attributes
        # Skip interface datatypes and FINN legacy parameters
        # Note: PE is included because modern kernels use it for stream tiling
        skip_attrs = {'SIMD', 'ram_style'}
        
        for attr_name in nodeattr_types:
            # Skip interface datatype attributes (end with DataType)
            if attr_name.endswith('DataType'):
                continue
                
            # Skip FINN legacy attributes
            if attr_name in skip_attrs:
                continue
                
            # Extract the parameter value
            try:
                value = self.get_nodeattr(attr_name)
                if value is not None:
                    params[attr_name] = value
            except AttributeError:
                # Parameter not found - skip it
                continue
        
        return ParameterBinding(params) if params else None
    
    def _get_finn_attribute_mapping(self) -> Dict[str, str]:
        """
        Generate FINN attribute names for all interfaces.
        
        Returns:
            Dictionary mapping interface names to FINN attribute names
        """
        attrs = {}
        
        # Get categorized inputs
        regular_inputs = self._kernel_def.get_regular_inputs()
        weight_inputs = self._kernel_def.get_weight_inputs()
        outputs = self._kernel_def.output_definitions
        
        # Map regular inputs
        if len(regular_inputs) == 1:
            attrs[regular_inputs[0].name] = "inputDataType"
        else:
            for i, inp in enumerate(regular_inputs):
                attrs[inp.name] = f"input{i}DataType"
        
        # Map weights
        if len(weight_inputs) == 1:
            attrs[weight_inputs[0].name] = "weightDataType"
        else:
            for i, weight in enumerate(weight_inputs):
                attrs[weight.name] = f"weight{i}DataType"
        
        # Map outputs
        if len(outputs) == 1:
            attrs[outputs[0].name] = "outputDataType"
        else:
            for i, out in enumerate(outputs):
                attrs[out.name] = f"output{i}DataType"
        
        return attrs
    
    
    
    
    # FINN Abstract Method Implementations
    
    def get_input_datatype(self, ind=0) -> DataType:
        """
        Get FINN DataType of input stream.
        
        Args:
            ind: Input index (default 0)
            
        Returns:
            QONNX DataType for the specified input
            
        Raises:
            IndexError: If index exceeds available inputs
        """
        self._ensure_kernel_model()
        input_models = list(self._kernel_model.input_models)
        
        if ind >= len(input_models):
            raise IndexError(
                f"Input index {ind} exceeds available inputs ({len(input_models)})"
            )
        
        return input_models[ind].datatype
    
    def get_output_datatype(self, ind=0) -> DataType:
        """
        Get FINN DataType of output stream.
        
        Args:
            ind: Output index (default 0)
            
        Returns:
            QONNX DataType for the specified output
            
        Raises:
            IndexError: If index exceeds available outputs
        """
        self._ensure_kernel_model()
        output_models = list(self._kernel_model.output_models)
        
        if ind >= len(output_models):
            raise IndexError(
                f"Output index {ind} exceeds available outputs ({len(output_models)})"
            )
        
        return output_models[ind].datatype
    
    def get_normal_input_shape(self, ind=0) -> List[int]:
        """
        Get normal (tensor) shape of input.
        
        Args:
            ind: Input index (default 0)
            
        Returns:
            List representing tensor dimensions
        """
        self._ensure_kernel_model()
        input_models = list(self._kernel_model.input_models)
        
        if ind >= len(input_models):
            raise IndexError(
                f"Input index {ind} exceeds available inputs ({len(input_models)})"
            )
        
        return list(input_models[ind].tensor_dims)
    
    def get_normal_output_shape(self, ind=0) -> List[int]:
        """
        Get normal (tensor) shape of output.
        
        Args:
            ind: Output index (default 0)
            
        Returns:
            List representing tensor dimensions
        """
        self._ensure_kernel_model()
        output_models = list(self._kernel_model.output_models)
        
        if ind >= len(output_models):
            raise IndexError(
                f"Output index {ind} exceeds available outputs ({len(output_models)})"
            )
        
        return list(output_models[ind].tensor_dims)
    
    def get_folded_input_shape(self, ind=0) -> List[int]:
        """
        Get folded shape for hardware implementation.
        
        Folded shape represents how data is organized for streaming:
        [num_blocks_dim0, num_blocks_dim1, ..., folded_block_dim0, folded_block_dim1, ...]
        
        Where folded_block_dim = block_dim / sdim
        
        Args:
            ind: Input index (default 0)
            
        Returns:
            List representing folded dimensions
        """
        self._ensure_kernel_model()
        input_models = list(self._kernel_model.input_models)
        
        if ind >= len(input_models):
            raise IndexError(
                f"Input index {ind} exceeds available inputs ({len(input_models)})"
            )
        
        iface = input_models[ind]
        
        # Calculate number of blocks in each dimension
        num_blocks = []
        for t, b in zip(iface.tensor_dims, iface.block_dims):
            num_blocks.append(math.ceil(t / b))
        
        # Calculate folded block dimensions (block_dims / sdim)
        folded_block = []
        for bd, sd in zip(iface.block_dims, iface.sdim):
            if bd % sd != 0:
                raise ValueError(
                    f"Block dimension {bd} not divisible by SDIM {sd} "
                    f"for input '{iface.definition.name}'"
                )
            folded_block.append(bd // sd)
        
        # Concatenate: [num_blocks..., folded_block_dims...]
        return num_blocks + folded_block
    
    def get_folded_output_shape(self, ind=0) -> List[int]:
        """
        Get folded shape for hardware output.
        
        Similar to input folding but uses output streaming rates.
        
        Args:
            ind: Output index (default 0)
            
        Returns:
            List representing folded dimensions
        """
        self._ensure_kernel_model()
        output_models = list(self._kernel_model.output_models)
        
        if ind >= len(output_models):
            raise IndexError(
                f"Output index {ind} exceeds available outputs ({len(output_models)})"
            )
        
        iface = output_models[ind]
        
        # Calculate number of blocks in each dimension
        num_blocks = []
        for t, b in zip(iface.tensor_dims, iface.block_dims):
            num_blocks.append(math.ceil(t / b))
        
        # For outputs, use streaming_rate instead of sdim
        # If not set, assume no folding (streaming_rate = block_dims)
        if hasattr(iface, '_streaming_rate') and iface._streaming_rate is not None:
            # Streaming rate is a single value, apply where it makes sense
            sr = iface.streaming_rate
            folded_block = []
            for bd in iface.block_dims:
                if sr > bd:
                    # Streaming rate exceeds block dim - no folding on this dimension
                    folded_block.append(1)
                elif bd % sr != 0:
                    # Try to find a divisor that works
                    folded_block.append(max(1, bd // sr))
                else:
                    folded_block.append(bd // sr)
        else:
            # No streaming rate set - no folding
            folded_block = [1] * len(iface.block_dims)
        
        return num_blocks + folded_block
    
    def get_instream_width(self, ind=0) -> int:
        """
        Get input stream width in bits.
        
        Stream width = datatype_bits * product(sdim)
        
        Args:
            ind: Input index (default 0)
            
        Returns:
            Width in bits
        """
        self._ensure_kernel_model()
        input_models = list(self._kernel_model.input_models)
        
        if ind >= len(input_models):
            raise IndexError(
                f"Input index {ind} exceeds available inputs ({len(input_models)})"
            )
        
        iface = input_models[ind]
        
        # Width = datatype bits * streaming elements
        datatype_bits = iface.datatype.bitwidth()
        streaming_elements = int(np.prod(iface.sdim))
        
        return datatype_bits * streaming_elements
    
    def get_outstream_width(self, ind=0) -> int:
        """
        Get output stream width in bits.
        
        Stream width = datatype_bits * product(streaming_rate)
        
        Args:
            ind: Output index (default 0)
            
        Returns:
            Width in bits
        """
        self._ensure_kernel_model()
        output_models = list(self._kernel_model.output_models)
        
        if ind >= len(output_models):
            raise IndexError(
                f"Output index {ind} exceeds available outputs ({len(output_models)})"
            )
        
        iface = output_models[ind]
        
        # Width = datatype bits * streaming elements
        datatype_bits = iface.datatype.bitwidth()
        
        # Use streaming_rate if available, otherwise assume 1
        if hasattr(iface, 'streaming_rate') and iface.streaming_rate:
            streaming_elements = int(np.prod(iface.streaming_rate))
        else:
            streaming_elements = 1
        
        return datatype_bits * streaming_elements
    
    # Node Attribute Management
    
    def get_nodeattr_types(self) -> Dict[str, Any]:
        """
        Define node attributes including auto-generated datatype attributes.
        
        If CodegenBinding is present, it will auto-generate algorithm parameters
        and other RTL-specific attributes.
        
        Returns:
            Dictionary of attribute definitions
        """
        # Start with parent class attributes
        attrs = super().get_nodeattr_types()
        
        # Use new KernelModeling architecture for attribute mapping
        finn_attrs = self._get_finn_attribute_mapping()
        for interface_name, attr_name in finn_attrs.items():
            # All datatype attributes are required
            attrs[attr_name] = ("s", True, "")
        
        # Add standard attributes
        attrs["SIMD"] = ("i", False, 1)  # Maps to input SDIM
        attrs["PE"] = ("i", False, 1)    # Maps to weight/output SDIM
        
        # Memory style for operations with weights
        if self._has_weight_inputs():
            attrs["ram_style"] = ("s", False, "auto", {"auto", "block", "distributed", "ultra"})
        
        return attrs
    
    def _has_weight_inputs(self) -> bool:
        """
        Check if kernel has weight inputs.
        
        Returns:
            True if any input has is_weight=True
        """
        return self._kernel_def.has_weights()
    
    def _apply_legacy_attributes(self):
        """
        Apply operation-specific legacy attribute mappings.
        
        This method is called after model creation to allow subclasses
        to interpret legacy FINN attributes (like numInputVectors, PE, SIMD)
        and apply them to the kernel model (e.g., modifying shapes or SDIM).
        
        Default implementation does nothing. Subclasses should override
        to handle their specific legacy attributes.
        """
        pass
    
    def _build_kernel_model(self):
        """
        Build and configure the KernelModel for proper initialization.
        
        This method ensures the KernelModel has proper SDIM configuration
        and output rates computed after creation.
        """
        if not self._kernel_model:
            return
        
        try:
            # Configure default SDIM for all inputs (set to 1 for all dimensions)
            sdim_params = self._kernel_model.get_sdim_parameters()
            if sdim_params:
                default_config = {}
                for intf_name, param_info in sdim_params.items():
                    # Set SDIM to 1 for all free dimensions (conservative default)
                    default_config[intf_name] = 1
                
                # Apply the default configuration
                self._kernel_model.configure_sdim(default_config)
            
            # Recompute output rates after SDIM configuration
            self._kernel_model.compute_output_rates()
        
        except Exception as e:
            # Log warning but don't fail
            import warnings
            warnings.warn(f"Failed to build KernelModel for {self.onnx_node.name}: {e}")
    
    def _get_interface_model(self, interface_name: str) -> Union[InputInterface, OutputInterface, None]:
        """
        Get interface model by name.
        
        Args:
            interface_name: Name of the interface (compiler name)
            
        Returns:
            Interface model or None if not found
        """
        self._ensure_kernel_model()
        
        # Check inputs
        for inp in self._kernel_model.input_models:
            if inp.definition.name == interface_name:
                return inp
        
        # Check outputs
        for out in self._kernel_model.output_models:
            if out.definition.name == interface_name:
                return out
        
        return None
    
    def _get_interface_datatype(self, interface_name: str) -> Optional[DataType]:
        """
        Get the datatype of an interface by name.
        
        Args:
            interface_name: Name of the interface
            
        Returns:
            QONNX DataType or None if not found
        """
        interface = self._get_interface_model(interface_name)
        if interface and hasattr(interface, 'datatype'):
            return interface.datatype
        return None
    
    def get_interface_index(self, name: str) -> tuple[int, bool]:
        """Get interface index by name.
        
        Args:
            name: Interface name to look up
            
        Returns:
            Tuple of (index, is_input) or (-1, False) if not found
        """
        self._ensure_kernel_model()
        
        # Check inputs
        for i, inp in enumerate(self._kernel_model.input_models):
            if inp.definition.name == name:
                return (i, True)
        
        # Check outputs  
        for i, out in enumerate(self._kernel_model.output_models):
            if out.definition.name == name:
                return (i, False)
                
        return (-1, False)
    
    def get_exp_cycles(self) -> int:
        """
        Get expected execution cycles.
        
        Returns:
            Expected cycles for one inference
        """
        self._ensure_kernel_model()
        
        try:
            metrics = self._kernel_model.calculate_performance_metrics()
            
            # Look for initiation interval in aggregate metrics
            if "aggregate" in metrics:
                if "initiation_interval" in metrics["aggregate"]:
                    return int(metrics["aggregate"]["initiation_interval"])
                elif "latency" in metrics["aggregate"]:
                    return int(metrics["aggregate"]["latency"])
        
        except Exception:
            # If metrics fail, return conservative estimate
            pass
        
        # Default: assume one cycle per output element
        return self.get_number_output_values()
    
    # Optional Method Implementations
    
    def verify_node(self) -> List[str]:
        """
        Verify node configuration.
        
        Returns:
            List of verification messages
        """
        info_messages = []
        
        # Check backend
        backend = self.get_nodeattr("backend")
        if backend == "fpgadataflow":
            info_messages.append("✓ Backend set correctly to 'fpgadataflow'")
        else:
            info_messages.append(f"✗ Backend '{backend}' should be 'fpgadataflow'")
        
        # Check required datatypes
        self._ensure_kernel_model()
        
        for i, inp_model in enumerate(self._kernel_model.input_models):
            dtype = inp_model.datatype
            if dtype:
                info_messages.append(f"✓ Input {i} has datatype {dtype}")
            else:
                info_messages.append(f"✗ Input {i} missing datatype")
        
        for i, out_model in enumerate(self._kernel_model.output_models):
            dtype = out_model.datatype
            if dtype:
                info_messages.append(f"✓ Output {i} has datatype {dtype}")
            else:
                info_messages.append(f"✗ Output {i} missing datatype")
        
        # Check SDIM configuration
        if self._sdim_config:
            info_messages.append(f"✓ SDIM configured: {self._sdim_config}")
        else:
            info_messages.append("ℹ No SDIM configuration (using defaults)")
        
        return info_messages
    
    def execute_node(self, context, graph):
        """
        Execute node in simulation.
        
        This is a basic pass-through implementation. Subclasses should
        override for actual computation.
        
        Args:
            context: Execution context with tensors
            graph: ONNX graph
        """
        # Simple pass-through: copy first input to first output
        node = self.onnx_node
        
        if len(node.input) > 0 and len(node.output) > 0:
            if node.input[0] in context:
                # Basic copy - subclasses should implement actual logic
                context[node.output[0]] = context[node.input[0]].copy()
    
    def make_shape_compatible_op(self, model):
        """
        Called by QONNX InferShapes to get a shape-compatible op.
        
        This is our primary initialization point - we ensure KernelModel
        exists before shape inference proceeds.
        
        Args:
            model: ONNX ModelWrapper
            
        Returns:
            Shape-compatible ONNX operation
        """
        # Ensure KernelModel is initialized with current graph state
        if not self._kernel_model:
            self._analyze_and_create_model(model)
            self._build_kernel_model()
        
        # Call parent implementation which creates const shape op
        return super().make_shape_compatible_op(model)
    
    def infer_node_datatype(self, model, node=None):
        """
        Infer node datatypes during graph transformation.
        
        This method supports both QONNX (1 arg) and FINN (2 arg) signatures.
        We use this hook to create/update our KernelModel with shapes.
        
        Args:
            model: ONNX model wrapper
            node: ONNX node (optional, defaults to self.onnx_node)
        """
        # Create/update KernelModel with shapes from ONNX
        if not self._kernel_model:
            self._analyze_and_create_model(model)
            self._build_kernel_model()
    
    def export_template_parameters(self) -> Dict[str, Any]:
        """Export KernelModel parameters for RTL template generation.
        
        Extracts all interface properties and parameters in a format
        suitable for RTL code generation templates.
        
        Returns:
            Dictionary with parameter names and values
        """
        self._ensure_kernel_model()
        params = {}
        
        # Export interface properties
        for inp in self._kernel_model.input_models:
            name_upper = inp.definition.name.upper()
            params[f"{name_upper}_WIDTH"] = inp.datatype.bitwidth()
            params[f"{name_upper}_SIGNED"] = 1 if inp.datatype.signed() else 0
            
            # Add block dimensions if they exist
            if hasattr(inp, 'block_dims') and inp.block_dims:
                for i, bdim in enumerate(inp.block_dims):
                    params[f"{name_upper}_BDIM{i}"] = bdim
            
            # Add stream dimensions if they exist
            if hasattr(inp, 'sdim') and inp.sdim:
                for i, sdim in enumerate(inp.sdim):
                    params[f"{name_upper}_SDIM{i}"] = sdim
        
        # Export output properties
        for out in self._kernel_model.output_models:
            name_upper = out.definition.name.upper()
            params[f"{name_upper}_WIDTH"] = out.datatype.bitwidth()
            params[f"{name_upper}_SIGNED"] = 1 if out.datatype.signed() else 0
            
            # Add block dimensions if they exist
            if hasattr(out, 'block_dims') and out.block_dims:
                for i, bdim in enumerate(out.block_dims):
                    params[f"{name_upper}_BDIM{i}"] = bdim
        
        # Add parameter binding values if they exist
        if self._kernel_model.parameter_binding:
            for param_name, param_value in self._kernel_model.parameter_binding.params.items():
                params[param_name.upper()] = param_value
        
        return params