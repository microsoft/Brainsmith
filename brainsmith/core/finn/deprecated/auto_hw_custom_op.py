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

from qonnx.core import ModelWrapper

# Kernel Modeling imports
from brainsmith.core.dataflow import (
    KernelSchema,
    KernelModel,
    InputInterface,
    OutputInterface
)
from brainsmith.core.dataflow.base import ParameterBinding


class AutoHWCustomOp(HWCustomOp, ABC):
    """
    Base class for HWCustomOp implementations using Kernel Modeling.
    
    This class bridges FINN's HWCustomOp interface with the Kernel Modeling
    system, providing automatic implementation of all required methods through
    delegation to KernelModel interfaces.
    
    The class implements a three-tier architecture:
    1. Static: KernelSchema defines interfaces and constraints (class attribute)
    2. Runtime: KernelModel instantiated with concrete types and shapes
    3. Dynamic: SDIM configuration for parallelism control
    
    Initialization Flow:
        The KernelModel is automatically initialized when InferShapes is run
        on the model. This happens via the make_shape_compatible_op() hook,
        ensuring the model is ready before any shape-dependent operations.
    
    Class Attributes:
        kernel_schema: KernelSchema providing static schema (must be overridden)
    """
    
    # Subclasses must override this class attribute
    kernel_schema: KernelSchema = None
    
    def __init__(self, onnx_node, **kwargs):
        """
        Initialize AutoHWCustomOp.
        
        Args:
            onnx_node: ONNX node containing attributes and connections
            **kwargs: Additional arguments passed to HWCustomOp
        """
        super().__init__(onnx_node, **kwargs)
        
        # Validate subclass defined the schema
        if self.kernel_schema is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define kernel_schema class attribute"
            )
    
    def get_kernel_model(self, model: ModelWrapper) -> KernelModel:
        """
        Get fresh kernel model by creating from current state.
        
        Always creates a new instance to ensure it reflects current attributes.
        
        Returns:
            KernelModel instance with current configuration
            
        Raises:
            RuntimeError: If ModelWrapper is not available
        """
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError(
                f"Cannot create KernelModel for {self.__class__.__name__}: "
                f"ModelWrapper not available. The node must be added to a model first."
            )
        
        # Create input models using factory method
        input_models = []
        for position, inp_schema in enumerate(self.kernel_schema.input_schemas):
            model = self._create_input_model(inp_schema, position)
            if model is not None:  # Skip optional missing inputs
                input_models.append(model)
        
        # Create output models using factory method
        output_models = []
        for position, out_schema in enumerate(self.kernel_schema.output_schemas):
            model = self._create_output_model(out_schema, position)
            output_models.append(model)
        
        # Combine parameters from all models
        all_params = {}
        for model in input_models + output_models:
            if hasattr(model, 'parameter_binding') and model.parameter_binding:
                all_params.update(model.parameter_binding.parameters)
        
        from brainsmith.core.dataflow.base import ParameterBinding
        param_binding = ParameterBinding(all_params) if all_params else None
        
        # Create kernel model
        from brainsmith.core.dataflow.kernel_model import KernelModel
        kernel_model = KernelModel(
            input_models=input_models,
            output_models=output_models,
            schema=self.kernel_schema,
            parameter_binding=param_binding
        )
        
        # Apply configuration
        self._configure_model_defaults(kernel_model)
        self._apply_legacy_attributes_to_model(kernel_model)
        
        return kernel_model
    
    def update_node_model(self, model):
        """
        Update model context. KernelModel is created fresh on access.
        
        Args:
            model: ModelWrapper with access to ONNX graph
        """
        # Just store the model reference
        self.model = model
    
    def set_model_context(self, model):
        """
        Set the model context.
        
        Args:
            model: ModelWrapper containing this node
        """
        self.model = model
    
    def _configure_model_defaults(self, kernel_model: KernelModel) -> None:
        """
        Configure default settings for a kernel model.
        
        Args:
            kernel_model: The model to configure
        """
        # Configure default SDIM for all inputs (set to 1 for all dimensions)
        sdim_params = kernel_model.get_sdim_parameters()
        if sdim_params:
            default_config = {}
            for intf_name, param_info in sdim_params.items():
                # Set SDIM to 1 for all free dimensions (conservative default)
                default_config[intf_name] = 1
            
            # Apply the default configuration
            kernel_model.configure_sdim(default_config)
        
        # Recompute output rates after SDIM configuration
        kernel_model.compute_output_rates()
    
    def _apply_legacy_attributes_to_model(self, kernel_model: KernelModel) -> None:
        """
        Apply operation-specific legacy attribute mappings to a model.
        
        This method allows subclasses to interpret legacy FINN attributes
        and apply them to the kernel model.
        
        Args:
            kernel_model: The model to configure
        """
        # Default implementation does nothing
        # Subclasses should override to handle their specific legacy attributes
        pass
    
    def _enrich_attrs_with_onnx_info(self, attrs: Dict[str, Any]) -> None:
        """
        Enrich attributes dictionary with information extracted from ONNX model.
        
        Adds standardized attributes:
        - {port_name}_shape: Shape from ONNX tensor
        - {port_name}_dtype: Datatype name from ONNX tensor  
        - input_{index}_shape: Fallback by index
        - input_{index}_dtype: Fallback by index
        - output_{index}_shape: For outputs
        - output_{index}_dtype: For outputs
        
        Args:
            attrs: Dictionary to enrich with ONNX information
        """
        # Add input tensor information
        for i, tensor_name in enumerate(self.onnx_node.input):
            if tensor_name == "":
                continue
                
            # Get shape and datatype
            shape = self.model.get_tensor_shape(tensor_name)
            dtype = self.model.get_tensor_datatype(tensor_name)
            
            # Add by index (always available)
            attrs[f"input_{i}_shape"] = list(shape)
            attrs[f"input_{i}_dtype"] = dtype.name
            
            # Try to add by port name
            port_name = self._get_input_port_name(i)
            if port_name:
                attrs[f"{port_name}_shape"] = list(shape)
                attrs[f"{port_name}_dtype"] = dtype.name
        
        # Add output tensor information
        for i, tensor_name in enumerate(self.onnx_node.output):
            if tensor_name == "":
                continue
                
            # Get shape and datatype
            shape = self.model.get_tensor_shape(tensor_name)
            dtype = self.model.get_tensor_datatype(tensor_name)
            
            # Add by index (always available)
            attrs[f"output_{i}_shape"] = list(shape)
            attrs[f"output_{i}_dtype"] = dtype.name
            
            # Try to add by port name
            port_name = self._get_output_port_name(i)
            if port_name:
                attrs[f"{port_name}_shape"] = list(shape)
                attrs[f"{port_name}_dtype"] = dtype.name
    
    def _get_input_port_name(self, index: int) -> Optional[str]:
        """Get the port name for an input by index."""
        if index < len(self.kernel_schema.input_schemas):
            return self.kernel_schema.input_schemas[index].name
        return None
    
    def _get_output_port_name(self, index: int) -> Optional[str]:
        """Get the port name for an output by index."""
        if index < len(self.kernel_schema.output_schemas):
            return self.kernel_schema.output_schemas[index].name
        return None
    
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
        for i, inp_schema in enumerate(self.kernel_schema.input_schemas):
            if i >= len(self.onnx_node.input):
                if not inp_schema.optional:
                    raise ValueError(f"Missing required input '{inp_schema.name}' at position {i}")
                continue
            
            tensor_name = self.onnx_node.input[i]
            if not tensor_name:
                if not inp_schema.optional:
                    raise ValueError(f"Missing required input '{inp_schema.name}' at position {i}")
                continue
            
            # Get tensor info
            shape = model.get_tensor_shape(tensor_name)
            dtype = model.get_tensor_datatype(tensor_name)
            has_initializer = model.get_initializer(tensor_name) is not None
            
            # Validate weight expectations
            if inp_schema.is_weight and not has_initializer:
                raise ValueError(
                    f"Input '{inp_schema.name}' at position {i} is defined as a weight "
                    f"but ONNX tensor '{tensor_name}' has no initializer"
                )
            elif not inp_schema.is_weight and has_initializer:
                raise ValueError(
                    f"Input '{inp_schema.name}' at position {i} is not defined as a weight "
                    f"but ONNX tensor '{tensor_name}' has an initializer"
                )
            
            # Store the spec
            specs[inp_schema.name] = (tuple(shape), dtype)
        
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
        for i, out_schema in enumerate(self.kernel_schema.output_schemas):
            if i >= len(self.onnx_node.output):
                raise ValueError(f"Missing output '{out_schema.name}' at position {i}")
            
            tensor_name = self.onnx_node.output[i]
            if not tensor_name:
                raise ValueError(f"Missing output '{out_schema.name}' at position {i}")
            
            # Get tensor info
            shape = model.get_tensor_shape(tensor_name)
            dtype = model.get_tensor_datatype(tensor_name)
            
            specs[out_schema.name] = (tuple(shape), dtype)
        
        return specs
    
    def _extract_input_specs(self) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
        """
        Extract input specifications from current ONNX state.
        
        Returns:
            Dictionary mapping input names to (shape, datatype) tuples
        """
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("ModelWrapper not available")
        
        return self._extract_input_specs_from_onnx(self.model)
    
    def _extract_output_specs(self) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
        """
        Extract output specifications from current ONNX state.
        
        Returns:
            Dictionary mapping output names to (shape, datatype) tuples
        """
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("ModelWrapper not available")
        
        return self._extract_output_specs_from_onnx(self.model)
    
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
        regular_inputs = self.kernel_schema.get_regular_inputs()
        weight_inputs = self.kernel_schema.get_weight_inputs()
        outputs = self.kernel_schema.output_schemas
        
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
    
    # Model Factory Methods

    from brainsmith.core.dataflow.input_interface import InputInterface
    from qonnx.core.datatype import DataType
    from brainsmith.core.dataflow.base import ParameterBinding


    def get_input_model(self, model, position=0) -> 'InputInterface':
        """Get the InputInterface for a specific input position."""
        schema = self.kernel_schema.inputs[position]
        tensor = self.onnx_node.input[position]
        shape = model.get_tensor_shape(tensor)
        dtype = model.get_tensor_datatype(tensor)


        tensor = model.get_tensor(schema.name)
        return self._create_input_model(schema)
    
    
                dt = model.get_tensor_datatype(i2c_input)
    

    
    def execute_node(self, context, graph):
        node = self.onnx_node
        in_act = context[node.input[0]]
        # ensure that shape is compatible
        in_act = in_act.reshape(self.get_normal_input_shape())

        

        if self.get_nodeattr("dynamic_input"):
            mvau_w = context[node.input[1]]
        else:
            mvau_w_init = [x for x in graph.initializer if x.name == node.input[1]][0]
            mvau_w = np_helper.to_array(mvau_w_init)



            

    def _create_input_model(self, schema: 'InputSchema') -> 'InputInterface':
 



    def _create_input_model(self, 
                           schema: 'InputSchema', 
                           position: int) -> 'InputInterface':
        """Create InputInterface from schema and ONNX position.
        
        Args:
            schema: Input schema with constraints
            position: Position in ONNX node inputs
            
        Returns:
            InputInterface instance or None if optional and missing
            
        Raises:
            ValueError: If required input missing or validation fails
            KeyError: If required nodeattr not found
        """
        from brainsmith.core.dataflow.input_interface import InputInterface
        from qonnx.core.datatype import DataType
        from brainsmith.core.dataflow.base import ParameterBinding
        
        # Check ONNX input exists at position
        if position >= len(self.onnx_node.input):
            if schema.optional:
                return None
            raise ValueError(f"No ONNX input at position {position} for '{schema.name}'")
        
        tensor_name = self.onnx_node.input[position]
        if not tensor_name:
            if schema.optional:
                return None
            raise ValueError(f"Empty ONNX input at position {position} for '{schema.name}'")
        
        # Get ONNX tensor info
        tensor_shape = list(self.model.get_tensor_shape(tensor_name))
        onnx_dtype = self.model.get_tensor_datatype(tensor_name)
        
        # Check weight expectations
        has_initializer = self.model.get_initializer(tensor_name) is not None
        if schema.is_weight and not has_initializer:
            raise ValueError(
                f"Input '{schema.name}' at position {position} is defined as a weight "
                f"but ONNX tensor '{tensor_name}' has no initializer"
            )
        elif not schema.is_weight and has_initializer:
            raise ValueError(
                f"Input '{schema.name}' at position {position} is not defined as a weight "
                f"but ONNX tensor '{tensor_name}' has an initializer"
            )
        
        # Resolve datatype
        if schema.datatype_attr:
            dtype_value = self.get_nodeattr(schema.datatype_attr)
            if dtype_value is None:
                raise KeyError(f"Datatype attribute '{schema.datatype_attr}' not found")
            datatype = DataType[dtype_value] if isinstance(dtype_value, str) else dtype_value
        else:
            # Use ONNX datatype
            datatype = onnx_dtype
        
        # Validate datatype constraints
        if schema.datatype_constraints:
            if not schema.validates_datatype(datatype):
                valid_types = schema._get_valid_type_names()
                raise ValueError(
                    f"Datatype {datatype.name} doesn't satisfy constraints "
                    f"for input '{schema.name}'. Valid types: {valid_types}"
                )
        
        # Get all node attributes for resolution
        node_attrs = {}
        for attr_name in self.get_nodeattr_types():
            try:
                node_attrs[attr_name] = self.get_nodeattr(attr_name)
            except:
                pass  # Skip attributes that can't be retrieved
        
        # Resolve tiling templates
        block_dims = self._resolve_tiling_template(
            schema.block_tiling, tensor_shape, node_attrs
        )
        stream_dims = self._resolve_tiling_template(
            schema.stream_tiling, block_dims, node_attrs
        )
        
        # Extract parameters for this interface
        params = {}
        for key, value in node_attrs.items():
            if key.isupper():
                if isinstance(value, list) and len(value) == 1:
                    params[key] = value[0]
                elif isinstance(value, int):
                    params[key] = value
        param_binding = ParameterBinding(params) if params else None
        
        # Create model
        return InputInterface(
            tensor_dims=tensor_shape,
            block_dims=block_dims,
            stream_dims=stream_dims,
            datatype=datatype,
            schema=schema,
            parameter_binding=param_binding
        )
    
    def _create_output_model(self,
                            schema: 'OutputSchema',
                            position: int) -> 'OutputInterface':
        """Create OutputInterface from schema and ONNX position.
        
        Args:
            schema: Output schema with constraints
            position: Position in ONNX node outputs
            
        Returns:
            OutputInterface instance
            
        Raises:
            ValueError: If output missing or validation fails
            KeyError: If required nodeattr not found
        """
        from brainsmith.core.dataflow.output_interface import OutputInterface
        from qonnx.core.datatype import DataType
        from brainsmith.core.dataflow.base import ParameterBinding
        
        # Check ONNX output exists at position
        if position >= len(self.onnx_node.output):
            raise ValueError(f"No ONNX output at position {position} for '{schema.name}'")
        
        tensor_name = self.onnx_node.output[position]
        if not tensor_name:
            raise ValueError(f"Empty ONNX output at position {position} for '{schema.name}'")
        
        # Get ONNX tensor info
        tensor_shape = list(self.model.get_tensor_shape(tensor_name))
        onnx_dtype = self.model.get_tensor_datatype(tensor_name)
        
        # Resolve datatype
        if schema.datatype_attr:
            dtype_value = self.get_nodeattr(schema.datatype_attr)
            if dtype_value is None:
                raise KeyError(f"Datatype attribute '{schema.datatype_attr}' not found")
            datatype = DataType[dtype_value] if isinstance(dtype_value, str) else dtype_value
        else:
            # Use ONNX datatype
            datatype = onnx_dtype
        
        # Validate datatype constraints
        if schema.datatype_constraints:
            if not schema.validates_datatype(datatype):
                valid_types = schema._get_valid_type_names()
                raise ValueError(
                    f"Datatype {datatype.name} doesn't satisfy constraints "
                    f"for output '{schema.name}'. Valid types: {valid_types}"
                )
        
        # Get all node attributes for resolution
        node_attrs = {}
        for attr_name in self.get_nodeattr_types():
            try:
                node_attrs[attr_name] = self.get_nodeattr(attr_name)
            except:
                pass  # Skip attributes that can't be retrieved
        
        # Resolve block tiling (outputs don't have stream tiling)
        block_dims = self._resolve_tiling_template(
            schema.block_tiling, tensor_shape, node_attrs
        )
        
        # Extract parameters
        params = {}
        for key, value in node_attrs.items():
            if key.isupper():
                if isinstance(value, list) and len(value) == 1:
                    params[key] = value[0]
                elif isinstance(value, int):
                    params[key] = value
        param_binding = ParameterBinding(params) if params else None
        
        # Create model
        return OutputInterface(
            tensor_dims=tensor_shape,
            block_dims=block_dims,
            datatype=datatype,
            schema=schema,
            parameter_binding=param_binding
        )
    
    def _resolve_tiling_template(self,
                                template: Optional[List[Union[int, str]]],
                                reference_shape: List[int],
                                node_attrs: Dict[str, Any]) -> List[int]:
        """Resolve tiling template against reference shape.
        
        Args:
            template: Tiling template with 1, ":", nodeattr names, or ints
            reference_shape: Shape to tile against
            node_attrs: Node attributes for resolving references
            
        Returns:
            Resolved tile dimensions
            
        Raises:
            KeyError: If nodeattr reference not found
            ValueError: If tile doesn't divide dimension evenly
        """
        if not template:
            return list(reference_shape)
        
        # Make a copy to avoid modifying original
        template = list(template)
        
        # Left-pad with 1s if template shorter than shape
        if len(template) < len(reference_shape):
            template = [1] * (len(reference_shape) - len(template)) + template
        
        resolved = []
        for i, (expr, ref_dim) in enumerate(zip(template, reference_shape)):
            if expr == 1:
                resolved.append(1)
            elif expr == ":":
                resolved.append(ref_dim)
            elif isinstance(expr, str):
                # Node attribute reference
                value = node_attrs.get(expr)
                if value is None:
                    raise KeyError(f"Tiling parameter '{expr}' not found in node_attrs")
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                resolved.append(int(value))
            elif isinstance(expr, int):
                resolved.append(expr)
            else:
                raise ValueError(f"Invalid tiling expression: {expr}")
        
        # Validate divisibility
        for j, (ref, tile) in enumerate(zip(reference_shape, resolved)):
            if ref % tile != 0:
                raise ValueError(
                    f"Tile size {tile} doesn't evenly divide dimension {ref} "
                    f"at position {j}"
                )
        
        return resolved
    
    
    
    
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
        kernel_model = self.get_kernel_model()
        input_models = list(kernel_model.input_models)
        
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
        kernel_model = self.get_kernel_model()
        output_models = list(kernel_model.output_models)
        
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
        kernel_model = self.get_kernel_model()
        input_models = list(kernel_model.input_models)
        
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
        kernel_model = self.get_kernel_model()
        output_models = list(kernel_model.output_models)
        
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
        kernel_model = self.get_kernel_model()
        input_models = list(kernel_model.input_models)
        
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
                    f"for input '{iface.schema.name}'"
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
        kernel_model = self.get_kernel_model()
        output_models = list(kernel_model.output_models)
        
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
        kernel_model = self.get_kernel_model()
        input_models = list(kernel_model.input_models)
        
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
        kernel_model = self.get_kernel_model()
        output_models = list(kernel_model.output_models)
        
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
        return self.kernel_schema.has_weights()
    
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
    
    def _get_interface_model(self, interface_name: str) -> Union[InputInterface, OutputInterface, None]:
        """
        Get interface model by name.
        
        Args:
            interface_name: Name of the interface (compiler name)
            
        Returns:
            Interface model or None if not found
        """
        kernel_model = self.get_kernel_model()
        
        # Check inputs
        for inp in kernel_model.input_models:
            if inp.schema.name == interface_name:
                return inp
        
        # Check outputs
        for out in kernel_model.output_models:
            if out.schema.name == interface_name:
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
        kernel_model = self.get_kernel_model()
        
        # Check inputs
        for i, inp in enumerate(kernel_model.input_models):
            if inp.schema.name == name:
                return (i, True)
        
        # Check outputs  
        for i, out in enumerate(kernel_model.output_models):
            if out.schema.name == name:
                return (i, False)
                
        return (-1, False)
    
    def get_exp_cycles(self) -> int:
        """
        Get expected execution cycles.
        
        Returns:
            Expected cycles for one inference
        """
        try:
            kernel_model = self.get_kernel_model()
            metrics = kernel_model.calculate_performance_metrics()
            
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
        try:
            kernel_model = self.get_kernel_model()
            
            for i, inp_model in enumerate(kernel_model.input_models):
                dtype = inp_model.datatype
                if dtype:
                    info_messages.append(f"✓ Input {i} has datatype {dtype}")
                else:
                    info_messages.append(f"✗ Input {i} missing datatype")
            
            for i, out_model in enumerate(kernel_model.output_models):
                dtype = out_model.datatype
                if dtype:
                    info_messages.append(f"✓ Output {i} has datatype {dtype}")
                else:
                    info_messages.append(f"✗ Output {i} missing datatype")
        except Exception as e:
            info_messages.append(f"✗ Cannot verify datatypes: {e}")
        
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
        
        Args:
            model: ONNX ModelWrapper
            
        Returns:
            Shape-compatible ONNX operation
        """
        # Call parent implementation which creates const shape op
        return super().make_shape_compatible_op(model)
    
    def infer_node_datatype(self, model, node=None):
        """
        Infer node datatypes during graph transformation.
        
        This method supports both QONNX (1 arg) and FINN (2 arg) signatures.
        
        Args:
            model: ONNX model wrapper
            node: ONNX node (optional, defaults to self.onnx_node)
        """
        # No-op: KernelModel is created fresh on access
        pass
    
    def export_template_parameters(self) -> Dict[str, Any]:
        """Export KernelModel parameters for RTL template generation.
        
        Extracts all interface properties and parameters in a format
        suitable for RTL code generation templates.
        
        Returns:
            Dictionary with parameter names and values
        """
        kernel_model = self.get_kernel_model()
        params = {}
        
        # Export interface properties
        for inp in kernel_model.input_models:
            name_upper = inp.schema.name.upper()
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
        for out in kernel_model.output_models:
            name_upper = out.schema.name.upper()
            params[f"{name_upper}_WIDTH"] = out.datatype.bitwidth()
            params[f"{name_upper}_SIGNED"] = 1 if out.datatype.signed() else 0
            
            # Add block dimensions if they exist
            if hasattr(out, 'block_dims') and out.block_dims:
                for i, bdim in enumerate(out.block_dims):
                    params[f"{name_upper}_BDIM{i}"] = bdim
        
        # Add parameter binding values if they exist
        if kernel_model.parameter_binding:
            for param_name, param_value in kernel_model.parameter_binding.params.items():
                params[param_name.upper()] = param_value
        
        return params