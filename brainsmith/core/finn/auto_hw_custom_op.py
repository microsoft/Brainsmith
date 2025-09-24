############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Base class for automatic hardware custom operators.

Uses cached kernel model state for performance while providing
refresh mechanism via transforms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union, Sequence, Set

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.core.dataflow import KernelSchema
from brainsmith.core.dataflow.models import (
    KernelModel,
    InputModel,
    OutputModel,
    create_kernel_model,
    update_kernel_stream_config
)


class AutoHWCustomOp(HWCustomOp, ABC):
    """Base class for automatic hardware custom operators.
    
    Key features:
    - Caches kernel model for performance
    - Provides refresh_kernel_model() for state updates
    - All model access through get_kernel_model()
    """
    
    # Subclasses must define this class attribute
    kernel_schema: KernelSchema = None
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize with cached model state."""
        super().__init__(onnx_node, **kwargs)
        
        if self.kernel_schema is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define kernel_schema class attribute"
            )
        
        # Cache for kernel model - will be created on first access
        self._kernel_model: Optional[KernelModel] = None

    def get_kernel_model(self) -> KernelModel:
        """Get the cached kernel model.
        
        Returns cached model if available, otherwise raises error.
        Call refresh_kernel_model() first to populate cache.
        """
        if self._kernel_model is None:
            raise RuntimeError(
                "Kernel model not initialized. Call refresh_kernel_model() first."
            )
        return self._kernel_model
    
    def refresh_kernel_model(self, model: ModelWrapper) -> None:
        """Refresh the cached kernel model from current state.
        
        This should be called by transforms when shapes or types change.
        
        Args:
            model: The global ModelWrapper instance (not cached)
        """
        self._kernel_model = self._create_kernel_model(model)
    
    def _create_kernel_model(self, model: ModelWrapper) -> KernelModel:
        """Create fresh kernel model from current nodeattrs.
        
        Returns:
            Fresh KernelModel reflecting current nodeattrs
        """
        # Create input models
        input_models = []
        for i in range(len(self.kernel_schema.inputs)):
            inp_model = self._create_input_model(i, model)
            if inp_model is not None:  # Skip optional missing
                input_models.append(inp_model)
        
        # Create output models
        output_models = []
        for i in range(len(self.kernel_schema.outputs)):
            out_model = self._create_output_model(i, model)
            output_models.append(out_model)
        
        # Extract parameters
        parameters = self._extract_parameters()
        
        # Create kernel model
        return create_kernel_model(
            name=self.kernel_schema.name,
            inputs=input_models,
            outputs=output_models,
            parameters=parameters,
            clock_freq_mhz=self.get_nodeattr("clock_freq_mhz", 100.0)
        )
    
    def _resolve_interface_datatype(
        self,
        schema,
        position: int,
        is_input: bool
    ) -> DataType:
        """Resolve datatype for an interface from nodeattr.
        
        Args:
            schema: Interface schema
            position: Interface position
            is_input: True for input, False for output
            
        Returns:
            Resolved QONNX DataType
        """
        dtype_attr = schema.datatype_attr
        if not dtype_attr:
            prefix = "input" if is_input else "output"
            dtype_attr = f"{prefix}{position}Datatype"
        
        return DataType[self.get_nodeattr(dtype_attr)]
    
    def _create_input_model(
        self, 
        position: int, 
        model: ModelWrapper
    ) -> Optional[InputModel]:
        """Create input model from schema and ONNX info."""
        
        schema = self.kernel_schema.inputs[position]

        # Check if input exists
        if position >= len(self.onnx_node.input):
            if schema.optional:
                return None
            raise ValueError(
                f"Required input '{schema.name}' missing at position {position}"
            )
        
        tensor_name = self.onnx_node.input[position]
        if not tensor_name and schema.optional:
            return None
        
        # Get tensor info from ONNX
        tensor_shape = model.get_tensor_shape(tensor_name)
        
        # Resolve datatype
        datatype = self._resolve_interface_datatype(schema, position, is_input=True)
        # TODO: Add datatype constraint validation when implemented in schema
        
        # Resolve dimensions from templates
        block_dims = self._resolve_tiling(schema.block_tiling, tensor_shape)
        stream_dims = self._resolve_tiling(schema.stream_tiling, block_dims)
        
        # Create model directly
        return InputModel(
            name=schema.name,
            tensor_dims=tuple(tensor_shape),
            block_dims=tuple(block_dims),
            datatype=datatype,
            stream_dims=tuple(stream_dims),
            is_weight=schema.is_weight
        )
    
    def _create_output_model(
        self,
        position: int,
        model: ModelWrapper
    ) -> OutputModel:
        """Create output model from schema and ONNX info."""
        
        schema = self.kernel_schema.outputs[position]
        tensor_name = self.onnx_node.output[position]
        
        # Get tensor info from ONNX
        tensor_shape = model.get_tensor_shape(tensor_name)
        
        # Resolve datatype
        datatype = self._resolve_interface_datatype(schema, position, is_input=False)
        
        # Resolve block dimensions
        block_dims = self._resolve_tiling(schema.block_tiling or [":"], tensor_shape)
        
        # Create model directly
        return OutputModel(
            name=schema.name,
            tensor_dims=tuple(tensor_shape),
            block_dims=tuple(block_dims),
            datatype=datatype,
            streaming_rate=1  # Will be computed by kernel model
        )
    
    def _resolve_tiling(
        self,
        template: List[Union[int, str]],
        shape: Sequence[int]
    ) -> Tuple[int, ...]:
        """Resolve tiling template to concrete dimensions.
        
        Handles both nodeattr resolution and tiling application in one method.
        
        Args:
            template: List containing literals, ":", and nodeattr names
            shape: Tensor shape to tile against
            
        Returns:
            Tuple of concrete dimensions
            
        Raises:
            ValueError: If validation fails or nodeattrs not found
        """
        if not template:
            raise ValueError("Tiling template cannot be empty")
        
        if not shape:
            raise ValueError("Shape cannot be empty")
        
        # Stage 1: Resolve nodeattr references to integers
        resolved_template = []
        for item in template:
            if isinstance(item, str) and item != ":":
                # This is a nodeattr reference
                value = self.get_nodeattr(item)
                if value is None:
                    raise ValueError(f"Nodeattr '{item}' not found")
                # Handle FINN's list encoding
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                resolved_template.append(int(value))
            else:
                # Keep literals and ":" as-is
                resolved_template.append(item)
        
        # Stage 2: Validate and apply tiling
        # Check dimension compatibility
        if len(resolved_template) > len(shape):
            raise ValueError(
                f"Template has {len(resolved_template)} dimensions "
                f"but shape has only {len(shape)}"
            )
        
        # Left-pad result with 1s if shape has more dims
        padding = len(shape) - len(resolved_template)
        result = [1] * padding
        
        # Process each template item aligned to the right of shape
        for i, (item, dim_size) in enumerate(zip(resolved_template, shape[padding:])):
            actual_idx = padding + i
            
            if item == ":":
                # Full dimension
                result.append(dim_size)
                
            elif isinstance(item, int):
                if item <= 0:
                    raise ValueError(
                        f"Dimension {actual_idx}: Value must be positive, got {item}"
                    )
                elif dim_size % item != 0:
                    raise ValueError(
                        f"Dimension {actual_idx}: {item} does not evenly divide {dim_size}"
                    )
                else:
                    result.append(item)
            else:
                # This should never happen after nodeattr resolution
                raise ValueError(
                    f"Invalid template item at {i}: {item} (type: {type(item).__name__})"
                )
        
        # Stage 3: Final validation - ensure result tiles evenly into shape
        final_result = tuple(result)
        for i, (tile, shape_dim) in enumerate(zip(final_result, shape)):
            if shape_dim % tile != 0:
                raise ValueError(
                    f"Dimension {i}: Tiling value {tile} does not evenly divide "
                    f"shape dimension {shape_dim}"
                )
        
        return final_result
    
    def _extract_parameters(self) -> Dict[str, Any]:
        """Extract parameter values from nodeattrs."""
        params = {}
        
        # Common kernel parameters
        common_params = [
            "CHANNELS", "PE", "SIMD", "K", "S", 
            "BATCH", "GROUPS", "DEPTHWISE"
        ]
        
        for param in common_params:
            if self.hasNodeAttr(param):
                value = self.get_nodeattr(param)
                # Unwrap single-element lists
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                params[param] = value
        
        return params
    
    # Override HWCustomOp methods to use cached models
    
    def get_folded_output_shape(self, ind=0):
        """Get folded output shape using cached model."""
        model = self.get_kernel_model()
        output = model.outputs[ind]
        
        # Create folded shape
        folded = []
        for tensor_dim, block_dim in zip(output.tensor_dims, output.block_dims):
            if block_dim < tensor_dim:
                # Dimension is folded
                folded.extend([tensor_dim // block_dim, block_dim])
            else:
                folded.append(tensor_dim)
        
        return folded
    
    def get_number_output_values(self):
        """Get total output values using cached model."""
        model = self.get_kernel_model()
        total = 0
        for output in model.outputs:
            total += prod(output.tensor_dims)
        return total
    
    def get_exp_cycles(self):
        """Get expected cycles using cached model."""
        model = self.get_kernel_model()
        return model.initiation_interval
    
    def get_input_datatype(self, ind=0) -> DataType:
        """Get input datatype from cached model."""
        return self.get_kernel_model().inputs[ind].datatype
    
    def get_output_datatype(self, ind=0) -> DataType:
        """Get output datatype from cached model."""
        return self.get_kernel_model().outputs[ind].datatype
    
    def get_normal_input_shape(self, ind=0) -> List[int]:
        """Get normal input shape from cached model."""
        return list(self.get_kernel_model().inputs[ind].tensor_dims)
    
    def get_normal_output_shape(self, ind=0) -> List[int]:
        """Get normal output shape from cached model."""
        return list(self.get_kernel_model().outputs[ind].tensor_dims)
    
    def get_folded_input_shape(self, ind=0) -> List[int]:
        """Get folded input shape using cached model."""
        model = self.get_kernel_model()
        inp = model.inputs[ind]
        
        # Create folded shape
        folded = []
        for tensor_dim, block_dim in zip(inp.tensor_dims, inp.block_dims):
            if block_dim < tensor_dim:
                # Dimension is folded
                folded.extend([tensor_dim // block_dim, block_dim])
            else:
                folded.append(tensor_dim)
        
        return folded
    
    def get_instream_width(self, ind=0) -> int:
        """Get input stream width in bits."""
        inp = self.get_kernel_model().inputs[ind]
        return inp.streaming_bandwidth * inp.datatype.bitwidth()
    
    def get_outstream_width(self, ind=0) -> int:
        """Get output stream width in bits."""
        out = self.get_kernel_model().outputs[ind]
        return out.streaming_rate * out.datatype.bitwidth()
    
    def get_template_values(self):
        """Get template values from cached model."""
        model = self.get_kernel_model()
        
        templates = {}
        
        # Input parameters
        for inp in model.inputs:
            templates[f"{inp.name}_block_dims"] = inp.block_dims
            templates[f"{inp.name}_stream_dims"] = inp.stream_dims
        
        # Output parameters  
        for out in model.outputs:
            templates[f"{out.name}_block_dims"] = out.block_dims
        
        # Kernel parameters
        templates.update(model.parameters)
        
        return templates
    
    def get_sdim_parameters(self):
        """Get SDIM configuration options."""
        # Build from scratch each time
        params = {}
        
        for i, inp_schema in enumerate(self.kernel_schema.inputs):
            if inp_schema.stream_tiling:
                # This input supports streaming configuration
                params[inp_schema.name] = {
                    "dimensions": len(inp_schema.stream_tiling),
                    "template": inp_schema.stream_tiling
                }
        
        return params
    
    def calculate_performance_for_sdim(
        self,
        sdim_config: Dict[str, Union[int, List[int]]]
    ) -> Dict[str, Any]:
        """Calculate performance metrics for specific SDIM config.
        
        Args:
            sdim_config: Maps input names to stream dimensions
            
        Returns:
            Performance metrics dictionary
        """
        # Get base model
        base_model = self.get_kernel_model()
        
        # Apply SDIM configuration
        configured_model = update_kernel_stream_config(base_model, sdim_config)
        
        # Return metrics
        return configured_model.calculate_metrics()
    
    def get_input_bandwidth_mbps(self, sdim_config=None):
        """Get total input bandwidth."""
        if sdim_config:
            model = update_kernel_stream_config(
                self.get_kernel_model(),
                sdim_config
            )
        else:
            model = self.get_kernel_model()
        
        total_bits = sum(inp.bandwidth_bits for inp in model.inputs)
        return total_bits * model.clock_freq_mhz / 8.0
    
    def get_output_bandwidth_mbps(self):
        """Get total output bandwidth."""
        model = self.get_kernel_model()
        total_bits = sum(out.bandwidth_bits for out in model.outputs)
        return total_bits * model.clock_freq_mhz / 8.0


def prod(shape):
    """Product of shape elements."""
    result = 1
    for x in shape:
        result *= x
    return result


def extract_tiling_parameters(template: List[Union[int, str]]) -> Set[str]:
    """Extract parameter names from a tiling template.
    
    Args:
        template: Tiling template with literals, ":", and parameter names
        
    Returns:
        Set of parameter names (excluding ":")
        
    Example:
        >>> extract_tiling_parameters([1, "PE", ":", "SIMD"])
        {"PE", "SIMD"}
    """
    return {item for item in template if isinstance(item, str) and item != ":"}