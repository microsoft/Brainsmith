############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Base class for automatic hardware custom operators.

Creates fresh kernel models on every access to ensure consistency
with current nodeattr values.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union, Sequence, Set

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType
from qonnx.core import ModelWrapper

from brainsmith.core.dataflow import KernelSchema
from brainsmith.core.dataflow.models import (
    KernelModel,
    InputModel,
    OutputModel,
    create_input_model,
    create_output_model,
    create_kernel_model,
    update_kernel_stream_config
)


class AutoHWCustomOp(HWCustomOp, ABC):
    """Base class for automatic hardware custom operators.
    
    Key features:
    - Creates fresh kernel models on every access
    - All model access through create_kernel_model()
    - Ensures consistency with current nodeattr values
    """
    
    # Subclasses must define this class attribute
    kernel_schema: KernelSchema = None
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize without creating any models."""
        super().__init__(onnx_node, **kwargs)
        
        if self.kernel_schema is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define kernel_schema class attribute"
            )
        
        self._kernel_model_state: Optional[KernelModel] = None

    def get_df_model(self) -> KernelModel:
        """Get the dataflow model for this operator."""
        if self._kernel_model_state is None:
            self._kernel_model_state = self._create_kernel_model()
        return self._kernel_model_state

    def _create_kernel_model(self, model: ModelWrapper) -> KernelModel:
        """Create fresh kernel model from current nodeattrs.
        
        This is the primary method for getting a kernel model. Always
        creates a new instance from current state.
        
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
        
        # TAFK TODO: Extract relationships
        
        # Create kernel model
        return create_kernel_model(
            name=self.kernel_schema.name,
            inputs=input_models,
            outputs=output_models,
            clock_freq_mhz=self.get_nodeattr("clock_freq_mhz", 100.0) # Need to remove # TAFK TODO
        )
    
    def _create_input_model(
        self, 
        ind: int, 
        model: ModelWrapper
    ) -> Optional[InputModel]:

        schema = self.kernel_schema.inputs[ind]

        # Check if input exists
        if ind >= len(self.onnx_node.input):
            if schema.optional:
                return None
            raise ValueError(
                f"Required input '{schema.name}' missing at ind {ind}"
            )
        
        tensor_name = self.onnx_node.input[ind]
        if not tensor_name and schema.optional:
            return None
        
        # Get tensor info from ONNX
        tensor_shape = model.get_tensor_shape(tensor_name)
        
        # Resolve datatype
        datatype_attr = schema.datatype_attr if schema.datatype_attr else f"input{ind}Datatype"
        datatype = self.get_node_attr(datatype_attr)
        if not schema.validate_datatype(datatype): # TAFK TODO: Make validate_datatype real
            raise ValueError(
                f"Input '{schema.name}' datatype {datatype} does not satisfy schema constraints"
            )

        # Resolve dimensions from templates
        block_dims = self._resolve_tiling(schema.block_tiling, tensor_shape)
        stream_dims = self._resolve_tiling(schema.stream_tiling, block_dims)
        
        return create_input_model(
            name=schema.name,
            tensor_dims=tensor_shape,
            block_dims=block_dims,
            datatype=datatype,
            stream_dims=stream_dims,
            is_weight=schema.is_weight
        )
    
    def _create_output_model(
        self,
        ind: int,
        model: ModelWrapper
    ) -> OutputModel:
        """Create output model from schema and ind."""
        
        schema = self.kernel_schema.outputs[ind]
        tensor_name = self.onnx_node.output[ind]
        
        # Get tensor info from ONNX
        tensor_shape = model.get_tensor_shape(tensor_name)
        
        # Resolve datatype
        dtype_attr = schema.datatype_attr if schema.datatype_attr else f"input{ind}Datatype"
        datatype = DataType[self.get_nodeattr(dtype_attr)]
        
        # Resolve block dimensions
        block_dims = self._resolve_tiling(schema.block_tiling or [":"], tensor_shape)
        
        return create_output_model(
            name=schema.name,
            tensor_dims=tensor_shape,
            block_dims=block_dims,
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
                ) # TAFK TODO: Make more rich, with node name etc.
        
        return final_result
    
    # Automate functions with Kernel Model
    
    # FINN Abstract Method Implementations
    
    def get_input_datatype(self, ind=0) -> DataType:
        return self.get_df_model().inputs[ind].datatype
    
    def get_output_datatype(self, ind=0) -> DataType:
        return self.get_df_model().outputs[ind].datatype
    
    def get_normal_input_shape(self, ind=0) -> List[int]:
        return self.get_df_model().inputs[ind].tensor_dims
    
    def get_normal_output_shape(self, ind=0) -> List[int]:
        return self.get_df_model().outputs[ind].tensor_dims

    def get_folded_input_shape(self, ind=0) -> List[int]:
        return self.get_df_model().inputs[ind].stream_dims
    
    def get_folded_output_shape(self, ind=0) -> List[int]:
        return self.get_df_model().outputs[ind].stream_dims

    def get_instream_width(self, ind=0) -> int:
        return self.get_df_model().inputs[ind].stream_width

    def get_outstream_width(self, ind=0) -> int:
        return self.get_df_model().outputs[ind].stream_width
    
    def get_exp_cycles(self):
        return self.get_df_model().initiation_interval
