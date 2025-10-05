############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Factory for creating KernelModels from resolved configs and tensor contexts.

This module implements Phase 2 of the two-phase model creation process,
taking resolved configurations and tensor information to create complete
KernelModel instances.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import math

from qonnx.core.datatype import DataType

from .resolved_config import ResolvedKernelConfig, ResolvedInterfaceConfig
from .tensor_context import TensorContext, TensorInfo
from .models import (
    KernelModel,
    InputModel,
    OutputModel,
    create_kernel_model,
    create_input_model,
    create_output_model
)


class KernelModelFactory:
    """Factory for creating KernelModels from resolved configs and tensor context."""
    
    @staticmethod
    def create_model(
        config: ResolvedKernelConfig,
        context: TensorContext,
        datatype_resolver: Dict[str, DataType]
    ) -> KernelModel:
        """Create complete KernelModel from resolved config and tensor context.
        
        Args:
            config: Resolved configuration from Phase 1
            context: Tensor information from ONNX graph
            datatype_resolver: Maps datatype attribute names to DataType values
            
        Returns:
            Complete KernelModel instance
        """
        # Create input models
        input_models = []
        for i, inp_config in enumerate(config.inputs):
            tensor_info = context.get_input_info(i)
            if tensor_info:
                inp_model = KernelModelFactory._create_input_model(
                    inp_config,
                    tensor_info,
                    config.parameters,
                    datatype_resolver.get(inp_config.datatype_attr)
                )
                if inp_model:  # Skip optional missing
                    input_models.append(inp_model)
            elif not inp_config.optional:
                raise ValueError(
                    f"Required input '{inp_config.name}' at position {i} missing"
                )
        
        # Create output models
        output_models = []
        for i, out_config in enumerate(config.outputs):
            tensor_info = context.get_output_info(i)
            if not tensor_info:
                raise ValueError(
                    f"Output '{out_config.name}' at position {i} missing"
                )
            
            out_model = KernelModelFactory._create_output_model(
                out_config,
                tensor_info,
                config.parameters,
                datatype_resolver.get(out_config.datatype_attr)
            )
            output_models.append(out_model)
        
        # Create kernel model
        return create_kernel_model(
            name=config.kernel_name,
            inputs=input_models,
            outputs=output_models,
            parameters=config.parameters,
            clock_freq_mhz=config.clock_freq_mhz
        )
    
    @staticmethod
    def _create_input_model(
        config: ResolvedInterfaceConfig,
        tensor_info: TensorInfo,
        parameters: Dict[str, Any],
        datatype: Optional[DataType]
    ) -> Optional[InputModel]:
        """Create InputModel from config and tensor info."""
        if not tensor_info:
            return None
        
        # Use provided datatype or fall back to tensor datatype
        final_datatype = datatype or tensor_info.datatype
        
        # Resolve block dimensions
        block_shape = KernelModelFactory._resolve_tiling_params(
            config.block_params,
            tensor_info.shape,
            parameters
        )
        
        # Resolve stream dimensions if present
        stream_shape = block_shape  # Default to full block
        if config.stream_params:
            stream_shape = KernelModelFactory._resolve_tiling_params(
                config.stream_params,
                block_shape,  # Stream tiles against block dims
                parameters
            )
        
        return InputModel(
            name=config.name,
            tensor_shape=tensor_info.shape,
            block_shape=block_shape,
            datatype=final_datatype,
            stream_shape=stream_shape,
            is_weight=config.is_weight
        )
    
    @staticmethod
    def _create_output_model(
        config: ResolvedInterfaceConfig,
        tensor_info: TensorInfo,
        parameters: Dict[str, Any],
        datatype: Optional[DataType]
    ) -> OutputModel:
        """Create OutputModel from config and tensor info."""
        # Use provided datatype or fall back to tensor datatype
        final_datatype = datatype or tensor_info.datatype

        # Resolve block dimensions
        block_shape = KernelModelFactory._resolve_tiling_params(
            config.block_params,
            tensor_info.shape,
            parameters
        )

        return OutputModel(
            name=config.name,
            tensor_shape=tensor_info.shape,
            block_shape=block_shape,
            datatype=final_datatype
        )
    
    @staticmethod
    def _resolve_tiling_params(
        template: List[Union[int, str]],
        shape: Tuple[int, ...],
        parameters: Dict[str, Any]
    ) -> Tuple[int, ...]:
        """Resolve template with parameters to concrete dimensions.
        
        Args:
            template: Tiling template with literals, ":", and parameter names
            shape: Shape to tile against
            parameters: Parameter values to substitute
            
        Returns:
            Tuple of concrete dimensions
            
        Raises:
            ValueError: If validation fails
        """
        if not template:
            raise ValueError("Tiling template cannot be empty")
        
        if not shape:
            raise ValueError("Shape cannot be empty")
        
        # Stage 1: Substitute parameter values
        resolved_template = []
        for item in template:
            if isinstance(item, str) and item != ":":
                # This is a parameter reference
                if item not in parameters:
                    raise ValueError(f"Parameter '{item}' not found in parameters dict")
                value = parameters[item]
                # Handle FINN's list encoding
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                resolved_template.append(int(value))
            else:
                # Keep literals and ":" as-is
                resolved_template.append(item)
        
        # Stage 2: Apply tiling to shape
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
                raise ValueError(
                    f"Invalid template item at {i}: {item} (type: {type(item).__name__})"
                )
        
        # Stage 3: Final validation
        final_result = tuple(result)
        for i, (tile, shape_dim) in enumerate(zip(final_result, shape)):
            if shape_dim % tile != 0:
                raise ValueError(
                    f"Dimension {i}: Tiling value {tile} does not evenly divide "
                    f"shape dimension {shape_dim}"
                )
        
        return final_result