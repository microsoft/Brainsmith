############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Direct factory for kernel model creation.

Eliminates intermediate "contextualized" or "bound" schemas, creating models
directly from KernelSchema + TensorContext + NodeAttrs.

This approach provides:
- Simpler mental model with fewer concepts
- Direct path from inputs to model
- All validation happens inside factory
- Clean error messages at the right time
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .schemas import KernelSchema, InputSchema, OutputSchema
from .models import KernelModel, InputModel, OutputModel, create_kernel_model
from .tensor_context import TensorContext, TensorInfo
from .types import Shape
from .template_utils import resolve_template_params
from .shape_utils import create_folded_shape
from .validation import ValidationResult, ConstraintViolation
from .constraint_types import validate_datatype_against_constraints
from .relationships import RelationType

from qonnx.core.datatype import DataType


class DirectKernelFactory:
    """Direct factory: KernelSchema + TensorContext + NodeAttrs → KernelModel"""
    
    @staticmethod
    def create_model(
        schema: KernelSchema,
        tensor_context: TensorContext,
        nodeattrs: Dict[str, Any]
    ) -> KernelModel:
        """
        Create a KernelModel directly from inputs without intermediate objects.
        
        Args:
            schema: The kernel schema defining structure and constraints
            tensor_context: Actual tensor information from ONNX graph
            nodeattrs: Node attributes for parameterization
            
        Returns:
            KernelModel: Fully validated and ready-to-use model
            
        Raises:
            ValueError: If validation fails at any stage
        """
        # Step 1: Validate tensor context satisfies schema constraints
        DirectKernelFactory._validate_tensor_context(schema, tensor_context)
        
        # Step 2: Build input models with validation
        input_models = DirectKernelFactory._create_input_models(
            schema, tensor_context, nodeattrs
        )
        
        # Step 3: Build output models with validation
        output_models = DirectKernelFactory._create_output_models(
            schema, tensor_context, nodeattrs
        )
        
        # Step 4: Extract performance attributes
        clock_freq = nodeattrs.get("clock_freq_mhz", 100)  # Default 100 MHz
        
        # Step 5: Build and validate final model
        model = create_kernel_model(
            name=schema.name,
            inputs=input_models,
            outputs=output_models,
            parameters=nodeattrs,
            clock_freq_mhz=clock_freq
        )
        
        # Step 6: Validate relationships and streaming
        DirectKernelFactory._validate_model_relationships(model, schema)
        DirectKernelFactory._validate_streaming_config(model)
        
        return model
    
    @staticmethod
    def _validate_tensor_context(schema: KernelSchema, context: TensorContext) -> None:
        """Validate that tensor context satisfies schema constraints."""
        # Build name lookup for context tensors
        context_inputs = {t.name: t for t in context.inputs}
        context_outputs = {t.name: t for t in context.outputs}
        
        # Check required inputs exist
        for inp in schema.inputs:
            if not inp.optional and inp.name not in context_inputs:
                raise ValueError(f"Required input '{inp.name}' not found in tensor context")
        
        # Check all outputs exist
        for out in schema.outputs:
            if out.name not in context_outputs:
                raise ValueError(f"Output '{out.name}' not found in tensor context")
        
        # Validate datatype constraints
        for inp in schema.inputs:
            if inp.name in context_inputs:
                tensor_info = context_inputs[inp.name]
                if inp.datatype_constraints and tensor_info.datatype:
                    if not validate_datatype_against_constraints(
                        tensor_info.datatype, inp.datatype_constraints
                    ):
                        raise ValueError(
                            f"Input '{inp.name}' datatype {tensor_info.datatype.name} "
                            f"violates schema constraints"
                        )
    
    @staticmethod
    def _create_input_models(
        schema: KernelSchema, 
        context: TensorContext, 
        nodeattrs: Dict[str, Any]
    ) -> List[InputModel]:
        """Create input models with full validation."""
        models = []
        
        # Build name lookup for context tensors
        context_inputs = {t.name: t for t in context.inputs}
        
        for inp_schema in schema.inputs:
            # Skip optional inputs not present
            if inp_schema.optional and inp_schema.name not in context_inputs:
                continue
            
            tensor_info = context_inputs[inp_schema.name]
            
            # Resolve block dimensions
            block_dims = DirectKernelFactory._resolve_dimensions(
                inp_schema.block_tiling, nodeattrs, tensor_info.shape, 
                f"input '{inp_schema.name}' block"
            )
            
            # Resolve stream dimensions (if specified)
            stream_dims = None
            if inp_schema.stream_tiling:
                stream_dims = DirectKernelFactory._resolve_dimensions(
                    inp_schema.stream_tiling, nodeattrs, block_dims,
                    f"input '{inp_schema.name}' stream"
                )
            
            # Validate streaming divides block
            DirectKernelFactory._validate_streaming_divides_block(
                stream_dims, block_dims, f"input '{inp_schema.name}'"
            )
            
            # Determine datatype (nodeattr can override if allowed)
            datatype = DirectKernelFactory._resolve_datatype(
                inp_schema, tensor_info, nodeattrs
            )
            
            # Create shape objects (convert lists to tuples)
            tensor_shape = tuple(tensor_info.shape)
            block_shape = tuple(block_dims)
            stream_shape = tuple(stream_dims) if stream_dims else None
            
            models.append(InputModel(
                name=inp_schema.name,
                tensor_dims=tensor_shape,
                block_dims=block_shape,
                stream_dims=stream_shape,
                datatype=datatype,
                is_weight=inp_schema.is_weight
            ))
        
        return models
    
    @staticmethod
    def _create_output_models(
        schema: KernelSchema,
        context: TensorContext,
        nodeattrs: Dict[str, Any]
    ) -> List[OutputModel]:
        """Create output models with full validation."""
        models = []
        
        # Build name lookup for context tensors
        context_outputs = {t.name: t for t in context.outputs}
        
        for out_schema in schema.outputs:
            tensor_info = context_outputs[out_schema.name]
            
            # Resolve block dimensions
            block_dims = DirectKernelFactory._resolve_dimensions(
                out_schema.block_tiling, nodeattrs, tensor_info.shape,
                f"output '{out_schema.name}' block"
            )
            
            # Outputs don't have stream tiling - they output at block rate
            stream_dims = None
            
            # Validate streaming divides block
            DirectKernelFactory._validate_streaming_divides_block(
                stream_dims, block_dims, f"output '{out_schema.name}'"
            )
            
            # Determine datatype
            datatype = DirectKernelFactory._resolve_datatype(
                out_schema, tensor_info, nodeattrs
            )
            
            # Create shape objects (convert lists to tuples)
            tensor_shape = tuple(tensor_info.shape)
            block_shape = tuple(block_dims)
            stream_shape = tuple(stream_dims) if stream_dims else None
            
            # Calculate streaming rate based on block dimensions
            # For outputs, streaming rate is the product of block dimensions
            streaming_rate = 1
            for dim in block_dims:
                streaming_rate *= dim
            
            models.append(OutputModel(
                name=out_schema.name,
                tensor_dims=tensor_shape,
                block_dims=block_shape,
                datatype=datatype,
                streaming_rate=streaming_rate
            ))
        
        return models
    
    @staticmethod
    def _resolve_dimensions(
        template: List[Any],
        nodeattrs: Dict[str, Any],
        reference_shape: List[int],
        context_name: str
    ) -> List[int]:
        """Resolve template dimensions to concrete values."""
        # For streaming, template can be shorter than reference shape
        # Pad with 1s at the beginning
        if len(template) < len(reference_shape):
            padding = len(reference_shape) - len(template)
            template = [1] * padding + template
        elif len(template) > len(reference_shape):
            raise ValueError(
                f"{context_name}: template length {len(template)} exceeds "
                f"tensor rank {len(reference_shape)}"
            )
        
        # Create param getter function for resolve_template_params
        def param_getter(name: str):
            if name not in nodeattrs:
                raise ValueError(f"Missing required parameter '{name}' for {context_name}")
            return nodeattrs[name]
        
        # First resolve parameters
        resolved_template = resolve_template_params(template, param_getter)
        
        # Then process each dimension
        resolved = []
        for i, (tmpl_resolved, tmpl_orig, ref) in enumerate(zip(resolved_template, template, reference_shape)):
            if tmpl_orig == ":":  # Full dimension
                value = ref
            elif isinstance(tmpl_resolved, int):
                value = tmpl_resolved
            else:
                # This should have been resolved
                raise ValueError(f"{context_name}[{i}]: unresolved template value {tmpl_resolved}")
            
            # Validate against reference
            if tmpl_orig == ":" and value != ref:
                raise ValueError(
                    f"{context_name}[{i}]: expected full dimension {ref}, got {value}"
                )
            elif value > ref:
                raise ValueError(
                    f"{context_name}[{i}]: resolved value {value} exceeds tensor dimension {ref}"
                )
            
            resolved.append(value)
        
        return resolved
    
    @staticmethod
    def _validate_streaming_divides_block(
        stream_dims: Optional[List[int]],
        block_dims: List[int],
        context_name: str
    ) -> None:
        """Validate streaming dimensions divide block dimensions."""
        if not stream_dims:
            return
            
        for i, (stream, block) in enumerate(zip(stream_dims, block_dims)):
            if block % stream != 0:
                raise ValueError(
                    f"{context_name}: stream[{i}]={stream} doesn't divide "
                    f"block[{i}]={block} evenly"
                )
    
    @staticmethod
    def _resolve_datatype(
        interface_schema: Any,
        tensor_info: TensorInfo,
        nodeattrs: Dict[str, Any]
    ) -> DataType:
        """
        Resolve datatype with validation.
        
        Priority:
        1. NodeAttr override (if allowed by constraints)
        2. Tensor context datatype
        3. Error if neither available
        """
        # Check for nodeattr override
        attr_name = f"{interface_schema.name}Datatype"
        if attr_name in nodeattrs:
            override_str = nodeattrs[attr_name]
            try:
                override_dt = DataType[override_str]
                # Validate against constraints
                if interface_schema.datatype_constraints:
                    if not validate_datatype_against_constraints(
                        override_dt, interface_schema.datatype_constraints
                    ):
                        raise ValueError(
                            f"NodeAttr datatype {override_str} violates constraints "
                            f"for {interface_schema.name}"
                        )
                return override_dt
            except KeyError:
                raise ValueError(f"Invalid datatype string: {override_str}")
        
        # Use tensor context datatype
        if tensor_info.datatype:
            return tensor_info.datatype
        
        # No datatype available
        raise ValueError(
            f"No datatype available for {interface_schema.name}. "
            f"Either tensor must have datatype or provide {attr_name} nodeattr"
        )
    
    @staticmethod
    def _validate_model_relationships(model: KernelModel, schema: KernelSchema) -> None:
        """Validate dimension relationships in final model."""
        # Build interface lookup
        interfaces = {}
        for inp in model.inputs:
            interfaces[inp.name] = inp.tensor_dims
        for out in model.outputs:
            interfaces[out.name] = out.tensor_dims
        
        # Check each relationship
        for rel in schema.relationships:
            source_shape = interfaces.get(rel.source_interface)
            target_shape = interfaces.get(rel.target_interface)
            
            if not source_shape or not target_shape:
                continue  # Already validated in earlier steps
            
            # Handle None dimensions (means total size)
            if rel.source_dim is None:
                source_val = 1
                for dim in source_shape:
                    source_val *= dim
            else:
                source_val = source_shape[rel.source_dim]
            
            if rel.target_dim is None:
                target_val = 1
                for dim in target_shape:
                    target_val *= dim
            else:
                target_val = target_shape[rel.target_dim]
            
            # Evaluate relationship
            valid = False
            if rel.relation == RelationType.EQUAL:
                valid = source_val == target_val
            elif rel.relation == RelationType.MULTIPLE:
                valid = source_val == rel.factor * target_val
            elif rel.relation == RelationType.DEPENDENT:
                # For DEPENDENT, we just validate it exists
                valid = True
                
            if not valid:
                raise ValueError(
                    f"Relationship {rel.relation.name} violated: "
                    f"{rel.source_interface}[{rel.source_dim}]={source_val} "
                    f"vs {rel.target_interface}[{rel.target_dim}]={target_val}"
                )
    
    @staticmethod
    def _validate_streaming_config(model: KernelModel) -> None:
        """Validate overall streaming configuration."""
        # Check bandwidth constraints
        for inp in model.inputs:
            if hasattr(inp, 'bandwidth_bits') and inp.bandwidth_bits > 1024:
                # Warning only - don't fail
                print(f"Warning: Input {inp.name} bandwidth {inp.bandwidth_bits} bits "
                      f"exceeds typical 1024 bit limit")