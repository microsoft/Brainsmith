############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

import torch
import numpy as np
import torch.nn.functional as F
from qonnx.core.datatype import DataType

from brainsmith.core.dataflow import (
    KernelSchema, InputSchema, OutputSchema,
    DatatypeConstraintGroup, RelationType,
    KernelValidator
)
from brainsmith.core.finn import AutoHWCustomOp
from brainsmith.core.plugins import kernel

@kernel(
    description="Hardware implementation of LayerNorm",
    author="Thomas Keller"
)
class LayerNorm(AutoHWCustomOp):
    """Hardware implementation of LayerNorm layer.
    
    LayerNorm normalizes inputs across the channel dimension with learned
    scale and bias parameters. This implementation supports configurable
    SIMD parallelism for the channel dimension.
    
    Shape assumptions:
    - Input shape: [..., NumChannels] where ... can be any number of batch dimensions
    - Output shape: Same as input shape
    - Normalization happens across the last dimension (channels)
    """

    kernel_schema = KernelSchema(
        name="LayerNorm",
        inputs=[InputSchema(
            name="input",
            datatype_constraints=[
                DatatypeConstraintGroup("FLOAT", 32, 32)  # FLOAT32 only
            ],
            block_tiling=[":", "NumChannels"],  # Full batch dims, tiled channels
            stream_tiling=["SIMD"],
            datatype_attr="inputDataType"
        )],
        outputs=[OutputSchema(
            name="output",
            datatype_constraints=[
                DatatypeConstraintGroup("FLOAT", 32, 32)  # FLOAT32 only
            ],
            block_tiling=[":", "NumChannels"],  # Match input tiling
            datatype_attr="outputDataType"
        )],
        relationships=[
            # Input and output shapes must be identical
            # Since we use ":" for batch dims, we ensure the channel dim matches via NumChannels parameter
        ],
        metadata={
            "description": "Layer normalization across channel dimension",
            "supports_streaming": True
        }
    )

    def get_nodeattr_types(self):
        """Define node attributes for LayerNorm."""
        my_attrs = super().get_nodeattr_types()
        my_attrs.update({
            # Tiling parameters
            "SIMD": ("i", True, 0),
            "NumChannels": ("i", True, 128),
            
            # Shape information (maintained for compatibility)
            "ifm_dim": ("ints", True, []),
            
            # LayerNorm-specific parameters
            "epsilon": ("f", True, 1e-5),
            
            # Datatype attributes (required by schema)
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            
            # Execution mode - supports python simulation
            "exec_mode": ("s", False, "python", {"python", "rtlsim", "cppsim"}),
        })
        return my_attrs

    def execute_node(self, context, graph):
        """Execute LayerNorm in Python simulation mode.
        
        Uses PyTorch implementation for functional verification.
        Note: This assumes weight and bias have been folded/removed.
        """
        node = self.onnx_node
        
        # Get input tensor
        in_values = context[node.input[0]]
        
        # Get shape info
        ishape = in_values.shape
        
        # Execute LayerNorm using PyTorch
        # Normalize over the last dimension (channels)
        in_act = torch.from_numpy(in_values)
        out_act = F.layer_norm(
            in_act, 
            normalized_shape=[ishape[-1]], 
            eps=self.get_nodeattr("epsilon")
        )
        
        # Store output
        context[node.output[0]] = np.asarray(out_act, dtype=np.float32)

    def verify_node(self):
        """Verify node configuration using the validation system.
        
        Validates:
        - SIMD divides evenly into NumChannels
        - Epsilon is positive
        - Model is properly built if available
        """
        # Basic parameter validation
        simd = self.get_nodeattr("SIMD")
        num_channels = self.get_nodeattr("NumChannels")
        epsilon = self.get_nodeattr("epsilon")
        
        if simd <= 0:
            raise ValueError(f"SIMD must be positive, got {simd}")
            
        if num_channels <= 0:
            raise ValueError(f"NumChannels must be positive, got {num_channels}")
            
        if num_channels % simd != 0:
            raise ValueError(
                f"SIMD ({simd}) must divide evenly into NumChannels ({num_channels})"
            )
            
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        
        # If kernel model is built, perform comprehensive validation
        if hasattr(self, '_kernel_model') and self._kernel_model is not None:
            validator = KernelValidator()
            result = validator.validate_model(
                self._kernel_model,
                self.kernel_schema,
                self._tensor_context
            )
            
            if not result.is_valid():
                # Collect error messages
                errors = []
                for violation in result.violations:
                    if violation.severity == "error":
                        errors.append(f"- {violation.message}")
                
                if errors:
                    error_msg = "LayerNorm validation failed:\n" + "\n".join(errors)
                    raise ValueError(error_msg)

    def get_normal_input_shape(self, ind=0):
        """Get normal (unfolded) input shape.
        
        Override needed for backward compatibility with ifm_dim attribute.
        """
        return self.get_nodeattr("ifm_dim")

    def get_normal_output_shape(self, ind=0):
        """Get normal (unfolded) output shape.
        
        For LayerNorm, output shape always matches input shape.
        """
        return self.get_normal_input_shape()

    def make_shape_compatible_op(self, model):
        """Create constant shape operation for shape inference."""
        return super().make_const_shape_op(self.get_normal_input_shape())
    
    # Note: The following methods are now handled by the AutoHWCustomOp base class:
    # - get_folded_input_shape() - Automatically computed from kernel model
    # - get_folded_output_shape() - Automatically computed from kernel model  
    # - get_number_output_values() - Automatically computed from kernel model
    # - get_input_datatype() - Uses DatatypeResolver with proper fallback
    # - get_output_datatype() - Uses DatatypeResolver with proper fallback
    # - get_instream_width() - Calculated from model's streaming bandwidth
    # - get_outstream_width() - Calculated from model's streaming rate
    # - infer_node_datatype() - Handled by transforms with proper validation
    
    # The base class provides automatic caching and invalidation, so we don't
    # need to manually manage any model state. Just ensure refresh_kernel_model()
    # is called by transforms when needed.
