############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Auto-generated HWCustomOp for test_kernel_e2e
# Generated from: /home/tafk/dev/brainsmith-2/brainsmith/tools/hw_kernel_gen/tests/test_kernel_e2e.sv
# Generation timestamp: 2025-06-30T07:36:27.569674
#
# This HWCustomOp uses the modern AutoHWCustomOp base class with explicit
# parameter definitions and no runtime CodegenBinding dependencies.
############################################################################

from typing import List, Dict, Tuple, Any
import numpy as np
from qonnx.core.datatype import DataType

from brainsmith.tools.hw_kernel_gen.auto_hw_custom_op_v2 import AutoHWCustomOp
from brainsmith.core.dataflow import (
    KernelDefinition,
    InputDefinition,
    OutputDefinition,
    RelationType,
    DatatypeConstraintGroup,
    parameterized_tiles,
    fixed_tiles
)


class TestKernelE2e(AutoHWCustomOp):
    """
    Auto-generated HWCustomOp for test_kernel_e2e kernel.
    
    Generated from RTL: /home/tafk/dev/brainsmith-2/brainsmith/tools/hw_kernel_gen/tests/test_kernel_e2e.sv
    Uses AutoHWCustomOp for automatic FINN method implementation.
    """
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize TestKernelE2e with KernelDefinition."""
        kernel_def = self._create_kernel_definition()
        super().__init__(onnx_node, kernel_def, **kwargs)
        
        # Set kernel-specific attributes
        self.kernel_name = "test_kernel_e2e"
        self.rtl_source = "/home/tafk/dev/brainsmith-2/brainsmith/tools/hw_kernel_gen/tests/test_kernel_e2e.sv"
    
    def get_nodeattr_types(self):
        """Explicit node attribute definitions."""
        return {
            "INPUT_BDIM": ('i', True, None),
            "INPUT_SDIM": ('i', True, None),
            "WEIGHT_BDIM": ('i', True, None),
            "WEIGHT_SDIM": ('i', True, None),
            "OUTPUT_BDIM": ('i', True, None),
            "PE": ('i', True, None),
            "ACTIVATION_TYPE": ('i', True, None),
            "s_axis_inputDataType": ('s', False, 'INT8'),
            "s_axis_weightsDataType": ('s', False, 'INT8'),
            "m_axis_outputDataType": ('s', False, 'INT8'),
        }
    
    def _create_kernel_definition(self) -> KernelDefinition:
        """Create simplified KernelDefinition with interface definitions only."""
        kernel_def = KernelDefinition(name="test_kernel_e2e")
        
        # Add input definitions
        input_def = InputDefinition(
            name="",
            datatype_constraints=[
                DatatypeConstraintGroup(
                    base_type="UINT",
                    min_width=8,
                    max_width=32
                ),
            ],
            block_dims_expr=parameterized_tiles("INPUT_BDIM"),
            description="Interface s_axis_input (input) - Direction: input"
        )
        kernel_def.add_input(input_def)
        
        # Add weight input definitions
        weight_def = InputDefinition(
            name="",
            datatype_constraints=[
                DatatypeConstraintGroup(
                    base_type="FIXED",
                    min_width=8,
                    max_width=16
                ),
            ],
            block_dims_expr=parameterized_tiles("WEIGHT_BDIM"),
            is_weight=True,
            description="Interface s_axis_weights (weight) - Direction: input"
        )
        kernel_def.add_input(weight_def)
        
        # Add output definitions
        output_def = OutputDefinition(
            name="",
            datatype_constraints=[
                DatatypeConstraintGroup(
                    base_type="UINT",
                    min_width=8,
                    max_width=32
                ),
            ],
            block_dims_expr=parameterized_tiles("OUTPUT_BDIM"),
            description="Interface m_axis_output (output) - Direction: output"
        )
        kernel_def.add_output(output_def)
        
        # Add relationships
        return kernel_def
    
    def _extract_input_specs(self) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
        """
        Extract input specifications from ONNX context.
        
        Returns:
            Dictionary mapping input names to (shape, datatype) tuples
        """
        specs = {}
        
        #  interface
        input_shape_0 = self.get_normal_input_shape(0)
        input_dtype_0 = self._get_interface_datatype("")
        specs[""] = (tuple(input_shape_0), input_dtype_0)
        
        #  interface
        input_shape_1 = self.get_normal_input_shape(1)
        input_dtype_1 = self._get_interface_datatype("")
        specs[""] = (tuple(input_shape_1), input_dtype_1)
        
        return specs
    
    def _extract_output_specs(self) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
        """
        Extract output specifications from ONNX context.
        
        Returns:
            Dictionary mapping output names to (shape, datatype) tuples
        """
        specs = {}
        
        #  interface
        # Derive output shape from kernel behavior
        output_shape_0 = self.get_normal_output_shape(0)
        output_dtype_0 = self._get_interface_datatype("")
        specs[""] = (tuple(output_shape_0), output_dtype_0)
        
        return specs
    


# Convenience function for FINN integration
def make_test_kernel_e2e_node(inputs, outputs, **node_attrs):
    """
    Create TestKernelE2e ONNX node.
    
    Required parameters:
    - INPUT_WIDTH: int
    - WEIGHT_WIDTH: int
    - WEIGHT_SIGNED: int
    - OUTPUT_WIDTH: int
    - ACC_WIDTH: int
    - ACC_SIGNED: int
    - THRESH_WIDTH: int
    - INPUT_BDIM: int
    - INPUT_SDIM: int
    - WEIGHT_BDIM: int
    - WEIGHT_SDIM: int
    - OUTPUT_BDIM: int
    - ACTIVATION_TYPE: int
    
    Interface datatype attributes:
    
    Optional parameters (with defaults):
    - PE: int = 1
    - MEM_DEPTH: int = 1024
    """
    import onnx.helper
    
    # Validate required parameters
    required = ['INPUT_WIDTH', 'WEIGHT_WIDTH', 'WEIGHT_SIGNED', 'OUTPUT_WIDTH', 'ACC_WIDTH', 'ACC_SIGNED', 'THRESH_WIDTH', 'INPUT_BDIM', 'INPUT_SDIM', 'WEIGHT_BDIM', 'WEIGHT_SDIM', 'OUTPUT_BDIM', 'ACTIVATION_TYPE']
    missing = [p for p in required if p not in node_attrs]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")
    
    return onnx.helper.make_node(
        "TestKernelE2e",
        inputs=inputs,
        outputs=outputs,
        domain="finn.custom_op.fpgadataflow",
        **node_attrs
    )