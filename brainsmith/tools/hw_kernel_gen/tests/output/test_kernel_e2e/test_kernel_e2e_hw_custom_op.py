############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Auto-generated HWCustomOp for test_kernel_e2e
# Generated from: /home/tafk/dev/brainsmith-2/brainsmith/tools/hw_kernel_gen/tests/test_kernel_e2e.sv
# Generation timestamp: 2025-06-25T13:56:08.949120
#
# PHASE 2: FINN INTEGRATION
# This HWCustomOp follows FINN's standard pattern with simple constructor
# and static interface metadata. Parameters are accessed via get_nodeattr().
############################################################################

from typing import List, Dict, Tuple, Any
import numpy as np
from qonnx.core.datatype import DataType

from brainsmith.dataflow.core import AutoHWCustomOp
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.qonnx_types import DatatypeConstraintGroup
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.block_chunking import BlockChunkingStrategy


class TestKernelE2e(AutoHWCustomOp):
    """
    Auto-generated HWCustomOp for test_kernel_e2e kernel.
    
    Generated from RTL: /home/tafk/dev/brainsmith-2/brainsmith/tools/hw_kernel_gen/tests/test_kernel_e2e.sv
    Follows FINN's standard HWCustomOp pattern with static interface metadata.
    
    RTL parameters are defined in get_nodeattr_types().
    """
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize TestKernelE2e following FINN's standard pattern."""
        super().__init__(onnx_node, **kwargs)
        
        # Set kernel-specific attributes
        self.kernel_name = "test_kernel_e2e"
        self.rtl_source = "/home/tafk/dev/brainsmith-2/brainsmith/tools/hw_kernel_gen/tests/test_kernel_e2e.sv"
    
    @staticmethod
    def get_interface_metadata() -> List[InterfaceMetadata]:
        """
        Return static interface metadata for dataflow interfaces (AXI-Stream only).
        
        Control and configuration interfaces are handled separately and not included here.
        All BDIM parameters have been validated during template generation
        to ensure they reference valid module parameters.
        """
        return [
            InterfaceMetadata(
                name="s_axis_input",
                interface_type=InterfaceType.INPUT,
                datatype_constraints=[
                    DatatypeConstraintGroup(
                        base_type="UINT",
                        min_width=8,
                        max_width=32
                    ),
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=[':', ':'],  # Validated symbolic shape
                    rindex=0
                )
            ),
            InterfaceMetadata(
                name="s_axis_weights",
                interface_type=InterfaceType.WEIGHT,
                datatype_constraints=[
                    DatatypeConstraintGroup(
                        base_type="FIXED",
                        min_width=8,
                        max_width=16
                    ),
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=['PE'],  # Validated symbolic shape
                    rindex=0
                )
            ),
            InterfaceMetadata(
                name="m_axis_output",
                interface_type=InterfaceType.OUTPUT,
                datatype_constraints=[
                    DatatypeConstraintGroup(
                        base_type="UINT",
                        min_width=8,
                        max_width=32
                    ),
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=[':', ':'],  # Validated symbolic shape
                    rindex=0
                )
            ),
        ]
    
    def get_nodeattr_types(self) -> Dict[str, Tuple[str, bool, Any]]:
        """
        Define ONNX node attributes for all RTL parameters.
        
        Parameters with whitelisted defaults are optional, all others are required.
        """
        # Start with parent class attributes
        my_attrs = {}
        
        # RTL parameters as node attributes (only exposed parameters)
        my_attrs["INPUT_BDIM"] = ("i", True, None)  # Required parameter
        my_attrs["WEIGHT_BDIM"] = ("i", True, None)  # Required parameter
        my_attrs["ACTIVATION_TYPE"] = ("i", True, None)  # Required parameter
        
        # Interface datatype attributes (high-level datatype specification)
        my_attrs["input0DataType"] = ('s', False, 'INT8')  # input interface datatype
        my_attrs["weight0DataType"] = ('s', False, 'INT8')  # weight interface datatype
        my_attrs["output0DataType"] = ('s', False, 'INT8')  # output interface datatype
        
        # Internal datatype attributes (e.g., thresholdDataType, accumulatorDataType)
        my_attrs["accumulatorDataType"] = ("s", False, "UINT8")  # Internal accumulator datatype
        my_attrs["thresholdDataType"] = ("s", False, "UINT8")  # Internal threshold datatype
        my_attrs["OUTPUTDataType"] = ("s", False, "UINT8")  # Internal OUTPUT datatype
        
        # Hardware-specific attributes from RTL analysis
        my_attrs["runtime_writeable_weights"] = ('i', False, 0, {0, 1})
        my_attrs["numInputVectors"] = ('ints', False, [1])
        
        # Base HWCustomOp attributes
        my_attrs.update({
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
            "numInputVectors": ("ints", False, [1]),
        })
        
        # Update with parent class attributes (FINN base classes)
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs
    
    # ===== Automatic Tensor Formatting Methods =====
    # These methods are automatically generated by the enhanced AutoHWCustomOp
    # using dataflow mathematics, eliminating the need for manual implementation
    
    
    
    
    # Note: Legacy methods like calc_simd(), calc_pe(), etc. are automatically
    # derived from the dataflow model parallelism configuration
    
    # Note: Datatype methods handled by AutoHWCustomOp parent class
    # Parent class validates datatypes against constraint groups at runtime
    # Note: Shape calculation methods handled by AutoHWCustomOp parent class
    # Parent class computes shapes from DataflowModel interfaces automatically
    
    # Note: Stream width methods handled by AutoHWCustomOp parent class
    # Parent class calculates stream widths from datatypes and parallelism automatically
    
    # ===== Resource Estimation Methods =====
    
    # Note: Cycle calculation and memory handling done by AutoHWCustomOp parent class
    
    def bram_estimation(self) -> int:
        """Estimate BRAM usage for test_kernel_e2e."""
        return 1
    
    def lut_estimation(self) -> int:
        """Estimate LUT usage for test_kernel_e2e."""
        return 2000
    
    def dsp_estimation(self) -> int:
        """Estimate DSP usage for test_kernel_e2e."""
        return 0
    
    def verify_node(self):
        """Verify kernel-specific constraints."""
        super().verify_node()
        
        # Verify all required parameters are present
        if self.get_nodeattr("INPUT_WIDTH") is None:
            raise ValueError(f"Required parameter 'INPUT_WIDTH' not specified")
        if self.get_nodeattr("WEIGHT_WIDTH") is None:
            raise ValueError(f"Required parameter 'WEIGHT_WIDTH' not specified")
        if self.get_nodeattr("WEIGHT_SIGNED") is None:
            raise ValueError(f"Required parameter 'WEIGHT_SIGNED' not specified")
        if self.get_nodeattr("OUTPUT_WIDTH") is None:
            raise ValueError(f"Required parameter 'OUTPUT_WIDTH' not specified")
        if self.get_nodeattr("ACC_WIDTH") is None:
            raise ValueError(f"Required parameter 'ACC_WIDTH' not specified")
        if self.get_nodeattr("ACC_SIGNED") is None:
            raise ValueError(f"Required parameter 'ACC_SIGNED' not specified")
        if self.get_nodeattr("THRESH_WIDTH") is None:
            raise ValueError(f"Required parameter 'THRESH_WIDTH' not specified")
        if self.get_nodeattr("INPUT_BDIM") is None:
            raise ValueError(f"Required parameter 'INPUT_BDIM' not specified")
        if self.get_nodeattr("INPUT_SDIM") is None:
            raise ValueError(f"Required parameter 'INPUT_SDIM' not specified")
        if self.get_nodeattr("WEIGHT_BDIM") is None:
            raise ValueError(f"Required parameter 'WEIGHT_BDIM' not specified")
        if self.get_nodeattr("ACTIVATION_TYPE") is None:
            raise ValueError(f"Required parameter 'ACTIVATION_TYPE' not specified")
        
        # Additional test_kernel_e2e-specific verification
        # TODO: Add kernel-specific constraint checks


# Convenience function for FINN integration
def make_test_kernel_e2e_node(inputs, outputs, **node_attrs):
    """
    Create TestKernelE2e ONNX node.
    
    Required algorithm parameters:
    - INPUT_BDIM: int
    - WEIGHT_BDIM: int
    - ACTIVATION_TYPE: int
    
    Interface datatype attributes:
    - input0DataType: str = "INT8"  # input interface datatype
    - weight0DataType: str = "INT8"  # weight interface datatype
    - output0DataType: str = "INT8"  # output interface datatype
    
    Note: RTL-level parameters (width, signed, format, etc.) are automatically derived 
    from interface datatypes by the RTLBackend and should not be specified directly.
    
    Optional parameters (with defaults):
    - MEM_DEPTH: int = 1024
    """
    import onnx.helper
    
    # Validate required algorithm parameters (only exposed parameters)
    required_algorithm_params = [p for p in ['INPUT_WIDTH', 'WEIGHT_WIDTH', 'WEIGHT_SIGNED', 'OUTPUT_WIDTH', 'ACC_WIDTH', 'ACC_SIGNED', 'THRESH_WIDTH', 'INPUT_BDIM', 'INPUT_SDIM', 'WEIGHT_BDIM', 'ACTIVATION_TYPE'] if p in ['INPUT_BDIM', 'WEIGHT_BDIM', 'ACTIVATION_TYPE', 'num_engines']]
    missing = [p for p in required_algorithm_params if p not in node_attrs]
    if missing:
        raise ValueError(f"Missing required algorithm parameters: {missing}")
    
    # Note: Interface datatype parameters are handled by RTLBackend, not node attributes
    
    return onnx.helper.make_node(
        "TestKernelE2e",
        inputs=inputs,
        outputs=outputs,
        domain="finn.custom_op.fpgadataflow",
        **node_attrs
    )