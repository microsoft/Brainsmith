"""
Proof of Concept: AutoHWCustomOp Subclass for Thresholding Operation

This demonstrates what our template should generate from ParsedKernelData.
Shows how to properly use DataflowModel as the heart of the HWCustomOp.
"""

from typing import List, Dict, Any, Tuple
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.block_chunking import DefaultChunkingStrategy
import numpy as np


class ThresholdingAxi(AutoHWCustomOp):
    """
    Auto-generated HWCustomOp for thresholding_axi kernel using DataflowModel.
    
    This demonstrates the clean integration where:
    1. Interface metadata is created from RTL analysis (ParsedKernelData)
    2. DataflowModel handles all shape/stream/resource calculations
    3. We only add kernel-specific attributes and optional overrides
    """
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize with interface metadata extracted from RTL."""
        
        # Create interface metadata from RTL analysis
        # This is what TemplateContextGenerator would produce from ParsedKernelData
        interface_metadata = [
            # Input stream interface from RTL
            InterfaceMetadata(
                name="s_axis_input",
                interface_type=InterfaceType.INPUT,
                allowed_datatypes=[
                    DataTypeConstraint(finn_type="UINT8", bit_width=8, signed=False),
                    DataTypeConstraint(finn_type="UINT16", bit_width=16, signed=False),
                    DataTypeConstraint(finn_type="INT8", bit_width=8, signed=True),
                    DataTypeConstraint(finn_type="INT16", bit_width=16, signed=True),
                ],
                # Chunking strategy - could be from pragma or default
                chunking_strategy=DefaultChunkingStrategy()
            ),
            
            # Threshold weights interface from RTL  
            InterfaceMetadata(
                name="s_axis_weights",
                interface_type=InterfaceType.WEIGHT,
                allowed_datatypes=[
                    DataTypeConstraint(finn_type="UINT8", bit_width=8, signed=False),
                    DataTypeConstraint(finn_type="UINT16", bit_width=16, signed=False),
                    DataTypeConstraint(finn_type="INT8", bit_width=8, signed=True),
                    DataTypeConstraint(finn_type="INT16", bit_width=16, signed=True),
                ],
                chunking_strategy=DefaultChunkingStrategy()
            ),
            
            # Output stream interface from RTL
            InterfaceMetadata(
                name="m_axis_output", 
                interface_type=InterfaceType.OUTPUT,
                allowed_datatypes=[
                    DataTypeConstraint(finn_type="UINT1", bit_width=1, signed=False),
                    DataTypeConstraint(finn_type="UINT2", bit_width=2, signed=False),
                    DataTypeConstraint(finn_type="UINT8", bit_width=8, signed=False),
                ],
                chunking_strategy=DefaultChunkingStrategy()
            ),
        ]
        
        # Initialize parent with interface metadata
        # This creates the DataflowModel internally!
        super().__init__(onnx_node, interface_metadata=interface_metadata, **kwargs)
        
        # Store kernel-specific info
        self.kernel_name = "thresholding_axi"
        self.rtl_source = "examples/thresholding/thresholding_axi.sv"
    
    def get_nodeattr_types(self) -> Dict[str, Tuple[str, bool, Any]]:
        """
        Define node attributes combining DataflowModel attributes and kernel-specific ones.
        
        The parent class provides dataflow attributes like:
        - {interface_name}_parallel for each interface
        - {interface_name}_dtype for each interface  
        - resource_estimation_mode
        - enable_constraint_validation
        
        We add kernel-specific RTL parameters.
        """
        # Get base dataflow attributes from parent
        attrs = super().get_enhanced_nodeattr_types()
        
        # Add RTL parameters extracted from ParsedKernelData
        kernel_attrs = {
            # RTL parameters
            "N": ("i", False, 0),  # Number of thresholds
            "C": ("i", False, 1),  # Channels
            "PE": ("i", False, 1),  # Processing elements (parallelism)
            "SIGNED": ("i", False, 1),  # Signed comparison
            "BIAS": ("i", False, 0),  # Bias value
            
            # Algorithm-specific attributes inferred from RTL
            "NumChannels": ("i", True, 1),  # Maps to C parameter
            "numSteps": ("i", True, 1),  # Maps to N parameter  
            "ActVal": ("i", False, 0),  # Maps to BIAS parameter
            
            # Runtime configuration
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
            "numInputVectors": ("ints", False, [1]),
        }
        
        attrs.update(kernel_attrs)
        return attrs
    
    # ===== Optional Kernel-Specific Overrides =====
    # Most functionality is handled by AutoHWCustomOp via DataflowModel
    # We only override if we need kernel-specific behavior
    
    def get_exp_cycles(self) -> int:
        """
        Override cycle calculation with kernel-specific formula.
        
        For thresholding: cycles = NumChannels / PE * num_input_vectors
        """
        # We can use the parent's calculation or provide custom one
        channels = self.get_nodeattr("NumChannels") or 1
        pe = self.get_nodeattr("PE") or 1
        num_vectors = np.prod(self.get_nodeattr("numInputVectors") or [1])
        
        return (channels // pe) * num_vectors
    
    def verify_node(self):
        """Add kernel-specific verification."""
        # Parent class handles interface verification
        super().verify_node()
        
        # Add thresholding-specific checks
        channels = self.get_nodeattr("NumChannels") or self.get_nodeattr("C") or 1
        pe = self.get_nodeattr("PE") or 1
        
        if channels % pe != 0:
            raise ValueError(f"NumChannels ({channels}) must be divisible by PE ({pe})")
    
    def get_hw_compatible_threshold_tensor(self, orig_thres_matrix):
        """
        Kernel-specific method for threshold data preparation.
        
        This is an example of functionality that's specific to thresholding
        and wouldn't be handled by the generic DataflowModel.
        """
        channels = self.get_nodeattr("NumChannels") or 1
        pe = self.get_nodeattr("PE") or 1
        
        # Ensure correct shape
        if orig_thres_matrix.shape[0] == 1:
            # Broadcast single threshold to all channels
            thres_matrix = np.tile(orig_thres_matrix, (channels, 1))
        else:
            thres_matrix = orig_thres_matrix
            
        # Validate shape
        assert thres_matrix.shape[0] == channels, f"Threshold matrix must have {channels} rows"
        
        # Distribute between PEs (interleaving)
        # This is specific to how the RTL kernel expects data
        if pe > 1:
            # Reshape for PE distribution
            tmem = channels // pe
            n_thres = thres_matrix.shape[1]
            
            # Interleave rows for PEs
            result = np.zeros((pe, tmem, n_thres), dtype=thres_matrix.dtype)
            for p in range(pe):
                result[p] = thres_matrix[p::pe]
                
            return result.reshape(1, pe, tmem, n_thres)
        else:
            return thres_matrix.reshape(1, 1, *thres_matrix.shape)
    
    # ===== Methods Fully Handled by DataflowModel =====
    # These are provided by AutoHWCustomOp and don't need overriding:
    #
    # - get_input_datatype() - from DataflowModel interfaces
    # - get_output_datatype() - from DataflowModel interfaces  
    # - get_normal_input_shape() - from DataflowModel tensor reconstruction
    # - get_folded_input_shape() - from DataflowModel with parallelism
    # - get_instream_width() - from DataflowModel stream calculations
    # - get_outstream_width() - from DataflowModel stream calculations
    # - estimate_bram_usage() - from DataflowModel resource requirements
    # - estimate_lut_usage() - from DataflowModel resource requirements
    # - estimate_dsp_usage() - from DataflowModel resource requirements


def demonstrate_usage():
    """Show how the generated HWCustomOp works with FINN workflow."""
    
    # Simulate ONNX node creation
    class FakeONNXNode:
        def __init__(self):
            self.input = ["input_tensor", "threshold_tensor"]
            self.output = ["output_tensor"]
            self.name = "Threshold_0"
            self.op_type = "ThresholdingAxi"
    
    # Create instance - DataflowModel is built automatically!
    onnx_node = FakeONNXNode()
    hw_op = ThresholdingAxi(onnx_node)
    
    print("=== Generated ThresholdingAxi HWCustomOp ===")
    print(f"Kernel name: {hw_op.kernel_name}")
    print(f"RTL source: {hw_op.rtl_source}")
    
    print("\n=== DataflowModel Integration ===")
    print(f"Input interfaces: {hw_op.input_interfaces}")
    print(f"Output interfaces: {hw_op.output_interfaces}")
    print(f"Weight interfaces: {hw_op.weight_interfaces}")
    
    print("\n=== Node Attributes ===")
    attrs = hw_op.get_nodeattr_types()
    print("Dataflow attributes (from AutoHWCustomOp):")
    for name, spec in attrs.items():
        if any(name.endswith(suffix) for suffix in ["_parallel", "_dtype", "_mode", "_validation"]):
            print(f"  {name}: {spec}")
    
    print("\nKernel-specific attributes:")
    for name, spec in attrs.items():
        if not any(name.endswith(suffix) for suffix in ["_parallel", "_dtype", "_mode", "_validation"]):
            print(f"  {name}: {spec}")
    
    print("\n=== Shape Calculations (via DataflowModel) ===")
    # These would normally come from ONNX tensor shapes
    # AutoHWCustomOp handles this automatically
    print(f"Normal input shape: {hw_op.get_normal_input_shape()}")
    print(f"Folded input shape: {hw_op.get_folded_input_shape()}")
    
    print("\n=== Stream Widths (via DataflowModel) ===")
    print(f"Input stream width: {hw_op.get_instream_width()} bits")
    print(f"Output stream width: {hw_op.get_outstream_width()} bits")
    
    print("\n=== Resource Estimation (via DataflowModel) ===")
    print(f"BRAM usage: {hw_op.estimate_bram_usage()}")
    print(f"LUT usage: {hw_op.estimate_lut_usage()}")
    print(f"DSP usage: {hw_op.estimate_dsp_usage()}")
    
    print("\n=== Parallelism Configuration ===")
    # Update parallelism - DataflowModel recalculates everything!
    hw_op.update_parallelism(iPar={"s_axis_input": 4}, wPar={})
    print(f"After setting input parallelism to 4:")
    print(f"  Expected cycles: {hw_op.get_exp_cycles()}")
    print(f"  Current parallelism: {hw_op.get_current_parallelism()}")


if __name__ == "__main__":
    demonstrate_usage()