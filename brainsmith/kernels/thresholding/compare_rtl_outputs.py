#!/usr/bin/env python3
"""
Generate and compare RTL outputs from FINN and Brainsmith implementations.
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path
from onnx import helper, TensorProto, numpy_helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.util.basic import qonnx_make_model
from qonnx.transformation.infer_shapes import InferShapes

# Import implementations
from finn.custom_op.fpgadataflow.rtl.thresholding_rtl import Thresholding_rtl
from brainsmith.kernels.thresholding.thresholding_axi_rtl import ThresholdingAxiRTL


def generate_finn_rtl():
    """Generate FINN RTL and return the wrapper content."""
    
    # Create FINN node with UINT16->UINT8 to avoid bug
    node = helper.make_node(
        "Thresholding",
        inputs=["inp", "thresh"],
        outputs=["outp"],
        domain="finn.custom_op.fpgadataflow",
        name="Thresholding_0",
        backend="fpgadataflow",
        NumChannels=8,
        PE=2,
        inputDataType="UINT16",
        weightDataType="INT16",
        outputDataType="UINT8",
        ActVal=0,
        numSteps=255,
        depth_trigger_uram=1024,
        depth_trigger_bram=256,
        deep_pipeline=1,
        runtime_writeable_weights=1,
    )
    
    # Create model
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 8])
    thresh = helper.make_tensor_value_info("thresh", TensorProto.FLOAT, [8, 255])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 8])
    
    thresh_vals = np.arange(0, 255, dtype=np.float32)
    thresh_vals = np.tile(thresh_vals, (8, 1))
    thresh_init = numpy_helper.from_array(thresh_vals, name="thresh")
    
    graph = helper.make_graph([node], "finn_graph", [inp], [outp], [thresh_init])
    model = helper.make_model(graph)
    model_wrapper = ModelWrapper(model)
    
    model_wrapper.set_tensor_datatype("inp", DataType["UINT16"])
    model_wrapper.set_tensor_datatype("thresh", DataType["INT16"])
    model_wrapper.set_tensor_datatype("outp", DataType["UINT8"])
    
    model_wrapper = model_wrapper.transform(InferShapes())
    node = model_wrapper.graph.node[0]
    
    # Generate RTL
    rtl_inst = Thresholding_rtl(node)
    rtl_inst.make_shape_compatible_op(model_wrapper)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        rtl_inst.set_nodeattr("code_gen_dir_ipgen", temp_dir)
        rtl_inst.generate_hdl(model_wrapper, fpgapart="xczu3eg-sbva484-1-e", clk=5.0)
        
        # Read the wrapper file
        wrapper_path = os.path.join(temp_dir, "Thresholding_0.v")
        if os.path.exists(wrapper_path):
            with open(wrapper_path, 'r') as f:
                return f.read()
    
    return None


def generate_brainsmith_rtl():
    """Generate Brainsmith RTL and return the wrapper content."""
    
    # Create Brainsmith node
    node = helper.make_node(
        "ThresholdingAxi",
        inputs=["inp", "thresh"],
        outputs=["outp"],
        domain="brainsmith.kernels.thresholding",
        name="ThresholdingAxi_0",
        backend="fpgadataflow",
        CHANNELS=8,
        PE=2,
        inputDataType="UINT16",
        outputDataType="UINT8",
        thresholdDataType="INT16",
        input_FPARG=0,
        BIAS=0,
        THRESHOLDS_PATH="",
        DEPTH_TRIGGER_URAM=1024,
        DEPTH_TRIGGER_BRAM=256,
        DEEP_PIPELINE=1,
        width=16,
        USE_AXILITE=1,
        numSteps=255,
        ActVal=0,
    )
    
    # Create model
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 8])
    thresh = helper.make_tensor_value_info("thresh", TensorProto.FLOAT, [8, 255])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 8])
    
    thresh_vals = np.arange(0, 255, dtype=np.float32)
    thresh_vals = np.tile(thresh_vals, (8, 1))
    thresh_init = numpy_helper.from_array(thresh_vals, name="thresh")
    
    graph = helper.make_graph([node], "bs_graph", [inp], [outp], [thresh_init])
    model = helper.make_model(graph)
    model_wrapper = ModelWrapper(model)
    
    model_wrapper.set_tensor_datatype("inp", DataType["UINT16"])
    model_wrapper.set_tensor_datatype("thresh", DataType["INT16"])
    model_wrapper.set_tensor_datatype("outp", DataType["UINT8"])
    
    model_wrapper = model_wrapper.transform(InferShapes())
    node = model_wrapper.graph.node[0]
    
    # Generate RTL
    rtl_inst = ThresholdingAxiRTL(node)
    rtl_inst.make_shape_compatible_op(model_wrapper)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        rtl_inst.set_nodeattr("code_gen_dir_ipgen", temp_dir)
        rtl_inst.generate_hdl(model_wrapper, fpgapart="xczu3eg-sbva484-1-e", clk=5.0)
        
        # Read the wrapper file
        wrapper_path = os.path.join(temp_dir, "ThresholdingAxi_0.v")
        if os.path.exists(wrapper_path):
            with open(wrapper_path, 'r') as f:
                return f.read()
    
    return None


def compare_rtl_wrappers():
    """Generate and compare RTL wrappers from both implementations."""
    
    print("🔍 Comparing RTL Generation: FINN vs Brainsmith")
    print("="*60)
    
    # Generate RTL from both
    print("\n📝 Generating FINN RTL...")
    finn_rtl = generate_finn_rtl()
    
    print("📝 Generating Brainsmith RTL...")
    bs_rtl = generate_brainsmith_rtl()
    
    if not finn_rtl:
        print("❌ Failed to generate FINN RTL")
        return
    
    if not bs_rtl:
        print("❌ Failed to generate Brainsmith RTL")
        return
    
    # Save to files for comparison
    output_dir = "rtl_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/finn_wrapper.v", 'w') as f:
        f.write(finn_rtl)
    
    with open(f"{output_dir}/brainsmith_wrapper.v", 'w') as f:
        f.write(bs_rtl)
    
    print(f"\n💾 RTL files saved to {output_dir}/")
    
    # Extract key information
    print("\n📊 Key Differences:")
    print("-"*60)
    
    # Module names
    finn_module = "thresholding_axi_wrapper" if "thresholding_axi_wrapper" in finn_rtl else "Thresholding_0"
    bs_module = "thresholding_axi_wrapper"
    print(f"Module name:")
    print(f"  FINN:       {finn_module}")
    print(f"  Brainsmith: {bs_module}")
    
    # Count parameters
    finn_params = finn_rtl.count("parameter")
    bs_params = bs_rtl.count("parameter")
    print(f"\nParameter count:")
    print(f"  FINN:       {finn_params}")
    print(f"  Brainsmith: {bs_params}")
    
    # Interface comparison
    print("\nInterfaces:")
    print("  FINN:       AXI-Stream (in/out) + optional AXI-Lite")
    print("  Brainsmith: AXI-Stream (in/out) + AXI-Lite (threshold config)")
    
    # Extract and compare parameters
    print("\nParameter mapping:")
    print("-"*40)
    
    # Common parameters
    print("Common concepts:")
    print("  Channels:     NumChannels (FINN) vs CHANNELS (Brainsmith)")
    print("  Parallelism:  PE (both)")
    print("  Bias:         ActVal (FINN) vs BIAS (Brainsmith)")
    print("  Memory cfg:   depth_trigger_* (both)")
    
    # Unique to each
    print("\nUnique to FINN:")
    print("  - runtime_writeable_weights")
    print("  - Separate template file references")
    
    print("\nUnique to Brainsmith:")
    print("  - input_FPARG (floating-point support)")
    print("  - THRESHOLDS_PATH (external threshold storage)")
    print("  - Explicit width parameters per interface")
    print("  - USE_AXILITE (explicit config interface control)")
    
    # Module instantiation style
    print("\nInstantiation style:")
    print("  FINN:       Uses template substitution extensively")
    print("  Brainsmith: Direct parameter mapping from RTL")
    
    print("\n✅ Comparison complete! Check rtl_comparison/ for full files.")


if __name__ == "__main__":
    compare_rtl_wrappers()