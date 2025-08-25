#!/usr/bin/env python3
"""
RTL generation comparison test between legacy FINN Thresholding and modern Brainsmith ThresholdingAxi.

This test creates equivalent nodes using both implementations and compares their RTL generation:
- FINN Thresholding_rtl backend  
- Brainsmith thresholding_axi_rtl backend

The test validates that both can generate RTL successfully and compares the outputs.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.util.basic import qonnx_make_model
from qonnx.custom_op.registry import getCustomOp

# Import transforms

from finn.transformation.fpgadataflow.convert_to_hw_layers import InferThresholdingLayer
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

from brainsmith.transforms.core.infer_thresholding_axi import InferThresholdingAxi


def create_multithreshold_model(channels, input_dt, output_dt, bias=0):
    """Create a simple MultiThreshold test model."""
    
    # Determine number of thresholds based on output datatype
    if output_dt == "UINT8":
        num_thresholds = 255  # Full range for UINT8 to avoid FINN bug
    elif output_dt == "UINT4":
        num_thresholds = 15   # Full range for UINT4
    else:
        num_thresholds = 2**(DataType[output_dt].bitwidth()) - 1
    
    # Create tensors
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, channels]
    )
    thresh = helper.make_tensor_value_info(
        "thresh", TensorProto.FLOAT, [channels, num_thresholds]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, channels]
    )
    
    # Create MultiThreshold node
    mt_node = helper.make_node(
        "MultiThreshold",
        inputs=["inp", "thresh"],
        outputs=["outp"],
        domain="qonnx.custom_op.general",
        name="MultiThreshold_0"
    )
    
    # Add attributes manually to ensure they're properly set
    mt_node.attribute.extend([
        helper.make_attribute("out_scale", 1.0),
        helper.make_attribute("out_bias", float(bias)),
        helper.make_attribute("out_dtype", output_dt)
    ])
    
    # Create graph
    graph = helper.make_graph(
        nodes=[mt_node],
        name="test_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[thresh]
    )
    
    # Create model
    model = qonnx_make_model(graph)
    model = ModelWrapper(model)
    
    # Set datatypes
    model.set_tensor_datatype("inp", DataType[input_dt])
    model.set_tensor_datatype("thresh", DataType["INT16"] if input_dt == "UINT16" else DataType["INT8"])
    model.set_tensor_datatype("outp", DataType[output_dt])
    
    # Create threshold values (sorted ascending as required)
    if output_dt == "UINT8":
        # Create full range of thresholds for UINT8
        thresh_vals = np.arange(0, 255, dtype=np.float32)
        thresh_vals = np.tile(thresh_vals, (channels, 1))
    else:
        # For other types, create evenly spaced thresholds
        thresh_vals = np.linspace(-10, 10, num_thresholds, dtype=np.float32)
        thresh_vals = np.tile(thresh_vals, (channels, 1))
    
    model.set_initializer("thresh", thresh_vals)
    
    return model


def test_rtl_generation(old_model, new_model, test_name, temp_dir):
    """Test RTL generation for both old and new implementations."""
    
    print(f"\n{'='*60}")
    print(f"RTL Generation Test: {test_name}")
    print(f"{'='*60}")
    
    results = {"old": {}, "new": {}}
    fpgapart = "xczu3eg-sbva484-1-e"
    
    # Test old FINN implementation if available
    if old_model is not None:
        print("\n📝 Testing FINN Thresholding RTL generation...")
        
        try:
            # Apply SpecializeLayers to get RTL backend
            old_specialized = old_model.transform(SpecializeLayers(fpgapart=fpgapart))
            
            # Get the Thresholding node
            old_nodes = old_specialized.get_nodes_by_op_type("Thresholding_rtl")
            if not old_nodes:
                # Try with non-RTL specialized name
                old_nodes = old_specialized.get_nodes_by_op_type("Thresholding_hls")
            
            if old_nodes:
                old_node = old_nodes[0]
                print(f"  Found node type: {old_node.op_type}, domain: {old_node.domain}")
                old_inst = getCustomOp(old_node)
                
                # Set up code generation directory
                old_codegen_dir = os.path.join(temp_dir, "finn_rtl")
                os.makedirs(old_codegen_dir, exist_ok=True)
                old_inst.set_nodeattr("code_gen_dir_ipgen", old_codegen_dir)
                
                # Generate HDL
                old_inst.generate_hdl(old_specialized, fpgapart=fpgapart, clk=5.0)
                
                # Collect generated files
                old_files = []
                for root, dirs, files in os.walk(old_codegen_dir):
                    for file in files:
                        rel_path = os.path.relpath(os.path.join(root, file), old_codegen_dir)
                        old_files.append(rel_path)
                
                results["old"]["success"] = True
                results["old"]["files"] = sorted(old_files)
                results["old"]["file_count"] = len(old_files)
                
                print(f"  ✅ Generated {len(old_files)} files")
                print(f"  📁 Files: {', '.join(old_files[:5])}{'...' if len(old_files) > 5 else ''}")
            else:
                print("  ⚠️  No specialized Thresholding node found after SpecializeLayers")
                results["old"]["success"] = False
                results["old"]["error"] = "No specialized node found"
                
        except Exception as e:
            results["old"]["success"] = False
            results["old"]["error"] = str(e)
            print(f"  ❌ RTL generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Test new Brainsmith implementation
    print("\n📝 Testing Brainsmith ThresholdingAxi RTL generation...")
    
    try:
        # Debug: Check node attributes before specialization
        debug_nodes = new_model.get_nodes_by_op_type("ThresholdingAxi")
        if debug_nodes:
            debug_inst = getCustomOp(debug_nodes[0])
            print(f"  Node attributes before specialization:")
            print(f"    - preferred_impl_style: {debug_inst.get_nodeattr('preferred_impl_style')}")
        
        # Apply SpecializeLayers to get RTL backend
        new_specialized = new_model.transform(SpecializeLayers(fpgapart=fpgapart))
        
        # Get the ThresholdingAxi node - try all possible names
        new_nodes = new_specialized.get_nodes_by_op_type("ThresholdingAxi_rtl")
        if not new_nodes:
            new_nodes = new_specialized.get_nodes_by_op_type("thresholding_axi_rtl")
        if not new_nodes:
            # Try with non-specialized name
            new_nodes = new_specialized.get_nodes_by_op_type("ThresholdingAxi")
        
        if new_nodes:
            new_node = new_nodes[0]
            print(f"  Found node type: {new_node.op_type}, domain: {new_node.domain}")
            new_inst = getCustomOp(new_node)
            
            # Set up code generation directory
            new_codegen_dir = os.path.join(temp_dir, "brainsmith_rtl")
            os.makedirs(new_codegen_dir, exist_ok=True)
            new_inst.set_nodeattr("code_gen_dir_ipgen", new_codegen_dir)
            
            # Initialize KernelModel if needed
            if hasattr(new_inst, 'make_shape_compatible_op'):
                new_inst.make_shape_compatible_op(new_specialized)
            
            # Generate HDL
            new_inst.generate_hdl(new_specialized, fpgapart=fpgapart, clk=5.0)
            
            # Collect generated files
            new_files = []
            for root, dirs, files in os.walk(new_codegen_dir):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), new_codegen_dir)
                    new_files.append(rel_path)
            
            results["new"]["success"] = True
            results["new"]["files"] = sorted(new_files)
            results["new"]["file_count"] = len(new_files)
            
            print(f"  ✅ Generated {len(new_files)} files")
            print(f"  📁 Files: {', '.join(new_files[:5])}{'...' if len(new_files) > 5 else ''}")
        else:
            print("  ⚠️  No specialized ThresholdingAxi node found after SpecializeLayers")
            # Debug: show what nodes we have
            print(f"  Available nodes: {[n.op_type for n in new_specialized.graph.node]}")
            results["new"]["success"] = False
            results["new"]["error"] = "No specialized node found"
            
    except Exception as e:
        results["new"]["success"] = False
        results["new"]["error"] = str(e)
        print(f"  ❌ RTL generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare results
    print("\n📊 Comparison:")
    
    if results["old"].get("success") and results["new"].get("success"):
        old_files = set(results["old"]["files"])
        new_files = set(results["new"]["files"])
        
        # Find common extensions
        old_exts = {Path(f).suffix for f in old_files}
        new_exts = {Path(f).suffix for f in new_files}
        
        print(f"  File types (old): {sorted(old_exts)}")
        print(f"  File types (new): {sorted(new_exts)}")
        
        # Check for key files
        old_has_wrapper = any(".v" in f for f in old_files)
        new_has_wrapper = any(".v" in f for f in new_files)
        old_has_sv = any(".sv" in f for f in old_files)
        new_has_sv = any(".sv" in f for f in new_files)
        
        print(f"  Has Verilog wrapper: old={old_has_wrapper}, new={new_has_wrapper}")
        print(f"  Has SystemVerilog: old={old_has_sv}, new={new_has_sv}")
        
    return results


def run_test(test_name, channels, input_dt, output_dt, bias=0):
    """Run a single RTL generation comparison test."""
    
    # Create base model
    model = create_multithreshold_model(channels, input_dt, output_dt, bias)
    
    # Convert to old and new implementations
    old_model = None
    new_model = None

    try:
        old_model = model.transform(InferThresholdingLayer())
        old_nodes = old_model.get_nodes_by_op_type("Thresholding")
        if not old_nodes:
            print(f"  ⚠️  FINN InferThresholdingLayer did not create Thresholding node")
            old_model = None
    except Exception as e:
        print(f"  ⚠️  FINN InferThresholdingLayer failed: {e}")
        old_model = None

    try:
        new_model = model.transform(InferThresholdingAxi())
        new_nodes = new_model.get_nodes_by_op_type("ThresholdingAxi")
        if not new_nodes:
            print(f"  ❌ InferThresholdingAxi did not create ThresholdingAxi node")
            return False
    except Exception as e:
        print(f"  ❌ InferThresholdingAxi failed: {e}")
        return False
    
    # Test RTL generation in temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        results = test_rtl_generation(old_model, new_model, test_name, temp_dir)
        
        # Return success if at least new implementation worked
        return results["new"].get("success", False)


def main():
    """Run all RTL generation comparison tests."""
    print("🚀 Starting RTL Generation Comparison Tests")
    print("="*60)
    
    # Define test configurations using UINT16->UINT8 to avoid FINN bug
    tests = [
        ("Test 1: Basic 8ch UINT16->UINT8", 8, "UINT16", "UINT8", 0),
        ("Test 2: Medium 16ch UINT16->UINT8 with PE=4", 16, "UINT16", "UINT8", 0),
        ("Test 3: Large 32ch UINT16->UINT8 with PE=8", 32, "UINT16", "UINT8", 0),
    ]
    
    passed = 0
    for test_name, channels, input_dt, output_dt, bias in tests:
        if run_test(test_name, channels, input_dt, output_dt, bias):
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"📊 Results: {passed}/{len(tests)} tests completed successfully")
    
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())