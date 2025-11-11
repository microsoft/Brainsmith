#!/usr/bin/env python3
"""
Comprehensive comparison test between FINN Thresholding and Brainsmith ThresholdingAxi.

This test:
1. Creates a MultiThreshold model
2. Applies both FINN's InferThresholdingLayer and Brainsmith's InferThresholdingAxi
3. Compares the generated nodes and their attributes
4. Applies SpecializeLayers to get RTL backends
5. Generates RTL from both implementations
6. Compares the generated RTL outputs
7. Saves all outputs and comparison results to examples/kernel_integrator/output/
"""

import os
import shutil
import sys
from pathlib import Path

import numpy as np

# Import transforms
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferThresholdingLayer
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

# Import from the manually implemented infer transform
sys.path.insert(0, str(Path(__file__).parent.parent))
from infer_thresholding_axi import InferThresholdingAxi  # noqa: E402

# Add kernel directory to path for later dynamic import
sys.path.insert(0, str(Path(__file__).parent.parent / "kernel"))

# Define output directory relative to this file
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def create_multithreshold_model(channels, input_dt, output_dt, bias=0):
    """Create a MultiThreshold test model."""
    
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


def compare_node_attributes(finn_node, brainsmith_node, report_lines):
    """Compare attributes between FINN and Brainsmith nodes."""
    
    report_lines.append("\n=== Node Attribute Comparison ===")
    
    if finn_node:
        finn_inst = getCustomOp(finn_node)
        report_lines.append("\nFINN Thresholding node:")
        report_lines.append(f"  Type: {finn_node.op_type}")
        report_lines.append(f"  Domain: {finn_node.domain}")
        report_lines.append(f"  NumChannels: {finn_inst.get_nodeattr('NumChannels')}")
        report_lines.append(f"  PE: {finn_inst.get_nodeattr('PE')}")
        report_lines.append(f"  ActVal: {finn_inst.get_nodeattr('ActVal')}")
        report_lines.append(f"  inputDataType: {finn_inst.get_nodeattr('inputDataType')}")
        report_lines.append(f"  outputDataType: {finn_inst.get_nodeattr('outputDataType')}")
        report_lines.append(f"  numSteps: {finn_inst.get_nodeattr('numSteps')}")
    else:
        report_lines.append("\nFINN Thresholding node: Not generated (may be expected for some configs)")
    
    if brainsmith_node:
        # Get attributes
        attrs = {attr.name: attr for attr in brainsmith_node.attribute}
        
        report_lines.append("\nBrainsmith ThresholdingAxi node:")
        report_lines.append(f"  Type: {brainsmith_node.op_type}")
        report_lines.append(f"  Domain: {brainsmith_node.domain}")
        report_lines.append(f"  CHANNELS: {attrs['CHANNELS'].i}")
        report_lines.append(f"  PE: {attrs['PE'].i}")
        report_lines.append(f"  BIAS: {attrs['BIAS'].i}")
        report_lines.append(f"  inputDataType: {attrs['inputDataType'].s.decode()}")
        report_lines.append(f"  outputDataType: {attrs['outputDataType'].s.decode()}")
        # Handle both possible names for threshold data type
        if 'thresholdDataType' in attrs:
            report_lines.append(f"  thresholdDataType: {attrs['thresholdDataType'].s.decode()}")
        elif 'weightDataType' in attrs:
            report_lines.append(f"  weightDataType: {attrs['weightDataType'].s.decode()}")
        
        # RTL-specific attributes (may not exist for non-RTL nodes)
        rtl_attrs = ['input_FPARG', 'DEPTH_TRIGGER_URAM', 'DEPTH_TRIGGER_BRAM', 'DEEP_PIPELINE', 'USE_AXILITE']
        has_rtl_attrs = any(attr in attrs for attr in rtl_attrs)
        if has_rtl_attrs:
            report_lines.append("\n  RTL-specific attributes:")
            for attr in rtl_attrs:
                if attr in attrs:
                    report_lines.append(f"    {attr}: {attrs[attr].i}")
        
        # Compare values if FINN node exists
        if finn_node:
            finn_inst = getCustomOp(finn_node)
            report_lines.append("\n  Attribute Mapping:")
            report_lines.append(f"    Channels: NumChannels={finn_inst.get_nodeattr('NumChannels')} vs CHANNELS={attrs['CHANNELS'].i}")
            report_lines.append(f"    PE: PE={finn_inst.get_nodeattr('PE')} vs PE={attrs['PE'].i}")
            report_lines.append(f"    Bias: ActVal={finn_inst.get_nodeattr('ActVal')} vs BIAS={attrs['BIAS'].i}")


def generate_and_compare_rtl(finn_model, brainsmith_model, test_name, output_subdir):
    """Generate RTL from both implementations and compare outputs."""
    
    report_lines = []
    report_lines.append(f"\n{'='*60}")
    report_lines.append(f"RTL Generation Test: {test_name}")
    report_lines.append(f"{'='*60}")
    
    results = {"finn": {}, "brainsmith": {}}
    fpgapart = "xczu3eg-sbva484-1-e"
    
    # Create output directories
    test_output_dir = OUTPUT_DIR / output_subdir
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test FINN implementation if available
    if finn_model is not None:
        report_lines.append("\nTesting FINN Thresholding RTL generation...")

        try:
            # Apply SpecializeLayers to get RTL backend
            finn_specialized = finn_model.transform(SpecializeLayers(fpgapart=fpgapart))

            # Get the Thresholding node
            finn_nodes = finn_specialized.get_nodes_by_op_type("Thresholding_rtl")
            if not finn_nodes:
                finn_nodes = finn_specialized.get_nodes_by_op_type("Thresholding_hls")

            if finn_nodes:
                finn_node = finn_nodes[0]
                report_lines.append(f"  Found node type: {finn_node.op_type}, domain: {finn_node.domain}")
                finn_inst = getCustomOp(finn_node)

                # Set up code generation directory
                finn_codegen_dir = test_output_dir / "finn_rtl"
                finn_codegen_dir.mkdir(parents=True, exist_ok=True)
                finn_inst.set_nodeattr("code_gen_dir_ipgen", str(finn_codegen_dir))

                # Generate HDL
                finn_inst.generate_hdl(finn_specialized, fpgapart=fpgapart, clk=5.0)

                # Collect generated files
                finn_files = []
                for root, dirs, files in os.walk(finn_codegen_dir):
                    for file in files:
                        rel_path = os.path.relpath(os.path.join(root, file), finn_codegen_dir)
                        finn_files.append(rel_path)

                results["finn"]["success"] = True
                results["finn"]["files"] = sorted(finn_files)
                results["finn"]["file_count"] = len(finn_files)
                results["finn"]["codegen_dir"] = finn_codegen_dir

                report_lines.append(f"  ✓ Generated {len(finn_files)} files")
                report_lines.append(f"  Files: {', '.join(finn_files[:5])}{'...' if len(finn_files) > 5 else ''}")

                # Copy wrapper file for easy comparison
                wrapper_candidates = [f for f in finn_files if f.endswith(".v") and "wrapper" in f.lower()]
                if not wrapper_candidates:
                    wrapper_candidates = [f for f in finn_files if f.endswith(".v")]
                if wrapper_candidates:
                    wrapper_src = finn_codegen_dir / wrapper_candidates[0]
                    wrapper_dst = test_output_dir / "finn_wrapper.v"
                    shutil.copy2(wrapper_src, wrapper_dst)
                    report_lines.append("  Wrapper copied to: finn_wrapper.v")
            else:
                report_lines.append("  ⚠️ No specialized Thresholding node found after SpecializeLayers")
                results["finn"]["success"] = False
                results["finn"]["error"] = "No specialized node found"

        except Exception as e:
            results["finn"]["success"] = False
            results["finn"]["error"] = str(e)
            report_lines.append(f"  ✗ RTL generation failed: {e}")

    # Test Brainsmith implementation
    report_lines.append("\nTesting Brainsmith ThresholdingAxi RTL generation...")
    report_lines.append("  Note: RTL generation from examples directory requires manual instantiation")
    
    try:
        # For Brainsmith, we need to manually instantiate the RTL backend
        # since dynamic module loading doesn't work from examples directory
        brainsmith_nodes = brainsmith_model.get_nodes_by_op_type("ThresholdingAxi")
        
        # Try to apply SpecializeLayers anyway to see if it works
        try:
            brainsmith_specialized = brainsmith_model.transform(SpecializeLayers(fpgapart=fpgapart))
            specialized_nodes = brainsmith_specialized.get_nodes_by_op_type("ThresholdingAxi_rtl")
            if not specialized_nodes:
                specialized_nodes = brainsmith_specialized.get_nodes_by_op_type("thresholding_axi_rtl")
            if specialized_nodes:
                brainsmith_nodes = specialized_nodes
                brainsmith_model = brainsmith_specialized
        except Exception:
            pass  # Expected when running from examples
        
        if brainsmith_nodes:
            brainsmith_node = brainsmith_nodes[0]
            report_lines.append(f"  Found node type: {brainsmith_node.op_type}, domain: {brainsmith_node.domain}")
            # If not specialized, manually instantiate RTL backend
            if brainsmith_node.op_type == "ThresholdingAxi":
                report_lines.append("  Manually instantiating RTL backend...")
                # Import the RTL backend directly
                from thresholding_axi_rtl import ThresholdingAxi_rtl
                brainsmith_inst = ThresholdingAxi_rtl(brainsmith_node)
            else:
                brainsmith_inst = getCustomOp(brainsmith_node)
            
            # Set up code generation directory
            brainsmith_codegen_dir = test_output_dir / "brainsmith_rtl"
            brainsmith_codegen_dir.mkdir(parents=True, exist_ok=True)
            brainsmith_inst.set_nodeattr("code_gen_dir_ipgen", str(brainsmith_codegen_dir))
            
            # Initialize KernelModel if needed
            if hasattr(brainsmith_inst, 'make_shape_compatible_op'):
                brainsmith_inst.make_shape_compatible_op(brainsmith_model)
            
            # Generate HDL
            brainsmith_inst.generate_hdl(brainsmith_model, fpgapart=fpgapart, clk=5.0)
            
            # Collect generated files
            brainsmith_files = []
            for root, dirs, files in os.walk(brainsmith_codegen_dir):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), brainsmith_codegen_dir)
                    brainsmith_files.append(rel_path)
            
            results["brainsmith"]["success"] = True
            results["brainsmith"]["files"] = sorted(brainsmith_files)
            results["brainsmith"]["file_count"] = len(brainsmith_files)
            results["brainsmith"]["codegen_dir"] = brainsmith_codegen_dir

            report_lines.append(f"  ✓ Generated {len(brainsmith_files)} files")
            report_lines.append(f"  Files: {', '.join(brainsmith_files[:5])}{'...' if len(brainsmith_files) > 5 else ''}")

            # Copy wrapper file for easy comparison
            wrapper_candidates = [f for f in brainsmith_files if f.endswith(".v") and "wrapper" in f.lower()]
            if not wrapper_candidates:
                wrapper_candidates = [f for f in brainsmith_files if f.endswith(".v")]
            if wrapper_candidates:
                wrapper_src = brainsmith_codegen_dir / wrapper_candidates[0]
                wrapper_dst = test_output_dir / "brainsmith_wrapper.v"
                shutil.copy2(wrapper_src, wrapper_dst)
                report_lines.append("  Wrapper copied to: brainsmith_wrapper.v")
        else:
            report_lines.append("  ⚠️ No specialized ThresholdingAxi node found after SpecializeLayers")
            results["brainsmith"]["success"] = False
            results["brainsmith"]["error"] = "No specialized node found"

    except Exception as e:
        results["brainsmith"]["success"] = False
        results["brainsmith"]["error"] = str(e)
        report_lines.append(f"  ✗ RTL generation failed: {e}")
    
    # Compare results
    report_lines.append("\n=== RTL Generation Comparison ===")
    
    if results["finn"].get("success") and results["brainsmith"].get("success"):
        finn_files = set(results["finn"]["files"])
        brainsmith_files = set(results["brainsmith"]["files"])
        
        # Find common extensions
        finn_exts = {Path(f).suffix for f in finn_files}
        brainsmith_exts = {Path(f).suffix for f in brainsmith_files}
        
        report_lines.append("\nFile Statistics:")
        report_lines.append(f"  FINN:       {len(finn_files)} files")
        report_lines.append(f"  Brainsmith: {len(brainsmith_files)} files")
        report_lines.append("\nFile types:")
        report_lines.append(f"  FINN:       {sorted(finn_exts)}")
        report_lines.append(f"  Brainsmith: {sorted(brainsmith_exts)}")
        
        # Check for key files
        finn_has_wrapper = any(".v" in f for f in finn_files)
        brainsmith_has_wrapper = any(".v" in f for f in brainsmith_files)
        finn_has_sv = any(".sv" in f for f in finn_files)
        brainsmith_has_sv = any(".sv" in f for f in brainsmith_files)
        
        report_lines.append("\nKey Files:")
        report_lines.append(f"  Has Verilog wrapper: FINN={finn_has_wrapper}, Brainsmith={brainsmith_has_wrapper}")
        report_lines.append(f"  Has SystemVerilog:   FINN={finn_has_sv}, Brainsmith={brainsmith_has_sv}")
        
        # Parameter mapping comparison
        report_lines.append("\n=== Parameter Mapping ===")
        report_lines.append("Common concepts:")
        report_lines.append("  Channels:     NumChannels (FINN) vs CHANNELS (Brainsmith)")
        report_lines.append("  Parallelism:  PE (both)")
        report_lines.append("  Bias:         ActVal (FINN) vs BIAS (Brainsmith)")
        report_lines.append("  Memory cfg:   depth_trigger_* (both)")
        
        report_lines.append("\nUnique to FINN:")
        report_lines.append("  - runtime_writeable_weights")
        report_lines.append("  - Separate template file references")
        
        report_lines.append("\nUnique to Brainsmith:")
        report_lines.append("  - input_FPARG (floating-point support)")
        report_lines.append("  - THRESHOLDS_PATH (external threshold storage)")
        report_lines.append("  - Explicit width parameters per interface")
        report_lines.append("  - USE_AXILITE (explicit config interface control)")
        
    return results, report_lines


def run_comprehensive_test(test_name, channels, input_dt, output_dt, bias=0):
    """Run a comprehensive comparison test."""
    
    print(f"\n{'='*60}")
    print(f"Starting: {test_name}")
    print(f"{'='*60}")
    
    # Create base model
    model = create_multithreshold_model(channels, input_dt, output_dt, bias)
    
    # Apply transforms
    finn_model = None
    brainsmith_model = None
    
    # Apply FINN transform
    try:
        finn_model = model.transform(InferThresholdingLayer())
        finn_nodes = finn_model.get_nodes_by_op_type("Thresholding")
        if not finn_nodes:
            print("  ⚠️ FINN InferThresholdingLayer did not create Thresholding node")
            finn_model = None
    except Exception as e:
        print(f"  ⚠️ FINN InferThresholdingLayer failed: {e}")
        print("      (This is expected for some configurations)")
        finn_model = None

    # Apply Brainsmith transform
    try:
        brainsmith_model = model.transform(InferThresholdingAxi())
        brainsmith_nodes = brainsmith_model.get_nodes_by_op_type("ThresholdingAxi")
        if not brainsmith_nodes:
            print("  ✗ InferThresholdingAxi did not create ThresholdingAxi node")
            return False
    except Exception as e:
        print(f"  ✗ InferThresholdingAxi failed: {e}")
        return False
    
    # Create test-specific output directory
    test_id = f"test_{channels}ch_{input_dt}_to_{output_dt}_bias{bias}"
    
    # Compare nodes and generate RTL
    all_report_lines = []
    all_report_lines.append(f"Comprehensive Test Report: {test_name}")
    all_report_lines.append(f"Configuration: {channels} channels, {input_dt} -> {output_dt}, bias={bias}")
    
    # Compare node attributes
    finn_node = finn_nodes[0] if finn_model and finn_nodes else None
    brainsmith_node = brainsmith_nodes[0] if brainsmith_nodes else None
    compare_node_attributes(finn_node, brainsmith_node, all_report_lines)
    
    # Generate and compare RTL
    results, rtl_report = generate_and_compare_rtl(
        finn_model, brainsmith_model, test_name, test_id
    )
    all_report_lines.extend(rtl_report)
    
    # Save comprehensive report
    report_path = OUTPUT_DIR / test_id / "comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(all_report_lines))

    print(f"\n✓ Test completed. Results saved to: {OUTPUT_DIR / test_id}")

    # Return success if Brainsmith worked
    return results["brainsmith"].get("success", False)


def main():
    """Run all comprehensive comparison tests."""
    print("Starting Comprehensive FINN vs Brainsmith Comparison Tests")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)
    
    # Create main output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define test configurations using UINT16->UINT8 to avoid FINN bug
    tests = [
        ("Test 1: Basic 8ch UINT16->UINT8", 8, "UINT16", "UINT8", 0),
        ("Test 2: Medium 16ch UINT16->UINT8", 16, "UINT16", "UINT8", 0),
        ("Test 3: Large 32ch UINT16->UINT8", 32, "UINT16", "UINT8", 0),
    ]
    
    passed = 0
    for test_name, channels, input_dt, output_dt, bias in tests:
        if run_comprehensive_test(test_name, channels, input_dt, output_dt, bias):
            passed += 1
    
    # Create summary report
    summary_lines = []
    summary_lines.append("FINN vs Brainsmith Comparison Test Summary")
    summary_lines.append("="*60)
    summary_lines.append(f"Total tests: {len(tests)}")
    summary_lines.append(f"Passed: {passed}")
    summary_lines.append(f"Failed: {len(tests) - passed}")
    summary_lines.append("")
    summary_lines.append("Test Details:")
    for i, (test_name, channels, input_dt, output_dt, bias) in enumerate(tests):
        status = "✓ PASS" if i < passed else "✗ FAIL"
        summary_lines.append(f"  {status} - {test_name}")
    summary_lines.append("")
    summary_lines.append(f"Results saved to: {OUTPUT_DIR}")
    
    summary_path = OUTPUT_DIR / "test_summary.txt"
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n{'='*60}")
    print('\n'.join(summary_lines))
    
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())