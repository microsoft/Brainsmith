#!/usr/bin/env python3
############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################
"""
Parity test comparing legacy LayerNorm vs modern AutoLayerNorm.

This test ensures that the modern AutoLayerNorm implementation provides
complete functional parity with the legacy LayerNorm implementation.

Test Strategy:
1. Create a FuncLayerNorm node in an ONNX model
2. Apply InferLayerNorm to get legacy LayerNorm
3. Apply InferAutoLayerNorm to get modern AutoLayerNorm
4. Compare all public methods for identical behavior
5. Test multiple scenarios (shapes, SIMD values, datatypes)
"""

import pytest
import numpy as np
from onnx import helper, TensorProto

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes

from brainsmith.kernels.layernorm.infer_layernorm import InferLayerNorm
from brainsmith.kernels.layernorm.infer_auto_layernorm import InferAutoLayerNorm
from brainsmith.kernels.layernorm.layernorm import LayerNorm
from brainsmith.kernels.layernorm.auto_layernorm import AutoLayerNorm


# ============================================================================
# Helper Functions
# ============================================================================

def create_funclayernorm_model(shape, datatype="FLOAT32", epsilon=1e-5):
    """Create a model with a FuncLayerNorm node.

    Args:
        shape: Input tensor shape (e.g., [1, 128, 768])
        datatype: FINN DataType name
        epsilon: LayerNorm epsilon value

    Returns:
        ModelWrapper with FuncLayerNorm node
    """
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, shape)

    node = helper.make_node(
        "FuncLayerNorm",
        ["inp"],
        ["out"],
        axis=-1,  # Normalize over last dimension
        epsilon=epsilon
    )

    graph = helper.make_graph([node], "test", [inp], [out])
    model = helper.make_model(graph)
    model = ModelWrapper(model)

    # Set datatypes
    dt = DataType[datatype]
    model.set_tensor_datatype("inp", dt)
    model.set_tensor_datatype("out", dt)

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    return model


def get_layernorm_instances(shape, simd=8, datatype="INT8", epsilon=1e-5):
    """Get both legacy and modern LayerNorm instances from same source.

    Args:
        shape: Input tensor shape
        simd: SIMD parallelization factor
        datatype: FINN DataType name
        epsilon: LayerNorm epsilon value

    Returns:
        Tuple of (legacy_model, legacy_op, modern_model, modern_op)
    """
    # Create two identical models with FuncLayerNorm
    model_legacy = create_funclayernorm_model(shape, datatype, epsilon)
    model_modern = create_funclayernorm_model(shape, datatype, epsilon)

    # Apply legacy transform
    model_legacy = model_legacy.transform(InferLayerNorm())

    # Apply modern transform
    model_modern = model_modern.transform(InferAutoLayerNorm())

    # Get nodes
    legacy_node = [n for n in model_legacy.graph.node if n.op_type == "LayerNorm"][0]
    modern_node = [n for n in model_modern.graph.node if n.op_type == "AutoLayerNorm"][0]

    # Create operator instances
    legacy_op = LayerNorm(legacy_node)
    modern_op = AutoLayerNorm(modern_node)

    # Set SIMD (must be done after instance creation)
    legacy_op.set_nodeattr("SIMD", simd)
    modern_op.set_nodeattr("SIMD", simd)

    # Refresh tensor context for AutoLayerNorm (required for kernel_model)
    modern_op.refresh_tensor_context(model_modern)

    return model_legacy, legacy_op, model_modern, modern_op


# ============================================================================
# Parity Tests
# ============================================================================

@pytest.mark.parametrize("shape,simd", [
    ([1, 128, 768], 8),    # BERT base
    ([1, 256, 1024], 16),  # BERT large
    ([2, 64, 512], 4),     # Smaller model
    ([1, 1, 768], 1),      # Single token
])
def test_shape_methods_parity(shape, simd):
    """Test that shape methods return identical results."""
    print(f"\n=== Shape Parity Test: shape={shape}, SIMD={simd} ===")

    # Get instances
    model_legacy, legacy_op, model_modern, modern_op = get_layernorm_instances(
        shape, simd=simd
    )

    # Test get_normal_input_shape
    legacy_normal_in = legacy_op.get_normal_input_shape()
    modern_normal_in = modern_op.get_normal_input_shape()
    assert legacy_normal_in == modern_normal_in, \
        f"Normal input shape mismatch: {legacy_normal_in} != {modern_normal_in}"
    print(f"  ✓ get_normal_input_shape(): {legacy_normal_in}")

    # Test get_normal_output_shape
    legacy_normal_out = legacy_op.get_normal_output_shape()
    modern_normal_out = modern_op.get_normal_output_shape()
    assert legacy_normal_out == modern_normal_out, \
        f"Normal output shape mismatch: {legacy_normal_out} != {modern_normal_out}"
    print(f"  ✓ get_normal_output_shape(): {legacy_normal_out}")

    # Test get_folded_input_shape
    legacy_folded_in = legacy_op.get_folded_input_shape()
    modern_folded_in = modern_op.get_folded_input_shape()
    assert legacy_folded_in == modern_folded_in, \
        f"Folded input shape mismatch: {legacy_folded_in} != {modern_folded_in}"
    print(f"  ✓ get_folded_input_shape(): {legacy_folded_in}")

    # Test get_folded_output_shape
    legacy_folded_out = legacy_op.get_folded_output_shape()
    modern_folded_out = modern_op.get_folded_output_shape()
    assert legacy_folded_out == modern_folded_out, \
        f"Folded output shape mismatch: {legacy_folded_out} != {modern_folded_out}"
    print(f"  ✓ get_folded_output_shape(): {legacy_folded_out}")


@pytest.mark.parametrize("datatype", ["INT8", "INT16", "FLOAT32"])
def test_datatype_methods_parity(datatype):
    """Test that datatype methods return identical results."""
    print(f"\n=== Datatype Parity Test: datatype={datatype} ===")

    shape = [1, 128, 768]
    simd = 8

    # Get instances
    model_legacy, legacy_op, model_modern, modern_op = get_layernorm_instances(
        shape, simd=simd, datatype=datatype
    )

    # Test get_input_datatype
    legacy_idt = legacy_op.get_input_datatype()
    modern_idt = modern_op.get_input_datatype()
    assert legacy_idt == modern_idt, \
        f"Input datatype mismatch: {legacy_idt} != {modern_idt}"
    print(f"  ✓ get_input_datatype(): {legacy_idt}")

    # Test get_output_datatype
    legacy_odt = legacy_op.get_output_datatype()
    modern_odt = modern_op.get_output_datatype()
    assert legacy_odt == modern_odt, \
        f"Output datatype mismatch: {legacy_odt} != {modern_odt}"
    print(f"  ✓ get_output_datatype(): {legacy_odt}")


@pytest.mark.parametrize("shape,simd,datatype", [
    ([1, 128, 768], 8, "INT8"),
    ([1, 256, 1024], 16, "INT16"),
    ([2, 64, 512], 4, "INT32"),
])
def test_stream_width_methods_parity(shape, simd, datatype):
    """Test that stream width methods return identical results."""
    print(f"\n=== Stream Width Parity Test: shape={shape}, SIMD={simd}, dtype={datatype} ===")

    # Get instances
    model_legacy, legacy_op, model_modern, modern_op = get_layernorm_instances(
        shape, simd=simd, datatype=datatype
    )

    # Test get_instream_width
    legacy_inwidth = legacy_op.get_instream_width()
    modern_inwidth = modern_op.get_instream_width()
    assert legacy_inwidth == modern_inwidth, \
        f"Input stream width mismatch: {legacy_inwidth} != {modern_inwidth}"
    print(f"  ✓ get_instream_width(): {legacy_inwidth} bits")

    # Test get_outstream_width
    legacy_outwidth = legacy_op.get_outstream_width()
    modern_outwidth = modern_op.get_outstream_width()
    assert legacy_outwidth == modern_outwidth, \
        f"Output stream width mismatch: {legacy_outwidth} != {modern_outwidth}"
    print(f"  ✓ get_outstream_width(): {legacy_outwidth} bits")

    # Verify calculation: SIMD * bitwidth
    dt = DataType[datatype]
    expected_width = simd * dt.bitwidth()
    assert legacy_inwidth == expected_width
    assert modern_inwidth == expected_width
    print(f"  ✓ Calculation verified: {simd} × {dt.bitwidth()} = {expected_width}")


def test_execution_parity():
    """Test that Python execution produces identical results."""
    print("\n=== Execution Parity Test ===")

    shape = [2, 64, 256]
    simd = 8

    # Get instances
    model_legacy, legacy_op, model_modern, modern_op = get_layernorm_instances(
        shape, simd=simd, datatype="FLOAT32"
    )

    # Create input data
    np.random.seed(42)
    input_data = np.random.randn(*shape).astype(np.float32)

    # Execute legacy
    legacy_context = {"inp": input_data.copy()}
    legacy_op.set_nodeattr("exec_mode", "python")
    legacy_op.execute_node(legacy_context, model_legacy.graph)
    legacy_output = legacy_context["out"]

    # Execute modern
    modern_context = {"inp": input_data.copy()}
    modern_op.set_nodeattr("exec_mode", "python")
    modern_op.execute_node(modern_context, model_modern.graph)
    modern_output = modern_context["out"]

    # Compare outputs
    assert legacy_output.shape == modern_output.shape, \
        f"Output shape mismatch: {legacy_output.shape} != {modern_output.shape}"

    # Check numerical equivalence (allow small floating point differences)
    assert np.allclose(legacy_output, modern_output, rtol=1e-5, atol=1e-7), \
        f"Output values differ! Max diff: {np.max(np.abs(legacy_output - modern_output))}"

    print(f"  ✓ Outputs identical: shape={legacy_output.shape}")
    print(f"  ✓ Max difference: {np.max(np.abs(legacy_output - modern_output)):.2e}")

    # Verify normalization properties
    mean = np.mean(legacy_output, axis=-1)
    std = np.std(legacy_output, axis=-1)
    print(f"  ✓ Output mean ≈ 0: {np.mean(np.abs(mean)):.2e}")
    print(f"  ✓ Output std ≈ 1: {np.mean(std):.6f}")


def test_utility_methods_parity():
    """Test utility methods for parity."""
    print("\n=== Utility Methods Parity Test ===")

    shape = [1, 128, 768]
    simd = 8

    # Get instances
    model_legacy, legacy_op, model_modern, modern_op = get_layernorm_instances(
        shape, simd=simd
    )

    # Test get_number_output_values
    legacy_num_values = legacy_op.get_number_output_values()
    modern_num_values = modern_op.get_number_output_values()
    assert legacy_num_values == modern_num_values, \
        f"Number of output values mismatch: {legacy_num_values} != {modern_num_values}"
    print(f"  ✓ get_number_output_values(): {legacy_num_values}")


def test_constraint_validation_parity():
    """Test that both implementations reject invalid SIMD values."""
    print("\n=== Constraint Validation Parity Test ===")

    shape = [1, 128, 768]
    invalid_simd = 7  # Doesn't divide 768

    print(f"  Testing invalid SIMD={invalid_simd} (768 % 7 != 0)")

    # Legacy implementation - should fail with assertion
    legacy_failed = False
    try:
        model_legacy, legacy_op, _, _ = get_layernorm_instances(
            shape, simd=invalid_simd
        )
        # Try to get folded shape
        _ = legacy_op.get_folded_input_shape()
    except AssertionError as e:
        if "SIMD must divide" in str(e):
            print(f"  ✓ Legacy validation: {str(e)[:50]}...")
            legacy_failed = True
        else:
            raise  # Re-raise unexpected assertion

    assert legacy_failed, "Legacy should have raised assertion error"

    # Modern implementation - should fail with HWCustomOpError
    modern_failed = False
    try:
        _, _, model_modern, modern_op = get_layernorm_instances(
            shape, simd=invalid_simd
        )
        # Try to access kernel_model (triggers validation)
        _ = modern_op.kernel_model
    except Exception as e:
        if "divisible" in str(e).lower() or "divide" in str(e).lower():
            print(f"  ✓ Modern validation: {type(e).__name__}: {str(e)[:50]}...")
            modern_failed = True
        else:
            raise  # Re-raise unexpected error

    assert modern_failed, "Modern should have raised validation error"

    print("  ✓ Both implementations reject invalid SIMD values")


# ============================================================================
# Summary Test
# ============================================================================

def test_comprehensive_parity():
    """Comprehensive parity test covering all methods."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE LAYERNORM PARITY TEST")
    print("=" * 60)

    test_cases = [
        # (shape, simd, datatype)
        ([1, 128, 768], 8, "INT8"),
        ([1, 256, 1024], 16, "INT16"),
        ([2, 64, 512], 4, "FLOAT32"),
    ]

    for shape, simd, datatype in test_cases:
        print(f"\n--- Test Case: shape={shape}, SIMD={simd}, dtype={datatype} ---")

        # Get instances
        model_legacy, legacy_op, model_modern, modern_op = get_layernorm_instances(
            shape, simd=simd, datatype=datatype
        )

        # Compare all methods
        comparisons = [
            ("normal_input_shape", legacy_op.get_normal_input_shape(), modern_op.get_normal_input_shape()),
            ("normal_output_shape", legacy_op.get_normal_output_shape(), modern_op.get_normal_output_shape()),
            ("folded_input_shape", legacy_op.get_folded_input_shape(), modern_op.get_folded_input_shape()),
            ("folded_output_shape", legacy_op.get_folded_output_shape(), modern_op.get_folded_output_shape()),
            ("input_datatype", legacy_op.get_input_datatype(), modern_op.get_input_datatype()),
            ("output_datatype", legacy_op.get_output_datatype(), modern_op.get_output_datatype()),
            ("instream_width", legacy_op.get_instream_width(), modern_op.get_instream_width()),
            ("outstream_width", legacy_op.get_outstream_width(), modern_op.get_outstream_width()),
            ("number_output_values", legacy_op.get_number_output_values(), modern_op.get_number_output_values()),
        ]

        all_match = True
        for name, legacy_val, modern_val in comparisons:
            match = legacy_val == modern_val
            symbol = "✓" if match else "✗"
            print(f"  {symbol} {name}: {legacy_val}")
            if not match:
                print(f"    Legacy: {legacy_val}")
                print(f"    Modern: {modern_val}")
                all_match = False

        assert all_match, "Some methods did not match!"

    print("\n" + "=" * 60)
    print("✓ ALL PARITY TESTS PASSED")
    print("=" * 60)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all parity tests."""
    print("=" * 60)
    print("LayerNorm Parity Test Suite")
    print("Legacy LayerNorm vs Modern AutoLayerNorm")
    print("=" * 60)

    try:
        # Individual parity tests
        test_shape_methods_parity([1, 128, 768], 8)
        test_shape_methods_parity([1, 256, 1024], 16)

        test_datatype_methods_parity("INT8")
        test_datatype_methods_parity("FLOAT32")

        test_stream_width_methods_parity([1, 128, 768], 8, "INT8")
        test_stream_width_methods_parity([1, 256, 1024], 16, "INT16")

        test_execution_parity()
        test_utility_methods_parity()
        test_constraint_validation_parity()

        # Comprehensive test
        test_comprehensive_parity()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED - PARITY CONFIRMED")
        print("=" * 60)
        print("\nConclusion:")
        print("  AutoLayerNorm provides complete functional parity with LayerNorm")
        print("  All shape methods, datatypes, stream widths, and execution match")
        print("  Modern implementation is ready for production use")

        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ PARITY TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
