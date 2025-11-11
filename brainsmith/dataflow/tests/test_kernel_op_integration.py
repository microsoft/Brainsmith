############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Integration tests for KernelOp two-phase caching with real kernels (Phase 3).

These tests validate end-to-end DSE workflows using actual kernel implementations
like LayerNorm and AddStreams.
"""

import time

import pytest
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.kernels.layernorm import LayerNorm


def create_layernorm_model():
    """Create a minimal ONNX model with LayerNorm for testing."""
    # Create input tensor
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 768])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 768])

    # Create LayerNorm node
    node = helper.make_node(
        "LayerNorm",
        inputs=["input"],
        outputs=["output"],
        name="layernorm_0",
        domain="brainsmith.kernels",
        SIMD=1,
        epsilon=1e-5,
        input0Datatype="FLOAT32",
        output0Datatype="FLOAT32",
    )

    graph = helper.make_graph([node], "layernorm_graph", [input_tensor], [output_tensor])

    model = helper.make_model(graph)
    model_w = ModelWrapper(model)

    return model_w, node


# ====================================================================
# Integration Test 1: End-to-End LayerNorm DSE
# ====================================================================


def test_end_to_end_layernorm_dse():
    """Test end-to-end DSE exploration with LayerNorm kernel.

    Validates:
    - All valid SIMD configurations can be explored
    - Total time <100ms for all configurations
    - All configurations pass validation
    """
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)

    # Get valid SIMD ranges
    valid_ranges = kernel_op.get_valid_ranges(model_w)
    assert "SIMD" in valid_ranges, "LayerNorm should have SIMD parameter"

    valid_simd = sorted(valid_ranges["SIMD"])
    print(f"\n  Valid SIMD values: {len(valid_simd)} divisors of 768")

    # Measure DSE exploration
    start = time.time()
    successful_configs = 0

    for simd in valid_simd:
        try:
            kernel_op.set_nodeattr("SIMD", simd)
            configured = kernel_op.get_design_point(model_w)

            # Verify configuration is correct
            assert configured.params["SIMD"] == simd
            assert configured.input_list[0].stream_shape[-1] == simd

            successful_configs += 1

        except Exception as e:
            pytest.fail(f"Configuration SIMD={simd} failed: {e}")

    elapsed = time.time() - start

    # Assertions
    assert successful_configs == len(valid_simd), "All configs should succeed"

    # Target: <100ms for typical ~18 configs, use generous 500ms for CI
    assert elapsed < 0.5, f"DSE took {elapsed*1000:.2f}ms, target <500ms"

    print(f"  Explored {successful_configs} configs in {elapsed*1000:.1f}ms")
    print(f"  Average: {elapsed*1000/successful_configs:.2f}ms per config")

    # Verify speedup claim: should be much faster than rebuilding each time
    # Theoretical old time: 18 configs × 10ms = 180ms
    # New time should be: 10ms + (18 × <1ms) < 30ms
    # So speedup should be >6x even with generous timing
    theoretical_old_time = len(valid_simd) * 0.010  # 10ms per rebuild
    speedup = theoretical_old_time / elapsed
    print(f"  Theoretical speedup vs full rebuild: {speedup:.1f}x")


def test_layernorm_design_space_stability():
    """Test that LayerNorm design space remains stable across reconfigurations."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)

    # Get initial design space
    design_space = kernel_op.get_design_space(model_w)

    # Get valid SIMD values
    valid_simd = sorted(kernel_op.get_valid_ranges(model_w)["SIMD"])

    # Explore all configurations
    for simd in valid_simd:
        kernel_op.set_nodeattr("SIMD", simd)
        kernel_op.get_design_point(model_w)

        # Invariant model should NEVER change
        assert (
            kernel_op._design_space is design_space
        ), f"Invariant model invalidated at SIMD={simd}"


def test_layernorm_derived_dimension_across_configs():
    """Test that DerivedDim resolution works correctly across all configs."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)

    valid_simd = sorted(kernel_op.get_valid_ranges(model_w)["SIMD"])

    for simd in valid_simd:
        kernel_op.set_nodeattr("SIMD", simd)
        configured = kernel_op.get_design_point(model_w)

        # Output stream shape should match input (DerivedDim("input", -1))
        input_stream_shape = configured.input_list[0].stream_shape
        output_stream_shape = configured.output_list[0].stream_shape

        assert (
            output_stream_shape == input_stream_shape
        ), f"DerivedDim failed at SIMD={simd}: {output_stream_shape} != {input_stream_shape}"


# ====================================================================
# Integration Test 2: Performance Validation
# ====================================================================


def test_layernorm_performance_meets_targets():
    """Test that LayerNorm meets Phase 3 performance targets."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)

    # Target 1: First get_design_point() call ~10ms
    start = time.time()
    kernel_op.get_design_point(model_w)
    first_call_time = time.time() - start

    # Use generous 50ms for CI variability
    assert first_call_time < 0.05, f"First build took {first_call_time*1000:.2f}ms, target <50ms"

    print(f"\n  First build: {first_call_time*1000:.2f}ms")

    # Target 2: Cache hit <0.1ms (average over 100 calls)
    start = time.time()
    for _ in range(100):
        kernel_op.get_design_point(model_w)
    cache_hit_time = (time.time() - start) / 100

    # Use generous 1ms for CI variability
    assert cache_hit_time < 0.001, f"Cache hit took {cache_hit_time*1000:.2f}ms avg, target <1ms"

    print(f"  Cache hit: {cache_hit_time*1000:.3f}ms avg (100 calls)")

    # Target 3: Reconfiguration <1ms
    kernel_op.set_nodeattr("SIMD", 2)
    start = time.time()
    kernel_op.get_design_point(model_w)
    reconfig_time = time.time() - start

    # Use generous 5ms for CI variability
    assert reconfig_time < 0.005, f"Reconfiguration took {reconfig_time*1000:.2f}ms, target <5ms"

    print(f"  Reconfiguration: {reconfig_time*1000:.2f}ms")


def test_layernorm_memory_efficiency():
    """Test that configured models are memory-efficient (flyweight pattern)."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)

    # Create multiple configurations
    valid_simd = sorted(kernel_op.get_valid_ranges(model_w)["SIMD"])[:5]

    configurations = []
    for simd in valid_simd:
        kernel_op.set_nodeattr("SIMD", simd)
        configured = kernel_op.get_design_point(model_w)
        configurations.append(configured)

    # All configurations should share same design space
    invariants = [cfg.design_space for cfg in configurations]
    assert all(
        inv is invariants[0] for inv in invariants
    ), "All configurations should share same design space (flyweight)"

    print(f"\n  {len(configurations)} configurations share 1 design space")


# ====================================================================
# Integration Test 3: Backward Compatibility
# ====================================================================


def test_layernorm_backward_compatibility():
    """Test that LayerNorm works exactly as before (backward compatibility)."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)

    # Legacy usage pattern should still work
    kernel_op.set_nodeattr("SIMD", 8)

    # Legacy method calls
    tensor_shape = kernel_op.get_normal_input_shape(ind=0, model_w=model_w)
    assert tensor_shape == [1, 1, 768]

    folded_shape = kernel_op.get_folded_input_shape(ind=0, model_w=model_w)
    assert len(folded_shape) == 4  # [fold_factors..., flattened_stream]

    stream_width = kernel_op.get_instream_width(ind=0, model_w=model_w)
    assert stream_width > 0

    # design_point property should work
    ki = kernel_op.design_point
    assert ki is not None
    assert hasattr(ki, "inputs")
    assert hasattr(ki, "outputs")


# ====================================================================
# Integration Test 4: Real-World DSE Scenarios
# ====================================================================


def test_realistic_dse_workflow():
    """Test realistic DSE workflow: explore, filter, select best."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)

    # Step 1: Get valid ranges
    valid_ranges = kernel_op.get_valid_ranges(model_w)
    all_simd = sorted(valid_ranges["SIMD"])

    # Step 2: Filter to reasonable range (e.g., 4 <= SIMD <= 128)
    filtered_simd = [s for s in all_simd if 4 <= s <= 128]

    print(f"\n  Total valid configs: {len(all_simd)}")
    print(f"  Filtered configs: {len(filtered_simd)}")

    # Step 3: Explore filtered space
    results = []
    for simd in filtered_simd:
        kernel_op.set_nodeattr("SIMD", simd)
        configured = kernel_op.get_design_point(model_w)

        # Simulate profiling (just capture configuration)
        results.append(
            {
                "simd": simd,
                "stream_width": configured.input_list[0].stream_width_bits,
                "cycles": configured.initiation_interval,
            }
        )

    assert len(results) == len(filtered_simd), "All filtered configs should be valid"

    # Step 4: Select "best" (e.g., highest SIMD that fits)
    best = max(results, key=lambda r: r["simd"])
    print(f"  Best config: SIMD={best['simd']}, cycles={best['cycles']}")

    # Step 5: Apply best configuration
    kernel_op.set_nodeattr("SIMD", best["simd"])
    final_instance = kernel_op.get_design_point(model_w)

    assert final_instance.params["SIMD"] == best["simd"]


def test_multi_kernel_dse_simulation():
    """Test DSE with multiple kernel instances (simulates full model DSE)."""
    # Create multiple LayerNorm instances (simulating multi-layer model)
    num_layers = 3
    kernel_ops = []

    for i in range(num_layers):
        model_w, node = create_layernorm_model()
        node.name = f"layernorm_{i}"
        kernel_op = LayerNorm(node)
        kernel_ops.append((kernel_op, model_w))

    # Explore same configuration across all layers
    test_simd_values = [1, 8, 64, 256]

    for simd in test_simd_values:
        for kernel_op, model_w in kernel_ops:
            kernel_op.set_nodeattr("SIMD", simd)
            configured = kernel_op.get_design_point(model_w)
            assert configured.params["SIMD"] == simd

    print(f"\n  Successfully configured {num_layers} kernels × {len(test_simd_values)} configs")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
