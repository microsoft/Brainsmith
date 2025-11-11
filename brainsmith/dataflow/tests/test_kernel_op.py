############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Tests for KernelOp two-phase caching system (Phase 3).

These tests validate the integration of the two-phase kernel construction
system into KernelOp, ensuring:
- Efficient DSE performance via two-phase caching
- Backward compatibility with existing code
- Correct cache invalidation behavior
"""

import time
from unittest.mock import MagicMock, patch

import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

import brainsmith.dataflow as df

# Test fixture: Simple kernel schema for testing (matches 3D tensor [1, 1, 768])
TEST_SCHEMA = df.KernelSchema(
    name="TestKernel",
    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[df.FULL_DIM, df.FULL_DIM, df.FULL_DIM],  # 3D tensor
            stream_tiling=[1, 1, "SIMD"],  # Stream last dimension only
        )
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[df.FULL_DIM, df.FULL_DIM, df.FULL_DIM],
            stream_tiling=[1, 1, df.DerivedDim("input", -1)],  # Match input streaming
            datatype=df.DerivedDatatype("input"),
        )
    ],
    constraints=[
        df.IsDynamic("input"),
    ],
)


# Test kernel implementation
class TestKernel(df.KernelOp):
    """Simple test kernel for KernelOp caching tests."""

    @classmethod
    def build_schema(cls, node, model):
        return TEST_SCHEMA

    @classmethod
    def get_inference_pattern(cls):
        return df.InferencePattern(source_ops=["Test"])

    @classmethod
    def infer_from(cls, node, model, insert_index):
        raise NotImplementedError("Not needed for these tests")

    def execute_node(self, context, graph):
        """Dummy execute_node for testing (not actually used)."""
        # Simple pass-through for testing
        node = self.onnx_node
        context[node.output[0]] = context[node.input[0]]


def create_test_model_and_node():
    """Create a minimal ONNX model and node for testing."""
    # Create simple model with one node
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 768])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 768])

    node = helper.make_node(
        "Test",
        inputs=["input"],
        outputs=["output"],
        name="test_node",
        domain="brainsmith.kernels",
        SIMD=1,
        input0Datatype="FLOAT32",
        output0Datatype="FLOAT32",
    )

    graph = helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])

    model = helper.make_model(graph)
    model_w = ModelWrapper(model)

    return model_w, node


# ====================================================================
# Test 1: Two-Phase Caching Tests
# ====================================================================


def test_design_space_cached_across_reconfigurations():
    """Test that design space is built once and reused across reconfigurations."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # First call builds design space
    invariant1 = kernel_op.get_design_space(model_w)

    # Second call returns same instance (cached)
    invariant2 = kernel_op.get_design_space(model_w)

    assert invariant1 is invariant2, "Invariant model should be cached"

    # Change parallelization param
    kernel_op.set_nodeattr("SIMD", 2)

    # Invariant model should still be cached (not invalidated)
    invariant3 = kernel_op.get_design_space(model_w)

    assert invariant1 is invariant3, "Invariant model should NOT be invalidated by param change"


def test_configuration_rebuilt_on_param_change():
    """Test that configured model is rebuilt when parallelization params change."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # First configuration
    kernel_op.set_nodeattr("SIMD", 1)
    configured1 = kernel_op.get_design_point(model_w)

    # Change param
    kernel_op.set_nodeattr("SIMD", 2)
    configured2 = kernel_op.get_design_point(model_w)

    # Should be different instances (reconfigured)
    assert configured1 is not configured2, "Configured model should be rebuilt on param change"

    # But should share same design space
    assert configured1.design_space is configured2.design_space, "Should share design space"


def test_configuration_cached_when_params_unchanged():
    """Test that configured model is cached when params don't change."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # First call
    configured1 = kernel_op.get_design_point(model_w)

    # Second call with same params
    configured2 = kernel_op.get_design_point(model_w)

    assert configured1 is configured2, "Configured model should be cached when params unchanged"


# ====================================================================
# Test 2: get_design_space() Tests
# ====================================================================


def test_get_design_space_builds_once():
    """Test that get_design_space() builds only on first call."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Patch module-level builder function to track calls
    with patch("brainsmith.dataflow.kernel_op.build_kernel_design_space") as mock_build:
        mock_build.return_value = MagicMock()

        # First call should build
        kernel_op.get_design_space(model_w)
        assert mock_build.call_count == 1

        # Second call should use cache
        kernel_op.get_design_space(model_w)
        assert mock_build.call_count == 1, "Should not rebuild (cache hit)"


def test_get_design_space_returns_same_instance():
    """Test that get_design_space() always returns same instance."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    instance1 = kernel_op.get_design_space(model_w)
    instance2 = kernel_op.get_design_space(model_w)
    instance3 = kernel_op.get_design_space(model_w)

    assert instance1 is instance2 is instance3, "Should return same instance"


# ====================================================================
# Test 3: get_design_point() Tests
# ====================================================================


def test_get_design_point_first_call_builds_both_models():
    """Test that first get_design_point() call builds both design_space and configured."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    assert kernel_op._design_space is None
    assert kernel_op._configuration is None

    # First call builds both
    configured = kernel_op.get_design_point(model_w)

    assert kernel_op._design_space is not None
    assert kernel_op._configuration is not None
    assert configured is kernel_op._configuration


def test_get_design_point_cache_hit_with_same_params():
    """Test that get_design_point() returns cached instance with unchanged params."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    configured1 = kernel_op.get_design_point(model_w)
    configured2 = kernel_op.get_design_point(model_w)

    assert configured1 is configured2, "Should return cached model"


def test_get_design_point_reconfigures_with_new_params():
    """Test that get_design_point() reconfigures when params change."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Initial configuration
    configured1 = kernel_op.get_design_point(model_w)
    assert configured1.params["SIMD"] == 1

    # Change param
    kernel_op.set_nodeattr("SIMD", 2)
    configured2 = kernel_op.get_design_point(model_w)

    assert configured2.params["SIMD"] == 2
    assert configured1 is not configured2, "Should reconfigure"


# ====================================================================
# Test 4: get_valid_ranges() Tests
# ====================================================================


def test_get_valid_ranges_returns_divisor_sets():
    """Test that get_valid_ranges() returns correct divisor sets."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    valid_ranges = kernel_op.get_valid_ranges(model_w)

    assert "SIMD" in valid_ranges
    assert isinstance(valid_ranges["SIMD"], set)

    # For channel dimension of 768, should have divisors
    expected_divisors = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 768}
    assert valid_ranges["SIMD"] == expected_divisors


def test_valid_ranges_match_block_shapes():
    """Test that valid ranges are divisors of block shape dimensions."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    design_space = kernel_op.get_design_space(model_w)
    valid_ranges = kernel_op.get_valid_ranges(model_w)

    # Get block shape for input (use input_list for positional access)
    block_shape = design_space.input_list[0].block_shape

    # All valid SIMD values should divide the last dimension
    for simd in valid_ranges["SIMD"]:
        assert (
            block_shape[-1] % simd == 0
        ), f"SIMD={simd} should divide block_shape[-1]={block_shape[-1]}"


# ====================================================================
# Test 5: set_nodeattr() Invalidation Tests
# ====================================================================


def test_set_nodeattr_invalidates_configured_only():
    """Test that set_nodeattr() invalidates only configured model, not design_space."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Build both models
    kernel_op.get_design_point(model_w)
    invariant_before = kernel_op._design_space

    # Change param
    kernel_op.set_nodeattr("SIMD", 2)

    # Invariant should be unchanged
    assert kernel_op._design_space is invariant_before, "Invariant should NOT be invalidated"

    # Configured should be invalidated
    assert kernel_op._configuration is None, "Configured should be invalidated"
    assert kernel_op._current_params is None, "Current params should be cleared"


def test_set_nodeattr_preserves_design_space():
    """Test that design space is never invalidated by set_nodeattr()."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Build models
    kernel_op.get_design_point(model_w)
    invariant_original = kernel_op._design_space

    # Change params multiple times
    for simd in [2, 4, 8, 16, 32]:
        kernel_op.set_nodeattr("SIMD", simd)
        assert (
            kernel_op._design_space is invariant_original
        ), f"Invariant should persist (SIMD={simd})"


def test_set_nodeattr_no_op_when_value_unchanged():
    """Test that set_nodeattr() is no-op when value doesn't change."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Build models
    configured = kernel_op.get_design_point(model_w)

    # Set same value
    kernel_op.set_nodeattr("SIMD", 1)

    # Configured should still be cached (no change)
    assert kernel_op._configuration is configured, "Should not invalidate when value unchanged"


# ====================================================================
# Test 6: Performance Tests
# ====================================================================


def test_reconfiguration_performance_target_1ms():
    """Test that reconfiguration completes in <1ms (target performance)."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Build design space first
    kernel_op.get_design_space(model_w)

    # Measure reconfiguration time
    kernel_op.set_nodeattr("SIMD", 2)
    start = time.time()
    kernel_op.get_design_point(model_w)
    elapsed = time.time() - start

    # Target: <1ms (0.001s), use generous 5ms for CI variability
    assert elapsed < 0.005, f"Reconfiguration took {elapsed*1000:.2f}ms, target <5ms"


def test_100_configurations_under_100ms():
    """Test that 100 DSE configurations complete in <100ms."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Get valid SIMD values
    valid_simd = kernel_op.get_valid_ranges(model_w)["SIMD"]
    simd_values = sorted(valid_simd)

    # Limit to 100 configs (repeat if needed)
    configs = (simd_values * (100 // len(simd_values) + 1))[:100]

    # Measure DSE simulation
    start = time.time()
    for simd in configs:
        kernel_op.set_nodeattr("SIMD", simd)
        kernel_op.get_design_point(model_w)
    elapsed = time.time() - start

    # Target: <100ms, use generous 500ms for CI variability
    assert elapsed < 0.5, f"100 configs took {elapsed*1000:.2f}ms, target <500ms"

    print(
        f"  Performance: {len(configs)} configs in {elapsed*1000:.1f}ms ({elapsed*1000/len(configs):.2f}ms avg)"
    )


def test_cache_hit_under_01ms():
    """Test that cache hit (same params) completes in <0.1ms."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Build initial model
    kernel_op.get_design_point(model_w)

    # Measure cache hit time (average over 100 calls to reduce timing noise)
    start = time.time()
    for _ in range(100):
        kernel_op.get_design_point(model_w)
    elapsed = time.time() - start

    avg_time = elapsed / 100

    # Target: <0.1ms (0.0001s) average, use generous 1ms for CI variability
    assert avg_time < 0.001, f"Cache hit took {avg_time*1000:.2f}ms avg, target <1ms"


# ====================================================================
# Test 7: Backward Compatibility Tests
# ====================================================================


def test_legacy_design_point_property_still_works():
    """Test that legacy design_point property still works."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Build model via get_design_point()
    configured = kernel_op.get_design_point(model_w)

    # Access via property should return same instance
    assert kernel_op.design_point is configured


def test_shape_queries_delegate_correctly():
    """Test that shape queries work correctly with KernelDesignPoint."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Build model
    kernel_op.get_design_point(model_w)

    # Test shape queries
    tensor_shape = kernel_op.get_normal_input_shape(ind=0, model_w=model_w)
    assert tensor_shape == [1, 1, 768]

    output_shape = kernel_op.get_normal_output_shape(ind=0, model_w=model_w)
    assert output_shape == [1, 1, 768]


def test_datatype_queries_delegate_correctly():
    """Test that datatype queries work correctly with KernelDesignPoint."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Build model
    kernel_op.get_design_point(model_w)

    # Test datatype queries
    input_dt = kernel_op.get_input_datatype(ind=0)
    assert input_dt == DataType["FLOAT32"]

    output_dt = kernel_op.get_output_datatype(ind=0)
    assert output_dt == DataType["FLOAT32"]


# ====================================================================
# Test 8: Error Handling Tests
# ====================================================================


def test_get_design_space_requires_model_wrapper():
    """Test that get_design_space() raises error without ModelWrapper."""
    _, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    with pytest.raises(Exception) as exc_info:
        kernel_op.get_design_space(None)

    assert "ModelWrapper" in str(exc_info.value)


def test_design_point_property_requires_prior_build():
    """Test that design_point property raises error if not built."""
    _, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    with pytest.raises(RuntimeError) as exc_info:
        _ = kernel_op.design_point

    assert "get_design_point" in str(exc_info.value) or "build_design_space" in str(exc_info.value)


# ====================================================================
# Test 9: Property-Based API Tests
# ====================================================================


def test_design_space_property_before_build_raises():
    """Test that design_space property raises before build_design_space()."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    with pytest.raises(RuntimeError) as exc_info:
        _ = kernel_op.design_space

    assert "build_design_space" in str(exc_info.value)


def test_design_point_property_before_build_raises():
    """Test that design_point property raises before build_design_space()."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    with pytest.raises(RuntimeError) as exc_info:
        _ = kernel_op.design_point

    assert "build_design_space" in str(exc_info.value)


def test_build_design_space_enables_properties():
    """Test that build_design_space() enables property access."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Initialize by calling a public API method
    kernel_op.infer_node_datatype(model_w)

    # Properties should work
    assert kernel_op.design_space is not None
    assert kernel_op.design_point is not None
    assert kernel_op.design_space.name == "TestKernel"


def test_build_design_space_idempotent():
    """Test that initialization is idempotent (doesn't rebuild when called multiple times)."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Build twice
    kernel_op.infer_node_datatype(model_w)
    ds1 = kernel_op.design_space
    km1 = kernel_op.design_point

    kernel_op.infer_node_datatype(model_w)
    ds2 = kernel_op.design_space
    km2 = kernel_op.design_point

    # Should return same instances (no rebuild)
    assert ds1 is ds2
    assert km1 is km2


# ====================================================================
# Test 10: Schema-Based Invalidation Tests
# ====================================================================


def test_structural_nodeattr_invalidates_design_space():
    """Test that structural nodeattr changes invalidate design space."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Build
    kernel_op.infer_node_datatype(model_w)
    ds1 = kernel_op.design_space

    # Change structural nodeattr (datatype)
    kernel_op.set_nodeattr("input0Datatype", "INT8")

    # Design space should be invalidated
    assert kernel_op._design_space is None

    # Rebuild
    kernel_op.infer_node_datatype(model_w)
    ds2 = kernel_op.design_space

    # Should be different instance
    assert ds1 is not ds2


def test_parametric_nodeattr_preserves_design_space():
    """Test that parametric nodeattr changes preserve design space."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Build
    kernel_op.infer_node_datatype(model_w)
    ds1 = kernel_op.design_space
    km1 = kernel_op.design_point

    # Change parametric nodeattr (SIMD)
    kernel_op.set_nodeattr("SIMD", 2)

    # Design space should NOT be invalidated
    assert kernel_op._design_space is ds1

    # But configuration should be invalidated
    assert kernel_op._configuration is None

    # Rebuild
    kernel_op.infer_node_datatype(model_w)
    ds2 = kernel_op.design_space
    km2 = kernel_op.design_point

    # Same design space, different configuration
    assert ds1 is ds2
    assert km1 is not km2


def test_execution_nodeattr_invalidates_config_only():
    """Test that execution nodeattr changes invalidate config only."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Add execution parameter to schema for testing
    kernel_op.kernel_schema.kernel_params["epsilon"] = ("f", True, 1e-5)

    # Build
    kernel_op.infer_node_datatype(model_w)
    ds1 = kernel_op.design_space

    # Change execution nodeattr
    kernel_op.set_nodeattr("epsilon", 1e-6)

    # Design space should NOT be invalidated
    assert kernel_op._design_space is ds1

    # But configuration should be invalidated (conservative)
    assert kernel_op._configuration is None


# ====================================================================
# Test 11: Explicit Invalidation Tests
# ====================================================================


def test_explicit_invalidation_works():
    """Test that invalidate_design_space() works."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Build
    kernel_op.infer_node_datatype(model_w)
    ds1 = kernel_op.design_space

    # Explicitly invalidate
    kernel_op.invalidate()

    # Should be invalidated
    assert kernel_op._design_space is None

    # Rebuild
    kernel_op.infer_node_datatype(model_w)
    ds2 = kernel_op.design_space

    # Different instance
    assert ds1 is not ds2


def test_build_after_invalidation_rebuilds():
    """Test that build_design_space() rebuilds after invalidation."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Build
    kernel_op.infer_node_datatype(model_w)

    # Invalidate
    kernel_op.invalidate()

    # Properties should raise
    with pytest.raises(RuntimeError):
        _ = kernel_op.design_space
    with pytest.raises(RuntimeError):
        _ = kernel_op.design_point

    # Rebuild
    kernel_op.infer_node_datatype(model_w)

    # Properties should work again
    ds2 = kernel_op.design_space
    km2 = kernel_op.design_point
    assert ds2 is not None
    assert km2 is not None


# ====================================================================
# Test 12: Backward Compatibility Tests
# ====================================================================


def test_old_api_still_works():
    """Test that old get_design_space() / get_design_point() still work."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Old API should still work
    ds = kernel_op.get_design_space(model_w)
    km = kernel_op.get_design_point(model_w)

    assert ds is not None
    assert km is not None


def test_get_valid_ranges_builds_automatically():
    """Test that get_valid_ranges() builds design space automatically."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Should build automatically
    valid_ranges = kernel_op.get_valid_ranges(model_w)

    assert "SIMD" in valid_ranges
    assert kernel_op._design_space is not None


def test_infer_node_datatype_builds_automatically():
    """Test that infer_node_datatype() builds design space automatically."""
    model_w, node = create_test_model_and_node()
    kernel_op = TestKernel(node)

    # Should build automatically
    kernel_op.infer_node_datatype(model_w)

    assert kernel_op._design_space is not None
    assert kernel_op._configuration is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
