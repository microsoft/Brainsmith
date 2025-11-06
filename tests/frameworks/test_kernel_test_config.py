"""Unit tests for KernelTestConfig dataclass (v2.5)."""

import pytest
from qonnx.core.datatype import DataType

from tests.frameworks.test_config import KernelTestConfig


class TestKernelTestConfigValidation:
    """Test validation logic in KernelTestConfig.__post_init__()."""

    def test_minimal_valid_config(self):
        """Test minimal valid configuration."""
        config = KernelTestConfig(
            operation="Add",
            input_shapes={"input": (1, 64), "param": (1, 64)},
            input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
        )
        assert config.operation == "Add"
        assert config.input_shapes == {"input": (1, 64), "param": (1, 64)}

    def test_missing_operation_raises(self):
        """Test that missing operation raises ValueError."""
        with pytest.raises(ValueError, match="operation is required"):
            KernelTestConfig(
                operation="",
                input_shapes={"input": (1, 64)},
                input_dtypes={"input": DataType["INT8"]},
            )

    def test_missing_input_shapes_raises(self):
        """Test that missing input_shapes raises ValueError."""
        with pytest.raises(ValueError, match="input_shapes is required"):
            KernelTestConfig(
                operation="Add",
                input_shapes={},
                input_dtypes={"input": DataType["INT8"]},
            )

    def test_missing_input_dtypes_raises(self):
        """Test that missing input_dtypes raises ValueError."""
        with pytest.raises(ValueError, match="input_dtypes is required"):
            KernelTestConfig(
                operation="Add",
                input_shapes={"input": (1, 64)},
                input_dtypes={},
            )

    def test_mismatched_keys_raises(self):
        """Test that mismatched input_shapes and input_dtypes keys raise ValueError."""
        with pytest.raises(ValueError, match="do not match"):
            KernelTestConfig(
                operation="Add",
                input_shapes={"input": (1, 64), "param": (1, 64)},
                input_dtypes={"input": DataType["INT8"]},  # Missing "param"
            )

    def test_invalid_stream_index_raises(self):
        """Test that out-of-range stream index raises ValueError."""
        with pytest.raises(ValueError, match="input_streams index.*out of range"):
            KernelTestConfig(
                operation="Add",
                input_shapes={"input": (1, 64), "param": (1, 64)},
                input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
                input_streams={5: 8},  # Index 5 out of range [0, 1]
            )

    def test_negative_stream_index_raises(self):
        """Test that negative stream index raises ValueError."""
        with pytest.raises(ValueError, match="input_streams index.*out of range"):
            KernelTestConfig(
                operation="Add",
                input_shapes={"input": (1, 64)},
                input_dtypes={"input": DataType["INT8"]},
                input_streams={-1: 8},  # Negative index
            )


class TestKernelTestConfigDefaults:
    """Test default values in KernelTestConfig."""

    def test_default_backend_variants(self):
        """Test that backend_variants defaults to ["hls"]."""
        config = KernelTestConfig(
            operation="Add",
            input_shapes={"input": (1, 64)},
            input_dtypes={"input": DataType["INT8"]},
        )
        assert config.backend_variants == ["hls"]

    def test_default_stream_configs_none(self):
        """Test that stream configs default to None."""
        config = KernelTestConfig(
            operation="Add",
            input_shapes={"input": (1, 64)},
            input_dtypes={"input": DataType["INT8"]},
        )
        assert config.input_streams is None
        assert config.output_streams is None

    def test_default_tolerances_none(self):
        """Test that tolerances default to None."""
        config = KernelTestConfig(
            operation="Add",
            input_shapes={"input": (1, 64)},
            input_dtypes={"input": DataType["INT8"]},
        )
        assert config.tolerance_python is None
        assert config.tolerance_cppsim is None
        assert config.tolerance_rtlsim is None


class TestKernelTestConfigTestId:
    """Test auto-generation of test_id."""

    def test_auto_generated_test_id_basic(self):
        """Test auto-generated test_id for basic config."""
        config = KernelTestConfig(
            operation="Add",
            input_shapes={"input": (1, 64), "param": (1, 64)},
            input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
        )
        assert config.test_id == "add_int8_1-64"

    def test_auto_generated_test_id_with_stream(self):
        """Test auto-generated test_id includes stream config."""
        config = KernelTestConfig(
            operation="Add",
            input_shapes={"input": (1, 64), "param": (1, 64)},
            input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
            input_streams={0: 8},
        )
        assert config.test_id == "add_int8_1-64_pe8"

    def test_auto_generated_test_id_with_multiple_streams(self):
        """Test auto-generated test_id with multiple stream configs."""
        config = KernelTestConfig(
            operation="MatMul",
            input_shapes={"input": (1, 64), "weight": (64, 128)},
            input_dtypes={"input": DataType["INT8"], "weight": DataType["INT8"]},
            input_streams={0: 8},
            output_streams={0: 16},
        )
        assert "pe8" in config.test_id
        assert "ope16" in config.test_id

    def test_auto_generated_test_id_with_dse_dimensions(self):
        """Test auto-generated test_id includes key DSE dimensions."""
        config = KernelTestConfig(
            operation="MatMul",
            input_shapes={"input": (1, 64), "weight": (64, 128)},
            input_dtypes={"input": DataType["INT8"], "weight": DataType["INT8"]},
            dse_dimensions={"SIMD": 16, "mem_mode": "internal"},
        )
        assert "simd16" in config.test_id
        assert "mem_modeinternal" in config.test_id

    def test_explicit_test_id_not_overridden(self):
        """Test that explicit test_id is not overridden."""
        config = KernelTestConfig(
            operation="Add",
            input_shapes={"input": (1, 64)},
            input_dtypes={"input": DataType["INT8"]},
            test_id="my_custom_id",
        )
        assert config.test_id == "my_custom_id"

    def test_test_id_with_4d_shapes(self):
        """Test test_id generation with 4D shapes."""
        config = KernelTestConfig(
            operation="Conv",
            input_shapes={"input": (1, 8, 8, 32)},
            input_dtypes={"input": DataType["INT8"]},
        )
        assert config.test_id == "conv_int8_1-8-8-32"


class TestKernelTestConfigMethods:
    """Test helper methods in KernelTestConfig."""

    def test_get_tolerance_python(self):
        """Test get_tolerance() for python mode."""
        config = KernelTestConfig(
            operation="Add",
            input_shapes={"input": (1, 64)},
            input_dtypes={"input": DataType["INT8"]},
            tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        )
        assert config.get_tolerance("python") == {"rtol": 1e-7, "atol": 1e-9}

    def test_get_tolerance_cppsim(self):
        """Test get_tolerance() for cppsim mode."""
        config = KernelTestConfig(
            operation="Add",
            input_shapes={"input": (1, 64)},
            input_dtypes={"input": DataType["INT8"]},
            tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
        )
        assert config.get_tolerance("cppsim") == {"rtol": 1e-5, "atol": 1e-6}

    def test_get_tolerance_rtlsim(self):
        """Test get_tolerance() for rtlsim mode."""
        config = KernelTestConfig(
            operation="Add",
            input_shapes={"input": (1, 64)},
            input_dtypes={"input": DataType["INT8"]},
            tolerance_rtlsim={"rtol": 1e-3, "atol": 1e-4},
        )
        assert config.get_tolerance("rtlsim") == {"rtol": 1e-3, "atol": 1e-4}

    def test_get_tolerance_none(self):
        """Test get_tolerance() returns None when not configured."""
        config = KernelTestConfig(
            operation="Add",
            input_shapes={"input": (1, 64)},
            input_dtypes={"input": DataType["INT8"]},
        )
        assert config.get_tolerance("python") is None
        assert config.get_tolerance("cppsim") is None

    def test_get_tolerance_invalid_mode_raises(self):
        """Test get_tolerance() raises for invalid mode."""
        config = KernelTestConfig(
            operation="Add",
            input_shapes={"input": (1, 64)},
            input_dtypes={"input": DataType["INT8"]},
        )
        with pytest.raises(ValueError, match="Invalid execution_mode"):
            config.get_tolerance("invalid_mode")

    def test_has_backend_testing_true(self):
        """Test has_backend_testing() returns True when fpgapart configured."""
        config = KernelTestConfig(
            operation="Add",
            input_shapes={"input": (1, 64)},
            input_dtypes={"input": DataType["INT8"]},
            fpgapart="xc7z020clg400-1",
        )
        assert config.has_backend_testing() is True

    def test_has_backend_testing_false(self):
        """Test has_backend_testing() returns False when fpgapart not configured."""
        config = KernelTestConfig(
            operation="Add",
            input_shapes={"input": (1, 64)},
            input_dtypes={"input": DataType["INT8"]},
        )
        assert config.has_backend_testing() is False

    def test_clone_basic(self):
        """Test clone() creates modified copy."""
        config1 = KernelTestConfig(
            operation="Add",
            input_shapes={"input": (1, 64)},
            input_dtypes={"input": DataType["INT8"]},
            input_streams={0: 8},
            test_id="config1",
        )

        config2 = config1.clone(input_streams={0: 16}, test_id="config2")

        # Original unchanged
        assert config1.input_streams == {0: 8}
        assert config1.test_id == "config1"

        # Clone modified
        assert config2.input_streams == {0: 16}
        assert config2.test_id == "config2"

        # Other fields unchanged
        assert config2.operation == "Add"
        assert config2.input_shapes == {"input": (1, 64)}

    def test_clone_multiple_changes(self):
        """Test clone() with multiple field changes."""
        config1 = KernelTestConfig(
            operation="Add",
            input_shapes={"input": (1, 64)},
            input_dtypes={"input": DataType["INT8"]},
        )

        config2 = config1.clone(
            input_streams={0: 8},
            fpgapart="xc7z020clg400-1",
            tolerance_python={"rtol": 1e-5},
        )

        assert config2.input_streams == {0: 8}
        assert config2.fpgapart == "xc7z020clg400-1"
        assert config2.tolerance_python == {"rtol": 1e-5}


class TestKernelTestConfigFullExample:
    """Test complete configuration examples."""

    def test_full_config_with_backend_testing(self):
        """Test fully-configured example with backend testing."""
        config = KernelTestConfig(
            test_id="add_int8_1-64_pe8_simd16",
            operation="Add",
            input_shapes={"input": (1, 64), "param": (1, 64)},
            input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
            input_streams={0: 8},
            dse_dimensions={"SIMD": 16},
            fpgapart="xc7z020clg400-1",
            backend_variants=["hls"],
            tolerance_python={"rtol": 1e-7, "atol": 1e-9},
            tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
            tolerance_rtlsim={"rtol": 1e-3, "atol": 1e-4},
        )

        assert config.test_id == "add_int8_1-64_pe8_simd16"
        assert config.has_backend_testing() is True
        assert config.get_tolerance("python") == {"rtol": 1e-7, "atol": 1e-9}
        assert config.get_tolerance("cppsim") == {"rtol": 1e-5, "atol": 1e-6}

    def test_minimal_config_python_only(self):
        """Test minimal configuration for Python-only testing."""
        config = KernelTestConfig(
            operation="Add",
            input_shapes={"input": (1, 64), "param": (1, 64)},
            input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
        )

        assert config.test_id == "add_int8_1-64"
        assert config.has_backend_testing() is False
        assert config.backend_variants == ["hls"]
        assert config.tolerance_python is None
