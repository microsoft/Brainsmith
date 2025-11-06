"""Tests for extended data generation with FP8 support."""

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from tests.support.data_generation import generate_test_data


class TestGenerateTestData:
    """Test data generation for all DataType formats."""

    def test_int8_type(self):
        """Test INT8 type generation."""
        dt = DataType["INT8"]
        data = generate_test_data(dt, (10, 10))

        assert data.dtype == np.float32
        assert data.shape == (10, 10)
        assert np.all(data >= dt.min())
        assert np.all(data <= dt.max())
        # Check we actually got integer values (in float32 container)
        assert np.all(data == np.floor(data))

    def test_arbitrary_bitwidth_int_types(self):
        """Test INT types with arbitrary bitwidths (INT9, INT11, etc.)."""
        for bitwidth in [9, 11, 16]:
            dt = DataType[f"INT{bitwidth}"]
            data = generate_test_data(dt, (10, 10))

            assert data.dtype == np.float32
            assert data.shape == (10, 10)
            assert np.all(data >= dt.min())
            assert np.all(data <= dt.max())
            assert np.all(data == np.floor(data))

    def test_uint_types(self):
        """Test UINT types."""
        for bitwidth in [4, 8, 16]:
            dt = DataType[f"UINT{bitwidth}"]
            data = generate_test_data(dt, (10, 10))

            assert data.dtype == np.float32
            assert np.all(data >= 0)
            assert np.all(data <= dt.max())
            assert np.all(data == np.floor(data))

    def test_bipolar(self):
        """Test BIPOLAR type."""
        data = generate_test_data(DataType["BIPOLAR"], (100,))

        assert data.dtype == np.float32
        # BIPOLAR should only have -1 and +1
        unique_values = set(data.flatten())
        assert unique_values.issubset({-1.0, 1.0})
        # Should have both values (with high probability)
        assert len(unique_values) == 2

    def test_binary(self):
        """Test BINARY type."""
        data = generate_test_data(DataType["BINARY"], (100,))

        assert data.dtype == np.float32
        # BINARY should only have 0 and 1
        unique_values = set(data.flatten())
        assert unique_values.issubset({0.0, 1.0})
        # Should have both values (with high probability)
        assert len(unique_values) == 2

    def test_float32(self):
        """Test standard FLOAT32."""
        data = generate_test_data(DataType["FLOAT32"], (10, 10))

        assert data.dtype == np.float32
        assert data.shape == (10, 10)
        # FLOAT32 uses normal distribution, just check it's reasonable
        assert np.all(np.abs(data) < 10.0)  # Very loose bound

    def test_float16(self):
        """Test FLOAT16."""
        data = generate_test_data(DataType["FLOAT16"], (10, 10))

        # gen_finn_dt_tensor returns float16 for FLOAT16
        assert data.dtype == np.float16
        assert data.shape == (10, 10)

    def test_arbprec_float_e5m2(self):
        """Test arbitrary precision float E5M2 format (common FP8)."""
        # E5M2: 5-bit exponent, 2-bit mantissa
        dt = DataType["FLOAT<5,2,15>"]
        data = generate_test_data(dt, (10, 10))

        assert data.dtype == np.float32
        assert data.shape == (10, 10)
        # Values should be within representable range
        assert np.all(np.abs(data) <= dt.max())

    def test_arbprec_float_custom(self):
        """Test custom arbitrary precision float format."""
        # Custom format: 5-bit exp, 10-bit mantissa, bias=15
        dt = DataType["FLOAT<5,10,15>"]
        data = generate_test_data(dt, (10, 10))

        assert data.dtype == np.float32
        assert data.shape == (10, 10)
        assert np.all(np.abs(data) <= dt.max())

    def test_mixed_types(self):
        """Test generating mixed-type inputs (INT + FLOAT)."""
        inputs = {
            "int_input": generate_test_data(DataType["INT8"], (16, 9)),
            "float_input": generate_test_data(DataType["FLOAT<5,10,15>"], (16, 9)),
            "bipolar_input": generate_test_data(DataType["BIPOLAR"], (16, 9)),
        }

        # All should have correct shapes
        for name, data in inputs.items():
            assert data.shape == (16, 9), f"{name} has wrong shape"

        # Check types
        assert inputs["int_input"].dtype == np.float32
        assert inputs["float_input"].dtype == np.float32
        assert inputs["bipolar_input"].dtype == np.float32

        # Check value ranges
        assert np.all(inputs["int_input"] >= -128)
        assert np.all(inputs["int_input"] <= 127)
        assert set(inputs["bipolar_input"].flatten()).issubset({-1.0, 1.0})

    def test_seed_reproducibility_int(self):
        """Test that seed produces reproducible results for INT types."""
        data1 = generate_test_data(DataType["INT8"], (10, 10), seed=42)
        data2 = generate_test_data(DataType["INT8"], (10, 10), seed=42)

        assert np.array_equal(data1, data2)

    def test_seed_reproducibility_float(self):
        """Test that seed produces reproducible results for FLOAT types."""
        data1 = generate_test_data(DataType["FLOAT<5,10,15>"], (10, 10), seed=42)
        data2 = generate_test_data(DataType["FLOAT<5,10,15>"], (10, 10), seed=42)

        assert np.array_equal(data1, data2)

    def test_seed_affects_randomness(self):
        """Test that different seeds produce different results."""
        data1 = generate_test_data(DataType["INT8"], (100,), seed=42)
        data2 = generate_test_data(DataType["INT8"], (100,), seed=43)

        # Should be different (with overwhelming probability)
        assert not np.array_equal(data1, data2)

    def test_shape_variations(self):
        """Test various tensor shapes."""
        shapes = [
            (10,),  # 1D
            (10, 10),  # 2D
            (5, 5, 5),  # 3D
            (2, 3, 4, 5),  # 4D
        ]

        for shape in shapes:
            data = generate_test_data(DataType["INT8"], shape)
            assert data.shape == shape

    @pytest.mark.skip(reason="Cannot easily create invalid DataType - QONNX validates at construction")
    def test_invalid_datatype(self):
        """Test that invalid DataType raises error."""
        # Note: QONNX validates DataTypes at construction time, so we can't
        # easily create an invalid one to test error handling
        pass

    def test_int_values_are_integers(self):
        """Verify INT types generate actual integer values (in float32 container)."""
        data = generate_test_data(DataType["INT9"], (100,))

        # All values should be integers
        assert np.all(data == np.round(data))

        # Values should be in correct range
        assert np.all(data >= DataType["INT9"].min())
        assert np.all(data <= DataType["INT9"].max())

    def test_float_values_can_be_fractional(self):
        """Verify FLOAT types can generate fractional values."""
        data = generate_test_data(DataType["FLOAT<5,10,15>"], (1000,))

        # At least some values should be non-integer (with high probability)
        # FP8 with 10-bit mantissa should have plenty of fractional values
        has_fractional = np.any(data != np.round(data))
        assert has_fractional, "Expected some fractional values in FP8 data"
