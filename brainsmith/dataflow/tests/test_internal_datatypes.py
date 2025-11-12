############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Integration tests for internal datatype support in kernel schemas.

This module tests the complete flow of internal datatypes from schema
definition through resolution during model building.
"""

from dataclasses import dataclass

import pytest
from qonnx.core.datatype import DataType

from brainsmith.dataflow import (
    ComputedDatatype,
    # Datatype sources
    DerivedDatatype,
    # Schemas
    InputSchema,
    KernelSchema,
    OutputSchema,
    UnionDatatype,
    WidenedDatatype,
)


# Mock interface for testing
@dataclass
class MockInterface:
    """Mock interface for testing internal datatype resolution."""

    name: str
    datatype: DataType


@pytest.fixture
def param_getter():
    """Mock parameter getter."""
    params = {"PE": 16, "SIMD": 8}
    return lambda name: params.get(name)


# ============================================================================
# Internal Datatype Schema Definition Tests
# ============================================================================


class TestInternalDatatypeSchemaDefinition:
    """Tests for defining internal datatypes in KernelSchema."""

    def test_simple_internal_datatype(self):
        """Test schema with single internal datatype."""
        schema = KernelSchema(
            name="TestKernel",
            inputs=[InputSchema("input")],
            outputs=[OutputSchema("output")],
            internal_datatypes={"accumulator": DerivedDatatype("input")},
        )

        assert "accumulator" in schema.internal_datatypes
        assert isinstance(schema.internal_datatypes["accumulator"], DerivedDatatype)

    def test_multiple_internal_datatypes(self):
        """Test schema with multiple internal datatypes."""
        schema = KernelSchema(
            name="TestKernel",
            inputs=[InputSchema("input"), InputSchema("weight")],
            outputs=[OutputSchema("output")],
            internal_datatypes={
                "accumulator": WidenedDatatype("input", extra_bits=8),
                "bias": DerivedDatatype("input"),
            },
        )

        assert len(schema.internal_datatypes) == 2
        assert "accumulator" in schema.internal_datatypes
        assert "bias" in schema.internal_datatypes

    def test_internal_datatype_name_conflict_with_input(self):
        """Test that internal datatype names cannot conflict with input names."""
        with pytest.raises(ValueError, match="conflicts with interface name"):
            KernelSchema(
                name="TestKernel",
                inputs=[InputSchema("input")],
                outputs=[OutputSchema("output")],
                internal_datatypes={
                    "input": DerivedDatatype("input")  # Same name as input
                },
            )

    def test_internal_datatype_name_conflict_with_output(self):
        """Test that internal datatype names cannot conflict with output names."""
        with pytest.raises(ValueError, match="conflicts with interface name"):
            KernelSchema(
                name="TestKernel",
                inputs=[InputSchema("input")],
                outputs=[OutputSchema("output")],
                internal_datatypes={
                    "output": DerivedDatatype("input")  # Same name as output
                },
            )

    def test_protected_attr_names_includes_internals(self):
        """Test that protected_attr_names includes internal datatype attributes."""
        schema = KernelSchema(
            name="TestKernel",
            inputs=[InputSchema("input")],
            outputs=[OutputSchema("output")],
            internal_datatypes={
                "accumulator": DerivedDatatype("input"),
                "bias": DerivedDatatype("input"),
            },
        )

        protected = schema.protected_attr_names
        assert "_accumulatorDatatype" in protected
        assert "_biasDatatype" in protected
        assert "_input0Datatype" in protected
        assert "_output0Datatype" in protected

    def test_get_nodeattr_types_includes_internals(self):
        """Test that get_nodeattr_types includes internal datatype attributes."""
        schema = KernelSchema(
            name="TestKernel",
            inputs=[InputSchema("input")],
            outputs=[OutputSchema("output")],
            internal_datatypes={"accumulator": DerivedDatatype("input")},
        )

        nodeattr_types = schema.get_nodeattr_types()

        # Check internal datatype attribute is registered
        assert "_accumulatorDatatype" in nodeattr_types
        attr_type, required, default = nodeattr_types["_accumulatorDatatype"]
        assert attr_type == "s"  # String type
        assert required is False  # Protected attributes are not required
        assert default == ""


# ============================================================================
# Internal Datatype Resolution Tests
# ============================================================================


class TestInternalDatatypeResolution:
    """Tests for resolving internal datatypes during model building."""

    def test_derived_internal_from_input(self, param_getter):
        """Test DerivedDatatype for internal referencing input."""
        interfaces = {"input": MockInterface("input", DataType["INT8"])}

        source = DerivedDatatype("input")
        result = source.resolve(interfaces, param_getter)
        assert result == DataType["INT8"]

    def test_widened_internal_from_input(self, param_getter):
        """Test WidenedDatatype for internal referencing input."""
        interfaces = {"input": MockInterface("input", DataType["INT8"])}

        source = WidenedDatatype("input", extra_bits=8)
        result = source.resolve(interfaces, param_getter)
        assert result == DataType["INT16"]  # INT8 + 8 bits

    def test_union_internal_from_multiple_inputs(self, param_getter):
        """Test UnionDatatype for internal referencing multiple inputs."""
        interfaces = {
            "input0": MockInterface("input0", DataType["INT4"]),
            "input1": MockInterface("input1", DataType["UINT4"]),
            "input2": MockInterface("input2", DataType["INT8"]),
        }

        source = UnionDatatype(("input0", "input1", "input2"))
        result = source.resolve(interfaces, param_getter)
        # Union of INT4 [-8,7], UINT4 [0,15], INT8 [-128,127] = INT8
        assert result == DataType["INT8"]

    def test_computed_internal_custom_logic(self, param_getter):
        """Test ComputedDatatype for internal with custom logic."""
        interfaces = {
            "input": MockInterface("input", DataType["INT8"]),
            "weight": MockInterface("weight", DataType["INT4"]),
        }

        def compute_matmul_acc(ifs, pg):
            """Compute accumulator for MatMul: wider of input + weight."""
            input_bits = ifs["input"].datatype.bitwidth()
            weight_bits = ifs["weight"].datatype.bitwidth()
            acc_bits = input_bits + weight_bits
            return DataType[f"INT{acc_bits}"]

        source = ComputedDatatype(compute_matmul_acc, "MatMul accumulator")
        result = source.resolve(interfaces, param_getter)
        assert result == DataType["INT12"]  # 8 + 4 bits

    def test_internal_not_found_error_message(self, param_getter):
        """Test that error message mentions interfaces/internals."""
        interfaces = {"input": MockInterface("input", DataType["INT8"])}

        source = DerivedDatatype("nonexistent")
        with pytest.raises(ValueError, match="Source.*'nonexistent' not found"):
            source.resolve(interfaces, param_getter)

        # Error should list available interfaces
        with pytest.raises(ValueError, match="Available interfaces/internals: input"):
            source.resolve(interfaces, param_getter)


# ============================================================================
# Output Referencing Internal Datatype Tests
# ============================================================================


class TestOutputReferencingInternalDatatype:
    """Tests for outputs referencing internal datatypes."""

    def test_output_derived_from_internal(self, param_getter):
        """Test output deriving datatype from internal."""
        # Simulate after internal resolution
        interfaces = {
            "input": MockInterface("input", DataType["INT8"]),
            "accumulator": MockInterface("accumulator", DataType["INT16"]),
        }

        # Output references internal
        source = DerivedDatatype("accumulator")
        result = source.resolve(interfaces, param_getter)
        assert result == DataType["INT16"]

    def test_output_widened_from_internal(self, param_getter):
        """Test output widening internal datatype."""
        # Simulate after internal resolution
        interfaces = {
            "input": MockInterface("input", DataType["INT8"]),
            "bias": MockInterface("bias", DataType["INT8"]),
        }

        # Output widens internal bias
        source = WidenedDatatype("bias", extra_bits=1)
        result = source.resolve(interfaces, param_getter)
        assert result == DataType["INT9"]

    def test_output_union_of_input_and_internal(self, param_getter):
        """Test output as union of input and internal datatypes."""
        # Simulate after internal resolution
        interfaces = {
            "input": MockInterface("input", DataType["INT8"]),
            "bias": MockInterface("bias", DataType["INT16"]),
        }

        # Output is union of input and internal
        source = UnionDatatype(("input", "bias"))
        result = source.resolve(interfaces, param_getter)
        assert result == DataType["INT16"]  # Wider type wins


# ============================================================================
# End-to-End Scenario Tests
# ============================================================================


class TestInternalDatatypeEndToEnd:
    """End-to-end tests for realistic kernel scenarios with internal datatypes."""

    def test_matmul_with_accumulator_internal(self, param_getter):
        """Test MatMul-like kernel with accumulator internal datatype."""
        schema = KernelSchema(
            name="MatMul",
            inputs=[InputSchema("input"), InputSchema("weight", is_weight=True)],
            outputs=[
                OutputSchema(
                    "output",
                    datatype=DerivedDatatype("accumulator"),  # Output uses accumulator
                )
            ],
            internal_datatypes={
                "accumulator": ComputedDatatype(
                    lambda ifs, pg: DataType[
                        f"INT{ifs['input'].datatype.bitwidth() + ifs['weight'].datatype.bitwidth() + 8}"
                    ],
                    "MatMul accumulator with headroom",
                )
            },
        )

        # Verify schema structure
        assert len(schema.internal_datatypes) == 1
        assert "accumulator" in schema.internal_datatypes
        assert schema.outputs[0].datatype.source_interface == "accumulator"

        # Simulate resolution
        interfaces = {
            "input": MockInterface("input", DataType["INT8"]),
            "weight": MockInterface("weight", DataType["INT4"]),
        }

        # Resolve internal
        acc_datatype = schema.internal_datatypes["accumulator"].resolve(interfaces, param_getter)
        assert acc_datatype == DataType["INT20"]  # 8 + 4 + 8 bits

        # Add internal to interfaces for output resolution
        interfaces["accumulator"] = MockInterface("accumulator", acc_datatype)

        # Resolve output
        output_datatype = schema.outputs[0].datatype.resolve(interfaces, param_getter)
        assert output_datatype == DataType["INT20"]

    def test_layernorm_with_bias_internal(self, param_getter):
        """Test LayerNorm-like kernel with bias internal datatype."""
        schema = KernelSchema(
            name="LayerNorm",
            inputs=[InputSchema("input")],
            outputs=[
                OutputSchema(
                    "output",
                    datatype=UnionDatatype(("input", "bias")),  # Output accommodates input and bias
                )
            ],
            internal_datatypes={
                "bias": DerivedDatatype("input")  # Bias has same type as input
            },
        )

        # Simulate resolution
        interfaces = {"input": MockInterface("input", DataType["INT8"])}

        # Resolve internal
        bias_datatype = schema.internal_datatypes["bias"].resolve(interfaces, param_getter)
        assert bias_datatype == DataType["INT8"]

        # Add internal to interfaces
        interfaces["bias"] = MockInterface("bias", bias_datatype)

        # Resolve output (union of INT8 and INT8 = INT8)
        output_datatype = schema.outputs[0].datatype.resolve(interfaces, param_getter)
        assert output_datatype == DataType["INT8"]

    def test_multiple_internals_chained(self, param_getter):
        """Test multiple internal datatypes where one references another indirectly."""
        # Note: Currently internals can only reference inputs, not other internals
        # This test documents current behavior and can be extended if chaining is added

        schema = KernelSchema(
            name="ComplexKernel",
            inputs=[InputSchema("input"), InputSchema("weight")],
            outputs=[
                OutputSchema(
                    "output",
                    datatype=UnionDatatype(("accumulator", "bias")),  # Output uses both internals
                )
            ],
            internal_datatypes={
                "accumulator": WidenedDatatype("input", extra_bits=8),
                "bias": DerivedDatatype("weight"),
            },
        )

        # Simulate resolution
        interfaces = {
            "input": MockInterface("input", DataType["INT8"]),
            "weight": MockInterface("weight", DataType["INT4"]),
        }

        # Resolve internals
        acc_datatype = schema.internal_datatypes["accumulator"].resolve(interfaces, param_getter)
        bias_datatype = schema.internal_datatypes["bias"].resolve(interfaces, param_getter)
        assert acc_datatype == DataType["INT16"]  # INT8 + 8 bits
        assert bias_datatype == DataType["INT4"]

        # Add internals to interfaces
        interfaces["accumulator"] = MockInterface("accumulator", acc_datatype)
        interfaces["bias"] = MockInterface("bias", bias_datatype)

        # Resolve output (union of INT16 and INT4 = INT16)
        output_datatype = schema.outputs[0].datatype.resolve(interfaces, param_getter)
        assert output_datatype == DataType["INT16"]


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestInternalDatatypeErrorHandling:
    """Tests for error handling with internal datatypes."""

    def test_internal_references_nonexistent_input(self, param_getter):
        """Test error when internal references non-existent input."""
        schema = KernelSchema(
            name="TestKernel",
            inputs=[InputSchema("input")],
            outputs=[OutputSchema("output")],
            internal_datatypes={"accumulator": DerivedDatatype("nonexistent")},
        )

        interfaces = {"input": MockInterface("input", DataType["INT8"])}

        # Should fail during resolution
        with pytest.raises(ValueError, match="'nonexistent' not found"):
            schema.internal_datatypes["accumulator"].resolve(interfaces, param_getter)

    def test_output_references_nonexistent_internal(self, param_getter):
        """Test error when output references non-existent internal."""
        # Note: This error happens at runtime, not schema definition time
        schema = KernelSchema(
            name="TestKernel",
            inputs=[InputSchema("input")],
            outputs=[OutputSchema("output", datatype=DerivedDatatype("nonexistent_internal"))],
            internal_datatypes={"accumulator": DerivedDatatype("input")},
        )

        interfaces = {
            "input": MockInterface("input", DataType["INT8"]),
            "accumulator": MockInterface("accumulator", DataType["INT16"]),
        }

        # Should fail when output tries to reference non-existent internal
        with pytest.raises(ValueError, match="'nonexistent_internal' not found"):
            schema.outputs[0].datatype.resolve(interfaces, param_getter)

    def test_internal_circular_reference_not_allowed(self, param_getter):
        """Test that internals cannot reference other internals (current limitation)."""
        # This documents current behavior: internals can only reference inputs

        schema = KernelSchema(
            name="TestKernel",
            inputs=[InputSchema("input")],
            outputs=[OutputSchema("output")],
            internal_datatypes={
                "accumulator": DerivedDatatype("input"),
                "bias": DerivedDatatype("accumulator"),  # Try to reference another internal
            },
        )

        interfaces = {"input": MockInterface("input", DataType["INT8"])}

        # First internal resolves fine
        acc_dt = schema.internal_datatypes["accumulator"].resolve(interfaces, param_getter)
        assert acc_dt == DataType["INT8"]

        # Second internal fails because "accumulator" not in interfaces yet
        # (internals are resolved in order, but they can't see each other)
        with pytest.raises(ValueError, match="'accumulator' not found"):
            schema.internal_datatypes["bias"].resolve(interfaces, param_getter)
