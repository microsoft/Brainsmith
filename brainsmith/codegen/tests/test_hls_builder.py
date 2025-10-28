# Portions derived from FINN project
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# Licensed under BSD-3-Clause License
#
# Modifications and additions Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for HLSCodeBuilder."""

import pytest
from brainsmith.codegen.hls_builder import HLSCodeBuilder


class TestHLSCodeBuilderBasics:
    """Test basic builder functionality."""

    def test_empty_builder(self):
        """Empty builder generates empty code."""
        builder = HLSCodeBuilder()
        assert builder.generate() == []

    def test_comment(self):
        """Comment method generates C++ comment."""
        builder = HLSCodeBuilder()
        builder.comment("Test comment")
        assert builder.generate() == ["// Test comment"]

    def test_blank_line(self):
        """Blank line method adds empty line."""
        builder = HLSCodeBuilder()
        builder.comment("First")
        builder.blank_line()
        builder.comment("Second")
        assert builder.generate() == ["// First", "", "// Second"]

    def test_pragma(self):
        """Pragma method generates HLS pragma."""
        builder = HLSCodeBuilder()
        builder.pragma("PIPELINE II=1")
        assert builder.generate() == ["#pragma HLS PIPELINE II=1"]

    def test_method_chaining(self):
        """Methods return self for chaining."""
        builder = HLSCodeBuilder()
        result = builder.comment("Test").blank_line().pragma("UNROLL")
        assert result is builder
        assert len(builder.generate()) == 3


class TestArrayDeclaration:
    """Test array declaration functionality."""

    def test_declare_1d_array(self):
        """Declare 1D array."""
        builder = HLSCodeBuilder()
        builder.declare_array("buf", "int", (16,))
        assert builder.generate() == ["int buf[16];"]

    def test_declare_2d_array(self):
        """Declare 2D array."""
        builder = HLSCodeBuilder()
        builder.declare_array("matrix", "float", (8, 16))
        assert builder.generate() == ["float matrix[8][16];"]

    def test_declare_array_with_pragma(self):
        """Declare array with partition pragma."""
        builder = HLSCodeBuilder()
        builder.declare_array(
            "buf",
            "ap_uint<8>",
            (32,),
            "ARRAY_PARTITION variable=buf complete"
        )
        code = builder.generate()
        assert code == [
            "ap_uint<8> buf[32];",
            "#pragma HLS ARRAY_PARTITION variable=buf complete"
        ]


class TestAssignments:
    """Test assignment generation."""

    def test_simple_assignment(self):
        """Generate simple assignment."""
        builder = HLSCodeBuilder()
        builder.assign("x", "5")
        assert builder.generate() == ["x = 5;"]

    def test_expression_assignment(self):
        """Generate assignment with expression."""
        builder = HLSCodeBuilder()
        builder.assign("result", "a + b * c")
        assert builder.generate() == ["result = a + b * c;"]


class TestStreamOperations:
    """Test stream read/write operations."""

    def test_stream_read(self):
        """Generate stream read."""
        builder = HLSCodeBuilder()
        builder.stream_read("in_stream", "val")
        assert builder.generate() == ["in_stream.read(val);"]

    def test_stream_write(self):
        """Generate stream write."""
        builder = HLSCodeBuilder()
        builder.stream_write("out_stream", "result[i]")
        assert builder.generate() == ["out_stream.write(result[i]);"]


class TestForLoops:
    """Test for loop generation with context manager."""

    def test_empty_for_loop(self):
        """Generate empty for loop."""
        builder = HLSCodeBuilder()
        with builder.for_loop("i", "N"):
            pass
        code = builder.generate()
        assert code == [
            "for (unsigned i = 0; i < N; i++) {",
            "}"
        ]

    def test_for_loop_with_body(self):
        """Generate for loop with statements."""
        builder = HLSCodeBuilder()
        with builder.for_loop("i", "10"):
            builder.assign("sum", "sum + arr[i]")
        code = builder.generate()
        assert code == [
            "for (unsigned i = 0; i < 10; i++) {",
            "    sum = sum + arr[i];",
            "}"
        ]

    def test_for_loop_with_unroll(self):
        """Generate for loop with UNROLL pragma."""
        builder = HLSCodeBuilder()
        with builder.for_loop("pe", "PE", unroll=True):
            builder.assign("out[pe]", "lhs[pe] + rhs[pe]")
        code = builder.generate()
        assert code == [
            "for (unsigned pe = 0; pe < PE; pe++) {",
            "    #pragma HLS UNROLL",
            "    out[pe] = lhs[pe] + rhs[pe];",
            "}"
        ]

    def test_nested_for_loops(self):
        """Generate nested for loops with proper indentation."""
        builder = HLSCodeBuilder()
        with builder.for_loop("i", "M"):
            with builder.for_loop("j", "N"):
                builder.assign("matrix[i][j]", "0")
        code = builder.generate()
        assert code == [
            "for (unsigned i = 0; i < M; i++) {",
            "    for (unsigned j = 0; j < N; j++) {",
            "        matrix[i][j] = 0;",
            "    }",
            "}"
        ]


class TestIfBlocks:
    """Test if block generation."""

    def test_empty_if_block(self):
        """Generate empty if block."""
        builder = HLSCodeBuilder()
        with builder.if_block("x > 0"):
            pass
        code = builder.generate()
        assert code == [
            "if (x > 0) {",
            "}"
        ]

    def test_if_block_with_body(self):
        """Generate if block with statements."""
        builder = HLSCodeBuilder()
        with builder.if_block("read_en"):
            builder.stream_read("in_strm", "val")
            builder.assign("buffer[idx]", "val")
        code = builder.generate()
        assert code == [
            "if (read_en) {",
            "    in_strm.read(val);",
            "    buffer[idx] = val;",
            "}"
        ]

    def test_nested_if_blocks(self):
        """Generate nested if blocks."""
        builder = HLSCodeBuilder()
        with builder.if_block("a > 0"):
            with builder.if_block("b > 0"):
                builder.assign("result", "a + b")
        code = builder.generate()
        assert code == [
            "if (a > 0) {",
            "    if (b > 0) {",
            "        result = a + b;",
            "    }",
            "}"
        ]


class TestComplexScenarios:
    """Test complex multi-level code generation."""

    def test_for_loop_with_if(self):
        """Generate for loop containing if block."""
        builder = HLSCodeBuilder()
        with builder.for_loop("i", "N"):
            with builder.if_block("i % 2 == 0"):
                builder.assign("sum", "sum + arr[i]")
        code = builder.generate()
        assert code == [
            "for (unsigned i = 0; i < N; i++) {",
            "    if (i % 2 == 0) {",
            "        sum = sum + arr[i];",
            "    }",
            "}"
        ]

    def test_realistic_elementwise_loop(self):
        """Generate realistic elementwise operation code."""
        builder = HLSCodeBuilder()
        builder.comment("Elementwise binary operation: Add")
        builder.declare_array("out", "OutType", (8,))
        builder.pragma("ARRAY_PARTITION variable=out complete dim=1")
        builder.blank_line()

        with builder.for_loop("i", "N"):
            builder.pragma("PIPELINE II=1")
            builder.stream_read("lhs_stream", "lhs_val")
            builder.stream_read("rhs_stream", "rhs_val")
            with builder.for_loop("pe", "PE", unroll=True):
                builder.assign("out[pe]", "lhs_val[pe] + rhs_val[pe]")
            builder.stream_write("out_stream", "out")

        code = builder.generate()
        assert len(code) == 14  # Verify realistic code length
        assert "// Elementwise binary operation: Add" in code
        assert "for (unsigned i = 0; i < N; i++) {" in code
        assert "    #pragma HLS PIPELINE II=1" in code


class TestRawCode:
    """Test raw code insertion."""

    def test_raw_code(self):
        """Insert raw C++ code."""
        builder = HLSCodeBuilder()
        builder.raw("const int MAX = 100;")
        assert builder.generate() == ["const int MAX = 100;"]

    def test_raw_code_with_indentation(self):
        """Raw code respects current indentation."""
        builder = HLSCodeBuilder()
        with builder.for_loop("i", "N"):
            builder.raw("// Custom code")
            builder.raw("custom_function(i);")
        code = builder.generate()
        assert code == [
            "for (unsigned i = 0; i < N; i++) {",
            "    // Custom code",
            "    custom_function(i);",
            "}"
        ]


class TestCustomIndentation:
    """Test custom indentation strings."""

    def test_custom_indent_2_spaces(self):
        """Use 2-space indentation."""
        builder = HLSCodeBuilder(indent_str="  ")
        with builder.for_loop("i", "N"):
            builder.assign("x", "i")
        code = builder.generate()
        assert code == [
            "for (unsigned i = 0; i < N; i++) {",
            "  x = i;",
            "}"
        ]

    def test_custom_indent_tabs(self):
        """Use tab indentation."""
        builder = HLSCodeBuilder(indent_str="\t")
        with builder.for_loop("i", "N"):
            builder.assign("x", "i")
        code = builder.generate()
        assert code == [
            "for (unsigned i = 0; i < N; i++) {",
            "\tx = i;",
            "}"
        ]
