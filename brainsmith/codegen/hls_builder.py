# Portions derived from FINN project
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# Licensed under BSD-3-Clause License
#
# Modifications and additions Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Structured HLS code generation with automatic indentation.

This module provides the HLSCodeBuilder class for generating well-formatted
HLS C++ code without manual indentation tracking.

Example:
    >>> builder = HLSCodeBuilder()
    >>> builder.comment("Main computation loop")
    >>> with builder.for_loop("i", "N"):
    ...     builder.stream_read("in_stream", "val")
    ...     builder.assign("out[i]", "val * 2")
    ...     builder.stream_write("out_stream", "out[i]")
    >>> code = builder.generate()
"""

from contextlib import contextmanager
from typing import Optional


class HLSCodeBuilder:
    """Structured HLS C++ code generation with auto-indentation.

    Provides a fluent API for generating HLS code with automatic indentation
    management through context managers. Eliminates manual string concatenation
    and indentation tracking.

    Attributes:
        indent_str: String used for one level of indentation (default: 4 spaces)
    """

    def __init__(self, indent_str: str = "    "):
        """Initialize code builder with empty code buffer.

        Args:
            indent_str: String for one indentation level (default: 4 spaces)
        """
        self._lines: list[str] = []
        self._indent_level: int = 0
        self._indent_str: str = indent_str

    def _add_line(self, line: str) -> None:
        """Add indented line to code buffer.

        Args:
            line: Code line to add (without indentation)
        """
        if line:  # Don't indent empty lines
            indented = self._indent_str * self._indent_level + line
            self._lines.append(indented)
        else:
            self._lines.append("")

    def comment(self, text: str) -> "HLSCodeBuilder":
        """Add C++ comment line.

        Args:
            text: Comment text (without // prefix)

        Returns:
            Self for method chaining
        """
        self._add_line(f"// {text}")
        return self

    def blank_line(self) -> "HLSCodeBuilder":
        """Add blank line for readability.

        Returns:
            Self for method chaining
        """
        self._lines.append("")
        return self

    def pragma(self, pragma_text: str) -> "HLSCodeBuilder":
        """Add HLS pragma directive.

        Args:
            pragma_text: Pragma content (without #pragma HLS prefix)

        Returns:
            Self for method chaining

        Example:
            >>> builder.pragma("PIPELINE II=1")
            # Generates: #pragma HLS PIPELINE II=1
        """
        self._add_line(f"#pragma HLS {pragma_text}")
        return self

    def declare_array(
        self,
        name: str,
        dtype: str,
        dims: tuple[int, ...],
        pragma: Optional[str] = None
    ) -> "HLSCodeBuilder":
        """Declare multi-dimensional array with optional pragma.

        Args:
            name: Variable name
            dtype: C++ type (e.g., "int", "ap_uint<8>")
            dims: Array dimensions as tuple
            pragma: Optional pragma for array (e.g., "ARRAY_PARTITION")

        Returns:
            Self for method chaining

        Example:
            >>> builder.declare_array("buffer", "int", (16, 32))
            # Generates: int buffer[16][32];
        """
        dim_str = "".join(f"[{d}]" for d in dims)
        self._add_line(f"{dtype} {name}{dim_str};")
        if pragma:
            self.pragma(pragma)
        return self

    def assign(self, lhs: str, rhs: str) -> "HLSCodeBuilder":
        """Generate assignment statement.

        Args:
            lhs: Left-hand side expression
            rhs: Right-hand side expression

        Returns:
            Self for method chaining

        Example:
            >>> builder.assign("result", "a + b")
            # Generates: result = a + b;
        """
        self._add_line(f"{lhs} = {rhs};")
        return self

    def stream_read(self, stream: str, var: str) -> "HLSCodeBuilder":
        """Generate HLS stream read operation.

        Args:
            stream: Stream variable name
            var: Variable to read into

        Returns:
            Self for method chaining

        Example:
            >>> builder.stream_read("in_strm", "val")
            # Generates: in_strm.read(val);
        """
        self._add_line(f"{stream}.read({var});")
        return self

    def stream_write(self, stream: str, expr: str) -> "HLSCodeBuilder":
        """Generate HLS stream write operation.

        Args:
            stream: Stream variable name
            expr: Expression to write

        Returns:
            Self for method chaining

        Example:
            >>> builder.stream_write("out_strm", "result[i]")
            # Generates: out_strm.write(result[i]);
        """
        self._add_line(f"{stream}.write({expr});")
        return self

    def raw(self, line: str) -> "HLSCodeBuilder":
        """Add raw C++ code line with current indentation.

        Use for code not covered by other methods.

        Args:
            line: C++ code line

        Returns:
            Self for method chaining
        """
        self._add_line(line)
        return self

    @contextmanager
    def for_loop(self, var: str, bound: str, unroll: bool = False):
        """Context manager for C++ for loop with auto-indentation.

        Args:
            var: Loop variable name
            bound: Loop bound expression
            unroll: Whether to add UNROLL pragma

        Yields:
            None (use context manager syntax)

        Example:
            >>> with builder.for_loop("i", "N"):
            ...     builder.assign("sum", "sum + arr[i]")
            # Generates:
            # for (unsigned i = 0; i < N; i++) {
            #     sum = sum + arr[i];
            # }
        """
        self._add_line(f"for (unsigned {var} = 0; {var} < {bound}; {var}++) {{")
        self._indent_level += 1
        if unroll:
            self.pragma("UNROLL")
        try:
            yield
        finally:
            self._indent_level -= 1
            self._add_line("}")

    @contextmanager
    def if_block(self, condition: str):
        """Context manager for C++ if block with auto-indentation.

        Args:
            condition: Boolean condition expression

        Yields:
            None (use context manager syntax)

        Example:
            >>> with builder.if_block("x > 0"):
            ...     builder.assign("result", "x")
            # Generates:
            # if (x > 0) {
            #     result = x;
            # }
        """
        self._add_line(f"if ({condition}) {{")
        self._indent_level += 1
        try:
            yield
        finally:
            self._indent_level -= 1
            self._add_line("}")

    def generate(self) -> list[str]:
        """Generate final code as list of lines.

        Returns:
            List of C++ code lines with proper indentation
        """
        return self._lines.copy()
