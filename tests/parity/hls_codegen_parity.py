"""HLS code generation parity testing mixin.

This module provides HLSCodegenParityMixin, which adds detailed validation
of HLS backend code generation methods to the base parity testing framework.

The mixin validates the template method pattern used by HLSBackend:
- global_includes(): Header includes
- defines(): Macro definitions
- pragmas(): HLS synthesis pragmas
- docompute(): Main computation body
- blackboxfunction(): Function signature
- strm_decl(): Stream declarations (optional)
- dataoutstrm(): Output stream handling (optional)

These tests complement test_cppsim_execution_parity() by validating individual
code generation methods rather than just end-to-end compilation/execution.

Key Benefits:
- Catches code generation bugs before expensive HLS synthesis
- Validates template structure and substitution correctness
- Provides detailed diffs when code generation diverges
- Runs quickly (no C++ compilation required)

Usage:
    class TestMyKernelHLSParity(ParityTestBase, HLSCodegenParityMixin):
        # Implement required abstract methods from ParityTestBase
        ...
        # Automatically inherits 7 HLS code generation tests
"""

import pytest
from typing import List
from finn.util.fpgadataflow import is_hls_node  # Leverage FINN utility


class HLSCodegenParityMixin:
    """Mixin adding HLS code generation validation to parity tests.

    Provides 7 additional test methods that validate HLSBackend template methods:
    - test_global_includes_parity()
    - test_defines_parity()
    - test_pragmas_parity()
    - test_docompute_parity()
    - test_blackboxfunction_parity()
    - test_strm_decl_parity()
    - test_dataoutstrm_parity()

    These tests validate that manual and auto implementations generate
    structurally identical C++ code through the HLS template method pattern.

    Requirements:
    - Must be mixed into a ParityTestBase subclass
    - Both manual and auto ops must inherit from HLSBackend
    - Tests auto-skip if backends are not HLS
    """

    def _normalize_code_lines(self, code_list: List[str]) -> List[str]:
        """Normalize code lines for comparison.

        Removes whitespace variations that don't affect semantics:
        - Strips leading/trailing whitespace
        - Removes empty lines
        - Normalizes internal whitespace

        Args:
            code_list: List of code lines from code_gen_dict

        Returns:
            Normalized list of non-empty lines
        """
        if not isinstance(code_list, list):
            code_list = [code_list]

        normalized = []
        for line in code_list:
            # Strip whitespace
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Normalize internal whitespace (collapse multiple spaces)
            line = ' '.join(line.split())
            normalized.append(line)

        return normalized

    def _compare_code_sections(
        self,
        manual_code: List[str],
        auto_code: List[str],
        section_name: str,
        op_class_name: str
    ) -> None:
        """Compare two code sections with detailed error reporting.

        Args:
            manual_code: Code from manual implementation
            auto_code: Code from auto implementation
            section_name: Name of code section (e.g., "$GLOBALS$")
            op_class_name: Operator class name for error messages

        Raises:
            AssertionError: If code sections differ
        """
        manual_normalized = self._normalize_code_lines(manual_code)
        auto_normalized = self._normalize_code_lines(auto_code)

        if manual_normalized != auto_normalized:
            # Build detailed diff message
            diff_msg = [
                f"\n{section_name} code generation mismatch for {op_class_name}:",
                "",
                "Manual implementation:",
            ]
            for i, line in enumerate(manual_normalized, 1):
                diff_msg.append(f"  {i}. {line}")

            diff_msg.append("")
            diff_msg.append("Auto implementation:")
            for i, line in enumerate(auto_normalized, 1):
                diff_msg.append(f"  {i}. {line}")

            diff_msg.append("")
            diff_msg.append("Line-by-line differences:")
            max_lines = max(len(manual_normalized), len(auto_normalized))
            for i in range(max_lines):
                manual_line = manual_normalized[i] if i < len(manual_normalized) else "<missing>"
                auto_line = auto_normalized[i] if i < len(auto_normalized) else "<missing>"

                if manual_line != auto_line:
                    diff_msg.append(f"  Line {i+1}:")
                    diff_msg.append(f"    Manual: {manual_line}")
                    diff_msg.append(f"    Auto:   {auto_line}")

            pytest.fail('\n'.join(diff_msg))

    @pytest.mark.parity
    @pytest.mark.hls
    def test_global_includes_parity(self):
        """Test global_includes() code generation parity.

        Validates that both backends generate identical C++ header includes.
        Common includes: streamtools.h, mvau.hpp, activations.hpp

        Example output:
            ['#include "streamtools.h"']
            ['#include "mvau.hpp"']
        """
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Skip if not HLS backends
        if not is_hls_node(manual_op.onnx_node):
            pytest.skip(f"{manual_op.__class__.__name__} is not an HLS backend")
        if not is_hls_node(auto_op.onnx_node):
            pytest.skip(f"{auto_op.__class__.__name__} is not an HLS backend")

        # Call global_includes() and extract from code_gen_dict
        manual_op.global_includes()
        auto_op.global_includes()

        manual_includes = manual_op.code_gen_dict.get("$GLOBALS$", [])
        auto_includes = auto_op.code_gen_dict.get("$GLOBALS$", [])

        self._compare_code_sections(
            manual_includes,
            auto_includes,
            "$GLOBALS$ (global_includes)",
            manual_op.__class__.__name__
        )

    @pytest.mark.parity
    @pytest.mark.hls
    def test_defines_parity(self):
        """Test defines() code generation parity.

        Validates that both backends generate identical C++ macro definitions.
        Common defines: Layer dimensions, bit-widths, parallelization factors

        Example output:
            ['#define InWidth 28']
            ['#define OutWidth 10']
            ['#define SIMD 49']
        """
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Skip if not HLS backends
        if not is_hls_node(manual_op.onnx_node):
            pytest.skip(f"{manual_op.__class__.__name__} is not an HLS backend")
        if not is_hls_node(auto_op.onnx_node):
            pytest.skip(f"{auto_op.__class__.__name__} is not an HLS backend")

        # Call defines() and extract from code_gen_dict
        manual_op.defines("var")
        auto_op.defines("var")

        manual_defines = manual_op.code_gen_dict.get("$DEFINES$", [])
        auto_defines = auto_op.code_gen_dict.get("$DEFINES$", [])

        self._compare_code_sections(
            manual_defines,
            auto_defines,
            "$DEFINES$ (defines)",
            manual_op.__class__.__name__
        )

    @pytest.mark.parity
    @pytest.mark.hls
    def test_pragmas_parity(self):
        """Test pragmas() code generation parity.

        Validates that both backends generate identical HLS synthesis pragmas.
        Common pragmas: Interface types (axis, ap_ctrl_none), array partitioning

        Example output:
            ['#pragma HLS INTERFACE axis port=in0_V']
            ['#pragma HLS INTERFACE axis port=out_V']
            ['#pragma HLS INTERFACE ap_ctrl_none port=return']
        """
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Skip if not HLS backends
        if not is_hls_node(manual_op.onnx_node):
            pytest.skip(f"{manual_op.__class__.__name__} is not an HLS backend")
        if not is_hls_node(auto_op.onnx_node):
            pytest.skip(f"{auto_op.__class__.__name__} is not an HLS backend")

        # Call pragmas() and extract from code_gen_dict
        manual_op.pragmas()
        auto_op.pragmas()

        manual_pragmas = manual_op.code_gen_dict.get("$PRAGMAS$", [])
        auto_pragmas = auto_op.code_gen_dict.get("$PRAGMAS$", [])

        self._compare_code_sections(
            manual_pragmas,
            auto_pragmas,
            "$PRAGMAS$ (pragmas)",
            manual_op.__class__.__name__
        )

    @pytest.mark.parity
    @pytest.mark.hls
    def test_docompute_parity(self):
        """Test docompute() code generation parity.

        Validates that both backends generate identical computation kernel calls.
        This is the core logic that invokes finn-hlslib functions.

        Example output:
            ['AddStreams_Batch<8, ap_int<8>, ap_int<8>, ap_int<9>, 3136> (in0_V, in1_V, out0_V, 1);']
        """
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Skip if not HLS backends
        if not is_hls_node(manual_op.onnx_node):
            pytest.skip(f"{manual_op.__class__.__name__} is not an HLS backend")
        if not is_hls_node(auto_op.onnx_node):
            pytest.skip(f"{auto_op.__class__.__name__} is not an HLS backend")

        # Call docompute() and extract from code_gen_dict
        manual_op.docompute()
        auto_op.docompute()

        manual_compute = manual_op.code_gen_dict.get("$DOCOMPUTE$", [])
        auto_compute = auto_op.code_gen_dict.get("$DOCOMPUTE$", [])

        self._compare_code_sections(
            manual_compute,
            auto_compute,
            "$DOCOMPUTE$ (docompute)",
            manual_op.__class__.__name__
        )

    @pytest.mark.parity
    @pytest.mark.hls
    def test_blackboxfunction_parity(self):
        """Test blackboxfunction() code generation parity.

        Validates that both backends generate identical function signatures.
        This is the top-level function that wraps the computation.

        Example output:
            ['void AddStreams_test(hls::stream<ap_uint<64>> &in0_V,
                                   hls::stream<ap_uint<64>> &in1_V,
                                   hls::stream<ap_uint<72>> &out0_V)']
        """
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Skip if not HLS backends
        if not is_hls_node(manual_op.onnx_node):
            pytest.skip(f"{manual_op.__class__.__name__} is not an HLS backend")
        if not is_hls_node(auto_op.onnx_node):
            pytest.skip(f"{auto_op.__class__.__name__} is not an HLS backend")

        # Call blackboxfunction() and extract from code_gen_dict
        manual_op.blackboxfunction()
        auto_op.blackboxfunction()

        manual_func = manual_op.code_gen_dict.get("$BLACKBOXFUNCTION$", [])
        auto_func = auto_op.code_gen_dict.get("$BLACKBOXFUNCTION$", [])

        self._compare_code_sections(
            manual_func,
            auto_func,
            "$BLACKBOXFUNCTION$ (blackboxfunction)",
            manual_op.__class__.__name__
        )

    @pytest.mark.parity
    @pytest.mark.hls
    def test_strm_decl_parity(self):
        """Test strm_decl() code generation parity (optional).

        Validates stream declaration generation if the method exists.
        Many operators don't implement this (return empty list).

        Example output:
            ['hls::stream<ap_uint<64>> in0_V ("in0_V");']
            ['hls::stream<ap_uint<64>> in1_V ("in1_V");']
        """
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Skip if not HLS backends
        if not is_hls_node(manual_op.onnx_node):
            pytest.skip(f"{manual_op.__class__.__name__} is not an HLS backend")
        if not is_hls_node(auto_op.onnx_node):
            pytest.skip(f"{auto_op.__class__.__name__} is not an HLS backend")

        # Skip if neither implements strm_decl
        if not hasattr(manual_op, 'strm_decl') and not hasattr(auto_op, 'strm_decl'):
            pytest.skip(f"{manual_op.__class__.__name__} does not implement strm_decl()")

        # Call strm_decl() if it exists
        if hasattr(manual_op, 'strm_decl'):
            manual_op.strm_decl()
        if hasattr(auto_op, 'strm_decl'):
            auto_op.strm_decl()

        manual_decl = manual_op.code_gen_dict.get("$STREAMDECLARATIONS$", [])
        auto_decl = auto_op.code_gen_dict.get("$STREAMDECLARATIONS$", [])

        self._compare_code_sections(
            manual_decl,
            auto_decl,
            "$STREAMDECLARATIONS$ (strm_decl)",
            manual_op.__class__.__name__
        )

    @pytest.mark.parity
    @pytest.mark.hls
    def test_dataoutstrm_parity(self):
        """Test dataoutstrm() code generation parity (optional).

        Validates output stream handling if the method exists.
        Only some operators (e.g., Thresholding, MVAU) implement this.

        Example output:
            ['hls::stream<ap_uint<8>> out_V ("out_V");']
        """
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Skip if not HLS backends
        if not is_hls_node(manual_op.onnx_node):
            pytest.skip(f"{manual_op.__class__.__name__} is not an HLS backend")
        if not is_hls_node(auto_op.onnx_node):
            pytest.skip(f"{auto_op.__class__.__name__} is not an HLS backend")

        # Skip if neither implements dataoutstrm
        if not hasattr(manual_op, 'dataoutstrm') and not hasattr(auto_op, 'dataoutstrm'):
            pytest.skip(f"{auto_op.__class__.__name__} does not implement dataoutstrm()")

        # Call dataoutstrm() if it exists
        if hasattr(manual_op, 'dataoutstrm'):
            manual_op.dataoutstrm()
        if hasattr(auto_op, 'dataoutstrm'):
            auto_op.dataoutstrm()

        manual_out = manual_op.code_gen_dict.get("$DATAOUTSTREAM$", [])
        auto_out = auto_op.code_gen_dict.get("$DATAOUTSTREAM$", [])

        self._compare_code_sections(
            manual_out,
            auto_out,
            "$DATAOUTSTREAM$ (dataoutstrm)",
            manual_op.__class__.__name__
        )


__all__ = ['HLSCodegenParityMixin']
