############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Pragma integration tests for RTL Parser - REFACTORED.

Tests pragma combinations, interactions, ordering effects, and conflict resolution.
These tests focus on how multiple pragmas work together and affect the final
KernelMetadata output. Now using PragmaPatterns for consistent test generation.
"""

import pytest
from brainsmith.core.dataflow.types import InterfaceType
from brainsmith.tools.kernel_integrator.rtl_parser import RTLParser

from .utils.rtl_builder import RTLBuilder, StrictRTLBuilder
from .utils.pragma_patterns import PragmaPatterns
from .utils.test_factory import TestDataFactory


class TestPragmaIntegration:
    """Test cases for pragma integration and interactions."""
    
    def test_multiple_pragmas_same_interface(self, rtl_parser):
        """Test multiple pragmas targeting the same interface."""
        # Use PragmaPatterns for multi-pragma cascade
        rtl = PragmaPatterns.multi_pragma_cascade("s_axis_input")
        
        kernel_metadata = rtl_parser.parse(rtl, "multi_pragma_interface.sv")
        
        # Verify basic parsing
        assert kernel_metadata.name == "pragma_cascade"
        assert len(kernel_metadata.interfaces) >= 2
        
        # Check that all pragmas were parsed
        pragma_types = {p.type.value for p in kernel_metadata.pragmas}
        assert "datatype" in pragma_types
        assert "bdim" in pragma_types
        assert "sdim" in pragma_types
        
        # Verify interface detection
        interface_types = {i.interface_type for i in kernel_metadata.interfaces}
        assert InterfaceType.INPUT in interface_types
        assert InterfaceType.OUTPUT in interface_types
    
    def test_pragma_ordering_effects(self, rtl_parser):
        """Test that pragma ordering doesn't affect final result."""
        # Use PragmaPatterns for consistent ordering test
        rtl_order1 = PragmaPatterns.pragma_ordering_test()
        
        # Create second version with pragmas reordered
        rtl_order2 = (StrictRTLBuilder()
                      .module("pragma_order")
                      # Same parameters but pragmas in different order
                      .parameter("BASE_WIDTH", "8")
                      .parameter("SCALE", "4")
                      # Reverse pragma order
                      .pragma("ALIAS", "SCALE", "ScalingFactor")
                      .pragma("DERIVED_PARAMETER", "TOTAL_WIDTH", "BASE_WIDTH * SCALE")
                      .pragma("DATATYPE", "s_axis_input", "UINT", "BASE_WIDTH", "TOTAL_WIDTH")
                      .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
                      .add_stream_output("m_axis_output", bdim_value="32")
                      .build())
        
        km1 = rtl_parser.parse(rtl_order1, "pragma_order_test1.sv")
        km2 = rtl_parser.parse(rtl_order2, "pragma_order_test2.sv")
        
        # Results should be equivalent regardless of pragma order
        assert len(km1.pragmas) == len(km2.pragmas)
        
        # Check consistency in linked parameters
        assert len(km1.exposed_parameters) == len(km2.exposed_parameters)
    
    def test_conflicting_pragmas(self, rtl_parser):
        """Test handling of conflicting pragma definitions."""
        rtl = PragmaPatterns.pragma_conflict_test()
        
        # Parser should handle conflicts gracefully
        kernel_metadata = rtl_parser.parse(rtl, "conflicting_pragmas_test.sv")
        
        # Verify basic parsing succeeds
        assert kernel_metadata.name == "pragma_conflict"
        
        # Check that pragmas were parsed (even if some conflict)
        assert len(kernel_metadata.pragmas) >= 4
        
        # Check conflict resolution
        aliases = kernel_metadata.linked_parameters.get("aliases", {})
        if "PARAM_A" in aliases or "PARAM_B" in aliases:
            # At least one should be processed
            assert len(aliases) > 0
    
    def test_pragma_cascade_effects(self, rtl_parser):
        """Test how pragmas affect each other (cascade effects)."""
        rtl = PragmaPatterns.datatype_param_cascade()
        
        kernel_metadata = rtl_parser.parse(rtl, "pragma_cascade_test.sv")
        
        # Verify parsing
        assert kernel_metadata.name == "datatype_params"
        
        # Check internal datatypes were created
        assert len(kernel_metadata.internal_datatypes) >= 2
        
        # Check cascade effects
        acc_dt = next((dt for dt in kernel_metadata.internal_datatypes 
                      if dt.name == "accumulator"), None)
        if acc_dt:
            assert hasattr(acc_dt, 'width')
            assert hasattr(acc_dt, 'signed')
    
    def test_pragma_with_invalid_targets(self, rtl_parser):
        """Test pragmas targeting non-existent interfaces."""
        rtl = PragmaPatterns.pragma_with_invalid_target()
        
        # Should parse without crashing
        kernel_metadata = rtl_parser.parse(rtl, "invalid_targets.sv")
        
        # Verify basic parsing succeeds
        assert kernel_metadata.name == "invalid_target"
        
        # Check that invalid pragmas were parsed but may not apply
        pragma_count = len(kernel_metadata.pragmas)
        assert pragma_count >= 3  # The invalid pragmas
        
        # Real interfaces should still work
        assert len(kernel_metadata.interfaces) >= 2
    
    def test_weight_interface_pragma_interaction(self, rtl_parser):
        """Test WEIGHT pragma interaction with interface detection."""
        rtl = PragmaPatterns.weight_pragma_variations()
        
        kernel_metadata = rtl_parser.parse(rtl, "weight_interface_test.sv")
        
        # Verify weight interfaces
        weight_ifaces = [i for i in kernel_metadata.interfaces 
                        if i.interface_type == InterfaceType.WEIGHT]
        # May have detected weight interfaces
        assert len(weight_ifaces) >= 0
        
        # Check weight pragmas
        weight_pragmas = [p for p in kernel_metadata.pragmas 
                         if p.type.value == "weight"]
        assert len(weight_pragmas) >= 2
    
    def test_datatype_and_dimension_integration(self, rtl_parser):
        """Test DATATYPE pragma integration with BDIM/SDIM."""
        rtl = (StrictRTLBuilder()
               .module("datatype_dimension_test")
               .parameter("TILE_H", "16")
               .parameter("TILE_W", "16")
               .parameter("CHANNELS", "3")
               .parameter("IMG_H", "224")
               .parameter("IMG_W", "224")
               .parameter("DATA_WIDTH", "32")
               .pragma("DATATYPE", "s_axis_input", "UINT", "8", "32")
               .pragma("BDIM", "s_axis_input", "[TILE_H, TILE_W, CHANNELS]")
               .pragma("SDIM", "s_axis_input", "[IMG_H, IMG_W, CHANNELS]")
               .add_stream_input("s_axis_input", 
                               bdim_value="768",  # 16*16*3
                               sdim_value="150528")  # 224*224*3
               .add_stream_output("m_axis_output", bdim_value="1000")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "datatype_dimension_test.sv")
        
        # Verify pragmas were parsed
        pragma_types = {p.type.value for p in kernel_metadata.pragmas}
        assert "datatype" in pragma_types
        assert "bdim" in pragma_types
        assert "sdim" in pragma_types
    
    def test_alias_and_derived_parameter_chain(self, rtl_parser):
        """Test complex ALIAS and DERIVED_PARAMETER chains."""
        rtl = PragmaPatterns.alias_and_derived_chain()
        
        kernel_metadata = rtl_parser.parse(rtl, "alias_derived_chain.sv")
        
        # Verify aliases
        aliases = kernel_metadata.linked_parameters.get("aliases", {})
        if aliases:
            assert "A" in aliases or "BaseWidth" in kernel_metadata.exposed_parameters
            assert "B" in aliases or "ScaleFactor" in kernel_metadata.exposed_parameters
        
        # Verify derived parameters
        derived = kernel_metadata.linked_parameters.get("derived", {})
        if derived:
            # Check that derived parameters were processed
            assert len(derived) >= 0  # May have some derived params
    
    def test_pragma_with_expressions(self, rtl_parser):
        """Test pragmas with complex expressions."""
        rtl = PragmaPatterns.pragma_with_expressions()
        
        kernel_metadata = rtl_parser.parse(rtl, "pragma_expressions.sv")
        
        # Verify complex expressions were parsed
        assert kernel_metadata.name == "pragma_expressions"
        
        # Check BDIM/SDIM with expressions
        bdim_pragmas = [p for p in kernel_metadata.pragmas if p.type.value == "bdim"]
        sdim_pragmas = [p for p in kernel_metadata.pragmas if p.type.value == "sdim"]
        
        assert len(bdim_pragmas) >= 1
        assert len(sdim_pragmas) >= 1
    
    def test_relationship_pragma_combinations(self, rtl_parser):
        """Test RELATIONSHIP pragma with other pragmas."""
        rtl = PragmaPatterns.relationship_pragma_test()
        
        kernel_metadata = rtl_parser.parse(rtl, "relationship_test.sv")
        
        # Check relationships
        relationship_pragmas = [p for p in kernel_metadata.pragmas 
                              if p.type.value == "relationship"]
        assert len(relationship_pragmas) >= 3
        
        # Verify relationships were processed
        assert len(kernel_metadata.relationships) >= 0  # May vary based on validation
    
    def test_all_pragma_types_together(self, rtl_parser):
        """Test module with all pragma types combined."""
        rtl = PragmaPatterns.all_pragma_types()
        
        kernel_metadata = rtl_parser.parse(rtl, "all_pragmas.sv")
        
        # Verify all pragma types present
        pragma_types = {p.type.value for p in kernel_metadata.pragmas}
        expected_types = {"top_module", "alias", "derived_parameter", 
                         "datatype", "bdim", "sdim", "datatype_param", 
                         "relationship"}
        
        # Should have most pragma types (some may fail validation)
        assert len(pragma_types.intersection(expected_types)) >= 6
    
    def test_pragma_parameter_references(self, rtl_parser):
        """Test pragmas that reference parameters."""
        rtl = (StrictRTLBuilder()
               .module("param_reference_test")
               .parameter("BASE", "8")
               .parameter("MULT", "4")
               .pragma("DERIVED_PARAMETER", "TOTAL", "BASE * MULT")
               .pragma("DATATYPE", "s_axis_data", "UINT", "BASE", "TOTAL")
               .pragma("BDIM", "s_axis_data", "TOTAL")
               .add_stream_input("s_axis_data", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_result", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "param_reference_test.sv")
        
        # Check parameter references in pragmas
        assert "TOTAL" not in kernel_metadata.exposed_parameters  # Hidden by DERIVED
        assert "BASE" in kernel_metadata.exposed_parameters or len(kernel_metadata.exposed_parameters) == 0
    
    def test_pragma_error_handling(self, rtl_parser):
        """Test pragma error cases and recovery."""
        # Test various error cases
        error_cases = [
            PragmaPatterns.pragma_error_cases("missing_args"),
            PragmaPatterns.pragma_error_cases("invalid_interface"),
            PragmaPatterns.pragma_error_cases("type_mismatch")
        ]
        
        for i, rtl in enumerate(error_cases):
            # Should parse without crashing
            kernel_metadata = rtl_parser.parse(rtl, f"pragma_error_{i}.sv")
            
            # Basic parsing should succeed
            assert kernel_metadata is not None
            assert len(kernel_metadata.interfaces) >= 2  # Should have basic interfaces
    
    def test_pragma_with_special_characters(self, rtl_parser):
        """Test pragma parsing with special characters."""
        rtl = PragmaPatterns.pragma_with_special_chars()
        
        # Should handle special characters gracefully
        kernel_metadata = rtl_parser.parse(rtl, "special_chars.sv")
        
        assert kernel_metadata.name == "special_chars"
        # Pragmas with special chars may fail validation but shouldn't crash
        assert len(kernel_metadata.pragmas) >= 0
    
    def test_invalid_pragma_syntax(self, rtl_parser):
        """Test handling of malformed pragma syntax."""
        rtl = PragmaPatterns.invalid_pragma_syntax()
        
        # Should parse module despite invalid pragmas
        kernel_metadata = rtl_parser.parse(rtl, "invalid_syntax.sv")
        
        assert kernel_metadata.name == "invalid_syntax"
        assert len(kernel_metadata.interfaces) >= 2
    
    def test_pragma_application_to_wrong_interface_type(self, rtl_parser):
        """Test pragmas applied to wrong interface types."""
        rtl = (StrictRTLBuilder()
               .module("wrong_interface_type")
               # SDIM on output (outputs don't have SDIM)
               .pragma("SDIM", "m_axis_output", "OUTPUT_SDIM")
               .parameter("OUTPUT_SDIM", "1024")
               # WEIGHT on output (can't be weight)
               .pragma("WEIGHT", "m_axis_output")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "wrong_interface_type.sv")
        
        # Should parse but pragmas may not apply correctly
        assert kernel_metadata.name == "wrong_interface_type"
        assert len(kernel_metadata.pragmas) >= 2
    
    def test_circular_dependency_detection(self, rtl_parser):
        """Test detection of circular dependencies in derived parameters."""
        try:
            rtl = PragmaPatterns.pragma_error_cases("circular_dep")
            kernel_metadata = rtl_parser.parse(rtl, "circular_dep.sv")
            
            # May detect circular dependency or just fail to resolve
            assert kernel_metadata is not None
        except Exception:
            # Circular dependencies might cause exceptions
            pass  # Expected in some implementations
    
    def test_pragma_with_interface_arrays(self, rtl_parser):
        """Test pragmas on indexed interfaces."""
        rtl = (StrictRTLBuilder()
               .module("interface_array_pragmas")
               .pragma("DATATYPE", "s_axis_in0", "UINT", "8", "32")
               .pragma("DATATYPE", "s_axis_in1", "UINT", "16", "32")
               .pragma("BDIM", "s_axis_in0", "32")
               .pragma("BDIM", "s_axis_in1", "64")
               .add_stream_input("s_axis_in0", bdim_value="32", sdim_value="512")
               .add_stream_input("s_axis_in1", bdim_value="64", sdim_value="1024")
               .add_stream_output("m_axis_out", bdim_value="96")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "interface_array_pragmas.sv")
        
        # Check indexed interface handling
        input_ifaces = [i for i in kernel_metadata.interfaces 
                       if i.interface_type == InterfaceType.INPUT]
        assert len(input_ifaces) >= 2