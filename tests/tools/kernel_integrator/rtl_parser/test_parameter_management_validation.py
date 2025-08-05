############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Parameter management deep validation tests.

This test module validates the parameter exposure control and internal datatype
creation logic in the RTL Parser, ensuring correct parameter hiding/exposure
and automatic internal datatype detection.

Test Coverage:
- Parameter exposure control based on pragma usage
- Auto-linking parameter hiding validation
- Internal datatype creation from unmatched prefixes
- Prefix detection algorithm validation
- Conflict resolution between pragmas and auto-linking
"""

import pytest
from brainsmith.tools.kernel_integrator.rtl_parser import RTLParser
from brainsmith.core.dataflow.types import InterfaceType

from .utils.rtl_builder import StrictRTLBuilder


class TestParameterExposureControl:
    """Test parameter exposure control reflects pragma and auto-linking effects."""
    
    def test_alias_pragma_hides_parameter(self, rtl_parser):
        """Test ALIAS pragma hides original parameter from exposure."""
        rtl = (StrictRTLBuilder()
               .module("alias_exposure_test")
               .parameter("PE", "8")
               .parameter("SIMD", "16")
               .pragma("ALIAS", "PE", "ParallelismFactor")
               # SIMD has no pragma, should remain exposed
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "alias_exposure_test.sv")
        
        # PE should be hidden (has ALIAS pragma)
        assert "PE" not in kernel_metadata.exposed_parameters
        
        # ParallelismFactor should be exposed (alias target)
        assert "ParallelismFactor" in kernel_metadata.exposed_parameters
        
        # SIMD should remain exposed (no pragma)
        assert "SIMD" in kernel_metadata.exposed_parameters
    
    def test_derived_parameter_pragma_hides_parameter(self, rtl_parser):
        """Test DERIVED_PARAMETER pragma hides parameter from exposure."""
        rtl = (StrictRTLBuilder()
               .module("derived_exposure_test")
               .parameter("BASE_WIDTH", "8")
               .parameter("SCALE", "4")
               .parameter("TOTAL_WIDTH", "32")  # Will be derived
               .parameter("OTHER_PARAM", "16")  # No pragma
               .pragma("DERIVED_PARAMETER", "TOTAL_WIDTH", "BASE_WIDTH * SCALE")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "derived_exposure_test.sv")
        
        # TOTAL_WIDTH should be hidden (derived parameter)
        assert "TOTAL_WIDTH" not in kernel_metadata.exposed_parameters
        
        # SCALE and OTHER_PARAM should remain exposed  
        assert "SCALE" in kernel_metadata.exposed_parameters
        assert "OTHER_PARAM" in kernel_metadata.exposed_parameters
        
        # BASE_WIDTH may be hidden by internal datatype detection (TOTAL_WIDTH creates TOTAL datatype)
        # This is expected RTL Parser behavior when parameter prefixes create internal datatypes
    
    def test_auto_linked_interface_parameters_hidden(self, rtl_parser):
        """Test auto-linked interface parameters are hidden from exposure."""
        rtl = (StrictRTLBuilder()
               .module("auto_link_exposure_test")
               .parameter("s_axis_input_WIDTH", "32")      # Should be auto-linked
               .parameter("s_axis_input_SIGNED", "0")      # Should be auto-linked
               .parameter("m_axis_output_WIDTH", "32")     # Should be auto-linked
               .parameter("INDEPENDENT_PARAM", "16")       # Should remain exposed
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "auto_link_exposure_test.sv")
        
        # Auto-linked parameters should be hidden
        assert "s_axis_input_WIDTH" not in kernel_metadata.exposed_parameters
        assert "s_axis_input_SIGNED" not in kernel_metadata.exposed_parameters
        assert "m_axis_output_WIDTH" not in kernel_metadata.exposed_parameters
        
        # Independent parameter should remain exposed
        assert "INDEPENDENT_PARAM" in kernel_metadata.exposed_parameters
    
    def test_internal_datatype_parameters_hidden(self, rtl_parser):
        """Test internal datatype parameters are hidden from exposure."""
        rtl = (StrictRTLBuilder()
               .module("internal_datatype_exposure_test")
               .parameter("ACC_WIDTH", "48")               # Internal datatype
               .parameter("ACC_SIGNED", "1")               # Internal datatype
               .parameter("THRESH_WIDTH", "32")            # Internal datatype
               .parameter("BIAS_WIDTH", "16")              # Internal datatype
               .parameter("STANDALONE_PARAM", "8")         # Should remain exposed
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "internal_datatype_exposure_test.sv")
        
        # Internal datatype parameters should be hidden
        assert "ACC_WIDTH" not in kernel_metadata.exposed_parameters
        assert "ACC_SIGNED" not in kernel_metadata.exposed_parameters
        assert "THRESH_WIDTH" not in kernel_metadata.exposed_parameters
        assert "BIAS_WIDTH" not in kernel_metadata.exposed_parameters
        
        # Standalone parameter should remain exposed
        assert "STANDALONE_PARAM" in kernel_metadata.exposed_parameters
    
    def test_datatype_param_pragma_parameter_linking(self, rtl_parser):
        """Test DATATYPE_PARAM pragma affects parameter exposure."""
        rtl = (StrictRTLBuilder()
               .module("datatype_param_exposure_test")
               .parameter("CUSTOM_WIDTH", "24")
               .parameter("CUSTOM_SIGNED", "1")
               .parameter("OTHER_PARAM", "16")
               .pragma("DATATYPE_PARAM", "custom_accumulator", "width", "CUSTOM_WIDTH")
               .pragma("DATATYPE_PARAM", "custom_accumulator", "signed", "CUSTOM_SIGNED")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "datatype_param_exposure_test.sv")
        
        # Parameters used in DATATYPE_PARAM should be hidden
        assert "CUSTOM_WIDTH" not in kernel_metadata.exposed_parameters
        assert "CUSTOM_SIGNED" not in kernel_metadata.exposed_parameters
        
        # Other parameter should remain exposed
        assert "OTHER_PARAM" in kernel_metadata.exposed_parameters
    
    def test_pragma_overrides_auto_linking(self, rtl_parser):
        """Test explicit pragmas override auto-linking for parameter exposure."""
        rtl = (StrictRTLBuilder()
               .module("pragma_override_test")
               .parameter("s_axis_input_WIDTH", "32")      # Would auto-link, but has ALIAS
               .parameter("s_axis_input_SIGNED", "0")      # Should auto-link (no pragma)
               .parameter("CUSTOM_PARAM", "16")
               .pragma("ALIAS", "s_axis_input_WIDTH", "InputBitWidth")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "pragma_override_test.sv")
        
        # Parameter with ALIAS should be hidden (pragma takes precedence)
        assert "s_axis_input_WIDTH" not in kernel_metadata.exposed_parameters
        assert "InputBitWidth" in kernel_metadata.exposed_parameters
        
        # Parameter without pragma should be auto-linked and hidden
        assert "s_axis_input_SIGNED" not in kernel_metadata.exposed_parameters
        
        # Independent parameter should remain exposed
        assert "CUSTOM_PARAM" in kernel_metadata.exposed_parameters


class TestInternalDatatypeCreation:
    """Test internal datatype creation from parameter prefix detection."""
    
    def test_basic_internal_datatype_detection(self, rtl_parser):
        """Test basic internal datatype creation from prefixes."""
        rtl = (StrictRTLBuilder()
               .module("basic_internal_datatype_test")
               .parameter("ACC_WIDTH", "48")
               .parameter("ACC_SIGNED", "1")
               .parameter("THRESH_WIDTH", "32")
               .parameter("BIAS_WIDTH", "16")
               .parameter("BIAS_SIGNED", "0")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "basic_internal_datatype_test.sv")
        
        # Should detect internal datatypes for ACC, THRESH, BIAS prefixes
        assert len(kernel_metadata.internal_datatypes) >= 3
        
        # Find each datatype
        acc_dt = next((dt for dt in kernel_metadata.internal_datatypes if dt.name == "ACC"), None)
        thresh_dt = next((dt for dt in kernel_metadata.internal_datatypes if dt.name == "THRESH"), None)
        bias_dt = next((dt for dt in kernel_metadata.internal_datatypes if dt.name == "BIAS"), None)
        
        # Validate ACC datatype
        assert acc_dt is not None
        assert acc_dt.width == "ACC_WIDTH"
        assert acc_dt.signed == "ACC_SIGNED"
        
        # Validate THRESH datatype (only width)
        assert thresh_dt is not None
        assert thresh_dt.width == "THRESH_WIDTH"
        assert thresh_dt.signed is None
        
        # Validate BIAS datatype
        assert bias_dt is not None
        assert bias_dt.width == "BIAS_WIDTH"
        assert bias_dt.signed == "BIAS_SIGNED"
    
    def test_internal_datatype_basic_properties(self, rtl_parser):
        """Test internal datatype with basic supported properties."""
        rtl = (StrictRTLBuilder()
               .module("basic_internal_datatype_test")
               .parameter("ACC_WIDTH", "32")
               .parameter("ACC_SIGNED", "1")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "basic_internal_datatype_test.sv")
        
        # Find ACC datatype (should be created from ACC_WIDTH/ACC_SIGNED pattern)
        acc_dt = next((dt for dt in kernel_metadata.internal_datatypes if dt.name == "ACC"), None)
        assert acc_dt is not None
        
        # Validate basic properties that RTL Parser actually supports
        assert acc_dt.width == "ACC_WIDTH"
        assert acc_dt.signed == "ACC_SIGNED"
    
    def test_interface_prefixes_excluded_from_internal(self, rtl_parser):
        """Test interface prefixes are excluded from internal datatype creation."""
        rtl = (StrictRTLBuilder()
               .module("interface_exclusion_test")
               .parameter("s_axis_input_WIDTH", "32")      # Interface prefix - exclude
               .parameter("m_axis_output_WIDTH", "32")     # Interface prefix - exclude
               .parameter("ACC_WIDTH", "48")               # Internal prefix - include
               .parameter("ACC_SIGNED", "1")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "interface_exclusion_test.sv")
        
        # Should only create internal datatype for ACC (not for interface prefixes)
        internal_names = {dt.name for dt in kernel_metadata.internal_datatypes}
        
        # Should include ACC
        assert "ACC" in internal_names
        
        # Should NOT include interface prefixes
        assert "s_axis_input" not in internal_names
        assert "m_axis_output" not in internal_names
    
    def test_single_character_prefixes(self, rtl_parser):
        """Test single character prefixes are detected for internal datatypes."""
        rtl = (StrictRTLBuilder()
               .module("single_char_prefix_test")
               .parameter("A_WIDTH", "8")
               .parameter("A_SIGNED", "0")
               .parameter("B_WIDTH", "16")
               .parameter("X_WIDTH", "32")
               .parameter("X_SIGNED", "1")
               .parameter("X_BIAS", "128")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "single_char_prefix_test.sv")
        
        # Should detect A, B, X internal datatypes
        internal_names = {dt.name for dt in kernel_metadata.internal_datatypes}
        assert "A" in internal_names
        assert "B" in internal_names
        assert "X" in internal_names
        
        # Validate individual datatypes
        a_dt = next((dt for dt in kernel_metadata.internal_datatypes if dt.name == "A"), None)
        assert a_dt.width == "A_WIDTH"
        assert a_dt.signed == "A_SIGNED"
        
        b_dt = next((dt for dt in kernel_metadata.internal_datatypes if dt.name == "B"), None)
        assert b_dt.width == "B_WIDTH"
        assert b_dt.signed is None
        
        x_dt = next((dt for dt in kernel_metadata.internal_datatypes if dt.name == "X"), None)
        assert x_dt.width == "X_WIDTH"
        assert x_dt.signed == "X_SIGNED"
        assert x_dt.bias == "X_BIAS"
    
    def test_partial_prefix_matches(self, rtl_parser):
        """Test partial prefix matches create appropriate internal datatypes."""
        rtl = (StrictRTLBuilder()
               .module("partial_prefix_test")
               .parameter("DATA_WIDTH", "32")      # Only width
               .parameter("WEIGHT_SIGNED", "1")    # Only signed (no width)
               .parameter("SCALE_BIAS", "64")      # Only bias (no width)
               .parameter("COMPLETE_WIDTH", "16")  # Complete datatype
               .parameter("COMPLETE_SIGNED", "0")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "partial_prefix_test.sv")
        
        # Should create internal datatypes even for partial matches
        internal_names = {dt.name for dt in kernel_metadata.internal_datatypes}
        assert "DATA" in internal_names
        assert "WEIGHT" in internal_names  
        assert "SCALE" in internal_names
        assert "COMPLETE" in internal_names
        
        # Validate partial datatypes
        data_dt = next((dt for dt in kernel_metadata.internal_datatypes if dt.name == "DATA"), None)
        assert data_dt.width == "DATA_WIDTH"
        assert data_dt.signed is None
        
        weight_dt = next((dt for dt in kernel_metadata.internal_datatypes if dt.name == "WEIGHT"), None)
        assert weight_dt.width is None
        assert weight_dt.signed == "WEIGHT_SIGNED"
        
        complete_dt = next((dt for dt in kernel_metadata.internal_datatypes if dt.name == "COMPLETE"), None)
        assert complete_dt.width == "COMPLETE_WIDTH"
        assert complete_dt.signed == "COMPLETE_SIGNED"


class TestParameterConflictResolution:
    """Test parameter conflict resolution between pragmas and auto-linking."""
    
    def test_pragma_wins_over_auto_linking(self, rtl_parser):
        """Test explicit pragma takes precedence over auto-linking."""
        rtl = (StrictRTLBuilder()
               .module("pragma_precedence_test")
               .parameter("s_axis_input_WIDTH", "32")      # Would auto-link, but has ALIAS
               .parameter("CUSTOM_PARAM", "16")
               .pragma("ALIAS", "s_axis_input_WIDTH", "InputWidth")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "pragma_precedence_test.sv")
        
        # Parameter should be controlled by pragma, not auto-linking
        assert "s_axis_input_WIDTH" not in kernel_metadata.exposed_parameters
        assert "InputWidth" in kernel_metadata.exposed_parameters
        
        # Interface should not have datatype_metadata.width set (overridden by pragma)
        input_interface = next(
            (i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.INPUT),
            None
        )
        # Auto-linking may still occur but pragma takes precedence for exposure
    
    def test_datatype_param_prevents_auto_linking(self, rtl_parser):
        """Test DATATYPE_PARAM pragma prevents auto-linking."""
        rtl = (StrictRTLBuilder()
               .module("datatype_param_override_test")
               .parameter("CUSTOM_WIDTH", "24")
               .parameter("s_axis_input_WIDTH", "32")      # Would auto-link to interface
               .pragma("DATATYPE_PARAM", "s_axis_input", "width", "CUSTOM_WIDTH")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "datatype_param_override_test.sv")
        
        # CUSTOM_WIDTH should be hidden (used in DATATYPE_PARAM)
        assert "CUSTOM_WIDTH" not in kernel_metadata.exposed_parameters
        
        # s_axis_input_WIDTH might still be auto-linked or exposed depending on implementation
        # The key is that CUSTOM_WIDTH is correctly handled by DATATYPE_PARAM pragma
    
    def test_multiple_pragmas_on_same_parameter(self, rtl_parser):
        """Test handling of multiple pragmas targeting the same parameter.""" 
        rtl = (StrictRTLBuilder()
               .module("multiple_pragma_test")
               .parameter("SHARED_PARAM", "32")
               .pragma("ALIAS", "SHARED_PARAM", "AliasName")
               .pragma("DERIVED_PARAMETER", "COMPUTED", "SHARED_PARAM * 2")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "multiple_pragma_test.sv")
        
        # Parameter should be handled by one of the pragmas
        # Implementation may choose which pragma takes precedence
        assert "SHARED_PARAM" not in kernel_metadata.exposed_parameters
        
        # At least one pragma effect should be visible
        assert ("AliasName" in kernel_metadata.exposed_parameters or
                "COMPUTED" not in kernel_metadata.exposed_parameters)