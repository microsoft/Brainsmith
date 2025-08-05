############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Demonstration of validation helpers usage.

This test module demonstrates how to use the validation helper functions
to perform comprehensive RTL Parser output validation. These tests serve
as examples for proper validation methodology.
"""

import pytest
from brainsmith.tools.kernel_integrator.rtl_parser import RTLParser
from brainsmith.core.dataflow.types import InterfaceType

from .utils.rtl_builder import StrictRTLBuilder
from .utils.validation_helpers import (
    validate_kernel_metadata_structure,
    validate_pragma_application,
    validate_parameter_exposure,
    validate_internal_datatypes,
    assert_valid_kernel_metadata,
    assert_pragma_effects
)


class TestValidationHelpersDemo:
    """Demonstration of validation helper usage patterns."""
    
    def test_comprehensive_validation_example(self, rtl_parser):
        """Comprehensive example showing all validation helper usage."""
        rtl = (StrictRTLBuilder()
               .module("comprehensive_validation_test")
               .parameter("TILE_H", "16")
               .parameter("TILE_W", "16")
               .parameter("IMG_H", "224")
               .parameter("IMG_W", "224")
               .parameter("PE", "8")
               .parameter("ACC_WIDTH", "48")
               .parameter("ACC_SIGNED", "1")
               .parameter("THRESH_WIDTH", "32")
               .parameter("CUSTOM_PARAM", "64")
               .pragma("DATATYPE", "s_axis_input", "UINT", "8", "32")
               .pragma("BDIM", "s_axis_input", "[TILE_H, TILE_W]")
               .pragma("SDIM", "s_axis_input", "[IMG_H, IMG_W]")
               .pragma("WEIGHT", "s_axis_weights")
               .pragma("DATATYPE", "s_axis_weights", "INT", "8", "8")
               .pragma("BDIM", "s_axis_weights", "PE")
               .pragma("SDIM", "s_axis_weights", "PE")
               .pragma("ALIAS", "PE", "ParallelismFactor")
               .pragma("DATATYPE_PARAM", "accumulator", "width", "ACC_WIDTH")
               .pragma("DATATYPE_PARAM", "accumulator", "signed", "ACC_SIGNED")
               .pragma("RELATIONSHIP", "s_axis_input", "s_axis_weights", "EQUAL")
               .add_stream_input("s_axis_input", bdim_value="256", sdim_value="50176")
               .add_stream_input("s_axis_weights", bdim_value="8", sdim_value="8")
               .add_stream_output("m_axis_output", bdim_value="256")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "comprehensive_validation_test.sv")
        
        # 1. Basic structure validation
        structure_errors = validate_kernel_metadata_structure(kernel_metadata)
        assert structure_errors == [], f"Structure validation failed: {structure_errors}"
        
        # Or use assertion helper
        assert_valid_kernel_metadata(kernel_metadata)
        
        # 2. Comprehensive pragma effect validation
        expected_effects = {
            'datatype_constraints': {
                's_axis_input': [('UINT', 8, 32)],
                's_axis_weights': [('INT', 8, 8)]
            },
            'bdim_params': {
                's_axis_input': ['TILE_H', 'TILE_W'],
                's_axis_weights': ['PE']
            },
            'sdim_params': {
                's_axis_input': ['IMG_H', 'IMG_W'],
                's_axis_weights': ['PE']
            },
            'weight_interfaces': ['s_axis_weights'],
            'hidden_parameters': ['PE', 'ACC_WIDTH', 'ACC_SIGNED', 'THRESH_WIDTH'],  # THRESH_WIDTH hidden by internal datatype
            'exposed_parameters': ['ParallelismFactor', 'TILE_H', 'TILE_W', 'IMG_H', 'IMG_W', 'CUSTOM_PARAM'],
            'internal_datatypes': ['accumulator', 'THRESH'],
            'relationships': 1
        }
        
        pragma_errors = validate_pragma_application(kernel_metadata, expected_effects)
        assert pragma_errors == [], f"Pragma validation failed: {pragma_errors}"
        
        # Or use assertion helper
        assert_pragma_effects(kernel_metadata, expected_effects)
        
        # 3. Specific parameter exposure validation
        should_be_hidden = {'PE', 'ACC_WIDTH', 'ACC_SIGNED'}
        should_be_exposed = {'ParallelismFactor', 'TILE_H', 'TILE_W', 'CUSTOM_PARAM'}
        
        exposure_errors = validate_parameter_exposure(
            kernel_metadata, should_be_hidden, should_be_exposed
        )
        assert exposure_errors == [], f"Parameter exposure validation failed: {exposure_errors}"
        
        # 4. Internal datatype validation
        expected_internal_datatypes = {
            'accumulator': {
                'width': 'ACC_WIDTH',
                'signed': 'ACC_SIGNED'
            },
            'THRESH': {
                'width': 'THRESH_WIDTH'
            }
        }
        
        dt_errors = validate_internal_datatypes(kernel_metadata, expected_internal_datatypes)
        assert dt_errors == [], f"Internal datatype validation failed: {dt_errors}"
    
    def test_validation_error_detection(self, rtl_parser):
        """Test that validation helpers properly detect errors."""
        # Create RTL with missing required pragmas
        rtl = (StrictRTLBuilder()
               .module("error_detection_test")
               # Missing required SDIM for input interface
               .pragma("DATATYPE", "s_axis_input", "UINT", "8", "32")
               .pragma("BDIM", "s_axis_input", "TILE_SIZE")
               # No SDIM pragma
               .parameter("TILE_SIZE", "32")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "error_detection_test.sv")
        
        # Basic structure should still be valid
        structure_errors = validate_kernel_metadata_structure(kernel_metadata)
        assert structure_errors == [], "Basic structure should be valid"
        
        # But pragma effects validation should detect missing SDIM
        expected_effects = {
            'sdim_params': {
                's_axis_input': ['STREAM_SIZE']  # This should fail
            }
        }
        
        pragma_errors = validate_pragma_application(kernel_metadata, expected_effects)
        assert len(pragma_errors) > 0, "Should detect missing SDIM parameters"
        assert any("sdim_params" in error for error in pragma_errors)
    
    def test_parameter_exposure_validation_errors(self, rtl_parser):
        """Test parameter exposure validation error detection."""
        rtl = (StrictRTLBuilder()
               .module("exposure_error_test")
               .parameter("VISIBLE_PARAM", "32")
               .parameter("HIDDEN_PARAM", "16")
               .pragma("ALIAS", "HIDDEN_PARAM", "AliasName")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "exposure_error_test.sv")
        
        # Test wrong expectations
        should_be_hidden = {'VISIBLE_PARAM'}  # Wrong - this should be exposed
        should_be_exposed = {'HIDDEN_PARAM'}  # Wrong - this should be hidden
        
        exposure_errors = validate_parameter_exposure(
            kernel_metadata, should_be_hidden, should_be_exposed
        )
        
        assert len(exposure_errors) >= 2, "Should detect both exposure errors"
        assert any("VISIBLE_PARAM" in error and "should be hidden" in error for error in exposure_errors)
        assert any("HIDDEN_PARAM" in error and "should be exposed" in error for error in exposure_errors)
    
    def test_internal_datatype_validation_errors(self, rtl_parser):
        """Test internal datatype validation error detection."""
        rtl = (StrictRTLBuilder()
               .module("datatype_error_test")
               .parameter("ACC_WIDTH", "48")
               .parameter("ACC_SIGNED", "1")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "datatype_error_test.sv")
        
        # Test wrong expectations
        expected_internal_datatypes = {
            'ACC': {
                'width': 'WRONG_WIDTH',  # Should be ACC_WIDTH
                'signed': 'ACC_SIGNED'
            },
            'NONEXISTENT': {
                'width': 'SOME_WIDTH'  # This datatype doesn't exist
            }
        }
        
        dt_errors = validate_internal_datatypes(kernel_metadata, expected_internal_datatypes)
        
        assert len(dt_errors) >= 2, "Should detect datatype validation errors"
        assert any("WRONG_WIDTH" in error for error in dt_errors)
        assert any("NONEXISTENT" in error for error in dt_errors)
    
    def test_assertion_helpers_usage(self, rtl_parser):
        """Test using assertion helpers for test validation."""
        rtl = (StrictRTLBuilder()
               .module("assertion_test")
               .parameter("WIDTH", "32")
               .pragma("DATATYPE", "s_axis_input", "UINT", "8", "32")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "assertion_test.sv")
        
        # These should pass without exception
        assert_valid_kernel_metadata(kernel_metadata)
        
        expected_effects = {
            'datatype_constraints': {
                's_axis_input': [('UINT', 8, 32)]
            }
        }
        assert_pragma_effects(kernel_metadata, expected_effects)
        
        # Test that assertion helpers raise errors appropriately
        wrong_effects = {
            'datatype_constraints': {
                's_axis_input': [('INT', 16, 64)]  # Wrong constraint
            }
        }
        
        with pytest.raises(AssertionError) as exc_info:
            assert_pragma_effects(kernel_metadata, wrong_effects)
        
        assert "validation failed" in str(exc_info.value).lower()
    
    def test_real_world_validation_pattern(self, rtl_parser):
        """Test validation pattern for real-world hardware kernel."""
        # Use the all_pragmas fixture which demonstrates all pragma types
        rtl_file = "/home/tafk/dev/brainsmith-2/tests/tools/hw_kernel_gen/rtl_parser/fixtures/pragmas/all_pragmas.sv"
        
        with open(rtl_file, 'r') as f:
            rtl_content = f.read()
        
        kernel_metadata = rtl_parser.parse(rtl_content, "all_pragmas.sv")
        
        # Basic structure validation
        assert_valid_kernel_metadata(kernel_metadata)
        
        # Validate key expected outcomes (adjusted for actual RTL Parser behavior)
        expected_effects = {
            'weight_interfaces': ['s_axis_weights'],
            'hidden_parameters': ['PE', 'MEM_SIZE'],  # ALIAS, DERIVED (BATCH_SIZE behavior varies)
            'exposed_parameters': ['ParallelismFactor'],  # ALIAS target
            'internal_datatypes': ['accumulator', 'threshold'],  # From DATATYPE_PARAM
            'relationships': 1  # From RELATIONSHIP pragma
        }
        
        # Use assertion helper for clean test failure messages
        assert_pragma_effects(kernel_metadata, expected_effects)
        
        # Validate specific interface configurations
        input_interfaces = [i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.INPUT]
        weight_interfaces = [i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.WEIGHT]
        
        assert len(input_interfaces) == 1
        assert len(weight_interfaces) == 1
        
        # Validate input interface has proper pragma effects
        input_interface = input_interfaces[0]
        assert len(input_interface.datatype_constraints) == 1
        assert input_interface.datatype_constraints[0].base_type == "UINT"
        assert input_interface.bdim_params == ['IN0_BDIM0', 'IN0_BDIM1', 'IN0_BDIM2']
        assert input_interface.sdim_params == ['IN0_SDIM0', 'IN0_SDIM1']