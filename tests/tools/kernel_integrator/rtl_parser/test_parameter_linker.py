############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Unit tests for the Parameter Linker component.

Tests automatic parameter linking functionality including:
- Interface parameter auto-linking
- Internal datatype detection
- Prefix-based grouping
- Exclusion handling
- Integration with kernel metadata
"""

import pytest
from pathlib import Path

from brainsmith.tools.kernel_integrator.rtl_parser.parameter_linker import ParameterLinker
from brainsmith.tools.kernel_integrator.types.rtl import Parameter
from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata, InterfaceMetadata, DatatypeMetadata
from brainsmith.core.dataflow.types import InterfaceType

from .utils.rtl_builder import RTLBuilder


class TestParameterLinker:
    """Test cases for Parameter Linker functionality."""
    
    def test_link_interface_parameters_basic(self, parameter_linker):
        """Test linking parameters to an interface."""
        parameters = [
            Parameter("in0_WIDTH", default_value="32"),
            Parameter("in0_SIGNED", default_value="1"),
            Parameter("in0_BIAS", default_value="127"),
            Parameter("OTHER_PARAM", default_value="16")
        ]
        
        # Link parameters for interface 'in0'
        dt_metadata = parameter_linker.link_interface_parameters("in0", parameters)
        
        assert dt_metadata is not None
        assert dt_metadata.name == "in0"
        assert dt_metadata.width == "in0_WIDTH"
        assert dt_metadata.signed == "in0_SIGNED"
        assert dt_metadata.bias == "in0_BIAS"
        assert dt_metadata.format is None  # Not found
    
    def test_link_interface_parameters_all_types(self, parameter_linker):
        """Test linking all supported parameter types."""
        parameters = [
            Parameter("data_WIDTH", default_value="32"),
            Parameter("data_SIGNED", default_value="1"),
            Parameter("data_FORMAT", default_value="FIXED"),
            Parameter("data_BIAS", default_value="0"),
            Parameter("data_FRACTIONAL_WIDTH", default_value="16"),
            Parameter("data_EXPONENT_WIDTH", default_value="8"),
            Parameter("data_MANTISSA_WIDTH", default_value="23")
        ]
        
        dt_metadata = parameter_linker.link_interface_parameters("data", parameters)
        
        assert dt_metadata is not None
        assert dt_metadata.width == "data_WIDTH"
        assert dt_metadata.signed == "data_SIGNED"
        assert dt_metadata.format == "data_FORMAT"
        assert dt_metadata.bias == "data_BIAS"
        assert dt_metadata.fractional_width == "data_FRACTIONAL_WIDTH"
        assert dt_metadata.exponent_width == "data_EXPONENT_WIDTH"
        assert dt_metadata.mantissa_width == "data_MANTISSA_WIDTH"
    
    def test_link_interface_parameters_no_match(self, parameter_linker):
        """Test when no parameters match the interface."""
        parameters = [
            Parameter("WIDTH", default_value="32"),
            Parameter("DEPTH", default_value="16"),
            Parameter("ENABLE", default_value="1")
        ]
        
        dt_metadata = parameter_linker.link_interface_parameters("stream", parameters)
        
        assert dt_metadata is None
    
    def test_link_interface_parameters_disabled(self):
        """Test when interface linking is disabled."""
        linker = ParameterLinker(enable_interface_linking=False)
        
        parameters = [
            Parameter("in0_WIDTH", default_value="32"),
            Parameter("in0_SIGNED", default_value="1")
        ]
        
        dt_metadata = linker.link_interface_parameters("in0", parameters)
        
        assert dt_metadata is None
    
    def test_link_internal_parameters_basic(self, parameter_linker):
        """Test detecting internal datatype parameters."""
        parameters = [
            Parameter("THRESH_WIDTH", default_value="16"),
            Parameter("THRESH_SIGNED", default_value="1"),
            Parameter("ACC_WIDTH", default_value="32"),
            Parameter("ACC_SIGNED", default_value="1"),
            Parameter("SCALE_WIDTH", default_value="8")
        ]
        
        internal_dts = parameter_linker.link_internal_parameters(parameters)
        
        assert len(internal_dts) == 3
        
        # Check each datatype
        thresh_dt = next(dt for dt in internal_dts if dt.name == "THRESH")
        assert thresh_dt.width == "THRESH_WIDTH"
        assert thresh_dt.signed == "THRESH_SIGNED"
        
        acc_dt = next(dt for dt in internal_dts if dt.name == "ACC")
        assert acc_dt.width == "ACC_WIDTH"
        assert acc_dt.signed == "ACC_SIGNED"
        
        scale_dt = next(dt for dt in internal_dts if dt.name == "SCALE")
        assert scale_dt.width == "SCALE_WIDTH"
        assert scale_dt.signed is None
    
    def test_link_internal_parameters_exclude_prefixes(self, parameter_linker):
        """Test excluding specific prefixes from internal linking."""
        parameters = [
            Parameter("in0_WIDTH", default_value="32"),
            Parameter("in0_SIGNED", default_value="1"),
            Parameter("out0_WIDTH", default_value="32"),
            Parameter("THRESH_WIDTH", default_value="16"),
            Parameter("THRESH_SIGNED", default_value="1")
        ]
        
        # Exclude interface prefixes
        internal_dts = parameter_linker.link_internal_parameters(
            parameters, 
            exclude_prefixes=["in0", "out0"]
        )
        
        # Should only get THRESH datatype
        assert len(internal_dts) == 1
        assert internal_dts[0].name == "THRESH"
    
    def test_link_internal_parameters_exclude_params(self, parameter_linker):
        """Test excluding specific parameters from internal linking."""
        parameters = [
            Parameter("ACC_WIDTH", default_value="32"),
            Parameter("ACC_SIGNED", default_value="1"),
            Parameter("THRESH_WIDTH", default_value="16"),
            Parameter("THRESH_SIGNED", default_value="1")
        ]
        
        # Exclude specific parameters (e.g., claimed by pragmas)
        internal_dts = parameter_linker.link_internal_parameters(
            parameters,
            exclude_parameters={"ACC_WIDTH", "ACC_SIGNED"}
        )
        
        # Should only get THRESH datatype
        assert len(internal_dts) == 1
        assert internal_dts[0].name == "THRESH"
    
    def test_link_internal_parameters_disabled(self):
        """Test when internal linking is disabled."""
        linker = ParameterLinker(enable_internal_linking=False)
        
        parameters = [
            Parameter("THRESH_WIDTH", default_value="16"),
            Parameter("THRESH_SIGNED", default_value="1")
        ]
        
        internal_dts = linker.link_internal_parameters(parameters)
        
        assert internal_dts == []
    
    def test_find_linked_parameters(self, parameter_linker):
        """Test finding parameters linked by a DatatypeMetadata."""
        dt_metadata = DatatypeMetadata(
            name="test",
            width="TEST_WIDTH",
            signed="TEST_SIGNED",
            bias="TEST_BIAS"
        )
        
        linked_params = parameter_linker.find_linked_parameters(dt_metadata)
        
        assert linked_params == {"TEST_WIDTH", "TEST_SIGNED", "TEST_BIAS"}
    
    def test_apply_to_kernel_metadata(self, parameter_linker):
        """Test applying parameter linking to kernel metadata."""
        # Create kernel metadata with interfaces
        kernel = KernelMetadata(
            name="test_kernel",
            source_file=Path("test.sv"),
            parameters=[
                Parameter("in0_WIDTH", default_value="32"),
                Parameter("in0_SIGNED", default_value="1"),
                Parameter("out0_WIDTH", default_value="32"),
                Parameter("THRESH_WIDTH", default_value="16"),
                Parameter("THRESH_SIGNED", default_value="1")
            ],
            interfaces=[
                InterfaceMetadata(
                    name="in0",
                    interface_type=InterfaceType.INPUT,
                    datatype_metadata=None  # Will be auto-linked
                ),
                InterfaceMetadata(
                    name="out0",
                    interface_type=InterfaceType.OUTPUT,
                    datatype_metadata=None  # Will be auto-linked
                )
            ],
            exposed_parameters=["in0_WIDTH", "in0_SIGNED", "out0_WIDTH", "THRESH_WIDTH", "THRESH_SIGNED"],
            pragmas=[],
            internal_datatypes=[]  # Will be auto-created
        )
        
        # Apply parameter linking
        parameter_linker.apply_to_kernel_metadata(kernel)
        
        # Check interface datatypes were linked
        in0_interface = next(i for i in kernel.interfaces if i.name == "in0")
        assert in0_interface.datatype_metadata is not None
        assert in0_interface.datatype_metadata.width == "in0_WIDTH"
        assert in0_interface.datatype_metadata.signed == "in0_SIGNED"
        
        out0_interface = next(i for i in kernel.interfaces if i.name == "out0")
        assert out0_interface.datatype_metadata is not None
        assert out0_interface.datatype_metadata.width == "out0_WIDTH"
        
        # Check internal datatypes were created
        assert kernel.internal_datatypes is not None
        assert len(kernel.internal_datatypes) == 1
        assert kernel.internal_datatypes[0].name == "THRESH"
        assert kernel.internal_datatypes[0].width == "THRESH_WIDTH"
        
        # Check exposed parameters were updated
        assert "in0_WIDTH" not in kernel.exposed_parameters
        assert "in0_SIGNED" not in kernel.exposed_parameters
        assert "out0_WIDTH" not in kernel.exposed_parameters
        assert "THRESH_WIDTH" not in kernel.exposed_parameters
        assert "THRESH_SIGNED" not in kernel.exposed_parameters
    
    def test_single_character_prefix(self, parameter_linker):
        """Test handling single character prefixes."""
        parameters = [
            Parameter("T_WIDTH", default_value="8"),
            Parameter("T_SIGNED", default_value="0"),
            Parameter("X_WIDTH", default_value="16")
        ]
        
        internal_dts = parameter_linker.link_internal_parameters(parameters)
        
        assert len(internal_dts) == 2
        
        # Check single character prefixes work
        t_dt = next(dt for dt in internal_dts if dt.name == "T")
        assert t_dt.width == "T_WIDTH"
        assert t_dt.signed == "T_SIGNED"
        
        x_dt = next(dt for dt in internal_dts if dt.name == "X")
        assert x_dt.width == "X_WIDTH"
    
    def test_no_prefix_parameters(self, parameter_linker):
        """Test parameters without prefixes are not linked."""
        parameters = [
            Parameter("WIDTH", default_value="32"),
            Parameter("SIGNED", default_value="1"),
            Parameter("FORMAT", default_value="FIXED")
        ]
        
        internal_dts = parameter_linker.link_internal_parameters(parameters)
        
        # No prefixes found, so no internal datatypes
        assert len(internal_dts) == 0
    
    def test_mixed_case_sensitivity(self, parameter_linker):
        """Test that parameter linking is case-sensitive."""
        parameters = [
            Parameter("data_WIDTH", default_value="32"),
            Parameter("DATA_WIDTH", default_value="16"),
            Parameter("Data_WIDTH", default_value="8")
        ]
        
        # Link for lowercase 'data'
        dt_metadata = parameter_linker.link_interface_parameters("data", parameters)
        assert dt_metadata is not None
        assert dt_metadata.width == "data_WIDTH"
        
        # Link for uppercase 'DATA'
        dt_metadata = parameter_linker.link_interface_parameters("DATA", parameters)
        assert dt_metadata is not None
        assert dt_metadata.width == "DATA_WIDTH"
        
        # Link for mixed case 'Data'
        dt_metadata = parameter_linker.link_interface_parameters("Data", parameters)
        assert dt_metadata is not None
        assert dt_metadata.width == "Data_WIDTH"