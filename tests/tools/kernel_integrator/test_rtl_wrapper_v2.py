"""Tests for RTL Wrapper generator V2."""
import pytest
from brainsmith.tools.kernel_integrator.generators.rtl_wrapper_v2 import RTLWrapperGeneratorV2
from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata, InterfaceMetadata
from brainsmith.tools.kernel_integrator.types.rtl import Parameter, Port, PortDirection
from brainsmith.core.dataflow.types import InterfaceType


class MockDatatypeMetadata:
    """Mock datatype metadata for testing."""
    def __init__(self, name, params):
        self.name = name
        self._params = params
    
    def get_all_parameters(self):
        return self._params


def test_rtl_wrapper_generator_properties():
    """Test generator properties."""
    gen = RTLWrapperGeneratorV2()
    assert gen.name == "rtl_wrapper"
    assert gen.template_file == "rtl_wrapper_minimal_v2.v.j2"
    assert gen.output_pattern == "{kernel_name}_wrapper.v"
    assert gen.get_output_filename("test_kernel") == "test_kernel_wrapper.v"


def test_categorize_parameters_general():
    """Test categorization of general parameters."""
    gen = RTLWrapperGeneratorV2()
    
    # Create parameters
    parameters = [
        Parameter(name="algo_param1"),
        Parameter(name="algo_param2"),
        Parameter(name="algo_param3")
    ]
    
    metadata = KernelMetadata(
        name="test_kernel",
        interfaces=[],
        parameters=parameters,
        source_file="test.v",
        internal_datatypes=[],
        linked_parameters={}
    )
    
    categorized = gen._categorize_parameters(metadata)
    
    # All should be general parameters
    assert len(categorized['general_parameters']) == 3
    assert len(categorized['axilite_parameters']) == 0
    assert len(categorized['internal_datatype_groups']) == 0
    assert len(categorized['interface_parameter_groups']) == 0
    
    # Check template param names were added
    for param in categorized['general_parameters']:
        assert param['template_param_name'] == f"${param['name'].upper()}$"


def test_categorize_parameters_with_interfaces():
    """Test categorization with interface parameters."""
    gen = RTLWrapperGeneratorV2()
    
    # Create interface with shape parameters
    interface = InterfaceMetadata(
        name="input",
        interface_type=InterfaceType.INPUT,
        bdim_params=["B", "M"],
        sdim_params=["K", "1"]  # "1" should be skipped
    )
    
    # Create parameters
    parameters = [
        Parameter(name="B"),
        Parameter(name="M"),
        Parameter(name="K"),
        Parameter(name="algo_param")
    ]
    
    metadata = KernelMetadata(
        name="test_kernel",
        interfaces=[interface],
        parameters=parameters,
        source_file="test.v",
        internal_datatypes=[],
        linked_parameters={}
    )
    
    categorized = gen._categorize_parameters(metadata)
    
    # Check categorization
    assert len(categorized['general_parameters']) == 1
    assert categorized['general_parameters'][0]['name'] == "algo_param"
    
    assert len(categorized['interface_parameter_groups']) == 1
    group = categorized['interface_parameter_groups'][0]
    assert group['interface'].name == "input"
    assert len(group['parameters']) == 3
    
    # Check order: BDIM first (B, M), then SDIM (K)
    param_names = [p['name'] for p in group['parameters']]
    assert param_names == ["B", "M", "K"]


def test_categorize_parameters_with_internal_datatypes():
    """Test categorization with internal datatype parameters."""
    gen = RTLWrapperGeneratorV2()
    
    # Create mock internal datatype
    dt_meta = MockDatatypeMetadata("custom_dt", ["WIDTH", "SIGNED"])
    
    # Create parameters
    parameters = [
        Parameter(name="WIDTH"),
        Parameter(name="SIGNED"),
        Parameter(name="algo_param")
    ]
    
    metadata = KernelMetadata(
        name="test_kernel",
        interfaces=[],
        parameters=parameters,
        source_file="test.v",
        internal_datatypes=[dt_meta],
        linked_parameters={}
    )
    
    categorized = gen._categorize_parameters(metadata)
    
    # Check categorization
    assert len(categorized['general_parameters']) == 1
    assert len(categorized['internal_datatype_groups']) == 1
    assert "custom_dt" in categorized['internal_datatype_groups']
    
    dt_params = categorized['internal_datatype_groups']["custom_dt"]
    assert len(dt_params) == 2
    assert dt_params[0]['name'] == "WIDTH"
    assert dt_params[1]['name'] == "SIGNED"


def test_categorize_parameters_with_axilite():
    """Test categorization with AXI-Lite parameters."""
    gen = RTLWrapperGeneratorV2()
    
    # Create parameters
    parameters = [
        Parameter(name="axi_param1"),
        Parameter(name="axi_param2"),
        Parameter(name="algo_param")
    ]
    
    metadata = KernelMetadata(
        name="test_kernel",
        interfaces=[],
        parameters=parameters,
        source_file="test.v",
        internal_datatypes=[],
        linked_parameters={
            "axilite": {
                "axi_param1": {},
                "axi_param2": {}
            }
        }
    )
    
    categorized = gen._categorize_parameters(metadata)
    
    # Check categorization
    assert len(categorized['general_parameters']) == 1
    assert len(categorized['axilite_parameters']) == 2
    
    axi_names = [p['name'] for p in categorized['axilite_parameters']]
    assert "axi_param1" in axi_names
    assert "axi_param2" in axi_names




def test_get_specific_vars():
    """Test complete specific variables generation."""
    gen = RTLWrapperGeneratorV2()
    
    # Create simple metadata
    metadata = KernelMetadata(
        name="test_kernel",
        interfaces=[],
        parameters=[Parameter(name="test_param")],
        source_file="test.v",
        top_module="test_top",
        internal_datatypes=[],
        linked_parameters={}
    )
    
    vars_dict = gen._get_specific_vars(metadata)
    
    # Now only returns categorized_parameters and generation_timestamp
    assert 'categorized_parameters' in vars_dict
    assert 'generation_timestamp' in vars_dict
    assert len(vars_dict) == 2
    
    # Check categorized parameters structure
    cat_params = vars_dict['categorized_parameters']
    assert 'general_parameters' in cat_params
    assert 'axilite_parameters' in cat_params
    assert 'internal_datatype_groups' in cat_params
    assert 'interface_parameter_groups' in cat_params