"""Tests for HWCustomOp generator V2."""
import pytest
from brainsmith.tools.kernel_integrator.generators.hw_custom_op_v2 import HWCustomOpGeneratorV2
from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata, InterfaceMetadata
from brainsmith.tools.kernel_integrator.types.rtl import Parameter, ParameterCategory
from brainsmith.core.dataflow.types import InterfaceType


def test_hw_custom_op_generator_properties():
    """Test generator properties."""
    gen = HWCustomOpGeneratorV2()
    assert gen.name == "hw_custom_op"
    assert gen.template_file == "hw_custom_op.py.j2"
    assert gen.output_pattern == "{kernel_name}.py"
    assert gen.get_output_filename("test_kernel") == "test_kernel.py"


def test_extract_shape_nodeattrs():
    """Test extraction of shape node attributes."""
    gen = HWCustomOpGeneratorV2()
    
    # Create test interfaces with shape expressions
    interfaces = [
        InterfaceMetadata(
            name="input",
            interface_type=InterfaceType.INPUT,
            bdim_shape=["B", "M"],
            sdim_shape=["K", "N"]
        ),
        InterfaceMetadata(
            name="output",
            interface_type=InterfaceType.OUTPUT,
            bdim_shape=["B", "P"],
            sdim_shape=[":", "N"]  # ":" should be skipped
        )
    ]
    
    metadata = KernelMetadata(
        name="test_kernel",
        interfaces=interfaces,
        parameters=[],
        source_file="test.v"
    )
    
    shape_attrs = gen._extract_shape_nodeattrs(metadata)
    
    # Should have 5 unique parameters: B, M, K, N, P
    assert len(shape_attrs) == 5
    
    # Check they're sorted by name
    names = [attr['name'] for attr in shape_attrs]
    assert names == ['B', 'K', 'M', 'N', 'P']
    
    # Check source comments
    b_attr = next(attr for attr in shape_attrs if attr['name'] == 'B')
    assert 'BDIM: input' in b_attr['source_comment']
    assert 'BDIM: output' in b_attr['source_comment']


def test_extract_datatype_attrs():
    """Test extraction of datatype attributes."""
    gen = HWCustomOpGeneratorV2()
    
    # Create parameters with datatype category
    parameters = [
        Parameter(
            name="input_dtype",
            category=ParameterCategory.DATATYPE,
            interface_name="input"
        ),
        Parameter(
            name="output_dtype",
            category=ParameterCategory.DATATYPE,
            interface_name="output"
        ),
        Parameter(
            name="algo_param",
            category=ParameterCategory.ALGORITHM,
            interface_name=None
        )
    ]
    
    metadata = KernelMetadata(
        name="test_kernel",
        interfaces=[],
        parameters=parameters,
        source_file="test.v"
    )
    
    datatype_attrs = gen._extract_datatype_attrs(metadata)
    
    assert len(datatype_attrs) == 2
    
    # Check attribute names
    attr_names = [attr['name'] for attr in datatype_attrs]
    assert 'inputDataType' in attr_names
    assert 'outputDataType' in attr_names
    
    # Check attribute spec
    input_attr = next(attr for attr in datatype_attrs if attr['name'] == 'inputDataType')
    assert input_attr['attr_spec'] == ("s", True, "")


def test_has_datatype_params():
    """Test datatype parameter detection."""
    gen = HWCustomOpGeneratorV2()
    
    # With datatype parameters
    metadata1 = KernelMetadata(
        name="test_kernel",
        interfaces=[],
        parameters=[Parameter(name="dt", category=ParameterCategory.DATATYPE)],
        source_file="test.v"
    )
    assert gen._has_datatype_params(metadata1) is True
    
    # Without datatype parameters
    metadata2 = KernelMetadata(
        name="test_kernel",
        interfaces=[],
        parameters=[Parameter(name="algo", category=ParameterCategory.ALGORITHM)],
        source_file="test.v"
    )
    assert gen._has_datatype_params(metadata2) is False


def test_get_specific_vars():
    """Test complete specific variables generation."""
    gen = HWCustomOpGeneratorV2()
    
    # Create comprehensive test metadata
    interfaces = [
        InterfaceMetadata(
            name="input",
            interface_type=InterfaceType.INPUT,
            bdim_shape=["B", "M"]
        ),
        InterfaceMetadata(
            name="output",
            interface_type=InterfaceType.OUTPUT
        ),
        InterfaceMetadata(
            name="weights",
            interface_type=InterfaceType.WEIGHT
        )
    ]
    
    parameters = [
        Parameter(
            name="input_dtype",
            category=ParameterCategory.DATATYPE,
            interface_name="input"
        )
    ]
    
    metadata = KernelMetadata(
        name="test_kernel",
        interfaces=interfaces,
        parameters=parameters,
        source_file="test.v",
        exposed_parameters=["B", "M"]
    )
    
    vars_dict = gen._get_specific_vars(metadata)
    
    # Check extracted attributes
    assert len(vars_dict['explicit_datatype_attrs']) == 1
    assert len(vars_dict['shape_nodeattrs']) == 2
    assert vars_dict['has_datatype_params'] is True
    
    # Check metadata
    assert vars_dict['class_name'] == 'TestKernel'
    assert vars_dict['source_file'] == 'test.v'
    
    # Check verification
    assert vars_dict['verification_required'] is True
    assert vars_dict['required_attributes'] == ["B", "M"]
    
    # Check timestamp exists
    assert 'generation_timestamp' in vars_dict