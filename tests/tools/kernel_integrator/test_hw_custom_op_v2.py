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
    assert gen.template_file == "hw_custom_op_v2.py.j2"
    assert gen.output_pattern == "{kernel_name}.py"
    assert gen.get_output_filename("test_kernel") == "test_kernel.py"








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
    
    # Now only returns generation_timestamp
    assert 'generation_timestamp' in vars_dict
    assert len(vars_dict) == 1