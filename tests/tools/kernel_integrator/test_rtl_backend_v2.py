"""Tests for RTL Backend generator V2."""
import pytest
from brainsmith.tools.kernel_integrator.generators.rtl_backend_v2 import RTLBackendGeneratorV2
from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata
from brainsmith.tools.kernel_integrator.types.rtl import Parameter, ParameterCategory, SourceType


def test_rtl_backend_generator_properties():
    """Test generator properties."""
    gen = RTLBackendGeneratorV2()
    assert gen.name == "rtl_backend"
    assert gen.template_file == "rtl_backend.py.j2"
    assert gen.output_pattern == "{kernel_name}_rtl.py"
    assert gen.get_output_filename("test_kernel") == "test_kernel_rtl.py"


def test_extract_rtl_nodeattrs():
    """Test extraction of RTL-specific node attributes."""
    gen = RTLBackendGeneratorV2()
    
    parameters = [
        # Algorithm parameter exposed as node attribute
        Parameter(
            name="algo_param",
            category=ParameterCategory.ALGORITHM,
            source_type=SourceType.RTL,
            default_value="42"
        ),
        # Path parameter (string type)
        Parameter(
            name="weights_PATH",
            category=ParameterCategory.ALGORITHM,
            source_type=SourceType.RTL
        ),
        # Control parameter with alias
        Parameter(
            name="ctrl_param",
            category=ParameterCategory.CONTROL,
            source_type=SourceType.NODEATTR_ALIAS,
            source_detail={"nodeattr_name": "control"}
        ),
        # Shape parameter (should be excluded)
        Parameter(
            name="BDIM",
            category=ParameterCategory.SHAPE,
            source_type=SourceType.RTL
        )
    ]
    
    metadata = KernelMetadata(
        name="test_kernel",
        interfaces=[],
        parameters=parameters,
        source_file="test.v"
    )
    
    nodeattrs = gen._extract_rtl_nodeattrs(metadata)
    
    # Should have 3 attributes (excluding shape parameter)
    assert len(nodeattrs) == 3
    assert "algo_param" in nodeattrs
    assert "weights_PATH" in nodeattrs
    assert "control" in nodeattrs  # Alias name
    
    # Check attribute specs
    assert nodeattrs["algo_param"] == ("i", True, 42)
    assert nodeattrs["weights_PATH"] == ("s", False, '')
    assert nodeattrs["control"] == ("i", True, None)


def test_generate_assignments():
    """Test parameter assignment generation."""
    gen = RTLBackendGeneratorV2()
    
    parameters = [
        # Algorithm parameter
        Parameter(
            name="algo_param",
            category=ParameterCategory.ALGORITHM,
            source_type=SourceType.RTL
        ),
        # Shape parameter from interface
        Parameter(
            name="M_BDIM",
            category=ParameterCategory.SHAPE,
            source_type=SourceType.INTERFACE_SHAPE,
            interface_name="input",
            source_detail={"dimension": 0, "shape_type": "bdim"}
        ),
        # Datatype width parameter
        Parameter(
            name="DT_WIDTH",
            category=ParameterCategory.DATATYPE,
            source_type=SourceType.INTERFACE_DATATYPE,
            interface_name="input",
            source_detail={"detail": "WIDTH"}
        )
    ]
    
    metadata = KernelMetadata(
        name="test_kernel",
        interfaces=[],
        parameters=parameters,
        source_file="test.v"
    )
    
    assignments = gen._generate_assignments(metadata)
    
    assert len(assignments) == 3
    
    # Check algorithm assignment
    algo_assign = next(a for a in assignments if a['template_var'] == 'algo_param')
    assert 'get_nodeattr("algo_param")' in algo_assign['assignment']
    
    # Check shape assignment
    shape_assign = next(a for a in assignments if a['template_var'] == 'M_BDIM')
    assert '_get_interface_bdim("input", 0)' in shape_assign['assignment']
    
    # Check datatype assignment
    dt_assign = next(a for a in assignments if a['template_var'] == 'DT_WIDTH')
    assert '_get_interface_datatype_width("input")' in dt_assign['assignment']


def test_create_shape_assignment():
    """Test shape parameter assignment creation."""
    gen = RTLBackendGeneratorV2()
    
    # BDIM parameter
    bdim_param = Parameter(
        name="B",
        category=ParameterCategory.SHAPE,
        source_type=SourceType.INTERFACE_SHAPE,
        interface_name="input",
        source_detail={"dimension": 0, "shape_type": "bdim"}
    )
    
    bdim_assign = gen._create_shape_assignment(bdim_param)
    assert bdim_assign['template_var'] == 'B'
    assert '_get_interface_bdim("input", 0)' in bdim_assign['assignment']
    
    # SDIM parameter
    sdim_param = Parameter(
        name="K",
        category=ParameterCategory.SHAPE,
        source_type=SourceType.INTERFACE_SHAPE,
        interface_name="input",
        source_detail={"dimension": 1, "shape_type": "sdim"}
    )
    
    sdim_assign = gen._create_shape_assignment(sdim_param)
    assert sdim_assign['template_var'] == 'K'
    assert '_get_interface_sdim("input", 1)' in sdim_assign['assignment']


def test_create_datatype_assignment():
    """Test datatype parameter assignment creation."""
    gen = RTLBackendGeneratorV2()
    
    # Width parameter
    width_param = Parameter(
        name="DT_W",
        category=ParameterCategory.DATATYPE,
        source_type=SourceType.INTERFACE_DATATYPE,
        interface_name="output",
        source_detail={"detail": "WIDTH"}
    )
    
    width_assign = gen._create_datatype_assignment(width_param)
    assert width_assign['template_var'] == 'DT_W'
    assert '_get_interface_datatype_width("output")' in width_assign['assignment']
    
    # Signedness parameter
    signed_param = Parameter(
        name="IS_SIGNED",
        category=ParameterCategory.DATATYPE,
        source_type=SourceType.INTERFACE_DATATYPE,
        interface_name="output",
        source_detail={"detail": "IS_SIGNED"}
    )
    
    signed_assign = gen._create_datatype_assignment(signed_param)
    assert signed_assign['template_var'] == 'IS_SIGNED'
    assert '_get_interface_datatype_is_signed("output")' in signed_assign['assignment']
    assert '"1" if' in signed_assign['assignment']  # Boolean to string conversion


def test_get_specific_vars():
    """Test complete specific variables generation."""
    gen = RTLBackendGeneratorV2()
    
    parameters = [
        Parameter(
            name="algo_param",
            category=ParameterCategory.ALGORITHM,
            source_type=SourceType.RTL
        )
    ]
    
    metadata = KernelMetadata(
        name="test_kernel",
        interfaces=[],
        parameters=parameters,
        source_file="test.v",
        top_module="test_top"
    )
    
    vars_dict = gen._get_specific_vars(metadata)
    
    assert vars_dict['class_name'] == 'TestKernel'
    assert vars_dict['source_file'] == 'test.v'
    assert vars_dict['finn_rtllib_module'] == 'test_top'
    assert len(vars_dict['rtl_specific_nodeattrs']) == 1
    assert len(vars_dict['explicit_parameter_assignments']) == 1
    assert vars_dict['supporting_rtl_files'] == []
    assert vars_dict['operation_description'] is None
    assert 'generation_timestamp' in vars_dict