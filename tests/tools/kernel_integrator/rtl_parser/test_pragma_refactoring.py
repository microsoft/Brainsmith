"""
Unit tests for refactored pragma system.

This module tests the behavior of pragmas after the refactoring to remove
source tracking and simplify the InterfacePragma pattern.
"""

import pytest
from brainsmith.core.dataflow.types import Direction, InterfaceType
from brainsmith.tools.kernel_integrator.types.metadata import (
    KernelMetadata, AXIStreamMetadata, AXILiteMetadata, ControlMetadata
)
from brainsmith.tools.kernel_integrator.types.rtl import Parameter, Port, PragmaType
from brainsmith.tools.kernel_integrator.rtl_parser.pragmas.dimension import BDimPragma, SDimPragma
from brainsmith.tools.kernel_integrator.rtl_parser.pragmas.interface import (
    DatatypePragma, DatatypeParamPragma, WeightPragma
)
from brainsmith.tools.kernel_integrator.rtl_parser.pragmas.parameter import (
    AliasPragma, DerivedParameterPragma, AxiLiteParamPragma
)
from brainsmith.tools.kernel_integrator.rtl_parser.pragmas.base import PragmaError


class TestPragmaRefactoring:
    """Test suite for the refactored pragma system."""
    
    def create_test_kernel(self):
        """Helper to create a basic test kernel with control interface."""
        control = ControlMetadata(name="control", ports={}, description=None)
        return KernelMetadata(
            name="test_kernel",
            source_file="test.sv",
            control=control,
            parameters=[],
            linked_parameters=[],
            inputs=[],
            outputs=[],
            config=[]
        )
    
    def create_axi_stream_interface(self, name: str, direction: Direction):
        """Helper to create an AXI-Stream interface."""
        return AXIStreamMetadata(
            name=name,
            ports={},
            description=None,
            direction=direction
        )
    
    def create_axi_lite_interface(self, name: str):
        """Helper to create an AXI-Lite interface."""
        return AXILiteMetadata(
            name=name,
            ports={},
            description=None
        )


class TestBDimPragma(TestPragmaRefactoring):
    """Test BDimPragma functionality."""
    
    def test_bdim_pragma_moves_parameters(self):
        """Test that BDIM pragma correctly moves parameters to interface."""
        # Create kernel with parameters
        kernel = self.create_test_kernel()
        kernel.parameters = [
            Parameter(name="TILE_H", default_value="16"),
            Parameter(name="TILE_W", default_value="32"),
            Parameter(name="OTHER_PARAM", default_value="8")
        ]
        
        # Create interface
        input_interface = self.create_axi_stream_interface("input0", Direction.INPUT)
        kernel.inputs.append(input_interface)
        
        # Create and apply pragma
        pragma = BDimPragma(
            type=PragmaType.BDIM,
            inputs={'positional': ['input0', ['TILE_H', 'TILE_W']]}
        )
        pragma.parsed_data = pragma._parse_inputs()
        pragma.apply_to_kernel(kernel)
        
        # Assert parameters moved
        assert len(kernel.parameters) == 1
        assert kernel.parameters[0].name == "OTHER_PARAM"
        assert len(input_interface.bdim_params) == 2
        assert input_interface.bdim_params[0].name == "TILE_H"
        assert input_interface.bdim_params[1].name == "TILE_W"
        assert input_interface.bdim_shape == [":", ":"]
    
    def test_bdim_pragma_with_shape(self):
        """Test BDIM pragma with explicit SHAPE specification."""
        kernel = self.create_test_kernel()
        kernel.parameters = [
            Parameter(name="BDIM0", default_value="16"),
            Parameter(name="BDIM1", default_value="32"),
            Parameter(name="BDIM2", default_value="3")
        ]
        
        output_interface = self.create_axi_stream_interface("output0", Direction.OUTPUT)
        kernel.outputs.append(output_interface)
        
        # Create pragma with SHAPE
        pragma = BDimPragma(
            type=PragmaType.BDIM,
            inputs={
                'positional': ['output0', ['BDIM0', 'BDIM1', 'BDIM2']],
                'named': {'SHAPE': ['TILE_H', 'TILE_W', ':']}
            }
        )
        pragma.parsed_data = pragma._parse_inputs()
        pragma.apply_to_kernel(kernel)
        
        # Assert shape applied correctly
        assert output_interface.bdim_shape == ['TILE_H', 'TILE_W', ':']
        assert len(output_interface.bdim_params) == 3
    
    def test_bdim_pragma_singleton_dimensions(self):
        """Test BDIM pragma with singleton dimensions."""
        kernel = self.create_test_kernel()
        kernel.parameters = [Parameter(name="KERNEL_SIZE", default_value="64")]
        
        weight_interface = self.create_axi_stream_interface("weights", Direction.INPUT)
        weight_interface.is_weight = True
        kernel.inputs.append(weight_interface)
        
        pragma = BDimPragma(
            type=PragmaType.BDIM,
            inputs={'positional': ['weights', ['1', 'KERNEL_SIZE']]}
        )
        pragma.parsed_data = pragma._parse_inputs()
        pragma.apply_to_kernel(kernel)
        
        # Only KERNEL_SIZE should be moved
        assert len(kernel.parameters) == 0
        assert len(weight_interface.bdim_params) == 1
        assert weight_interface.bdim_params[0].name == "KERNEL_SIZE"
    
    def test_bdim_pragma_rejects_control_interface(self):
        """Test that BDIM pragma rejects CONTROL interfaces."""
        kernel = self.create_test_kernel()
        kernel.parameters = [Parameter(name="SOME_PARAM", default_value="8")]
        
        pragma = BDimPragma(
            type=PragmaType.BDIM,
            inputs={'positional': ['control', ['SOME_PARAM']]}
        )
        pragma.parsed_data = pragma._parse_inputs()
        
        with pytest.raises(PragmaError, match="BDIM pragmas are only allowed on INPUT, OUTPUT, or WEIGHT"):
            pragma.apply_to_kernel(kernel)


class TestSDimPragma(TestPragmaRefactoring):
    """Test SDimPragma functionality."""
    
    def test_sdim_pragma_moves_parameters(self):
        """Test that SDIM pragma correctly moves parameters to interface."""
        kernel = self.create_test_kernel()
        kernel.parameters = [
            Parameter(name="STREAM_H", default_value="8"),
            Parameter(name="STREAM_W", default_value="16")
        ]
        
        input_interface = self.create_axi_stream_interface("input0", Direction.INPUT)
        kernel.inputs.append(input_interface)
        
        pragma = SDimPragma(
            type=PragmaType.SDIM,
            inputs={'positional': ['input0', ['STREAM_H', 'STREAM_W']]}
        )
        pragma.parsed_data = pragma._parse_inputs()
        pragma.apply_to_kernel(kernel)
        
        assert len(kernel.parameters) == 0
        assert len(input_interface.sdim_params) == 2
        assert input_interface.sdim_shape == ['STREAM_H', 'STREAM_W']  # Default uses param names
    
    def test_sdim_pragma_rejects_output_interface(self):
        """Test that SDIM pragma rejects OUTPUT interfaces."""
        kernel = self.create_test_kernel()
        kernel.parameters = [Parameter(name="SDIM", default_value="8")]
        
        output_interface = self.create_axi_stream_interface("output0", Direction.OUTPUT)
        kernel.outputs.append(output_interface)
        
        pragma = SDimPragma(
            type=PragmaType.SDIM,
            inputs={'positional': ['output0', ['SDIM']]}
        )
        pragma.parsed_data = pragma._parse_inputs()
        
        with pytest.raises(PragmaError, match="SDIM pragmas are only allowed on INPUT or WEIGHT"):
            pragma.apply_to_kernel(kernel)


class TestDatatypeParamPragma(TestPragmaRefactoring):
    """Test DatatypeParamPragma functionality."""
    
    def test_datatype_param_moves_width_parameter(self):
        """Test DATATYPE_PARAM pragma for width property."""
        kernel = self.create_test_kernel()
        kernel.parameters = [
            Parameter(name="INPUT0_WIDTH", default_value="8"),
            Parameter(name="OTHER_PARAM", default_value="16")
        ]
        
        input_interface = self.create_axi_stream_interface("input0", Direction.INPUT)
        kernel.inputs.append(input_interface)
        
        pragma = DatatypeParamPragma(
            type=PragmaType.DATATYPE_PARAM,
            inputs={'positional': ['input0', 'width', 'INPUT0_WIDTH']}
        )
        pragma.parsed_data = pragma._parse_inputs()
        pragma.apply_to_kernel(kernel)
        
        assert len(kernel.parameters) == 1
        assert kernel.parameters[0].name == "OTHER_PARAM"
        assert input_interface.dtype_params is not None
        assert input_interface.dtype_params.width is not None
        assert input_interface.dtype_params.width.name == "INPUT0_WIDTH"
        assert input_interface.dtype_params.width.kernel_value == "width"
    
    def test_datatype_param_moves_signed_parameter(self):
        """Test DATATYPE_PARAM pragma for signed property."""
        kernel = self.create_test_kernel()
        kernel.parameters = [Parameter(name="SIGNED_INPUT0", default_value="1")]
        
        input_interface = self.create_axi_stream_interface("input0", Direction.INPUT)
        kernel.inputs.append(input_interface)
        
        pragma = DatatypeParamPragma(
            type=PragmaType.DATATYPE_PARAM,
            inputs={'positional': ['input0', 'signed', 'SIGNED_INPUT0']}
        )
        pragma.parsed_data = pragma._parse_inputs()
        pragma.apply_to_kernel(kernel)
        
        assert input_interface.dtype_params is not None
        assert input_interface.dtype_params.signed is not None
        assert input_interface.dtype_params.signed.name == "SIGNED_INPUT0"
        assert input_interface.dtype_params.signed.kernel_value == "signed"
    
    def test_datatype_param_works_on_config_interface(self):
        """Test DATATYPE_PARAM pragma on CONFIG interface."""
        kernel = self.create_test_kernel()
        kernel.parameters = [Parameter(name="CONFIG_WIDTH", default_value="32")]
        
        config_interface = self.create_axi_lite_interface("config0")
        kernel.config.append(config_interface)
        
        pragma = DatatypeParamPragma(
            type=PragmaType.DATATYPE_PARAM,
            inputs={'positional': ['config0', 'width', 'CONFIG_WIDTH']}
        )
        pragma.parsed_data = pragma._parse_inputs()
        pragma.apply_to_kernel(kernel)
        
        assert config_interface.dtype_params is not None
        assert config_interface.dtype_params.width is not None
        assert config_interface.dtype_params.width.name == "CONFIG_WIDTH"
        assert config_interface.dtype_params.width.kernel_value == "width"


class TestAliasPragma(TestPragmaRefactoring):
    """Test AliasPragma functionality."""
    
    def test_alias_pragma_sets_kernel_value(self):
        """Test ALIAS pragma sets kernel_value on parameter."""
        kernel = self.create_test_kernel()
        kernel.parameters = [
            Parameter(name="PE", default_value="16"),
            Parameter(name="OTHER", default_value="8")
        ]
        
        pragma = AliasPragma(
            type=PragmaType.ALIAS,
            inputs={'positional': ['PE', 'parallelism_factor']}
        )
        pragma.parsed_data = pragma._parse_inputs()
        pragma.apply_to_kernel(kernel)
        
        # Parameter should stay in kernel.parameters but have kernel_value set
        assert len(kernel.parameters) == 2
        pe_param = next(p for p in kernel.parameters if p.name == "PE")
        assert pe_param.kernel_value == "parallelism_factor"
        assert pe_param.nodeattr_name == "parallelism_factor"
    
    def test_alias_pragma_validates_conflicts(self):
        """Test ALIAS pragma validates nodeattr name conflicts."""
        kernel = self.create_test_kernel()
        kernel.parameters = [
            Parameter(name="PE", default_value="16"),
            Parameter(name="parallelism_factor", default_value="8")  # Conflict!
        ]
        
        pragma = AliasPragma(
            type=PragmaType.ALIAS,
            inputs={'positional': ['PE', 'parallelism_factor']}
        )
        pragma.parsed_data = pragma._parse_inputs()
        
        with pytest.raises(PragmaError, match="conflicts with existing parameter"):
            pragma.apply_to_kernel(kernel)


class TestDerivedParameterPragma(TestPragmaRefactoring):
    """Test DerivedParameterPragma functionality."""
    
    def test_derived_parameter_adds_to_linked_parameters(self):
        """Test DERIVED_PARAMETER pragma adds to linked_parameters."""
        kernel = self.create_test_kernel()
        
        pragma = DerivedParameterPragma(
            type=PragmaType.DERIVED_PARAMETER,
            inputs={'positional': ['SIMD', 'self.get_input_datatype().bitwidth()']}
        )
        pragma.parsed_data = pragma._parse_inputs()
        pragma.apply_to_kernel(kernel)
        
        assert len(kernel.linked_parameters) == 1
        assert kernel.linked_parameters[0].name == "SIMD"
        assert kernel.linked_parameters[0].kernel_value == "self.get_input_datatype().bitwidth()"
    
    def test_multiple_derived_parameters(self):
        """Test multiple DERIVED_PARAMETER pragmas."""
        kernel = self.create_test_kernel()
        
        # Add first derived parameter
        pragma1 = DerivedParameterPragma(
            type=PragmaType.DERIVED_PARAMETER,
            inputs={'positional': ['SIMD', 'self.get_nodeattr("PE") * 2']}
        )
        pragma1.parsed_data = pragma1._parse_inputs()
        pragma1.apply_to_kernel(kernel)
        
        # Add second derived parameter
        pragma2 = DerivedParameterPragma(
            type=PragmaType.DERIVED_PARAMETER,
            inputs={'positional': ['MEM_DEPTH', 'self.calc_wmem()']}
        )
        pragma2.parsed_data = pragma2._parse_inputs()
        pragma2.apply_to_kernel(kernel)
        
        assert len(kernel.linked_parameters) == 2
        assert kernel.linked_parameters[0].name == "SIMD"
        assert kernel.linked_parameters[1].name == "MEM_DEPTH"


class TestAxiLiteParamPragma(TestPragmaRefactoring):
    """Test AxiLiteParamPragma functionality."""
    
    def test_axilite_param_enable(self):
        """Test AXILITE_PARAM pragma for enable property."""
        kernel = self.create_test_kernel()
        kernel.parameters = [Parameter(name="USE_AXILITE", default_value="1")]
        
        config_interface = self.create_axi_lite_interface("s_axilite_config")
        kernel.config.append(config_interface)
        
        pragma = AxiLiteParamPragma(
            type=PragmaType.AXILITE_PARAM,
            inputs={'positional': ['USE_AXILITE', 's_axilite_config', 'enable']}
        )
        pragma.parsed_data = pragma._parse_inputs()
        pragma.apply_to_kernel(kernel)
        
        assert len(kernel.parameters) == 0
        assert config_interface.enable_param is not None
        assert config_interface.enable_param.name == "USE_AXILITE"
    
    def test_axilite_param_data_width(self):
        """Test AXILITE_PARAM pragma for data_width property."""
        kernel = self.create_test_kernel()
        kernel.parameters = [Parameter(name="AXILITE_DATA_W", default_value="32")]
        
        config_interface = self.create_axi_lite_interface("s_axilite_config")
        kernel.config.append(config_interface)
        
        pragma = AxiLiteParamPragma(
            type=PragmaType.AXILITE_PARAM,
            inputs={'positional': ['AXILITE_DATA_W', 's_axilite_config', 'data_width']}
        )
        pragma.parsed_data = pragma._parse_inputs()
        pragma.apply_to_kernel(kernel)
        
        assert config_interface.data_width_param is not None
        assert config_interface.data_width_param.name == "AXILITE_DATA_W"
    
    def test_axilite_param_addr_width(self):
        """Test AXILITE_PARAM pragma for addr_width property."""
        kernel = self.create_test_kernel()
        kernel.parameters = [Parameter(name="AXILITE_ADDR_W", default_value="16")]
        
        config_interface = self.create_axi_lite_interface("s_axilite_config")
        kernel.config.append(config_interface)
        
        pragma = AxiLiteParamPragma(
            type=PragmaType.AXILITE_PARAM,
            inputs={'positional': ['AXILITE_ADDR_W', 's_axilite_config', 'addr_width']}
        )
        pragma.parsed_data = pragma._parse_inputs()
        pragma.apply_to_kernel(kernel)
        
        assert config_interface.addr_width_param is not None
        assert config_interface.addr_width_param.name == "AXILITE_ADDR_W"


class TestWeightPragma(TestPragmaRefactoring):
    """Test WeightPragma functionality."""
    
    def test_weight_pragma_marks_interface(self):
        """Test WEIGHT pragma marks interface as weight."""
        kernel = self.create_test_kernel()
        
        weights_interface = self.create_axi_stream_interface("weights_V", Direction.INPUT)
        kernel.inputs.append(weights_interface)
        
        pragma = WeightPragma(
            type=PragmaType.WEIGHT,
            inputs={'positional': ['weights_V']}
        )
        pragma.parsed_data = pragma._parse_inputs()
        pragma.apply_to_kernel(kernel)
        
        assert weights_interface.is_weight is True
    
    def test_weight_pragma_multiple_interfaces(self):
        """Test WEIGHT pragma with multiple interfaces."""
        kernel = self.create_test_kernel()
        
        weights1 = self.create_axi_stream_interface("weights1", Direction.INPUT)
        weights2 = self.create_axi_stream_interface("weights2", Direction.INPUT)
        bias = self.create_axi_stream_interface("bias", Direction.INPUT)
        kernel.inputs.extend([weights1, weights2, bias])
        
        pragma = WeightPragma(
            type=PragmaType.WEIGHT,
            inputs={'positional': ['weights1', 'weights2', 'bias']}
        )
        pragma.parsed_data = pragma._parse_inputs()
        pragma.apply_to_kernel(kernel)
        
        assert weights1.is_weight is True
        assert weights2.is_weight is True
        assert bias.is_weight is True


class TestDatatypePragma(TestPragmaRefactoring):
    """Test DatatypePragma functionality."""
    
    def test_datatype_pragma_adds_constraints(self):
        """Test DATATYPE pragma adds constraints to interface."""
        kernel = self.create_test_kernel()
        
        input_interface = self.create_axi_stream_interface("input0", Direction.INPUT)
        kernel.inputs.append(input_interface)
        
        pragma = DatatypePragma(
            type=PragmaType.DATATYPE,
            inputs={'positional': ['input0', 'UINT', '8', '16']}
        )
        pragma.parsed_data = pragma._parse_inputs()
        pragma.apply_to_kernel(kernel)
        
        assert hasattr(input_interface, 'datatype_constraints')
        assert len(input_interface.datatype_constraints) == 1
        constraint = input_interface.datatype_constraints[0]
        assert constraint.base_type == 'UINT'
        assert constraint.min_width == 8
        assert constraint.max_width == 16
    
    def test_datatype_pragma_multiple_types(self):
        """Test DATATYPE pragma with multiple allowed types."""
        kernel = self.create_test_kernel()
        
        input_interface = self.create_axi_stream_interface("input0", Direction.INPUT)
        kernel.inputs.append(input_interface)
        
        pragma = DatatypePragma(
            type=PragmaType.DATATYPE,
            inputs={'positional': ['input0', ['INT', 'UINT', 'FIXED'], '1', '32']}
        )
        pragma.parsed_data = pragma._parse_inputs()
        pragma.apply_to_kernel(kernel)
        
        assert len(input_interface.datatype_constraints) == 3
    
    def test_datatype_pragma_wildcard(self):
        """Test DATATYPE pragma with wildcard type."""
        kernel = self.create_test_kernel()
        
        output_interface = self.create_axi_stream_interface("output0", Direction.OUTPUT)
        kernel.outputs.append(output_interface)
        
        pragma = DatatypePragma(
            type=PragmaType.DATATYPE,
            inputs={'positional': ['output0', '*', '16', '16']}
        )
        pragma.parsed_data = pragma._parse_inputs()
        pragma.apply_to_kernel(kernel)
        
        assert len(output_interface.datatype_constraints) == 1
        assert output_interface.datatype_constraints[0].base_type == 'ANY'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])