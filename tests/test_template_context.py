"""
Tests for template context data structures and builders.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from brainsmith.tools.hw_kernel_gen.template_context import (
    BaseContext,
    InterfaceInfo,
    ParameterInfo,
    HWCustomOpContext,
    RTLBackendContext,
    TemplateContextBuilder,
    create_context_builder
)
from brainsmith.tools.hw_kernel_gen.config import PipelineConfig, GeneratorType
from brainsmith.tools.hw_kernel_gen.errors import ValidationError, CodeGenerationError


class TestBaseContext:
    """Test BaseContext class."""
    
    def test_default_base_context(self):
        """Test default base context initialization."""
        context = BaseContext()
        
        assert context.context_type == "BaseContext"
        assert context.generated_timestamp == ""
        assert context.generator_version == "1.0.0"
        assert context.module_name == ""
        assert context.file_name == ""
        assert context.class_name == ""
        assert context.include_debug_info is False
        assert context.include_documentation is True
        assert context.include_type_hints is True
    
    def test_base_context_with_values(self):
        """Test base context with provided values."""
        context = BaseContext(
            module_name="test_module",
            file_name="test_file.py",
            class_name="TestClass",
            include_debug_info=True
        )
        
        assert context.module_name == "test_module"
        assert context.file_name == "test_file.py"
        assert context.class_name == "TestClass"
        assert context.include_debug_info is True
    
    def test_base_context_to_dict(self):
        """Test converting base context to dictionary."""
        context = BaseContext(module_name="test", file_name="test.py")
        data = context.to_dict()
        
        assert isinstance(data, dict)
        assert data['module_name'] == "test"
        assert data['file_name'] == "test.py"
        assert data['context_type'] == "BaseContext"
    
    def test_base_context_validation_success(self):
        """Test successful base context validation."""
        context = BaseContext(module_name="test", file_name="test.py")
        # Should not raise an exception
        context.validate()
    
    def test_base_context_validation_missing_module_name(self):
        """Test base context validation with missing module name."""
        context = BaseContext(file_name="test.py")
        
        with pytest.raises(ValidationError) as exc_info:
            context.validate()
        
        assert "Module name is required" in str(exc_info.value)
    
    def test_base_context_validation_missing_file_name(self):
        """Test base context validation with missing file name."""
        context = BaseContext(module_name="test")
        
        with pytest.raises(ValidationError) as exc_info:
            context.validate()
        
        assert "File name is required" in str(exc_info.value)


class TestInterfaceInfo:
    """Test InterfaceInfo class."""
    
    def test_basic_interface_info(self):
        """Test basic interface info initialization."""
        interface = InterfaceInfo(name="data_in", direction="input")
        
        assert interface.name == "data_in"
        assert interface.direction == "input"
        assert interface.width is None
        assert interface.type == "wire"
        assert interface.description == ""
        assert interface.is_clock is False
        assert interface.is_reset is False
        assert interface.is_control is False
        assert interface.is_axi is False
        assert interface.axi_type is None
        assert interface.axi_signals == []
    
    def test_clock_signal_detection(self):
        """Test automatic clock signal detection."""
        interface = InterfaceInfo(name="ap_clk", direction="input")
        assert interface.is_clock is True
        
        interface = InterfaceInfo(name="system_clock", direction="input")
        assert interface.is_clock is True
    
    def test_reset_signal_detection(self):
        """Test automatic reset signal detection."""
        interface = InterfaceInfo(name="ap_rst_n", direction="input")
        assert interface.is_reset is True
        
        interface = InterfaceInfo(name="system_reset", direction="input")
        assert interface.is_reset is True
    
    def test_control_signal_detection(self):
        """Test automatic control signal detection."""
        interface = InterfaceInfo(name="ap_start", direction="input")
        assert interface.is_control is True
        
        interface = InterfaceInfo(name="ctrl_enable", direction="input")
        assert interface.is_control is True
    
    def test_axi_signal_detection(self):
        """Test automatic AXI signal detection."""
        interface = InterfaceInfo(name="s_axis_tdata", direction="input")
        assert interface.is_axi is True
        assert interface.axi_type == "s_axis"
        
        interface = InterfaceInfo(name="m_axis_tvalid", direction="output")
        assert interface.is_axi is True
        assert interface.axi_type == "m_axis"
        
        interface = InterfaceInfo(name="s_axi_awaddr", direction="input")
        assert interface.is_axi is True
        assert interface.axi_type == "s_axi"


class TestParameterInfo:
    """Test ParameterInfo class."""
    
    def test_basic_parameter_info(self):
        """Test basic parameter info initialization."""
        param = ParameterInfo(name="WIDTH", value=32)
        
        assert param.name == "WIDTH"
        assert param.value == 32
        assert param.type == "int"
        assert param.description == ""
        assert param.is_configurable is True
    
    def test_parameter_type_inference(self):
        """Test automatic parameter type inference."""
        # Boolean parameter
        param = ParameterInfo(name="ENABLE", value=True)
        assert param.type == "bool"
        
        # String parameter
        param = ParameterInfo(name="MODE", value="fast")
        assert param.type == "string"
        
        # Float parameter
        param = ParameterInfo(name="RATIO", value=1.5)
        assert param.type == "float"
        
        # Integer parameter (default)
        param = ParameterInfo(name="COUNT", value=10)
        assert param.type == "int"


class TestHWCustomOpContext:
    """Test HWCustomOpContext class."""
    
    def test_default_hw_custom_op_context(self):
        """Test default HW custom op context."""
        context = HWCustomOpContext()
        
        assert context.context_type == "HWCustomOpContext"
        assert context.rtl_file_path == ""
        assert context.top_module_name == ""
        assert context.interfaces == []
        assert context.input_interfaces == []
        assert context.output_interfaces == []
        assert context.parameters == []
        assert context.finn_datatype == "float32"
        assert context.input_shape == []
        assert context.output_shape == []
        assert context.generate_wrapper is True
        assert context.generate_testbench is False
        assert context.tdims == []
        assert context.tensor_configs == {}
    
    def test_hw_custom_op_context_class_name_derivation(self):
        """Test automatic class name derivation from module name."""
        context = HWCustomOpContext(top_module_name="my_custom_module")
        assert context.class_name == "MyCustomModule"
    
    def test_hw_custom_op_context_file_name_derivation(self):
        """Test automatic file name derivation from module name."""
        context = HWCustomOpContext(module_name="test_module")
        assert context.file_name == "test_module.py"
    
    def test_hw_custom_op_context_interface_separation(self):
        """Test interface separation by direction."""
        interfaces = [
            InterfaceInfo(name="data_in", direction="input"),
            InterfaceInfo(name="data_out", direction="output"),
            InterfaceInfo(name="enable", direction="input")
        ]
        
        context = HWCustomOpContext(interfaces=interfaces)
        
        assert len(context.input_interfaces) == 2
        assert len(context.output_interfaces) == 1
        assert context.input_interfaces[0].name == "data_in"
        assert context.input_interfaces[1].name == "enable"
        assert context.output_interfaces[0].name == "data_out"
    
    def test_get_axi_interfaces(self):
        """Test AXI interface grouping."""
        interfaces = [
            InterfaceInfo(name="s_axis_tdata", direction="input"),
            InterfaceInfo(name="s_axis_tvalid", direction="input"),
            InterfaceInfo(name="m_axis_tdata", direction="output"),
            InterfaceInfo(name="regular_signal", direction="input")
        ]
        
        context = HWCustomOpContext(interfaces=interfaces)
        axi_interfaces = context.get_axi_interfaces()
        
        assert "s_axis" in axi_interfaces
        assert "m_axis" in axi_interfaces
        assert len(axi_interfaces["s_axis"]) == 2
        assert len(axi_interfaces["m_axis"]) == 1
    
    def test_get_control_interfaces(self):
        """Test control interface filtering."""
        interfaces = [
            InterfaceInfo(name="ap_clk", direction="input"),
            InterfaceInfo(name="ap_rst_n", direction="input"),
            InterfaceInfo(name="ap_start", direction="input"),
            InterfaceInfo(name="data_in", direction="input")
        ]
        
        context = HWCustomOpContext(interfaces=interfaces)
        control_interfaces = context.get_control_interfaces()
        
        assert len(control_interfaces) == 3
        control_names = [iface.name for iface in control_interfaces]
        assert "ap_clk" in control_names
        assert "ap_rst_n" in control_names
        assert "ap_start" in control_names
        assert "data_in" not in control_names
    
    def test_get_data_interfaces(self):
        """Test data interface filtering."""
        interfaces = [
            InterfaceInfo(name="ap_clk", direction="input"),
            InterfaceInfo(name="data_in", direction="input"),
            InterfaceInfo(name="data_out", direction="output")
        ]
        
        context = HWCustomOpContext(interfaces=interfaces)
        data_interfaces = context.get_data_interfaces()
        
        assert len(data_interfaces) == 2
        data_names = [iface.name for iface in data_interfaces]
        assert "data_in" in data_names
        assert "data_out" in data_names
        assert "ap_clk" not in data_names
    
    def test_hw_custom_op_validation_success(self):
        """Test successful HW custom op validation."""
        interfaces = [
            InterfaceInfo(name="ap_clk", direction="input"),
            InterfaceInfo(name="ap_rst_n", direction="input"),
            InterfaceInfo(name="data_in", direction="input")
        ]
        
        context = HWCustomOpContext(
            module_name="test",
            file_name="test.py",
            top_module_name="test_top",
            interfaces=interfaces
        )
        
        # Should not raise an exception
        context.validate()
    
    def test_hw_custom_op_validation_missing_top_module(self):
        """Test validation with missing top module name."""
        context = HWCustomOpContext(
            module_name="test",
            file_name="test.py"
        )
        
        with pytest.raises(ValidationError) as exc_info:
            context.validate()
        
        assert "Top module name is required" in str(exc_info.value)
    
    def test_hw_custom_op_validation_no_interfaces(self):
        """Test validation with no interfaces."""
        context = HWCustomOpContext(
            module_name="test",
            file_name="test.py",
            top_module_name="test_top"
        )
        
        with pytest.raises(ValidationError) as exc_info:
            context.validate()
        
        assert "At least one interface is required" in str(exc_info.value)
    
    def test_hw_custom_op_validation_missing_clock(self):
        """Test validation with missing clock signal."""
        interfaces = [
            InterfaceInfo(name="ap_rst_n", direction="input"),
            InterfaceInfo(name="data_in", direction="input")
        ]
        
        context = HWCustomOpContext(
            module_name="test",
            file_name="test.py",
            top_module_name="test_top",
            interfaces=interfaces
        )
        
        with pytest.raises(ValidationError) as exc_info:
            context.validate()
        
        assert "Clock signal not found" in str(exc_info.value)
    
    def test_hw_custom_op_validation_missing_reset(self):
        """Test validation with missing reset signal."""
        interfaces = [
            InterfaceInfo(name="ap_clk", direction="input"),
            InterfaceInfo(name="data_in", direction="input")
        ]
        
        context = HWCustomOpContext(
            module_name="test",
            file_name="test.py",
            top_module_name="test_top",
            interfaces=interfaces
        )
        
        with pytest.raises(ValidationError) as exc_info:
            context.validate()
        
        assert "Reset signal not found" in str(exc_info.value)


class TestRTLBackendContext:
    """Test RTLBackendContext class."""
    
    def test_default_rtl_backend_context(self):
        """Test default RTL backend context."""
        context = RTLBackendContext()
        
        assert context.context_type == "RTLBackendContext"
        assert context.rtl_files == []
        assert context.main_rtl_file == ""
        assert context.synthesis_tool == "vivado"
        assert context.target_device == ""
        assert context.clock_frequency == 100.0
        assert context.modules == []
        assert context.dependencies == []
        assert context.backend_type == "hls"
        assert context.optimization_level == "2"
    
    def test_rtl_backend_context_file_name_derivation(self):
        """Test automatic file name derivation."""
        context = RTLBackendContext(module_name="test_backend")
        assert context.file_name == "test_backend_backend.py"
    
    def test_rtl_backend_validation_success(self):
        """Test successful RTL backend validation."""
        context = RTLBackendContext(
            module_name="test",
            file_name="test.py",
            rtl_files=["test.v"]
        )
        
        # Should not raise an exception
        context.validate()
    
    def test_rtl_backend_validation_no_rtl_files(self):
        """Test validation with no RTL files."""
        context = RTLBackendContext(
            module_name="test",
            file_name="test.py"
        )
        
        with pytest.raises(ValidationError) as exc_info:
            context.validate()
        
        assert "At least one RTL file is required" in str(exc_info.value)


class TestTemplateContextBuilder:
    """Test TemplateContextBuilder class."""
    
    def test_context_builder_initialization(self):
        """Test context builder initialization."""
        builder = TemplateContextBuilder()
        
        assert builder.config is None
        assert builder._context_cache == {}
    
    def test_context_builder_with_config(self):
        """Test context builder with configuration."""
        config = PipelineConfig()
        builder = TemplateContextBuilder(config)
        
        assert builder.config is config
    
    def test_build_hw_custom_op_context_basic(self):
        """Test building basic HW custom op context."""
        analysis_data = {
            'module_name': 'test_module',
            'top_module': 'test_top',
            'rtl_file': 'test.v',
            'interfaces': [
                {'name': 'ap_clk', 'direction': 'input'},
                {'name': 'ap_rst_n', 'direction': 'input'},
                {'name': 'data_in', 'direction': 'input', 'width': 32}
            ],
            'parameters': {
                'WIDTH': 32,
                'DEPTH': {'value': 1024, 'type': 'int', 'description': 'Buffer depth'}
            }
        }
        
        builder = TemplateContextBuilder()
        context = builder.build_hw_custom_op_context(analysis_data)
        
        assert isinstance(context, HWCustomOpContext)
        assert context.module_name == 'test_module'
        assert context.top_module_name == 'test_top'
        assert context.rtl_file_path == 'test.v'
        assert len(context.interfaces) == 3
        assert len(context.parameters) == 2
    
    def test_build_hw_custom_op_context_with_finn_config(self):
        """Test building HW custom op context with FINN configuration."""
        analysis_data = {
            'module_name': 'test_module',
            'top_module': 'test_top',
            'rtl_file': 'test.v',
            'interfaces': [
                {'name': 'ap_clk', 'direction': 'input'},
                {'name': 'ap_rst_n', 'direction': 'input'}
            ],
            'parameters': {},
            'finn_config': {
                'datatype': 'int8',
                'input_shape': [1, 3, 224, 224],
                'output_shape': [1, 1000]
            },
            'tdims': ['N', 'C', 'H', 'W'],
            'tensor_configs': {'input': {'dtype': 'int8'}}
        }
        
        builder = TemplateContextBuilder()
        context = builder.build_hw_custom_op_context(analysis_data)
        
        assert context.finn_datatype == 'int8'
        assert context.input_shape == [1, 3, 224, 224]
        assert context.output_shape == [1, 1000]
        assert context.tdims == ['N', 'C', 'H', 'W']
        assert context.tensor_configs == {'input': {'dtype': 'int8'}}
    
    def test_build_hw_custom_op_context_with_config(self):
        """Test building context with configuration overrides."""
        config = PipelineConfig()
        config.generation.include_debug_info = True
        config.generation.include_documentation = False
        
        analysis_data = {
            'module_name': 'test_module',
            'top_module': 'test_top',
            'rtl_file': 'test.v',
            'interfaces': [
                {'name': 'ap_clk', 'direction': 'input'},
                {'name': 'ap_rst_n', 'direction': 'input'}
            ],
            'parameters': {}
        }
        
        builder = TemplateContextBuilder(config)
        context = builder.build_hw_custom_op_context(analysis_data)
        
        assert context.include_debug_info is True
        assert context.include_documentation is False
    
    def test_build_hw_custom_op_context_caching(self):
        """Test context caching functionality."""
        analysis_data = {
            'module_name': 'test_module',
            'top_module': 'test_top',
            'rtl_file': 'test.v',
            'interfaces': [
                {'name': 'ap_clk', 'direction': 'input'},
                {'name': 'ap_rst_n', 'direction': 'input'}
            ],
            'parameters': {}
        }
        
        builder = TemplateContextBuilder()
        
        # First call should build and cache
        context1 = builder.build_hw_custom_op_context(analysis_data)
        
        # Second call should return cached result
        context2 = builder.build_hw_custom_op_context(analysis_data)
        
        assert context1 is context2  # Same object from cache
    
    def test_build_rtl_backend_context_basic(self):
        """Test building basic RTL backend context."""
        analysis_data = {
            'module_name': 'test_backend',
            'rtl_files': ['test1.v', 'test2.v'],
            'main_rtl_file': 'test1.v',
            'modules': ['test_module'],
            'dependencies': ['dependency1.v']
        }
        
        builder = TemplateContextBuilder()
        context = builder.build_rtl_backend_context(analysis_data)
        
        assert isinstance(context, RTLBackendContext)
        assert context.module_name == 'test_backend'
        assert context.rtl_files == ['test1.v', 'test2.v']
        assert context.main_rtl_file == 'test1.v'
        assert context.modules == ['test_module']
        assert context.dependencies == ['dependency1.v']
    
    def test_build_rtl_backend_context_with_backend_config(self):
        """Test building RTL backend context with backend configuration."""
        analysis_data = {
            'module_name': 'test_backend',
            'rtl_files': ['test.v'],
            'backend_config': {
                'synthesis_tool': 'quartus',
                'target_device': 'cyclone5',
                'clock_frequency': 200.0,
                'backend_type': 'verilog',
                'optimization_level': '3'
            }
        }
        
        builder = TemplateContextBuilder()
        context = builder.build_rtl_backend_context(analysis_data)
        
        assert context.synthesis_tool == 'quartus'
        assert context.target_device == 'cyclone5'
        assert context.clock_frequency == 200.0
        assert context.backend_type == 'verilog'
        assert context.optimization_level == '3'
    
    def test_build_context_error_handling(self):
        """Test error handling in context building."""
        invalid_data = {'invalid': 'data'}
        
        builder = TemplateContextBuilder()
        
        with pytest.raises(CodeGenerationError) as exc_info:
            builder.build_hw_custom_op_context(invalid_data)
        
        assert "Failed to build HW Custom Op context" in str(exc_info.value)
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        analysis_data = {
            'module_name': 'test_module',
            'top_module': 'test_top',
            'rtl_file': 'test.v',
            'interfaces': [
                {'name': 'ap_clk', 'direction': 'input'},
                {'name': 'ap_rst_n', 'direction': 'input'}
            ],
            'parameters': {}
        }
        
        builder = TemplateContextBuilder()
        
        # Build context to populate cache
        builder.build_hw_custom_op_context(analysis_data)
        assert len(builder._context_cache) > 0
        
        # Clear cache
        builder.clear_cache()
        assert len(builder._context_cache) == 0
    
    def test_get_cache_stats(self):
        """Test cache statistics."""
        builder = TemplateContextBuilder()
        
        stats = builder.get_cache_stats()
        assert isinstance(stats, dict)
        assert 'cache_size' in stats
        assert 'cached_contexts' in stats
        assert stats['cache_size'] == 0


class TestContextBuilderHelpers:
    """Test context builder helper functions."""
    
    def test_create_context_builder(self):
        """Test create_context_builder factory function."""
        builder = create_context_builder()
        
        assert isinstance(builder, TemplateContextBuilder)
        assert builder.config is None
    
    def test_create_context_builder_with_config(self):
        """Test create_context_builder with configuration."""
        config = PipelineConfig()
        builder = create_context_builder(config)
        
        assert isinstance(builder, TemplateContextBuilder)
        assert builder.config is config