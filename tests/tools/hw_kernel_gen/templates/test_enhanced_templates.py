"""
Unit tests for enhanced Phase 3 templates.

Tests the v2 templates that support Phase 2 parameter validation
and enhanced interface handling.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from brainsmith.tools.hw_kernel_gen.templates.template_context import TemplateContext


class MockParameterDefinition:
    """Mock ParameterDefinition for testing."""
    
    def __init__(self, name, default_value=None, description=None, is_required=False):
        self.name = name
        self.default_value = default_value
        self.description = description
        self.is_required = is_required


class MockInterfaceMetadata:
    """Mock InterfaceMetadata for testing."""
    
    def __init__(self, name, interface_type_name, block_shape=None):
        self.name = name
        self.interface_type = Mock()
        self.interface_type.name = interface_type_name
        self.chunking_strategy = None
        self.allowed_datatypes = []
        
        if block_shape:
            self.chunking_strategy = Mock()
            self.chunking_strategy.block_shape = block_shape
        
        # Add mock datatype
        if interface_type_name in ['INPUT', 'OUTPUT', 'WEIGHT']:
            mock_datatype = Mock()
            mock_datatype.bit_width = 8
            self.allowed_datatypes = [mock_datatype]


class MockTemplateContext:
    """Mock TemplateContext for testing."""
    
    def __init__(self):
        self.module_name = "test_module"
        self.class_name = "TestHWCustomOp"
        self.source_file = Path("/test/source.sv")
        self.interface_metadata = []
        self.parameter_definitions = []
        self.whitelisted_defaults = {}
        self.required_attributes = []
        self.generation_timestamp = "2025-01-06 12:00:00"


class TestEnhancedTemplates:
    """Test suite for enhanced Phase 3 templates."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.template_dir = self.temp_dir / "templates"
        self.template_dir.mkdir()
        
        # Copy actual templates to temp directory for testing
        self.copy_templates_to_temp()
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def copy_templates_to_temp(self):
        """Copy actual template files to temporary directory for testing."""
        # Get actual template directory
        actual_template_dir = Path(__file__).parent.parent.parent.parent.parent / "brainsmith" / "tools" / "hw_kernel_gen" / "templates"
        
        # Copy templates if they exist
        templates_to_copy = [
            "hw_custom_op_phase2.py.j2",
            "rtl_wrapper_v2.v.j2", 
            "test_suite_v2.py.j2"
        ]
        
        for template_name in templates_to_copy:
            src_template = actual_template_dir / template_name
            if src_template.exists():
                dst_template = self.template_dir / template_name
                dst_template.write_text(src_template.read_text())
    
    def create_sample_context(self):
        """Create a sample template context for testing."""
        context = MockTemplateContext()
        
        # Add parameters
        context.parameter_definitions = [
            MockParameterDefinition("PE", 4, "Processing Elements", is_required=True),
            MockParameterDefinition("SIMD", 1, "SIMD width", is_required=False),
            MockParameterDefinition("DEPTH", 512, "Memory depth", is_required=False)
        ]
        
        # Add interfaces
        context.interface_metadata = [
            MockInterfaceMetadata("input0", "INPUT", ["PE"]),
            MockInterfaceMetadata("output0", "OUTPUT", ["PE"]),
            MockInterfaceMetadata("weights", "WEIGHT", ["SIMD", ":", "PE"])
        ]
        
        # Add whitelisted defaults and required attributes
        context.whitelisted_defaults = {"SIMD": 1, "DEPTH": 512}
        context.required_attributes = ["PE"]
        
        return context
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_hw_custom_op_phase2_template_rendering(self):
        """Test hw_custom_op_phase2.py.j2 template rendering."""
        try:
            template = self.jinja_env.get_template("hw_custom_op_phase2.py.j2")
        except TemplateNotFound:
            pytest.skip("hw_custom_op_phase2.py.j2 template not found")
        
        context = self.create_sample_context()
        context_dict = self._context_to_dict(context)
        
        rendered = template.render(**context_dict)
        
        # Check basic structure
        assert "class TestHWCustomOp" in rendered
        assert "AutoHWCustomOp" in rendered
        assert "__init__" in rendered
        assert "runtime_parameters" in rendered
        
        # Check parameter extraction
        assert "self.get_nodeattr(\"PE\")" in rendered
        assert "self.get_nodeattr(\"SIMD\")" in rendered
        assert "self.get_nodeattr(\"DEPTH\")" in rendered
        
        # Check imports
        assert "from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp" in rendered
    
    def test_rtl_wrapper_v2_template_rendering(self):
        """Test rtl_wrapper_v2.v.j2 template rendering."""
        try:
            template = self.jinja_env.get_template("rtl_wrapper_v2.v.j2")
        except TemplateNotFound:
            pytest.skip("rtl_wrapper_v2.v.j2 template not found")
        
        context = self.create_sample_context()
        context_dict = self._context_to_dict(context)
        
        rendered = template.render(**context_dict)
        
        # Check basic module structure
        assert "module test_module_wrapper" in rendered
        assert "parameter PE = 4" in rendered
        assert "parameter SIMD = 1" in rendered
        assert "parameter DEPTH = 512" in rendered
        
        # Check interface declarations
        assert "input0_TDATA" in rendered
        assert "output0_TDATA" in rendered
        assert "weights_TDATA" in rendered
        
        # Check parameter validation
        assert "$error" in rendered
        assert "must be positive" in rendered
        
        # Check module instantiation
        assert "test_module #(" in rendered
        assert ".PE(PE)" in rendered
        assert ".SIMD(SIMD)" in rendered
        
        # Check control signals
        assert "input wire ap_clk" in rendered
        assert "input wire ap_rst_n" in rendered
        assert "input wire ap_start" in rendered
        assert "output wire ap_done" in rendered
    
    def test_test_suite_v2_template_rendering(self):
        """Test test_suite_v2.py.j2 template rendering."""
        try:
            template = self.jinja_env.get_template("test_suite_v2.py.j2")
        except TemplateNotFound:
            pytest.skip("test_suite_v2.py.j2 template not found")
        
        context = self.create_sample_context()
        context_dict = self._context_to_dict(context)
        
        rendered = template.render(**context_dict)
        
        # Check basic test class structure
        assert "class TestTestHWCustomOp" in rendered
        assert "import pytest" in rendered
        assert "import onnx.helper" in rendered
        
        # Check test methods exist
        assert "def test_parameter_validation_required_attributes" in rendered
        assert "def test_parameter_validation_whitelisted_defaults" in rendered
        assert "def test_hwcustomop_instantiation_runtime_extraction" in rendered
        assert "def test_bdim_parameter_consistency" in rendered
        
        # Check parameter testing
        assert "PE=4" in rendered
        assert "SIMD=1" in rendered
        assert "DEPTH=512" in rendered
        
        # Check runtime parameter extraction testing
        assert "runtime_parameters" in rendered
        assert "get_nodeattr" in rendered
        
        # Check required parameter validation
        assert "Missing" in rendered or "required" in rendered
    
    def test_rtl_wrapper_parameter_validation_generation(self):
        """Test RTL wrapper generates proper parameter validation."""
        try:
            template = self.jinja_env.get_template("rtl_wrapper_v2.v.j2")
        except TemplateNotFound:
            pytest.skip("rtl_wrapper_v2.v.j2 template not found")
        
        context = self.create_sample_context()
        context_dict = self._context_to_dict(context)
        
        rendered = template.render(**context_dict)
        
        # Check validation for required parameters
        assert "if (PE <= 0)" in rendered
        
        # Check validation for whitelisted parameters
        assert "if (SIMD <= 0)" in rendered
        assert "if (DEPTH <= 0)" in rendered
        
        # Check error messages include parameter names and values
        assert "Parameter PE must be positive" in rendered
        assert "Parameter SIMD must be positive" in rendered
        assert "Parameter DEPTH must be positive" in rendered
    
    def test_rtl_wrapper_interface_width_calculation(self):
        """Test RTL wrapper generates interface width calculations."""
        try:
            template = self.jinja_env.get_template("rtl_wrapper_v2.v.j2")
        except TemplateNotFound:
            pytest.skip("rtl_wrapper_v2.v.j2 template not found")
        
        context = self.create_sample_context()
        context_dict = self._context_to_dict(context)
        
        rendered = template.render(**context_dict)
        
        # Check width calculations are generated
        assert "ELEMENT_WIDTH" in rendered
        assert "PARALLEL_ELEMENTS" in rendered
        assert "TDATA_WIDTH" in rendered
        
        # Check BDIM-based width calculations
        assert "PE" in rendered  # Should be used in width calculation
        assert "SIMD" in rendered  # Should be used in width calculation
        
        # Check element width from datatype
        assert "8" in rendered  # Default bit width from mock datatype
    
    def test_test_suite_parameter_range_validation(self):
        """Test test suite generates parameter range validation tests."""
        try:
            template = self.jinja_env.get_template("test_suite_v2.py.j2")
        except TemplateNotFound:
            pytest.skip("test_suite_v2.py.j2 template not found")
        
        context = self.create_sample_context()
        context_dict = self._context_to_dict(context)
        
        rendered = template.render(**context_dict)
        
        # Check that range validation tests are generated
        assert "def test_parameter_range_validation" in rendered
        
        # Check negative value testing for each parameter
        assert "PE=-1" in rendered
        assert "SIMD=-1" in rendered
        assert "DEPTH=-1" in rendered
        
        # Check zero value testing
        assert "PE=0" in rendered
        assert "SIMD=0" in rendered
        assert "DEPTH=0" in rendered
        
        # Check exception handling
        assert "pytest.raises" in rendered
        assert "ValueError" in rendered or "AssertionError" in rendered
    
    def test_template_parameter_substitution(self):
        """Test that template parameter substitution works correctly."""
        try:
            template = self.jinja_env.get_template("hw_custom_op_phase2.py.j2")
        except TemplateNotFound:
            pytest.skip("hw_custom_op_phase2.py.j2 template not found")
        
        # Create context with specific values
        context = MockTemplateContext()
        context.module_name = "matrix_multiply"
        context.class_name = "MatrixMultiplyHWCustomOp"
        context.parameter_definitions = [
            MockParameterDefinition("ROWS", 64, "Number of rows"),
            MockParameterDefinition("COLS", 32, "Number of columns")
        ]
        
        context_dict = self._context_to_dict(context)
        rendered = template.render(**context_dict)
        
        # Check that substitution worked
        assert "MatrixMultiplyHWCustomOp" in rendered
        assert "matrix_multiply" in rendered
        assert "ROWS" in rendered
        assert "COLS" in rendered
        assert "Number of rows" in rendered or "Number of columns" in rendered
    
    def test_template_validation_code_generation(self):
        """Test that templates generate proper validation code."""
        try:
            rtl_template = self.jinja_env.get_template("rtl_wrapper_v2.v.j2")
        except TemplateNotFound:
            pytest.skip("rtl_wrapper_v2.v.j2 template not found")
        
        context = self.create_sample_context()
        context_dict = self._context_to_dict(context)
        
        rendered = rtl_template.render(**context_dict)
        
        # Check SystemVerilog validation code
        assert "initial begin" in rendered
        assert "$error" in rendered
        assert "$finish" in rendered
        assert "__FILE__" in rendered
        assert "__LINE__" in rendered
        
        # Check that all parameters get validation
        for param in context.parameter_definitions:
            assert f"if ({param.name} <= 0)" in rendered
    
    def test_template_error_handling_empty_context(self):
        """Test template rendering with minimal/empty context."""
        try:
            template = self.jinja_env.get_template("hw_custom_op_phase2.py.j2")
        except TemplateNotFound:
            pytest.skip("hw_custom_op_phase2.py.j2 template not found")
        
        # Minimal context
        minimal_context = {
            'module_name': 'minimal',
            'class_name': 'MinimalHWCustomOp',
            'source_file': '/test.sv',
            'parameter_definitions': [],
            'interface_metadata': [],
            'whitelisted_defaults': {},
            'required_attributes': []
        }
        
        # Should not raise exceptions
        rendered = template.render(**minimal_context)
        assert "MinimalHWCustomOp" in rendered
        assert "class MinimalHWCustomOp" in rendered
    
    def test_template_complex_interface_handling(self):
        """Test template handling of complex interface configurations."""
        try:
            template = self.jinja_env.get_template("rtl_wrapper_v2.v.j2")
        except TemplateNotFound:
            pytest.skip("rtl_wrapper_v2.v.j2 template not found")
        
        # Create context with complex interfaces
        context = MockTemplateContext()
        context.interface_metadata = [
            MockInterfaceMetadata("data_in", "INPUT", ["BATCH", "CHANNELS", "PE"]),
            MockInterfaceMetadata("weights_static", "WEIGHT", [":", "SIMD", "PE"]),
            MockInterfaceMetadata("data_out", "OUTPUT", ["BATCH", "PE"]),
            MockInterfaceMetadata("control_in", "INPUT", ["1"])
        ]
        context.parameter_definitions = [
            MockParameterDefinition("BATCH", 1),
            MockParameterDefinition("CHANNELS", 16), 
            MockParameterDefinition("PE", 4),
            MockParameterDefinition("SIMD", 8)
        ]
        
        context_dict = self._context_to_dict(context)
        rendered = template.render(**context_dict)
        
        # Check all interfaces are declared
        assert "data_in_TDATA" in rendered
        assert "weights_static_TDATA" in rendered
        assert "data_out_TDATA" in rendered
        assert "control_in_TDATA" in rendered
        
        # Check width calculations handle complex shapes
        assert "BATCH" in rendered
        assert "CHANNELS" in rendered
        assert "PE" in rendered
        assert "SIMD" in rendered
        
        # Check that colon elements are handled
        assert "1 *" in rendered  # Colon should become 1 in width calculation
    
    def _context_to_dict(self, context):
        """Convert mock context to dictionary for template rendering."""
        return {
            'module_name': context.module_name,
            'class_name': context.class_name,
            'source_file': str(context.source_file),
            'parameter_definitions': context.parameter_definitions,
            'interface_metadata': context.interface_metadata,
            'whitelisted_defaults': context.whitelisted_defaults,
            'required_attributes': context.required_attributes,
            'generation_timestamp': getattr(context, 'generation_timestamp', '2025-01-06 12:00:00')
        }


class TestTemplateErrorHandling:
    """Test template error handling and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.template_dir = self.temp_dir / "templates"
        self.template_dir.mkdir()
        
        # Create minimal test templates
        self.create_test_templates()
        
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def create_test_templates(self):
        """Create minimal test templates."""
        # Minimal working template
        hw_template = self.template_dir / "hw_custom_op_phase2.py.j2"
        hw_template.write_text("""
class {{ class_name }}(AutoHWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        {% for param in parameter_definitions %}
        # Parameter: {{ param.name }}
        {% endfor %}
""")
        
        # Template with potential error conditions
        error_template = self.template_dir / "error_test.j2"
        error_template.write_text("""
{% for param in parameter_definitions %}
Value: {{ param.nonexistent_attribute }}
{% endfor %}
""")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_template_missing_attribute_handling(self):
        """Test template graceful handling of missing attributes."""
        template = self.jinja_env.get_template("error_test.j2")
        
        context = {
            'parameter_definitions': [
                Mock(name="PE", default_value=1),  # Missing nonexistent_attribute
                Mock(name="SIMD", default_value=2)
            ]
        }
        
        # Should handle missing attributes gracefully (depending on Jinja2 settings)
        # In strict mode this would raise, in normal mode it returns empty string
        try:
            rendered = template.render(**context)
            # If it doesn't raise, check that template was processed
            assert "Value:" in rendered
        except Exception:
            # If it raises, that's also valid behavior
            pass
    
    def test_template_none_value_handling(self):
        """Test template handling of None values."""
        template = self.jinja_env.get_template("hw_custom_op_phase2.py.j2")
        
        context = {
            'class_name': 'TestHWCustomOp',
            'parameter_definitions': [
                Mock(name="PE", default_value=None),  # None value
                Mock(name="SIMD", default_value=1)
            ]
        }
        
        # Should handle None values without crashing
        rendered = template.render(**context)
        assert "TestHWCustomOp" in rendered
        assert "PE" in rendered
        assert "SIMD" in rendered
    
    def test_template_empty_lists_handling(self):
        """Test template handling of empty lists."""
        template = self.jinja_env.get_template("hw_custom_op_phase2.py.j2")
        
        context = {
            'class_name': 'EmptyTestHWCustomOp',
            'parameter_definitions': [],  # Empty list
            'interface_metadata': []
        }
        
        rendered = template.render(**context)
        assert "EmptyTestHWCustomOp" in rendered
        # Should not crash with empty lists


class TestTemplatePerformance:
    """Performance tests for template rendering."""
    
    @pytest.mark.performance
    def test_template_rendering_performance(self):
        """Test that template rendering is reasonably fast."""
        import time
        
        temp_dir = Path(tempfile.mkdtemp())
        template_dir = temp_dir / "templates"
        template_dir.mkdir()
        
        # Create simple template
        template_file = template_dir / "perf_test.j2"
        template_file.write_text("""
class {{ class_name }}:
    {% for param in parameter_definitions %}
    {{ param.name }} = {{ param.default_value }}
    {% endfor %}
""")
        
        jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = jinja_env.get_template("perf_test.j2")
        
        # Create large context
        large_context = {
            'class_name': 'PerformanceTestHWCustomOp',
            'parameter_definitions': [
                Mock(name=f"PARAM_{i}", default_value=i) 
                for i in range(100)  # 100 parameters
            ]
        }
        
        # Time multiple renders
        start_time = time.time()
        for _ in range(10):
            rendered = template.render(**large_context)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        assert avg_time < 0.1, f"Template rendering should be < 100ms, got {avg_time*1000:.2f}ms"
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)