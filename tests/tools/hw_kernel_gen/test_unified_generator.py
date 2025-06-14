"""
Unit tests for UnifiedGenerator class.

Tests the unified generator that replaces all legacy generator classes
and uses Phase 2 template context generation exclusively.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from brainsmith.tools.hw_kernel_gen.unified_generator import (
    UnifiedGenerator, 
    UnifiedGeneratorError
)


class MockKernelMetadata:
    """Mock KernelMetadata for testing to avoid circular imports."""
    
    def __init__(self, name="test_kernel"):
        self.name = name
        self.parameters = []
        self.interfaces = []


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
    
    def validate(self):
        """Mock validation that always passes."""
        return []


class TestUnifiedGenerator:
    """Test suite for UnifiedGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.template_dir = Path(self.temp_dir) / "templates"
        self.template_dir.mkdir()
        
        # Create mock template files
        self.create_mock_templates()
        
        self.generator = UnifiedGenerator(template_dir=self.template_dir)
        self.mock_kernel_metadata = MockKernelMetadata()
    
    def create_mock_templates(self):
        """Create mock template files for testing."""
        # Mock Phase 2 HWCustomOp template
        hw_template = self.template_dir / "hw_custom_op_phase2.py.j2"
        hw_template.write_text("""
class {{ class_name }}(AutoHWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        # Mock template content
""")
        
        # Mock RTL wrapper template
        rtl_template = self.template_dir / "rtl_wrapper_v2.v.j2"
        rtl_template.write_text("""
module {{ module_name }}_wrapper();
    // Mock RTL template content
endmodule
""")
        
        # Mock test suite template
        test_template = self.template_dir / "test_suite_v2.py.j2"
        test_template.write_text("""
class Test{{ class_name }}:
    def test_mock(self):
        pass
""")
        
        # Legacy fallback templates
        legacy_rtl = self.template_dir / "rtl_wrapper.v.j2"
        legacy_rtl.write_text("module {{ module_name }}_wrapper(); endmodule")
        
        legacy_test = self.template_dir / "test_suite.py.j2"
        legacy_test.write_text("class Test{{ class_name }}: pass")
    
    def test_initialization_default_template_dir(self):
        """Test UnifiedGenerator initialization with default template directory."""
        # Test with default template directory (should not raise)
        generator = UnifiedGenerator()
        assert generator.template_dir is not None
        assert "templates" in str(generator.template_dir)
        assert hasattr(generator, 'jinja_env')
        assert hasattr(generator, 'template_context_generator')
    
    def test_initialization_custom_template_dir(self):
        """Test UnifiedGenerator initialization with custom template directory."""
        generator = UnifiedGenerator(template_dir=self.template_dir)
        assert generator.template_dir == self.template_dir
        assert hasattr(generator, 'jinja_env')
        assert hasattr(generator, 'template_context_generator')
    
    def test_initialization_invalid_template_dir(self):
        """Test UnifiedGenerator initialization with invalid template directory."""
        # Patch at the module level where UnifiedGenerator imports it
        with patch('brainsmith.tools.hw_kernel_gen.unified_generator.Environment') as mock_env:
            mock_env.side_effect = Exception("Jinja2 initialization failed")
            
            with pytest.raises(UnifiedGeneratorError, match="Failed to initialize Jinja2 environment"):
                UnifiedGenerator(template_dir=Path("/nonexistent"))
    
    @patch('brainsmith.tools.hw_kernel_gen.unified_generator.TemplateContextGenerator')
    def test_generate_and_write_success(self, mock_context_generator_class):
        """Test successful generation with generate_and_write."""
        # Mock the template context generator
        mock_context_generator = Mock()
        mock_context = MockTemplateContext()
        mock_context_generator.generate_template_context.return_value = mock_context
        mock_context_generator._template_context_to_dict.return_value = {
            'class_name': 'TestHWCustomOp',
            'module_name': 'test_module'
        }
        mock_context_generator_class.return_value = mock_context_generator
        
        generator = UnifiedGenerator(template_dir=self.template_dir)
        generator.template_context_generator = mock_context_generator
        
        # Test dry-run mode (no file writing)
        result = generator.generate_and_write(self.mock_kernel_metadata, write_files=False)
        
        assert result is not None
        assert result.is_success()
        assert len(result.generated_files) == 3  # hw_custom_op, rtl_wrapper, test_suite
        assert "TestHWCustomOp" in result.generated_files["test_kernel_hw_custom_op.py"]
        mock_context_generator.generate_template_context.assert_called_once_with(self.mock_kernel_metadata)
    
    @patch('brainsmith.tools.hw_kernel_gen.unified_generator.TemplateContextGenerator')
    def test_generate_and_write_template_not_found(self, mock_context_generator_class):
        """Test generation with missing template."""
        # Remove the Phase 2 template
        phase2_template = self.template_dir / "hw_custom_op_phase2.py.j2"
        phase2_template.unlink()
        
        mock_context_generator = Mock()
        mock_context = MockTemplateContext()
        mock_context_generator.generate_template_context.return_value = mock_context
        mock_context_generator_class.return_value = mock_context_generator
        
        generator = UnifiedGenerator(template_dir=self.template_dir)
        generator.template_context_generator = mock_context_generator
        
        result = generator.generate_and_write(self.mock_kernel_metadata, write_files=False)
        
        # Should fail gracefully and report errors
        assert not result.is_success()
        assert len(result.errors) > 0
        assert any("Phase 2 template not found" in error for error in result.errors)
    
    @patch('brainsmith.tools.hw_kernel_gen.unified_generator.TemplateContextGenerator')
    def test_generate_and_write_validation_failure(self, mock_context_generator_class):
        """Test generation with template context validation failure."""
        mock_context_generator = Mock()
        mock_context = MockTemplateContext()
        
        # Mock the validate method to return validation errors
        with patch.object(mock_context, 'validate', return_value=["Validation error"]):
            mock_context_generator.generate_template_context.return_value = mock_context
            mock_context_generator_class.return_value = mock_context_generator
            
            generator = UnifiedGenerator(template_dir=self.template_dir)
            generator.template_context_generator = mock_context_generator
            
            result = generator.generate_and_write(self.mock_kernel_metadata, write_files=False)
            
            # Should fail gracefully and report validation errors
            assert not result.is_success()
            assert "Template context validation: Validation error" in result.errors
    
    
    @patch('brainsmith.tools.hw_kernel_gen.unified_generator.TemplateContextGenerator')
    def test_generate_and_write_rtl_fallback_to_legacy(self, mock_context_generator_class):
        """Test generation falls back to legacy RTL template."""
        # Remove v2 template but keep legacy
        v2_template = self.template_dir / "rtl_wrapper_v2.v.j2"
        v2_template.unlink()
        
        mock_context_generator = Mock()
        mock_context = MockTemplateContext()
        mock_context_generator.generate_template_context.return_value = mock_context
        mock_context_generator._template_context_to_dict.return_value = {
            'module_name': 'test_module'
        }
        mock_context_generator_class.return_value = mock_context_generator
        
        generator = UnifiedGenerator(template_dir=self.template_dir)
        generator.template_context_generator = mock_context_generator
        
        with patch('brainsmith.tools.hw_kernel_gen.unified_generator.logger') as mock_logger:
            result = generator.generate_and_write(self.mock_kernel_metadata, write_files=False)
            
            assert result.is_success()
            assert "test_module_wrapper" in result.generated_files["test_kernel_wrapper.v"]
            mock_logger.warning.assert_called()
    
    
    
    @patch('brainsmith.tools.hw_kernel_gen.unified_generator.TemplateContextGenerator')
    def test_generate_and_write_all_templates(self, mock_context_generator_class):
        """Test successful generation of all artifacts."""
        mock_context_generator = Mock()
        mock_context = MockTemplateContext()
        mock_context_generator.generate_template_context.return_value = mock_context
        mock_context_generator._template_context_to_dict.return_value = {
            'class_name': 'TestHWCustomOp',
            'module_name': 'test_module'
        }
        mock_context_generator_class.return_value = mock_context_generator
        
        generator = UnifiedGenerator(template_dir=self.template_dir)
        generator.template_context_generator = mock_context_generator
        
        result = generator.generate_and_write(self.mock_kernel_metadata, write_files=False)
        
        assert result.is_success()
        assert len(result.generated_files) == 3  # HWCustomOp, RTL wrapper, test suite
        
        # Check that all expected files are generated
        assert "test_kernel_hw_custom_op.py" in result.generated_files
        assert "test_kernel_wrapper.v" in result.generated_files
        assert "test_test_kernel.py" in result.generated_files
        
        # Check content contains expected elements
        assert "TestHWCustomOp" in result.generated_files["test_kernel_hw_custom_op.py"]
        assert "test_module_wrapper" in result.generated_files["test_kernel_wrapper.v"]
        assert "TestTestHWCustomOp" in result.generated_files["test_test_kernel.py"]
    
    @patch('brainsmith.tools.hw_kernel_gen.unified_generator.TemplateContextGenerator')
    def test_generate_and_write_partial_failure(self, mock_context_generator_class):
        """Test generation handles partial failures gracefully."""
        # Remove RTL and test templates to simulate failures
        rtl_template = self.template_dir / "rtl_wrapper_v2.v.j2"
        rtl_template.unlink()
        rtl_legacy = self.template_dir / "rtl_wrapper.v.j2"
        rtl_legacy.unlink()
        
        test_template = self.template_dir / "test_suite_v2.py.j2"
        test_template.unlink()
        test_legacy = self.template_dir / "test_suite.py.j2"
        test_legacy.unlink()
        
        mock_context_generator = Mock()
        mock_context = MockTemplateContext()
        mock_context_generator.generate_template_context.return_value = mock_context
        mock_context_generator._template_context_to_dict.return_value = {
            'class_name': 'TestHWCustomOp',
            'module_name': 'test_module'
        }
        mock_context_generator_class.return_value = mock_context_generator
        
        generator = UnifiedGenerator(template_dir=self.template_dir)
        generator.template_context_generator = mock_context_generator
        
        with patch('brainsmith.tools.hw_kernel_gen.unified_generator.logger') as mock_logger:
            result = generator.generate_and_write(self.mock_kernel_metadata, write_files=False)
            
            # Should still generate HWCustomOp even if others fail
            assert len(result.generated_files) == 1
            assert "test_kernel_hw_custom_op.py" in result.generated_files
            
            # Should have logged warnings for failures
            assert mock_logger.warning.call_count >= 2  # RTL wrapper and test suite failures
    
    def test_get_available_templates(self):
        """Test getting list of available templates."""
        generator = UnifiedGenerator(template_dir=self.template_dir)
        templates = generator.get_available_templates()
        
        assert isinstance(templates, list)
        assert "hw_custom_op_phase2.py.j2" in templates
        assert "rtl_wrapper_v2.v.j2" in templates
        assert "test_suite_v2.py.j2" in templates
    
    def test_validate_templates(self):
        """Test template validation functionality."""
        generator = UnifiedGenerator(template_dir=self.template_dir)
        status = generator.validate_templates()
        
        assert isinstance(status, dict)
        
        # Should find required Phase 2 templates
        assert status.get("hw_custom_op_phase2.py.j2") == True
        assert status.get("rtl_wrapper_v2.v.j2") == True
        assert status.get("test_suite_v2.py.j2") == True
        
        # Should find fallback templates
        assert status.get("rtl_wrapper.v.j2") == True
        assert status.get("test_suite.py.j2") == True
    
    def test_validate_templates_missing(self):
        """Test template validation with missing templates."""
        # Remove some templates
        hw_template = self.template_dir / "hw_custom_op_phase2.py.j2"
        hw_template.unlink()
        
        generator = UnifiedGenerator(template_dir=self.template_dir)
        status = generator.validate_templates()
        
        assert status.get("hw_custom_op_phase2.py.j2") == False
        assert status.get("rtl_wrapper_v2.v.j2") == True  # Still there
    
    def test_template_context_to_dict_delegation(self):
        """Test that _template_context_to_dict delegates correctly."""
        generator = UnifiedGenerator(template_dir=self.template_dir)
        mock_context = MockTemplateContext()
        
        # Mock the delegation call
        with patch.object(generator.template_context_generator, '_template_context_to_dict') as mock_method:
            mock_method.return_value = {'test': 'value'}
            
            result = generator._template_context_to_dict(mock_context)
            
            assert result == {'test': 'value'}
            mock_method.assert_called_once_with(mock_context)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestUnifiedGeneratorError:
    """Test suite for UnifiedGeneratorError exception."""
    
    def test_unified_generator_error_creation(self):
        """Test UnifiedGeneratorError can be created and raised."""
        with pytest.raises(UnifiedGeneratorError, match="Test error message"):
            raise UnifiedGeneratorError("Test error message")
    
    def test_unified_generator_error_inheritance(self):
        """Test UnifiedGeneratorError inherits from Exception."""
        error = UnifiedGeneratorError("Test")
        assert isinstance(error, Exception)


class TestUnifiedGeneratorIntegration:
    """Integration tests for UnifiedGenerator with real components."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.template_dir = Path(self.temp_dir) / "templates"
        self.template_dir.mkdir()
    
    def create_minimal_templates(self):
        """Create minimal but functional templates."""
        # Minimal working Phase 2 template
        hw_template = self.template_dir / "hw_custom_op_phase2.py.j2"
        hw_template.write_text("""
class {{ class_name }}(AutoHWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        self.runtime_parameters = {}
        {% for param in parameter_definitions %}
        self.runtime_parameters["{{ param.name }}"] = self.get_nodeattr("{{ param.name }}")
        {% endfor %}
""")
    
    @pytest.mark.integration
    def test_end_to_end_generation_flow(self):
        """Test complete end-to-end generation flow."""
        self.create_minimal_templates()
        
        # Create mock kernel metadata with some parameters
        mock_metadata = MockKernelMetadata("integration_test")
        
        generator = UnifiedGenerator(template_dir=self.template_dir)
        
        # Mock the template context generator to return minimal context
        with patch.object(generator, 'template_context_generator') as mock_generator:
            mock_context = MockTemplateContext()
            mock_context.class_name = "IntegrationTestHWCustomOp"
            mock_context.parameter_definitions = [
                Mock(name="PE", default_value=1),
                Mock(name="SIMD", default_value=1)
            ]
            mock_generator.generate_template_context.return_value = mock_context
            mock_generator._template_context_to_dict.return_value = {
                'class_name': 'IntegrationTestHWCustomOp',
                'parameter_definitions': mock_context.parameter_definitions
            }
            
            # Should be able to generate all templates
            result = generator.generate_and_write(mock_metadata, write_files=False)
            
            assert result.is_success()
            assert "integration_test_hw_custom_op.py" in result.generated_files
            hw_code = result.generated_files["integration_test_hw_custom_op.py"]
            assert "IntegrationTestHWCustomOp" in hw_code
            assert "runtime_parameters" in hw_code
    
    def teardown_method(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)