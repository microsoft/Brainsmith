"""
Tests for generator base interface and data structures.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from brainsmith.tools.hw_kernel_gen.generator_base import (
    GenerationStatus,
    ArtifactType,
    GeneratedArtifact,
    GenerationResult,
    GeneratorBase,
    create_generation_result,
    create_artifact
)
from brainsmith.tools.hw_kernel_gen.data_structures import (
    PipelineInputs,
    PipelineResults,
    PipelineStage,
    ParsedRTLData,
    RTLModule,
    RTLInterface,
    RTLSignal,
    InterfaceType,
    SignalDirection,
    create_pipeline_inputs,
    create_pipeline_results
)
from brainsmith.tools.hw_kernel_gen.template_context import BaseContext
from brainsmith.tools.hw_kernel_gen.errors import CodeGenerationError, ValidationError


class TestGeneratedArtifact:
    """Test GeneratedArtifact class."""
    
    def test_basic_artifact_creation(self):
        """Test basic artifact creation."""
        artifact = GeneratedArtifact(
            name="test_module",
            type=ArtifactType.PYTHON_FILE,
            content="print('Hello, World!')"
        )
        
        assert artifact.name == "test_module"
        assert artifact.type == ArtifactType.PYTHON_FILE
        assert artifact.content == "print('Hello, World!')"
        assert artifact.file_path == Path("test_module.py")
        assert artifact.encoding == "utf-8"
        assert artifact.is_valid is True
        assert artifact.line_count == 1
        assert artifact.size_bytes > 0
    
    def test_artifact_file_path_derivation(self):
        """Test automatic file path derivation for different types."""
        test_cases = [
            (ArtifactType.PYTHON_FILE, "test.py"),
            (ArtifactType.HEADER_FILE, "test.h"),
            (ArtifactType.CONFIG_FILE, "test.json"),
            (ArtifactType.DOCUMENTATION, "test.md"),
            (ArtifactType.TEST_FILE, "test_test.py"),
            (ArtifactType.BUILD_SCRIPT, "test.sh"),
            (ArtifactType.WRAPPER_FILE, "test_wrapper.py")
        ]
        
        for artifact_type, expected_name in test_cases:
            artifact = GeneratedArtifact(
                name="test",
                type=artifact_type,
                content="content"
            )
            assert artifact.file_path.name == expected_name
    
    def test_artifact_validation_success(self):
        """Test successful artifact validation."""
        artifact = GeneratedArtifact(
            name="test",
            type=ArtifactType.PYTHON_FILE,
            content="x = 1\nprint(x)"
        )
        
        assert artifact.validate_content() is True
        assert artifact.is_valid is True
        assert len(artifact.validation_errors) == 0
    
    def test_artifact_validation_empty_content(self):
        """Test validation with empty content."""
        artifact = GeneratedArtifact(
            name="test",
            type=ArtifactType.PYTHON_FILE,
            content=""
        )
        
        assert artifact.validate_content() is False
        assert artifact.is_valid is False
        assert "empty" in artifact.validation_errors[0].lower()
    
    def test_artifact_python_syntax_validation(self):
        """Test Python syntax validation."""
        artifact = GeneratedArtifact(
            name="test",
            type=ArtifactType.PYTHON_FILE,
            content="invalid python syntax {"
        )
        
        assert artifact.validate_content() is False
        assert artifact.is_valid is False
        assert any("syntax" in error.lower() for error in artifact.validation_errors)
    
    def test_artifact_json_validation(self):
        """Test JSON validation for config files."""
        # Valid JSON
        valid_artifact = GeneratedArtifact(
            name="config",
            type=ArtifactType.CONFIG_FILE,
            content='{"key": "value"}'
        )
        assert valid_artifact.validate_content() is True
        
        # Invalid JSON
        invalid_artifact = GeneratedArtifact(
            name="config",
            type=ArtifactType.CONFIG_FILE,
            content='{"key": invalid}'
        )
        assert invalid_artifact.validate_content() is False
    
    def test_artifact_write_to_file(self):
        """Test writing artifact to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            artifact = GeneratedArtifact(
                name="test",
                type=ArtifactType.PYTHON_FILE,
                content="print('test')"
            )
            
            artifact.write_to_file(output_dir)
            
            written_file = output_dir / "test.py"
            assert written_file.exists()
            assert written_file.read_text() == "print('test')"
    
    def test_artifact_write_overwrite_protection(self):
        """Test overwrite protection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            test_file = output_dir / "test.py"
            test_file.write_text("existing content")
            
            artifact = GeneratedArtifact(
                name="test",
                type=ArtifactType.PYTHON_FILE,
                content="new content"
            )
            
            # Should fail without overwrite
            with pytest.raises(CodeGenerationError) as exc_info:
                artifact.write_to_file(output_dir, overwrite=False)
            
            assert "already exists" in str(exc_info.value)
            
            # Should succeed with overwrite
            artifact.write_to_file(output_dir, overwrite=True)
            assert test_file.read_text() == "new content"
    
    def test_artifact_to_dict(self):
        """Test artifact serialization to dictionary."""
        artifact = GeneratedArtifact(
            name="test",
            type=ArtifactType.PYTHON_FILE,
            content="content",
            template_name="test.j2"
        )
        
        data = artifact.to_dict()
        
        assert data['name'] == "test"
        assert data['type'] == "python_file"
        assert data['content'] == "content"
        assert data['template_name'] == "test.j2"
        assert isinstance(data['file_path'], str)


class TestGenerationResult:
    """Test GenerationResult class."""
    
    def test_basic_result_creation(self):
        """Test basic result creation."""
        result = GenerationResult(
            status=GenerationStatus.SUCCESS,
            message="Generation completed successfully"
        )
        
        assert result.status == GenerationStatus.SUCCESS
        assert result.message == "Generation completed successfully"
        assert result.artifacts == []
        assert result.errors == []
        assert result.warnings == []
        assert result.generation_time == 0.0
    
    def test_add_artifact(self):
        """Test adding artifacts to result."""
        result = GenerationResult(status=GenerationStatus.SUCCESS)
        
        artifact = GeneratedArtifact(
            name="test",
            type=ArtifactType.PYTHON_FILE,
            content="test content"
        )
        
        result.add_artifact(artifact)
        
        assert len(result.artifacts) == 1
        assert result.total_lines == 1
        assert result.total_size > 0
    
    def test_add_error(self):
        """Test adding errors to result."""
        result = GenerationResult(status=GenerationStatus.SUCCESS)
        
        result.add_error("Test error")
        
        assert len(result.errors) == 1
        assert result.errors[0] == "Test error"
        assert result.status == GenerationStatus.PARTIAL
    
    def test_add_warning(self):
        """Test adding warnings to result."""
        result = GenerationResult(status=GenerationStatus.SUCCESS)
        
        result.add_warning("Test warning")
        
        assert len(result.warnings) == 1
        assert result.warnings[0] == "Test warning"
        assert result.status == GenerationStatus.SUCCESS  # Warnings don't change status
    
    def test_get_artifacts_by_type(self):
        """Test getting artifacts by type."""
        result = GenerationResult(status=GenerationStatus.SUCCESS)
        
        python_artifact = GeneratedArtifact("test1", ArtifactType.PYTHON_FILE, "content1")
        config_artifact = GeneratedArtifact("test2", ArtifactType.CONFIG_FILE, "content2")
        
        result.add_artifact(python_artifact)
        result.add_artifact(config_artifact)
        
        python_artifacts = result.get_artifacts_by_type(ArtifactType.PYTHON_FILE)
        config_artifacts = result.get_artifacts_by_type(ArtifactType.CONFIG_FILE)
        
        assert len(python_artifacts) == 1
        assert len(config_artifacts) == 1
        assert python_artifacts[0] is python_artifact
        assert config_artifacts[0] is config_artifact
    
    def test_has_errors(self):
        """Test error detection."""
        result = GenerationResult(status=GenerationStatus.SUCCESS)
        
        assert result.has_errors() is False
        
        result.add_error("Error")
        assert result.has_errors() is True
        
        # Test artifact validation errors
        result2 = GenerationResult(status=GenerationStatus.SUCCESS)
        invalid_artifact = GeneratedArtifact("test", ArtifactType.PYTHON_FILE, "")
        invalid_artifact.validation_errors = ["Validation error"]
        result2.add_artifact(invalid_artifact)
        
        assert result2.has_errors() is True
    
    def test_validate_all_artifacts(self):
        """Test validating all artifacts."""
        result = GenerationResult(status=GenerationStatus.SUCCESS)
        
        valid_artifact = GeneratedArtifact("test1", ArtifactType.PYTHON_FILE, "x = 1")
        invalid_artifact = GeneratedArtifact("test2", ArtifactType.PYTHON_FILE, "")
        
        result.add_artifact(valid_artifact)
        result.add_artifact(invalid_artifact)
        
        assert result.validate_all_artifacts() is False
        assert result.status == GenerationStatus.PARTIAL
        assert len(result.errors) > 0
    
    def test_write_all_artifacts(self):
        """Test writing all artifacts to files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            result = GenerationResult(status=GenerationStatus.SUCCESS)
            
            artifact1 = GeneratedArtifact("test1", ArtifactType.PYTHON_FILE, "content1")
            artifact2 = GeneratedArtifact("test2", ArtifactType.CONFIG_FILE, '{"key": "value"}')
            
            result.add_artifact(artifact1)
            result.add_artifact(artifact2)
            
            written_files = result.write_all_artifacts(output_dir)
            
            assert len(written_files) == 2
            assert (output_dir / "test1.py").exists()
            assert (output_dir / "test2.json").exists()
    
    def test_to_dict(self):
        """Test result serialization to dictionary."""
        result = GenerationResult(
            status=GenerationStatus.SUCCESS,
            message="Test",
            generator_type="TestGenerator"
        )
        
        artifact = GeneratedArtifact("test", ArtifactType.PYTHON_FILE, "content")
        result.add_artifact(artifact)
        
        data = result.to_dict()
        
        assert data['status'] == "success"
        assert data['message'] == "Test"
        assert data['generator_type'] == "TestGenerator"
        assert len(data['artifacts']) == 1
        assert data['artifacts'][0]['name'] == "test"


class TestGeneratorBase:
    """Test GeneratorBase abstract class."""
    
    def test_generator_base_cannot_be_instantiated(self):
        """Test that GeneratorBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            GeneratorBase()
    
    def test_concrete_generator_implementation(self):
        """Test a concrete generator implementation."""
        
        class TestGenerator(GeneratorBase):
            def generate(self, analysis_data, **kwargs):
                self.start_generation_timer()
                result = self.create_result(GenerationStatus.SUCCESS, "Test generation")
                
                artifact = self.create_artifact(
                    "test",
                    ArtifactType.PYTHON_FILE,
                    "print('generated')",
                    "test.j2"
                )
                result.add_artifact(artifact)
                
                return result
            
            def get_supported_templates(self):
                return ["test.j2", "example.j2"]
            
            def validate_input(self, analysis_data):
                if not analysis_data.get('module_name'):
                    raise ValidationError("Module name required")
        
        generator = TestGenerator()
        
        assert generator.generator_type == "TestGenerator"
        assert generator.version == "1.0.0"
        assert generator.get_supported_templates() == ["test.j2", "example.j2"]
    
    def test_generator_create_artifact(self):
        """Test artifact creation helper."""
        
        class TestGenerator(GeneratorBase):
            def generate(self, analysis_data, **kwargs):
                return self.create_result(GenerationStatus.SUCCESS)
            
            def get_supported_templates(self):
                return []
            
            def validate_input(self, analysis_data):
                pass
        
        generator = TestGenerator()
        
        artifact = generator.create_artifact(
            "test",
            ArtifactType.PYTHON_FILE,
            "print('test')",
            "test.j2",
            {"key": "value"}
        )
        
        assert artifact.name == "test"
        assert artifact.type == ArtifactType.PYTHON_FILE
        assert artifact.template_name == "test.j2"
        assert artifact.context_data == {"key": "value"}
    
    def test_generator_timing(self):
        """Test generation timing functionality."""
        
        class TestGenerator(GeneratorBase):
            def generate(self, analysis_data, **kwargs):
                self.start_generation_timer()
                import time
                time.sleep(0.01)  # Small delay for testing
                return self.create_result(GenerationStatus.SUCCESS)
            
            def get_supported_templates(self):
                return []
            
            def validate_input(self, analysis_data):
                pass
        
        generator = TestGenerator()
        result = generator.generate({})
        
        assert result.generation_time > 0
    
    def test_generator_render_template_without_manager(self):
        """Test template rendering without template manager."""
        
        class TestGenerator(GeneratorBase):
            def generate(self, analysis_data, **kwargs):
                return self.create_result(GenerationStatus.SUCCESS)
            
            def get_supported_templates(self):
                return []
            
            def validate_input(self, analysis_data):
                pass
        
        generator = TestGenerator()
        context = BaseContext(module_name="test")
        
        with pytest.raises(CodeGenerationError) as exc_info:
            generator.render_template("test.j2", context)
        
        assert "Template manager not initialized" in str(exc_info.value)
    
    def test_generator_build_context_without_builder(self):
        """Test context building without context builder."""
        
        class TestGenerator(GeneratorBase):
            def generate(self, analysis_data, **kwargs):
                return self.create_result(GenerationStatus.SUCCESS)
            
            def get_supported_templates(self):
                return []
            
            def validate_input(self, analysis_data):
                pass
            
            def build_context(self, analysis_data, **kwargs):
                return super().build_context(analysis_data, **kwargs)
        
        generator = TestGenerator()
        
        with pytest.raises(CodeGenerationError) as exc_info:
            generator.build_context({})
        
        assert "Context builder not initialized" in str(exc_info.value)
    
    def test_generator_info(self):
        """Test generator information."""
        
        class TestGenerator(GeneratorBase):
            def generate(self, analysis_data, **kwargs):
                return self.create_result(GenerationStatus.SUCCESS)
            
            def get_supported_templates(self):
                return ["test.j2"]
            
            def validate_input(self, analysis_data):
                pass
        
        generator = TestGenerator()
        info = generator.get_generator_info()
        
        assert info['type'] == "TestGenerator"
        assert info['version'] == "1.0.0"
        assert info['supported_templates'] == ["test.j2"]
        assert info['config_required'] is False
        assert info['template_manager_required'] is False
        assert info['context_builder_required'] is False


class TestDataStructures:
    """Test data structure classes."""
    
    def test_rtl_signal_basic(self):
        """Test basic RTL signal creation."""
        signal = RTLSignal(
            name="data_in",
            direction=SignalDirection.INPUT,
            width=32
        )
        
        assert signal.name == "data_in"
        assert signal.direction == SignalDirection.INPUT
        assert signal.width == 32
        assert signal.is_vector is True
        assert signal.interface_type == InterfaceType.DATA
    
    def test_rtl_signal_auto_classification(self):
        """Test automatic signal classification."""
        # Clock signal
        clk_signal = RTLSignal("ap_clk", SignalDirection.INPUT)
        assert clk_signal.interface_type == InterfaceType.CLOCK
        
        # Reset signal
        rst_signal = RTLSignal("ap_rst_n", SignalDirection.INPUT)
        assert rst_signal.interface_type == InterfaceType.RESET
        
        # Control signal
        ctrl_signal = RTLSignal("ap_start", SignalDirection.INPUT)
        assert ctrl_signal.interface_type == InterfaceType.CONTROL
        
        # AXI Stream
        axi_signal = RTLSignal("s_axis_tdata", SignalDirection.INPUT)
        assert axi_signal.interface_type == InterfaceType.AXI_STREAM
    
    def test_rtl_interface_basic(self):
        """Test basic RTL interface creation."""
        interface = RTLInterface(
            name="data_input",
            interface_type=InterfaceType.AXI_STREAM
        )
        
        signal1 = RTLSignal("tdata", SignalDirection.INPUT, 32)
        signal2 = RTLSignal("tvalid", SignalDirection.INPUT, 1)
        
        interface.add_signal(signal1)
        interface.add_signal(signal2)
        
        assert len(interface.signals) == 2
        assert interface.data_width == 32
    
    def test_rtl_module_basic(self):
        """Test basic RTL module creation."""
        module = RTLModule(
            name="test_module",
            file_path=Path("test.sv"),
            is_top_level=True
        )
        
        interface = RTLInterface("data", InterfaceType.DATA)
        module.add_interface(interface)
        
        assert module.name == "test_module"
        assert module.is_top_level is True
        assert len(module.interfaces) == 1
    
    def test_parsed_rtl_data(self):
        """Test parsed RTL data structure."""
        rtl_data = ParsedRTLData()
        
        module = RTLModule("test", Path("test.sv"), is_top_level=True)
        rtl_data.add_module(module)
        
        assert len(rtl_data.modules) == 1
        assert rtl_data.top_module is module
    
    def test_parsed_rtl_data_validation(self):
        """Test RTL data validation."""
        # Empty data should fail
        empty_data = ParsedRTLData()
        assert empty_data.validate() is False
        assert len(empty_data.errors) > 0
        
        # Valid data should pass
        valid_data = ParsedRTLData()
        module = RTLModule("test", Path("test.sv"))
        interface = RTLInterface("data", InterfaceType.DATA)
        signal = RTLSignal("clk", SignalDirection.INPUT)
        interface.add_signal(signal)
        module.add_interface(interface)
        valid_data.add_module(module)
        
        assert valid_data.validate() is True
    
    def test_pipeline_inputs_creation(self):
        """Test pipeline inputs creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rtl_file = Path(tmpdir) / "test.sv"
            rtl_file.write_text("module test(); endmodule")
            
            inputs = create_pipeline_inputs(
                rtl_files=[rtl_file],
                module_name="test_module",
                generator_type="hw_custom_op"
            )
            
            assert inputs.module_name == "test_module"
            assert inputs.generator_type == "hw_custom_op"
            assert len(inputs.rtl_files) == 1
    
    def test_pipeline_inputs_validation(self):
        """Test pipeline inputs validation."""
        # Valid inputs
        with tempfile.TemporaryDirectory() as tmpdir:
            rtl_file = Path(tmpdir) / "test.sv"
            rtl_file.write_text("module test(); endmodule")
            
            valid_inputs = PipelineInputs(
                rtl_files=[rtl_file],
                module_name="test"
            )
            
            # Should not raise
            valid_inputs.validate()
        
        # Invalid inputs
        invalid_inputs = PipelineInputs(
            rtl_files=[Path("/nonexistent.sv")],
            module_name=""
        )
        
        with pytest.raises(ValidationError):
            invalid_inputs.validate()
    
    def test_pipeline_results(self):
        """Test pipeline results structure."""
        inputs = PipelineInputs(rtl_files=[], module_name="test")
        results = create_pipeline_results("test-123", inputs)
        
        assert results.pipeline_id == "test-123"
        assert results.inputs is inputs
        assert results.status == "unknown"
        
        # Test stage completion
        results.mark_stage_complete(PipelineStage.RTL_PARSING, 1.5)
        assert results.stage_times[PipelineStage.RTL_PARSING] == 1.5
        
        # Test error handling
        results.add_error("Test error", PipelineStage.RTL_PARSING)
        assert len(results.errors) == 1
        assert len(results.stage_errors[PipelineStage.RTL_PARSING]) == 1
        
        # Test finalization
        results.finalize()
        assert results.status == "failed"  # Due to error
        assert results.completed_at != ""


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_generation_result(self):
        """Test generation result factory."""
        result = create_generation_result(
            GenerationStatus.SUCCESS,
            "Test message",
            "TestGenerator"
        )
        
        assert result.status == GenerationStatus.SUCCESS
        assert result.message == "Test message"
        assert result.generator_type == "TestGenerator"
    
    def test_create_artifact(self):
        """Test artifact factory."""
        artifact = create_artifact(
            "test",
            ArtifactType.PYTHON_FILE,
            "print('test')"
        )
        
        assert artifact.name == "test"
        assert artifact.type == ArtifactType.PYTHON_FILE
        assert artifact.content == "print('test')"
    
    def test_create_pipeline_results_with_auto_id(self):
        """Test pipeline results creation with auto-generated ID."""
        results = create_pipeline_results()
        
        assert len(results.pipeline_id) == 8  # UUID truncated to 8 chars
        assert results.inputs is None