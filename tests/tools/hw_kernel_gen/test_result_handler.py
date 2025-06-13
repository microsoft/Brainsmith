"""
Unit tests for ResultHandler class.

Tests the result handling system for unified generator that manages
generation results and writes generated files to the filesystem.
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from brainsmith.tools.hw_kernel_gen.result_handler import (
    GenerationResult, 
    ResultHandler
)
from brainsmith.tools.hw_kernel_gen.templates.template_context import TemplateContext


class MockKernelMetadata:
    """Mock KernelMetadata for testing."""
    
    def __init__(self, name="test_kernel"):
        self.name = name
        # Create mocks with proper name attributes
        pe_param = Mock()
        pe_param.name = "PE"
        simd_param = Mock()
        simd_param.name = "SIMD"
        self.parameters = [pe_param, simd_param]
        
        input_interface = Mock()
        input_interface.name = "input"
        output_interface = Mock()
        output_interface.name = "output"
        self.interfaces = [input_interface, output_interface]


class MockTemplateContext:
    """Mock TemplateContext for testing."""
    
    def __init__(self):
        # Create mocks with proper name attributes
        pe_param = Mock()
        pe_param.name = "PE"
        simd_param = Mock()
        simd_param.name = "SIMD"
        self.parameter_definitions = [pe_param, simd_param]
        
        input_interface = Mock()
        input_interface.name = "input"
        output_interface = Mock()
        output_interface.name = "output"
        self.interface_metadata = [input_interface, output_interface]
        
        self.required_attributes = ["PE"]
        self.whitelisted_defaults = {"SIMD": 1}


class TestGenerationResult:
    """Test suite for GenerationResult class."""
    
    def test_generation_result_creation(self):
        """Test basic GenerationResult creation."""
        result = GenerationResult(
            kernel_name="test_kernel",
            source_file=Path("/test/source.sv")
        )
        
        assert result.kernel_name == "test_kernel"
        assert result.source_file == Path("/test/source.sv")
        assert result.generated_files == {}
        assert result.template_context is None
        assert result.kernel_metadata is None
        assert result.validation_passed == True
        assert result.errors == []
        assert result.warnings == []
        assert result.generation_time_ms is None
    
    def test_generation_result_with_files(self):
        """Test GenerationResult with generated files."""
        files = {
            "hw_custom_op.py": "class TestHWCustomOp: pass",
            "rtl_wrapper.v": "module test_wrapper(); endmodule"
        }
        
        result = GenerationResult(
            kernel_name="test_kernel",
            source_file=Path("/test/source.sv"),
            generated_files=files
        )
        
        assert result.generated_files == files
        assert len(result.generated_files) == 2
    
    def test_add_error(self):
        """Test adding errors to GenerationResult."""
        result = GenerationResult(
            kernel_name="test_kernel",
            source_file=Path("/test/source.sv")
        )
        
        with patch('brainsmith.tools.hw_kernel_gen.result_handler.logger') as mock_logger:
            result.add_error("Test error message")
            
            assert len(result.errors) == 1
            assert result.errors[0] == "Test error message"
            assert result.validation_passed == False
            mock_logger.error.assert_called_once()
    
    def test_add_warning(self):
        """Test adding warnings to GenerationResult."""
        result = GenerationResult(
            kernel_name="test_kernel",
            source_file=Path("/test/source.sv")
        )
        
        with patch('brainsmith.tools.hw_kernel_gen.result_handler.logger') as mock_logger:
            result.add_warning("Test warning message")
            
            assert len(result.warnings) == 1
            assert result.warnings[0] == "Test warning message"
            assert result.validation_passed == True  # Warnings don't affect validation
            mock_logger.warning.assert_called_once()
    
    def test_is_success_with_files_and_validation(self):
        """Test is_success returns True when files exist and validation passed."""
        result = GenerationResult(
            kernel_name="test_kernel",
            source_file=Path("/test/source.sv"),
            generated_files={"test.py": "content"}
        )
        
        assert result.is_success() == True
    
    def test_is_success_no_files(self):
        """Test is_success returns False when no files generated."""
        result = GenerationResult(
            kernel_name="test_kernel",
            source_file=Path("/test/source.sv")
        )
        
        assert result.is_success() == False
    
    def test_is_success_validation_failed(self):
        """Test is_success returns False when validation failed."""
        result = GenerationResult(
            kernel_name="test_kernel",
            source_file=Path("/test/source.sv"),
            generated_files={"test.py": "content"},
            validation_passed=False
        )
        
        assert result.is_success() == False
    
    def test_get_summary(self):
        """Test get_summary returns correct summary information."""
        result = GenerationResult(
            kernel_name="test_kernel",
            source_file=Path("/test/source.sv"),
            generated_files={"test.py": "content", "wrapper.v": "module"},
            generation_time_ms=123.45
        )
        result.add_warning("Test warning")
        result.add_error("Test error")
        
        summary = result.get_summary()
        
        assert summary["kernel_name"] == "test_kernel"
        assert summary["source_file"] == str(Path("/test/source.sv"))
        assert summary["success"] == False  # Has errors
        assert summary["files_generated"] == 2
        assert summary["validation_passed"] == False  # add_error sets this
        assert summary["error_count"] == 1
        assert summary["warning_count"] == 1
        assert summary["generation_time_ms"] == 123.45


class TestResultHandler:
    """Test suite for ResultHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "output"
        self.handler = ResultHandler(self.output_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_result_handler_initialization(self):
        """Test ResultHandler initialization creates output directory."""
        assert self.output_dir.exists()
        assert self.output_dir.is_dir()
        assert self.handler.output_dir == self.output_dir
    
    def test_result_handler_initialization_existing_dir(self):
        """Test ResultHandler initialization with existing directory."""
        # Directory already exists from setup_method
        handler2 = ResultHandler(self.output_dir)
        
        assert handler2.output_dir == self.output_dir
        assert self.output_dir.exists()
    
    def test_result_handler_initialization_permission_error(self):
        """Test ResultHandler initialization with permission error."""
        # Mock permission error during directory creation
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("No permission")
            
            with pytest.raises(RuntimeError, match="No write permission"):
                ResultHandler(Path("/root/forbidden"))
    
    def test_result_handler_initialization_write_test_failure(self):
        """Test ResultHandler initialization with write test failure."""
        # Create directory but mock write test failure
        test_dir = self.temp_dir / "write_test"
        test_dir.mkdir()
        
        with patch('pathlib.Path.write_text') as mock_write:
            mock_write.side_effect = PermissionError("Write failed")
            
            with pytest.raises(RuntimeError, match="No write permission"):
                ResultHandler(test_dir)
    
    def test_write_result_success(self):
        """Test successful result writing."""
        files = {
            "hw_custom_op.py": "class TestHWCustomOp: pass",
            "rtl_wrapper.v": "module test_wrapper(); endmodule",
            "test_suite.py": "class TestTestKernel: pass"
        }
        
        result = GenerationResult(
            kernel_name="test_kernel",
            source_file=Path("/test/source.sv"),
            generated_files=files,
            generation_time_ms=100.0
        )
        
        with patch('brainsmith.tools.hw_kernel_gen.result_handler.logger') as mock_logger:
            kernel_dir = self.handler.write_result(result)
            
            # Check kernel directory was created
            assert kernel_dir.exists()
            assert kernel_dir.name == "test_kernel"
            
            # Check all files were written
            for filename, content in files.items():
                file_path = kernel_dir / filename
                assert file_path.exists()
                assert file_path.read_text() == content
            
            # Check metadata file was written
            metadata_file = kernel_dir / "generation_metadata.json"
            assert metadata_file.exists()
            
            # Check summary log was written
            summary_file = kernel_dir / "generation_summary.txt"
            assert summary_file.exists()
            
            mock_logger.info.assert_called()
    
    def test_write_result_with_template_context(self):
        """Test result writing with template context metadata."""
        result = GenerationResult(
            kernel_name="test_kernel",
            source_file=Path("/test/source.sv"),
            generated_files={"test.py": "content"},
            template_context=MockTemplateContext()
        )
        
        kernel_dir = self.handler.write_result(result)
        
        # Check metadata includes template context info
        metadata_file = kernel_dir / "generation_metadata.json"
        metadata = json.loads(metadata_file.read_text())
        
        assert "template_context" in metadata
        assert metadata["template_context"]["parameter_count"] == 2
        assert metadata["template_context"]["interface_count"] == 2
        assert metadata["template_context"]["required_parameters"] == ["PE"]
        assert metadata["template_context"]["whitelisted_defaults"] == {"SIMD": 1}
    
    def test_write_result_with_kernel_metadata(self):
        """Test result writing with kernel metadata."""
        result = GenerationResult(
            kernel_name="test_kernel",
            source_file=Path("/test/source.sv"),
            generated_files={"test.py": "content"},
            kernel_metadata=MockKernelMetadata()
        )
        
        kernel_dir = self.handler.write_result(result)
        
        # Check metadata includes kernel metadata info
        metadata_file = kernel_dir / "generation_metadata.json"
        metadata = json.loads(metadata_file.read_text())
        
        assert "kernel_metadata" in metadata
        assert metadata["kernel_metadata"]["parameter_count"] == 2
        assert metadata["kernel_metadata"]["interface_count"] == 2
        assert metadata["kernel_metadata"]["parameter_names"] == ["PE", "SIMD"]
    
    def test_write_result_file_write_error(self):
        """Test result writing with file write error."""
        result = GenerationResult(
            kernel_name="test_kernel",
            source_file=Path("/test/source.sv"),
            generated_files={"test.py": "content"}
        )
        
        # Mock file write failure for generated files
        with patch('pathlib.Path.write_text') as mock_write:
            mock_write.side_effect = [
                OSError("Write failed"),  # First call fails
                None,  # Subsequent calls succeed (metadata, summary)
                None
            ]
            
            with patch('brainsmith.tools.hw_kernel_gen.result_handler.logger'):
                kernel_dir = self.handler.write_result(result)
                
                # Should still create directory and metadata despite file error
                assert kernel_dir.exists()
                assert len(result.errors) == 1
                assert "Failed to write file test.py" in result.errors[0]
    
    def test_write_result_directory_creation_error(self):
        """Test result writing with major failure that causes exception."""
        result = GenerationResult(
            kernel_name="test_kernel",
            source_file=Path("/test/source.sv"),
            generated_files={"test.py": "content"}
        )
        
        # Mock a major failure that causes the whole operation to fail
        with patch.object(self.handler, '_write_metadata') as mock_write_metadata:
            mock_write_metadata.side_effect = Exception("Critical failure")
            
            with pytest.raises(RuntimeError, match="Failed to write result"):
                self.handler.write_result(result)
    
    def test_write_metadata_json_serialization_error(self):
        """Test metadata writing with JSON serialization issues."""
        result = GenerationResult(
            kernel_name="test_kernel",
            source_file=Path("/test/source.sv"),
            generated_files={"test.py": "content"}
        )
        
        kernel_dir = self.output_dir / "test_kernel"
        kernel_dir.mkdir()
        
        # Mock JSON dumps to fail
        with patch('json.dumps') as mock_dumps:
            mock_dumps.side_effect = TypeError("Not JSON serializable")
            
            with patch('brainsmith.tools.hw_kernel_gen.result_handler.logger') as mock_logger:
                self.handler._write_metadata(kernel_dir, result)
                
                # Should log warning but not crash
                mock_logger.warning.assert_called_once()
    
    def test_write_summary_log_content(self):
        """Test summary log content is correctly formatted."""
        result = GenerationResult(
            kernel_name="test_kernel",
            source_file=Path("/test/source.sv"),
            generated_files={"test.py": "content"},
            generation_time_ms=150.75
        )
        result.add_error("Test error")
        result.add_warning("Test warning")
        
        kernel_dir = self.output_dir / "test_kernel"
        kernel_dir.mkdir()
        
        self.handler._write_summary_log(kernel_dir, result, ["/path/to/test.py"])
        
        summary_file = kernel_dir / "generation_summary.txt"
        summary_content = summary_file.read_text()
        
        assert "Generation Summary for test_kernel" in summary_content
        assert "Success: False" in summary_content
        assert "Files Generated: 1" in summary_content
        assert "Generated Files:" in summary_content
        assert "/path/to/test.py" in summary_content
        assert "Errors:" in summary_content
        assert "Test error" in summary_content
        assert "Warnings:" in summary_content
        assert "Test warning" in summary_content
        assert "Generation Time: 150.75 ms" in summary_content
    
    def test_cleanup_failed_generation(self):
        """Test cleanup of failed generation directory."""
        # Create a kernel directory with some files
        kernel_dir = self.output_dir / "failed_kernel"
        kernel_dir.mkdir()
        (kernel_dir / "test_file.py").write_text("test content")
        
        assert kernel_dir.exists()
        
        with patch('brainsmith.tools.hw_kernel_gen.result_handler.logger') as mock_logger:
            self.handler.cleanup_failed_generation("failed_kernel")
            
            assert not kernel_dir.exists()
            mock_logger.info.assert_called_once()
    
    def test_cleanup_failed_generation_nonexistent(self):
        """Test cleanup of nonexistent directory."""
        with patch('brainsmith.tools.hw_kernel_gen.result_handler.logger') as mock_logger:
            self.handler.cleanup_failed_generation("nonexistent_kernel")
            
            # Should not raise error
            mock_logger.info.assert_not_called()
    
    def test_cleanup_failed_generation_permission_error(self):
        """Test cleanup with permission error."""
        kernel_dir = self.output_dir / "protected_kernel"
        kernel_dir.mkdir()
        
        with patch('shutil.rmtree') as mock_rmtree:
            mock_rmtree.side_effect = PermissionError("Permission denied")
            
            with patch('brainsmith.tools.hw_kernel_gen.result_handler.logger') as mock_logger:
                self.handler.cleanup_failed_generation("protected_kernel")
                
                mock_logger.warning.assert_called_once()
    
    def test_get_existing_results(self):
        """Test getting list of existing kernel results."""
        # Create some kernel directories
        (self.output_dir / "kernel1").mkdir()
        (self.output_dir / "kernel2").mkdir()
        (self.output_dir / ".hidden").mkdir()  # Should be ignored
        (self.output_dir / "regular_file.txt").write_text("test")  # Should be ignored
        
        results = self.handler.get_existing_results()
        
        assert set(results) == {"kernel1", "kernel2"}
        assert ".hidden" not in results
        assert "regular_file.txt" not in results
    
    def test_get_existing_results_no_output_dir(self):
        """Test getting existing results when output directory doesn't exist."""
        # Remove output directory
        shutil.rmtree(self.output_dir)
        
        results = self.handler.get_existing_results()
        assert results == []
    
    def test_get_existing_results_permission_error(self):
        """Test getting existing results with permission error."""
        with patch('pathlib.Path.iterdir') as mock_iterdir:
            mock_iterdir.side_effect = PermissionError("Permission denied")
            
            with patch('brainsmith.tools.hw_kernel_gen.result_handler.logger') as mock_logger:
                results = self.handler.get_existing_results()
                
                assert results == []
                mock_logger.warning.assert_called_once()
    
    def test_load_result_metadata_success(self):
        """Test loading result metadata successfully."""
        # Create metadata file
        kernel_dir = self.output_dir / "test_kernel"
        kernel_dir.mkdir()
        
        metadata = {
            "kernel_name": "test_kernel",
            "success": True,
            "files_generated": 2
        }
        
        metadata_file = kernel_dir / "generation_metadata.json"
        metadata_file.write_text(json.dumps(metadata))
        
        loaded_metadata = self.handler.load_result_metadata("test_kernel")
        
        assert loaded_metadata == metadata
        assert loaded_metadata["kernel_name"] == "test_kernel"
    
    def test_load_result_metadata_nonexistent(self):
        """Test loading metadata for nonexistent kernel."""
        loaded_metadata = self.handler.load_result_metadata("nonexistent_kernel")
        assert loaded_metadata is None
    
    def test_load_result_metadata_invalid_json(self):
        """Test loading metadata with invalid JSON."""
        kernel_dir = self.output_dir / "test_kernel"
        kernel_dir.mkdir()
        
        metadata_file = kernel_dir / "generation_metadata.json"
        metadata_file.write_text("invalid json content")
        
        with patch('brainsmith.tools.hw_kernel_gen.result_handler.logger') as mock_logger:
            loaded_metadata = self.handler.load_result_metadata("test_kernel")
            
            assert loaded_metadata is None
            mock_logger.warning.assert_called_once()
    
    def test_load_result_metadata_permission_error(self):
        """Test loading metadata with permission error."""
        kernel_dir = self.output_dir / "test_kernel"
        kernel_dir.mkdir()
        
        metadata_file = kernel_dir / "generation_metadata.json"
        metadata_file.write_text('{"test": "data"}')
        
        with patch('pathlib.Path.read_text') as mock_read:
            mock_read.side_effect = PermissionError("Permission denied")
            
            with patch('brainsmith.tools.hw_kernel_gen.result_handler.logger') as mock_logger:
                loaded_metadata = self.handler.load_result_metadata("test_kernel")
                
                assert loaded_metadata is None
                mock_logger.warning.assert_called_once()


class TestResultHandlerIntegration:
    """Integration tests for ResultHandler with real file operations."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "integration_output"
    
    def teardown_method(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.integration
    def test_end_to_end_result_handling(self):
        """Test complete end-to-end result handling flow."""
        handler = ResultHandler(self.output_dir)
        
        # Create comprehensive result
        files = {
            "matrix_mult_hw_custom_op.py": """
class MatrixMultHWCustomOp(AutoHWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        self.runtime_parameters = {
            "PE": self.get_nodeattr("PE"),
            "SIMD": self.get_nodeattr("SIMD")
        }
""",
            "matrix_mult_wrapper.v": """
module matrix_mult_wrapper #(
    parameter PE = 1,
    parameter SIMD = 1
) (
    input wire ap_clk,
    input wire ap_rst_n
);
    matrix_mult #(.PE(PE), .SIMD(SIMD)) dut ();
endmodule
""",
            "test_matrix_mult.py": """
class TestMatrixMultHWCustomOp:
    def test_parameter_extraction(self):
        assert True
"""
        }
        
        template_context = MockTemplateContext()
        kernel_metadata = MockKernelMetadata("matrix_mult")
        
        result = GenerationResult(
            kernel_name="matrix_mult",
            source_file=Path("/src/matrix_mult.sv"),
            generated_files=files,
            template_context=template_context,
            kernel_metadata=kernel_metadata,
            generation_time_ms=250.5
        )
        result.add_warning("Large parameter value detected")
        
        # Write result
        kernel_dir = handler.write_result(result)
        
        # Verify complete directory structure
        assert kernel_dir.exists()
        assert kernel_dir.name == "matrix_mult"
        
        # Verify all files exist and have correct content
        for filename, expected_content in files.items():
            file_path = kernel_dir / filename
            assert file_path.exists()
            actual_content = file_path.read_text()
            assert actual_content == expected_content
        
        # Verify metadata file
        metadata_file = kernel_dir / "generation_metadata.json"
        assert metadata_file.exists()
        metadata = json.loads(metadata_file.read_text())
        
        assert metadata["kernel_name"] == "matrix_mult"
        assert metadata["success"] == True
        assert metadata["warning_count"] == 1
        assert metadata["generation_time_ms"] == 250.5
        assert "template_context" in metadata
        assert "kernel_metadata" in metadata
        
        # Verify summary file
        summary_file = kernel_dir / "generation_summary.txt"
        assert summary_file.exists()
        summary_content = summary_file.read_text()
        
        assert "matrix_mult" in summary_content
        assert "Success: True" in summary_content
        assert "Files Generated: 3" in summary_content
        assert "Large parameter value detected" in summary_content
        
        # Test that handler can list this result
        existing_results = handler.get_existing_results()
        assert "matrix_mult" in existing_results
        
        # Test that handler can load metadata
        loaded_metadata = handler.load_result_metadata("matrix_mult")
        assert loaded_metadata["kernel_name"] == "matrix_mult"
        assert loaded_metadata["success"] == True