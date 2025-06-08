"""
Test suite for Enhanced Pragma Processor.

Tests pragma parsing, validation, constraint generation, and dataflow conversion.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from brainsmith.tools.hw_kernel_gen.enhanced_data_structures import RTLSignal, RTLInterface, RTLModule
from brainsmith.tools.hw_kernel_gen.enhanced_config import PipelineConfig, AnalysisConfig
from brainsmith.tools.hw_kernel_gen.analysis.enhanced_pragma_processor import (
    PragmaParser, PragmaValidator, DataflowPragmaConverter,
    PragmaProcessor, ParsedPragma, PragmaProcessingResult,
    create_pragma_processor, process_pragmas
)
from brainsmith.tools.hw_kernel_gen.analysis.analysis_patterns import (
    PragmaType, PragmaPattern, get_pragma_patterns
)


class TestPragmaParser:
    """Test pragma parsing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PragmaParser()
    
    def test_parser_initialization(self):
        """Test parser initialization."""
        assert len(self.parser.patterns) > 0
        assert len(self.parser._compiled_patterns) > 0
    
    def test_brainsmith_pragma_parsing(self):
        """Test parsing of Brainsmith pragmas."""
        pragma_text = "// @brainsmith interface input0 axis_stream"
        
        result = self.parser.parse_pragma_text(pragma_text, 1)
        
        assert result is not None
        assert result.pragma_type == PragmaType.BRAINSMITH
        assert result.directive == "interface"
        assert result.line_number == 1
        assert result.raw_text == pragma_text
    
    def test_hls_pragma_parsing(self):
        """Test parsing of HLS pragmas."""
        pragma_text = "#pragma HLS INTERFACE axis port=input0"
        
        result = self.parser.parse_pragma_text(pragma_text, 2)
        
        assert result is not None
        assert result.pragma_type == PragmaType.HLS
        assert result.directive == "INTERFACE"
        assert result.line_number == 2
    
    def test_interface_pragma_parsing(self):
        """Test parsing of interface pragmas."""
        pragma_text = "// @interface input0 type=axis_stream direction=input width=32"
        
        result = self.parser.parse_pragma_text(pragma_text, 3)
        
        assert result is not None
        assert result.pragma_type == PragmaType.INTERFACE
        assert result.directive == "input0"
        assert result.line_number == 3
    
    def test_invalid_pragma_parsing(self):
        """Test parsing of invalid pragmas."""
        pragma_text = "// This is just a comment"
        
        result = self.parser.parse_pragma_text(pragma_text, 1)
        
        assert result is None
    
    def test_parameter_extraction(self):
        """Test parameter extraction from pragmas."""
        pragma_text = "// @brainsmith parallelism input_par=4"
        
        result = self.parser.parse_pragma_text(pragma_text, 1)
        
        assert result is not None
        assert "parallelism" in result.parameters
    
    def test_reference_extraction(self):
        """Test reference extraction from pragmas."""
        pragma_text = "// @brainsmith interface input0 axis_stream"
        
        result = self.parser.parse_pragma_text(pragma_text, 1)
        
        assert result is not None
        assert "input0" in result.references or "axis_stream" in result.references
    
    def test_pragma_file_parsing(self):
        """Test parsing pragmas from a file."""
        # Create temporary file with pragmas
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.sv', delete=False)
        try:
            temp_file.write("""
// @brainsmith interface input0 axis_stream
// @brainsmith interface output0 axis_stream  
// @brainsmith parallelism input_par=4
// This is a regular comment
module test();
endmodule
""")
            temp_file.close()
            
            results = self.parser.parse_pragma_file(temp_file.name)
            
            # Should find 3 pragmas
            assert len(results) >= 3
            
            # Check pragma types
            pragma_types = {result.pragma_type for result in results}
            assert PragmaType.BRAINSMITH in pragma_types
            
        finally:
            Path(temp_file.name).unlink()
    
    def test_pragma_list_parsing(self):
        """Test parsing list of pragma texts."""
        pragma_texts = [
            "// @brainsmith interface input0 axis_stream",
            "// @interface output0 type=axis_stream direction=output",
            "#pragma HLS PIPELINE II=1"
        ]
        
        results = self.parser.parse_pragma_list(pragma_texts)
        
        assert len(results) == 3
        
        # Check that different pragma types were detected
        pragma_types = {result.pragma_type for result in results}
        assert len(pragma_types) >= 2
    
    def test_custom_patterns(self):
        """Test parsing with custom pragma patterns."""
        # Create custom pattern
        custom_pattern = PragmaPattern(
            PragmaType.CUSTOM,
            patterns=[r"//\s*@custom\s+(\w+)(.*)"],
            parameter_patterns={"param": r"param=(\w+)"}
        )
        
        parser = PragmaParser([custom_pattern])
        pragma_text = "// @custom directive param=value"
        
        result = parser.parse_pragma_text(pragma_text, 1)
        
        assert result is not None
        assert result.pragma_type == PragmaType.CUSTOM
        assert result.directive == "directive"


class TestPragmaValidator:
    """Test pragma validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = AnalysisConfig()
        self.validator = PragmaValidator(self.config)
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        assert self.validator.config is not None
    
    def create_test_module(self) -> RTLModule:
        """Create a test RTL module for validation."""
        signals = [
            RTLSignal("input0_tdata", "input", 32),
            RTLSignal("input0_tvalid", "input", 1),
            RTLSignal("output0_tdata", "output", 32),
            RTLSignal("ap_clk", "input", 1),
            RTLSignal("ap_rst_n", "input", 1)
        ]
        
        interface = RTLInterface("test_interface", "axi_stream", signals)
        
        return RTLModule(
            name="test_module",
            interfaces=[interface],
            parameters={"WIDTH": 32, "DEPTH": 64}
        )
    
    def test_valid_pragma_validation(self):
        """Test validation of valid pragma."""
        pragma = ParsedPragma(
            pragma_type=PragmaType.BRAINSMITH,
            raw_text="// @brainsmith interface input0 axis_stream",
            directive="interface",
            parameters={"interface": ["axis_stream", "input0"]},
            references=["input0", "axis_stream"]
        )
        
        module = self.create_test_module()
        result = self.validator.validate_pragma(pragma, module)
        
        # Should be valid
        success = result.success if hasattr(result, 'success') else result.get("success", True)
        assert success == True
    
    def test_invalid_reference_validation(self):
        """Test validation with invalid references."""
        pragma = ParsedPragma(
            pragma_type=PragmaType.BRAINSMITH,
            raw_text="// @brainsmith interface nonexistent axis_stream",
            directive="interface",
            parameters={"interface": ["axis_stream", "nonexistent"]},
            references=["nonexistent", "axis_stream"]
        )
        
        module = self.create_test_module()
        result = self.validator.validate_pragma(pragma, module)
        
        # Should have warnings about unknown references
        warnings = result.warnings if hasattr(result, 'warnings') else result.get("warnings", [])
        assert len(warnings) > 0
    
    def test_pragma_consistency_validation(self):
        """Test validation of pragma consistency."""
        pragmas = [
            ParsedPragma(
                pragma_type=PragmaType.BRAINSMITH,
                raw_text="// @brainsmith interface input0 axis_stream",
                directive="interface",
                parameters={"interface": ["axis_stream", "input0"]},
                is_valid=True
            ),
            ParsedPragma(
                pragma_type=PragmaType.BRAINSMITH,
                raw_text="// @brainsmith interface input0 axi_lite",
                directive="interface",
                parameters={"interface": ["axi_lite", "input0"]},  # Conflicting definition
                is_valid=True
            )
        ]
        
        module = self.create_test_module()
        result = self.validator.validate_pragma_consistency(pragmas, module)
        
        # Should have errors about conflicting definitions
        errors = result.errors if hasattr(result, 'errors') else result.get("errors", [])
        assert len(errors) > 0
    
    def test_duplicate_pragma_validation(self):
        """Test detection of duplicate pragmas."""
        pragmas = [
            ParsedPragma(
                pragma_type=PragmaType.BRAINSMITH,
                raw_text="// @brainsmith parallelism input_par=4",
                directive="parallelism",
                parameters={"parallelism": ["input_par", "4"]},
                is_valid=True
            ),
            ParsedPragma(
                pragma_type=PragmaType.BRAINSMITH,
                raw_text="// @brainsmith parallelism input_par=4",
                directive="parallelism",
                parameters={"parallelism": ["input_par", "4"]},  # Duplicate
                is_valid=True
            )
        ]
        
        result = self.validator.validate_pragma_consistency(pragmas)
        
        # Should have warnings about duplicates
        warnings = result.warnings if hasattr(result, 'warnings') else result.get("warnings", [])
        assert len(warnings) > 0
    
    def test_interface_coverage_validation(self):
        """Test validation of interface coverage."""
        # Create module with multiple interfaces
        interface1 = RTLInterface("input0", "axi_stream", [])
        interface2 = RTLInterface("output0", "axi_stream", [])
        module = RTLModule("test", interfaces=[interface1, interface2])
        
        # Only one pragma for interface coverage
        pragmas = [
            ParsedPragma(
                pragma_type=PragmaType.BRAINSMITH,
                raw_text="// @brainsmith interface input0 axis_stream",
                directive="interface",
                parameters={"interface": ["axis_stream", "input0"]},
                is_valid=True
            )
        ]
        
        # Enable coverage checking
        self.validator.config.validate_pragma_compatibility = True
        result = self.validator.validate_pragma_consistency(pragmas, module)
        
        # Should have warnings about missing coverage
        warnings = result.warnings if hasattr(result, 'warnings') else result.get("warnings", [])
        assert len(warnings) > 0


class TestDataflowPragmaConverter:
    """Test pragma to dataflow constraint conversion."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PipelineConfig()
        self.converter = DataflowPragmaConverter(self.config)
    
    def test_converter_initialization(self):
        """Test converter initialization."""
        assert self.converter.config is not None
    
    def test_interface_pragma_conversion(self):
        """Test conversion of interface pragmas."""
        pragmas = [
            ParsedPragma(
                pragma_type=PragmaType.INTERFACE,
                directive="input0",
                parameters={"interface": ["axis_stream", "input0"]},
                is_valid=True
            )
        ]
        
        constraints = self.converter.convert_pragmas_to_dataflow(pragmas)
        
        assert "interface_constraints" in constraints
        assert "input0" in constraints["interface_constraints"]
    
    def test_parallelism_pragma_conversion(self):
        """Test conversion of parallelism pragmas."""
        pragmas = [
            ParsedPragma(
                pragma_type=PragmaType.PARALLELISM,
                directive="parallelism",
                parameters={"parallelism": ["input_par", "4"]},
                is_valid=True
            )
        ]
        
        constraints = self.converter.convert_pragmas_to_dataflow(pragmas)
        
        assert "parallelism_constraints" in constraints
        assert "input_par" in constraints["parallelism_constraints"]
        assert constraints["parallelism_constraints"]["input_par"] == 4
    
    def test_dataflow_pragma_conversion(self):
        """Test conversion of dataflow pragmas."""
        pragmas = [
            ParsedPragma(
                pragma_type=PragmaType.DATAFLOW,
                directive="dataflow",
                parameters={"dataflow": ["chunking", "broadcast"]},
                is_valid=True
            )
        ]
        
        constraints = self.converter.convert_pragmas_to_dataflow(pragmas)
        
        assert "chunking_constraints" in constraints
        assert "chunking" in constraints["chunking_constraints"]
    
    @patch('brainsmith.tools.hw_kernel_gen.analysis.enhanced_pragma_processor.DATAFLOW_AVAILABLE', True)
    def test_parallelism_configuration_creation(self):
        """Test creation of parallelism configuration."""
        pragmas = [
            ParsedPragma(
                pragma_type=PragmaType.PARALLELISM,
                directive="parallelism",
                parameters={"input_par": "4", "weight_par": "2"},
                is_valid=True
            )
        ]
        
        # Mock ParallelismConfiguration
        with patch('brainsmith.tools.hw_kernel_gen.analysis.enhanced_pragma_processor.ParallelismConfiguration') as mock_config:
            mock_instance = Mock()
            mock_config.return_value = mock_instance
            
            result = self.converter.create_parallelism_configuration(pragmas)
            
            # Should create configuration
            assert result is mock_instance
            mock_config.assert_called_once()


class TestPragmaProcessor:
    """Test complete pragma processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PipelineConfig()
        self.processor = PragmaProcessor(self.config)
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        assert self.processor.config is not None
        assert self.processor.parser is not None
        assert self.processor.validator is not None
        assert self.processor.converter is not None
    
    def test_pragma_text_processing(self):
        """Test processing of pragma text."""
        pragma_text = "// @brainsmith interface input0 axis_stream"
        
        result = self.processor.process_pragmas(pragma_text)
        
        assert isinstance(result, PragmaProcessingResult)
        assert result.pragma_count == 1
        assert len(result.parsed_pragmas) == 1
        assert result.parsed_pragmas[0].pragma_type == PragmaType.BRAINSMITH
    
    def test_pragma_list_processing(self):
        """Test processing of pragma list."""
        pragma_texts = [
            "// @brainsmith interface input0 axis_stream",
            "// @brainsmith parallelism input_par=4",
            "// @interface output0 type=axis_stream direction=output"
        ]
        
        result = self.processor.process_pragmas(pragma_texts)
        
        assert result.pragma_count == 3
        assert len(result.parsed_pragmas) == 3
        
        # Check pragma types
        pragma_types = {pragma.pragma_type for pragma in result.parsed_pragmas}
        assert PragmaType.BRAINSMITH in pragma_types
        assert PragmaType.INTERFACE in pragma_types
    
    def test_pragma_file_processing(self):
        """Test processing pragmas from file."""
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.sv', delete=False)
        try:
            temp_file.write("""
// @brainsmith interface input0 axis_stream
// @brainsmith interface output0 axis_stream
// @brainsmith parallelism input_par=4
""")
            temp_file.close()
            
            result = self.processor.process_pragmas(temp_file.name)
            
            assert result.pragma_count >= 3
            assert len(result.parsed_pragmas) >= 3
            
        finally:
            Path(temp_file.name).unlink()
    
    def test_pragma_validation_processing(self):
        """Test processing with validation."""
        pragma_texts = [
            "// @brainsmith interface input0 axis_stream",
            "// @brainsmith interface nonexistent axis_stream"  # Invalid reference
        ]
        
        # Create module for validation
        signals = [RTLSignal("input0_tdata", "input", 32)]
        interface = RTLInterface("input0", "axi_stream", signals)
        module = RTLModule("test", interfaces=[interface], parameters={"WIDTH": 32})
        
        result = self.processor.process_pragmas(pragma_texts, module)
        
        assert result.pragma_count == 2
        # Some pragmas may be invalid due to references
        assert result.valid_pragma_count <= result.pragma_count
    
    def test_dataflow_constraint_generation(self):
        """Test dataflow constraint generation."""
        pragma_texts = [
            "// @brainsmith interface input0 axis_stream",
            "// @brainsmith parallelism input_par=4"
        ]
        
        # Enable dataflow
        self.processor.config.dataflow.mode = self.processor.config.dataflow.mode.__class__.DATAFLOW_ONLY
        
        result = self.processor.process_pragmas(pragma_texts)
        
        # Should generate constraints
        assert len(result.interface_constraints) > 0 or len(result.parallelism_constraints) > 0
    
    def test_processing_statistics(self):
        """Test processing statistics tracking."""
        pragma_texts = ["// @brainsmith interface input0 axis_stream"]
        
        initial_stats = self.processor.get_processing_statistics()
        initial_count = initial_stats["processing_count"]
        
        # Process pragmas
        self.processor.process_pragmas(pragma_texts)
        
        final_stats = self.processor.get_processing_statistics()
        assert final_stats["processing_count"] == initial_count + 1
        assert final_stats["total_processing_time"] > 0
    
    def test_empty_pragma_processing(self):
        """Test processing with no pragmas."""
        result = self.processor.process_pragmas([])
        
        assert result.pragma_count == 0
        assert len(result.parsed_pragmas) == 0
        assert result.valid_pragma_count == 0
    
    def test_malformed_pragma_handling(self):
        """Test handling of malformed pragmas."""
        pragma_texts = [
            "// @brainsmith interface input0 axis_stream",  # Valid
            "// @invalid_pragma_format",                    # Invalid
            "// Just a comment"                             # Not a pragma
        ]
        
        result = self.processor.process_pragmas(pragma_texts)
        
        # Should only process valid pragmas
        assert result.pragma_count == 1  # Only the valid one
        assert len(result.parsed_pragmas) == 1


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_pragma_processor(self):
        """Test pragma processor factory."""
        config = PipelineConfig()
        processor = create_pragma_processor(config)
        
        assert isinstance(processor, PragmaProcessor)
        assert processor.config is config
    
    def test_process_pragmas_convenience(self):
        """Test convenience function for pragma processing."""
        pragma_texts = ["// @brainsmith interface input0 axis_stream"]
        
        result = process_pragmas(pragma_texts)
        
        assert isinstance(result, PragmaProcessingResult)
        assert result.pragma_count == 1


class TestIntegration:
    """Integration tests for pragma processing components."""
    
    def test_end_to_end_pragma_processing(self):
        """Test complete end-to-end pragma processing."""
        # Create comprehensive pragma set
        pragma_texts = [
            "// @brainsmith interface input0 axis_stream",
            "// @brainsmith interface output0 axis_stream",
            "// @brainsmith parallelism input_par=4",
            "// @brainsmith parallelism weight_par=2",
            "// @interface control type=axi_lite direction=slave",
            "#pragma HLS PIPELINE II=1"
        ]
        
        # Create matching RTL module
        input_signals = [RTLSignal("input0_tdata", "input", 32)]
        output_signals = [RTLSignal("output0_tdata", "output", 32)]
        control_signals = [RTLSignal("s_axi_awaddr", "input", 32)]
        
        input_iface = RTLInterface("input0", "axi_stream", input_signals)
        output_iface = RTLInterface("output0", "axi_stream", output_signals)
        control_iface = RTLInterface("control", "axi_lite", control_signals)
        
        module = RTLModule(
            name="comprehensive_module",
            interfaces=[input_iface, output_iface, control_iface],
            parameters={"WIDTH": 32, "DEPTH": 64}
        )
        
        # Process with full configuration
        config = PipelineConfig()
        config.analysis.validate_pragma_compatibility = True
        
        processor = PragmaProcessor(config)
        result = processor.process_pragmas(pragma_texts, module)
        
        # Verify comprehensive processing
        assert result.pragma_count >= 4  # At least 4 valid pragmas
        assert result.valid_pragma_count > 0
        
        # Check constraint generation
        assert len(result.interface_constraints) > 0
        assert len(result.parallelism_constraints) > 0
        
        # Check validation was performed
        assert result.overall_validation is not None
        
        # Check processing performance
        stats = processor.get_processing_statistics()
        assert stats["processing_count"] > 0
        assert stats["total_processing_time"] > 0
    
    def test_error_recovery(self):
        """Test error recovery in pragma processing."""
        # Mix of valid and invalid pragmas
        pragma_texts = [
            "// @brainsmith interface input0 axis_stream",  # Valid
            "// @invalid_syntax_error",                     # Invalid syntax
            "// @brainsmith interface unknown_ref axis",   # Invalid reference
            "// @brainsmith parallelism input_par=4"       # Valid
        ]
        
        # Create minimal module
        signals = [RTLSignal("input0_tdata", "input", 32)]
        interface = RTLInterface("input0", "axi_stream", signals)
        module = RTLModule("test", interfaces=[interface])
        
        # Enable error recovery
        config = PipelineConfig()
        config.analysis.continue_on_parse_errors = True
        config.analysis.continue_on_validation_errors = True
        
        processor = PragmaProcessor(config)
        result = processor.process_pragmas(pragma_texts, module)
        
        # Should process successfully despite errors
        assert result.pragma_count >= 2  # At least some valid pragmas
        assert result.processing_time > 0
        
        # May have validation warnings but should not fail completely
        assert result.overall_validation is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])