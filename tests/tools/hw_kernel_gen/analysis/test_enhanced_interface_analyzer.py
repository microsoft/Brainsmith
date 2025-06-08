"""
Test suite for Enhanced Interface Analyzer.

Tests interface detection, classification, validation, and dataflow conversion.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from brainsmith.tools.hw_kernel_gen.enhanced_data_structures import RTLSignal, RTLInterface, RTLModule
from brainsmith.tools.hw_kernel_gen.enhanced_config import PipelineConfig, AnalysisConfig
from brainsmith.tools.hw_kernel_gen.analysis.enhanced_interface_analyzer import (
    InterfaceClassifier, InterfaceValidator, DataflowInterfaceConverter,
    InterfaceAnalyzer, InterfaceAnalysisResult,
    create_interface_analyzer, analyze_interfaces
)
from brainsmith.tools.hw_kernel_gen.analysis.analysis_patterns import (
    InterfaceType, SignalRole, InterfacePattern, SignalPattern,
    get_interface_patterns, create_custom_interface_pattern
)


class TestInterfaceClassifier:
    """Test interface classification functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = InterfaceClassifier()
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        assert len(self.classifier.patterns) > 0
        assert self.classifier._pattern_cache == {}
    
    def test_axi_stream_classification(self):
        """Test AXI-Stream interface classification."""
        # Create AXI-Stream signals
        signals = [
            RTLSignal("s_axis_tdata", "input", 32, interface_role="tdata"),
            RTLSignal("s_axis_tvalid", "input", 1, interface_role="tvalid"),
            RTLSignal("s_axis_tready", "output", 1, interface_role="tready"),
            RTLSignal("s_axis_tlast", "input", 1, interface_role="tlast")
        ]
        
        interface_type, confidence = self.classifier.get_best_classification("s_axis_input", signals)
        
        assert interface_type == InterfaceType.AXI_STREAM
        assert confidence > 0.7
    
    def test_axi_lite_classification(self):
        """Test AXI-Lite interface classification."""
        # Create AXI-Lite signals
        signals = [
            RTLSignal("s_axi_awaddr", "input", 32, interface_role="awaddr"),
            RTLSignal("s_axi_awvalid", "input", 1, interface_role="awvalid"),
            RTLSignal("s_axi_awready", "output", 1, interface_role="awready"),
            RTLSignal("s_axi_wdata", "input", 32, interface_role="wdata"),
            RTLSignal("s_axi_wvalid", "input", 1, interface_role="wvalid"),
            RTLSignal("s_axi_wready", "output", 1, interface_role="wready"),
        ]
        
        interface_type, confidence = self.classifier.get_best_classification("s_axi_control", signals)
        
        assert interface_type == InterfaceType.AXI_LITE
        assert confidence > 0.5
    
    def test_control_interface_classification(self):
        """Test control interface classification."""
        signals = [
            RTLSignal("ap_clk", "input", 1, interface_role="clock"),
            RTLSignal("ap_rst_n", "input", 1, interface_role="reset"),
            RTLSignal("ap_start", "input", 1, interface_role="enable"),
            RTLSignal("ap_done", "output", 1, interface_role="interrupt")
        ]
        
        interface_type, confidence = self.classifier.get_best_classification("control", signals)
        
        assert interface_type == InterfaceType.CONTROL
        assert confidence > 0.6
    
    def test_unknown_interface_classification(self):
        """Test unknown interface classification."""
        signals = [
            RTLSignal("unknown_signal1", "input", 8),
            RTLSignal("unknown_signal2", "output", 16)
        ]
        
        interface_type, confidence = self.classifier.get_best_classification("unknown", signals)
        
        assert interface_type == InterfaceType.UNKNOWN
        assert confidence == 0.0
    
    def test_partial_interface_classification(self):
        """Test classification with partial signal matches."""
        # Incomplete AXI-Stream (missing tready)
        signals = [
            RTLSignal("s_axis_tdata", "input", 32, interface_role="tdata"),
            RTLSignal("s_axis_tvalid", "input", 1, interface_role="tvalid")
        ]
        
        candidates = self.classifier.classify_interface("s_axis_input", signals)
        
        # Should still classify as AXI-Stream but with lower confidence
        axi_stream_candidates = [c for c in candidates if c[0] == InterfaceType.AXI_STREAM]
        assert len(axi_stream_candidates) > 0
        assert axi_stream_candidates[0][1] < 0.7  # Lower confidence due to missing signals
    
    def test_custom_patterns(self):
        """Test classification with custom patterns."""
        # Create custom pattern
        custom_patterns = [
            create_custom_interface_pattern(
                InterfaceType.CUSTOM,
                signal_patterns=[
                    SignalPattern(SignalRole.DATA, [".*_custom_data$"], required=True),
                    SignalPattern(SignalRole.VALID, [".*_custom_valid$"], required=True)
                ],
                prefix_patterns=["custom_"],
                required_signals={SignalRole.DATA, SignalRole.VALID},
                min_signals=2
            )
        ]
        
        classifier = InterfaceClassifier(get_interface_patterns() + custom_patterns)
        
        signals = [
            RTLSignal("custom_input_custom_data", "input", 32, interface_role="data"),
            RTLSignal("custom_input_custom_valid", "input", 1, interface_role="valid")
        ]
        
        interface_type, confidence = classifier.get_best_classification("custom_input", signals)
        
        assert interface_type == InterfaceType.CUSTOM
        assert confidence > 0.8


class TestInterfaceValidator:
    """Test interface validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = AnalysisConfig()
        self.validator = InterfaceValidator(self.config)
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        assert self.validator.config is not None
    
    def test_valid_axi_stream_validation(self):
        """Test validation of complete AXI-Stream interface."""
        signals = [
            RTLSignal("tdata", "input", 32, interface_role="tdata"),
            RTLSignal("tvalid", "input", 1, interface_role="tvalid"),
            RTLSignal("tready", "output", 1, interface_role="tready")
        ]
        
        interface = RTLInterface("test_axis", "axi_stream", signals)
        result = self.validator.validate_interface(interface, InterfaceType.AXI_STREAM)
        
        # Check validation success
        success = result.success if hasattr(result, 'success') else result.get("success", True)
        assert success == True
    
    def test_incomplete_axi_stream_validation(self):
        """Test validation of incomplete AXI-Stream interface."""
        # Missing tready signal
        signals = [
            RTLSignal("tdata", "input", 32, interface_role="tdata"),
            RTLSignal("tvalid", "input", 1, interface_role="tvalid")
        ]
        
        interface = RTLInterface("incomplete_axis", "axi_stream", signals)
        result = self.validator.validate_interface(interface, InterfaceType.AXI_STREAM)
        
        # Should have validation errors
        errors = result.errors if hasattr(result, 'errors') else result.get("errors", [])
        assert len(errors) > 0
    
    def test_signal_direction_validation(self):
        """Test validation of signal directions."""
        # Wrong direction for tready (should be output)
        signals = [
            RTLSignal("tdata", "input", 32, interface_role="tdata"),
            RTLSignal("tvalid", "input", 1, interface_role="tvalid"),
            RTLSignal("tready", "input", 1, interface_role="tready")  # Wrong direction
        ]
        
        interface = RTLInterface("test_axis", "axi_stream", signals)
        result = self.validator.validate_interface(interface, InterfaceType.AXI_STREAM)
        
        # Should have warnings about direction mismatch
        warnings = result.warnings if hasattr(result, 'warnings') else result.get("warnings", [])
        assert len(warnings) > 0
    
    def test_signal_width_validation(self):
        """Test validation of signal widths."""
        # Wrong width for tvalid (should be 1)
        signals = [
            RTLSignal("tdata", "input", 32, interface_role="tdata"),
            RTLSignal("tvalid", "input", 8, interface_role="tvalid"),  # Wrong width
            RTLSignal("tready", "output", 1, interface_role="tready")
        ]
        
        interface = RTLInterface("test_axis", "axi_stream", signals)
        result = self.validator.validate_interface(interface, InterfaceType.AXI_STREAM)
        
        # Should have warnings about width mismatch
        warnings = result.warnings if hasattr(result, 'warnings') else result.get("warnings", [])
        assert len(warnings) > 0


class TestDataflowInterfaceConverter:
    """Test dataflow interface conversion."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PipelineConfig()
        self.converter = DataflowInterfaceConverter(self.config)
    
    def test_converter_initialization(self):
        """Test converter initialization."""
        assert self.converter.config is not None
    
    @patch('brainsmith.tools.hw_kernel_gen.analysis.enhanced_interface_analyzer.DATAFLOW_AVAILABLE', False)
    def test_converter_without_dataflow(self):
        """Test converter when dataflow is not available."""
        signals = [
            RTLSignal("tdata", "input", 32, interface_role="tdata"),
            RTLSignal("tvalid", "input", 1, interface_role="tvalid"),
            RTLSignal("tready", "output", 1, interface_role="tready")
        ]
        
        interface = RTLInterface("test_axis", "axi_stream", signals)
        result = self.converter.convert_interface(interface, InterfaceType.AXI_STREAM)
        
        assert result is None
    
    def test_dataflow_type_inference(self):
        """Test dataflow interface type inference."""
        # Test AXI-Stream input interface
        signals_input = [
            RTLSignal("tdata", "input", 32, interface_role="tdata"),
            RTLSignal("tvalid", "input", 1, interface_role="tvalid"),
            RTLSignal("tready", "output", 1, interface_role="tready")
        ]
        interface_input = RTLInterface("input_axis", "axi_stream", signals_input)
        
        # Test AXI-Stream output interface
        signals_output = [
            RTLSignal("tdata", "output", 32, interface_role="tdata"),
            RTLSignal("tvalid", "output", 1, interface_role="tvalid"),
            RTLSignal("tready", "input", 1, interface_role="tready")
        ]
        interface_output = RTLInterface("output_axis", "axi_stream", signals_output)
        
        # The dataflow type inference should work even without full dataflow system
        with patch('brainsmith.tools.hw_kernel_gen.analysis.enhanced_interface_analyzer.DATAFLOW_AVAILABLE', True):
            # Mock the DataflowInterfaceType
            with patch('brainsmith.tools.hw_kernel_gen.analysis.enhanced_interface_analyzer.DataflowInterfaceType') as mock_type:
                mock_type.INPUT = "INPUT"
                mock_type.OUTPUT = "OUTPUT"
                mock_type.CONTROL = "CONTROL"
                
                input_type = self.converter.infer_dataflow_type(InterfaceType.AXI_STREAM, interface_input)
                output_type = self.converter.infer_dataflow_type(InterfaceType.AXI_STREAM, interface_output)
                
                assert input_type == "INPUT"
                assert output_type == "OUTPUT"


class TestInterfaceAnalyzer:
    """Test complete interface analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PipelineConfig()
        self.analyzer = InterfaceAnalyzer(self.config)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.config is not None
        assert self.analyzer.classifier is not None
        assert self.analyzer.validator is not None
        assert self.analyzer.converter is not None
    
    def create_test_module(self) -> RTLModule:
        """Create a test RTL module with various interfaces."""
        # AXI-Stream input interface
        axis_input_signals = [
            RTLSignal("s_axis_tdata", "input", 32, interface_role="tdata"),
            RTLSignal("s_axis_tvalid", "input", 1, interface_role="tvalid"),
            RTLSignal("s_axis_tready", "output", 1, interface_role="tready")
        ]
        axis_input = RTLInterface("s_axis_input", "axi_stream", axis_input_signals)
        
        # AXI-Stream output interface
        axis_output_signals = [
            RTLSignal("m_axis_tdata", "output", 32, interface_role="tdata"),
            RTLSignal("m_axis_tvalid", "output", 1, interface_role="tvalid"),
            RTLSignal("m_axis_tready", "input", 1, interface_role="tready")
        ]
        axis_output = RTLInterface("m_axis_output", "axi_stream", axis_output_signals)
        
        # Control interface
        control_signals = [
            RTLSignal("ap_clk", "input", 1, interface_role="clock"),
            RTLSignal("ap_rst_n", "input", 1, interface_role="reset"),
            RTLSignal("ap_start", "input", 1, interface_role="enable"),
            RTLSignal("ap_done", "output", 1, interface_role="interrupt")
        ]
        control = RTLInterface("control", "control", control_signals)
        
        return RTLModule(
            name="test_module",
            interfaces=[axis_input, axis_output, control],
            parameters={"WIDTH": 32, "DEPTH": 64}
        )
    
    def test_single_interface_analysis(self):
        """Test analysis of a single interface."""
        signals = [
            RTLSignal("s_axis_tdata", "input", 32, interface_role="tdata"),
            RTLSignal("s_axis_tvalid", "input", 1, interface_role="tvalid"),
            RTLSignal("s_axis_tready", "output", 1, interface_role="tready")
        ]
        interface = RTLInterface("s_axis_input", "axi_stream", signals)
        
        result = self.analyzer.analyze_single_interface(interface)
        
        assert isinstance(result, InterfaceAnalysisResult)
        assert result.interface_name == "s_axis_input"
        assert result.interface_type == InterfaceType.AXI_STREAM
        assert result.confidence > 0.7
        assert len(result.detected_signals) == 3
    
    def test_module_interface_analysis(self):
        """Test analysis of all interfaces in a module."""
        module = self.create_test_module()
        results = self.analyzer.analyze_interfaces(module)
        
        assert len(results) == 3  # Three interfaces
        
        # Check that we have the expected interface types
        interface_types = {result.interface_type for result in results}
        assert InterfaceType.AXI_STREAM in interface_types
        assert InterfaceType.CONTROL in interface_types
        
        # Check that all interfaces were analyzed
        interface_names = {result.interface_name for result in results}
        assert "s_axis_input" in interface_names
        assert "m_axis_output" in interface_names
        assert "control" in interface_names
    
    def test_analysis_with_validation(self):
        """Test analysis with validation enabled."""
        self.analyzer.config.validation.validate_interface_constraints = True
        
        module = self.create_test_module()
        results = self.analyzer.analyze_interfaces(module)
        
        # All results should have validation performed
        for result in results:
            assert result.validation_result is not None
            assert isinstance(result.is_valid, bool)
    
    def test_analysis_statistics(self):
        """Test analysis statistics tracking."""
        module = self.create_test_module()
        
        initial_stats = self.analyzer.get_analysis_statistics()
        initial_count = initial_stats["analysis_count"]
        
        # Perform analysis
        self.analyzer.analyze_interfaces(module)
        
        final_stats = self.analyzer.get_analysis_statistics()
        assert final_stats["analysis_count"] == initial_count + 1
        assert final_stats["total_analysis_time"] > 0
    
    def test_custom_patterns(self):
        """Test analysis with custom interface patterns."""
        # Create custom pattern
        custom_patterns = [
            create_custom_interface_pattern(
                InterfaceType.CUSTOM,
                signal_patterns=[
                    SignalPattern(SignalRole.DATA, [".*_custom_data$"], required=True),
                    SignalPattern(SignalRole.VALID, [".*_custom_valid$"], required=True)
                ],
                prefix_patterns=["custom_"],
                required_signals={SignalRole.DATA, SignalRole.VALID}
            )
        ]
        
        # Create module with custom interface
        custom_signals = [
            RTLSignal("custom_input_custom_data", "input", 32, interface_role="data"),
            RTLSignal("custom_input_custom_valid", "input", 1, interface_role="valid")
        ]
        custom_interface = RTLInterface("custom_input", "custom", custom_signals)
        
        module = RTLModule("test_custom", interfaces=[custom_interface])
        
        # Analyze with custom patterns
        results = self.analyzer.analyze_interfaces(module, custom_patterns)
        
        assert len(results) == 1
        assert results[0].interface_type == InterfaceType.CUSTOM
        assert results[0].confidence > 0.8


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_interface_analyzer(self):
        """Test interface analyzer factory."""
        config = PipelineConfig()
        analyzer = create_interface_analyzer(config)
        
        assert isinstance(analyzer, InterfaceAnalyzer)
        assert analyzer.config is config
    
    def test_analyze_interfaces_convenience(self):
        """Test convenience function for interface analysis."""
        # Create test module
        signals = [
            RTLSignal("s_axis_tdata", "input", 32, interface_role="tdata"),
            RTLSignal("s_axis_tvalid", "input", 1, interface_role="tvalid"),
            RTLSignal("s_axis_tready", "output", 1, interface_role="tready")
        ]
        interface = RTLInterface("s_axis_input", "axi_stream", signals)
        module = RTLModule("test_module", interfaces=[interface])
        
        # Use convenience function
        results = analyze_interfaces(module)
        
        assert len(results) == 1
        assert results[0].interface_type == InterfaceType.AXI_STREAM


class TestIntegration:
    """Integration tests for interface analysis components."""
    
    def test_end_to_end_analysis(self):
        """Test complete end-to-end interface analysis."""
        # Create a complex module with multiple interface types
        axi_stream_signals = [
            RTLSignal("s_axis_tdata", "input", 32, interface_role="tdata"),
            RTLSignal("s_axis_tvalid", "input", 1, interface_role="tvalid"),
            RTLSignal("s_axis_tready", "output", 1, interface_role="tready"),
            RTLSignal("s_axis_tlast", "input", 1, interface_role="tlast")
        ]
        axi_stream = RTLInterface("s_axis_input", "axi_stream", axi_stream_signals)
        
        axi_lite_signals = [
            RTLSignal("s_axi_awaddr", "input", 32, interface_role="awaddr"),
            RTLSignal("s_axi_awvalid", "input", 1, interface_role="awvalid"),
            RTLSignal("s_axi_awready", "output", 1, interface_role="awready"),
            RTLSignal("s_axi_wdata", "input", 32, interface_role="wdata"),
            RTLSignal("s_axi_wvalid", "input", 1, interface_role="wvalid"),
            RTLSignal("s_axi_wready", "output", 1, interface_role="wready")
        ]
        axi_lite = RTLInterface("s_axi_control", "axi_lite", axi_lite_signals)
        
        control_signals = [
            RTLSignal("ap_clk", "input", 1, interface_role="clock"),
            RTLSignal("ap_rst_n", "input", 1, interface_role="reset")
        ]
        control = RTLInterface("control", "control", control_signals)
        
        module = RTLModule(
            name="complex_module",
            interfaces=[axi_stream, axi_lite, control],
            parameters={"WIDTH": 32, "DEPTH": 64}
        )
        
        # Perform analysis
        config = PipelineConfig()
        config.validation.validate_interface_constraints = True
        
        analyzer = InterfaceAnalyzer(config)
        results = analyzer.analyze_interfaces(module)
        
        # Verify results
        assert len(results) == 3
        
        # Check interface types were correctly identified
        interface_types = {result.interface_type for result in results}
        assert InterfaceType.AXI_STREAM in interface_types
        assert InterfaceType.AXI_LITE in interface_types
        assert InterfaceType.CONTROL in interface_types
        
        # Check validation was performed
        for result in results:
            assert result.validation_result is not None
        
        # Check analysis performance
        stats = analyzer.get_analysis_statistics()
        assert stats["analysis_count"] > 0
        assert stats["total_analysis_time"] > 0
    
    def test_error_handling(self):
        """Test error handling in interface analysis."""
        # Create module with problematic interface
        empty_signals = []
        empty_interface = RTLInterface("empty", "unknown", empty_signals)
        module = RTLModule("error_module", interfaces=[empty_interface])
        
        # Analysis should handle empty interfaces gracefully
        analyzer = InterfaceAnalyzer()
        results = analyzer.analyze_interfaces(module)
        
        assert len(results) == 1
        assert results[0].interface_type == InterfaceType.UNKNOWN
        assert results[0].confidence == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])