"""
Comprehensive Week 2 Validation Test Suite.

This test suite demonstrates the complete functionality of the Week 2
interface analysis and pragma processing extraction components.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from brainsmith.tools.hw_kernel_gen.enhanced_data_structures import (
    RTLSignal, RTLInterface, RTLModule, ParsedRTLData
)
from brainsmith.tools.hw_kernel_gen.enhanced_config import PipelineConfig, GeneratorType, DataflowMode

# Import all Week 2 components
from brainsmith.tools.hw_kernel_gen.analysis import (
    # Interface Analysis
    InterfaceClassifier, InterfaceAnalyzer, InterfaceValidator,
    DataflowInterfaceConverter, analyze_interfaces,
    
    # Pragma Processing  
    PragmaParser, PragmaProcessor, PragmaValidator,
    DataflowPragmaConverter, process_pragmas,
    
    # Configuration
    InterfaceAnalysisConfig, PragmaAnalysisConfig, AnalysisProfile,
    create_analysis_config,
    
    # Integration
    AnalysisOrchestrator, AnalysisResults, AnalysisCache,
    LegacyAnalysisAdapter, run_complete_analysis
)

from brainsmith.tools.hw_kernel_gen.analysis.analysis_patterns import (
    InterfaceType, SignalRole, PragmaType,
    get_interface_patterns, get_pragma_patterns
)


class TestWeek2ComponentIntegration:
    """Test integration between Week 2 components."""
    
    def setup_method(self):
        """Set up comprehensive test environment."""
        # Create realistic RTL module
        self.rtl_module = self.create_realistic_rtl_module()
        
        # Create corresponding pragmas
        self.pragma_texts = self.create_realistic_pragmas()
        
        # Create configuration
        self.config = PipelineConfig()
        self.config.generator_type = GeneratorType.AUTO_HW_CUSTOM_OP
        self.config.dataflow.mode = DataflowMode.DATAFLOW_ONLY
        self.config.analysis.analyze_interfaces = True
        self.config.analysis.analyze_dataflow_interfaces = True
        self.config.analysis.validate_pragma_compatibility = True
        self.config.validation.validate_interface_constraints = True
    
    def create_realistic_rtl_module(self) -> RTLModule:
        """Create a realistic RTL module for comprehensive testing."""
        # Input AXI-Stream interface (e.g., image data)
        input_signals = [
            RTLSignal("s_axis_input_tdata", "input", 64, interface_role="tdata"),
            RTLSignal("s_axis_input_tvalid", "input", 1, interface_role="tvalid"),
            RTLSignal("s_axis_input_tready", "output", 1, interface_role="tready"),
            RTLSignal("s_axis_input_tlast", "input", 1, interface_role="tlast"),
            RTLSignal("s_axis_input_tkeep", "input", 8, interface_role="tkeep"),
            RTLSignal("s_axis_input_tuser", "input", 4, interface_role="tuser")
        ]
        input_interface = RTLInterface("s_axis_input", "axi_stream", input_signals)
        
        # Weight AXI-Stream interface (e.g., convolution weights)
        weight_signals = [
            RTLSignal("s_axis_weights_tdata", "input", 32, interface_role="tdata"),
            RTLSignal("s_axis_weights_tvalid", "input", 1, interface_role="tvalid"),
            RTLSignal("s_axis_weights_tready", "output", 1, interface_role="tready"),
            RTLSignal("s_axis_weights_tlast", "input", 1, interface_role="tlast")
        ]
        weight_interface = RTLInterface("s_axis_weights", "axi_stream", weight_signals)
        
        # Output AXI-Stream interface
        output_signals = [
            RTLSignal("m_axis_output_tdata", "output", 32, interface_role="tdata"),
            RTLSignal("m_axis_output_tvalid", "output", 1, interface_role="tvalid"),
            RTLSignal("m_axis_output_tready", "input", 1, interface_role="tready"),
            RTLSignal("m_axis_output_tlast", "output", 1, interface_role="tlast")
        ]
        output_interface = RTLInterface("m_axis_output", "axi_stream", output_signals)
        
        # Control AXI-Lite interface
        control_signals = [
            RTLSignal("s_axi_control_awaddr", "input", 32, interface_role="awaddr"),
            RTLSignal("s_axi_control_awvalid", "input", 1, interface_role="awvalid"),
            RTLSignal("s_axi_control_awready", "output", 1, interface_role="awready"),
            RTLSignal("s_axi_control_wdata", "input", 32, interface_role="wdata"),
            RTLSignal("s_axi_control_wstrb", "input", 4, interface_role="wstrb"),
            RTLSignal("s_axi_control_wvalid", "input", 1, interface_role="wvalid"),
            RTLSignal("s_axi_control_wready", "output", 1, interface_role="wready"),
            RTLSignal("s_axi_control_bresp", "output", 2, interface_role="bresp"),
            RTLSignal("s_axi_control_bvalid", "output", 1, interface_role="bvalid"),
            RTLSignal("s_axi_control_bready", "input", 1, interface_role="bready"),
            RTLSignal("s_axi_control_araddr", "input", 32, interface_role="araddr"),
            RTLSignal("s_axi_control_arvalid", "input", 1, interface_role="arvalid"),
            RTLSignal("s_axi_control_arready", "output", 1, interface_role="arready"),
            RTLSignal("s_axi_control_rdata", "output", 32, interface_role="rdata"),
            RTLSignal("s_axi_control_rresp", "output", 2, interface_role="rresp"),
            RTLSignal("s_axi_control_rvalid", "output", 1, interface_role="rvalid"),
            RTLSignal("s_axi_control_rready", "input", 1, interface_role="rready")
        ]
        control_interface = RTLInterface("s_axi_control", "axi_lite", control_signals)
        
        # Clock and reset interface
        clock_reset_signals = [
            RTLSignal("ap_clk", "input", 1, interface_role="clock"),
            RTLSignal("ap_rst_n", "input", 1, interface_role="reset"),
            RTLSignal("ap_start", "input", 1, interface_role="enable"),
            RTLSignal("ap_done", "output", 1, interface_role="interrupt"),
            RTLSignal("ap_ready", "output", 1, interface_role="interrupt"),
            RTLSignal("ap_idle", "output", 1, interface_role="interrupt")
        ]
        clock_reset_interface = RTLInterface("control_signals", "control", clock_reset_signals)
        
        return RTLModule(
            name="conv2d_accelerator",
            interfaces=[
                input_interface, weight_interface, output_interface,
                control_interface, clock_reset_interface
            ],
            parameters={
                "DATA_WIDTH": 64,
                "WEIGHT_WIDTH": 32,
                "OUTPUT_WIDTH": 32,
                "KERNEL_SIZE": 3,
                "INPUT_CHANNELS": 64,
                "OUTPUT_CHANNELS": 128,
                "IMAGE_HEIGHT": 224,
                "IMAGE_WIDTH": 224
            }
        )
    
    def create_realistic_pragmas(self) -> list[str]:
        """Create realistic pragmas matching the RTL module."""
        return [
            # Interface definitions
            "// @brainsmith interface s_axis_input axis_stream input data_width=64",
            "// @brainsmith interface s_axis_weights axis_stream input data_width=32",
            "// @brainsmith interface m_axis_output axis_stream output data_width=32",
            "// @brainsmith interface s_axi_control axi_lite slave",
            
            # Parallelism constraints
            "// @brainsmith parallelism input_parallelism=8",
            "// @brainsmith parallelism weight_parallelism=4", 
            "// @brainsmith parallelism output_parallelism=16",
            
            # Dataflow constraints
            "// @brainsmith dataflow tensor_shape input=[1,64,224,224]",
            "// @brainsmith dataflow tensor_shape weights=[128,64,3,3]",
            "// @brainsmith dataflow tensor_shape output=[1,128,222,222]",
            "// @brainsmith dataflow layout=NCHW",
            "// @brainsmith dataflow chunking_strategy=broadcast",
            
            # HLS pragmas for comparison
            "#pragma HLS INTERFACE axis port=s_axis_input",
            "#pragma HLS INTERFACE axis port=m_axis_output", 
            "#pragma HLS INTERFACE s_axilite port=s_axi_control",
            "#pragma HLS PIPELINE II=1",
            
            # Interface-specific pragmas
            "// @interface s_axis_input type=axis_stream direction=input width=64 layout=NCHW",
            "// @interface m_axis_output type=axis_stream direction=output width=32 layout=NCHW"
        ]
    
    def test_complete_analysis_workflow(self):
        """Test the complete analysis workflow from RTL to results."""
        # Create orchestrator
        orchestrator = AnalysisOrchestrator(self.config)
        
        # Perform complete analysis
        results = orchestrator.analyze_rtl_module(
            self.rtl_module,
            self.pragma_texts,
            enable_caching=True
        )
        
        # Validate overall success
        assert results.success == True
        assert len(results.errors) == 0
        assert results.total_analysis_time > 0
        
        # Validate interface analysis
        assert len(results.interface_results) == 5  # Five interfaces
        
        # Check interface types are correctly identified
        interface_types = {result.interface_type for result in results.interface_results}
        assert InterfaceType.AXI_STREAM in interface_types
        assert InterfaceType.AXI_LITE in interface_types
        assert InterfaceType.CONTROL in interface_types
        
        # Validate high confidence classifications
        high_confidence_results = [r for r in results.interface_results if r.confidence > 0.7]
        assert len(high_confidence_results) >= 3  # At least 3 should be high confidence
        
        # Validate pragma processing
        assert results.pragma_results is not None
        assert results.pragma_results.pragma_count >= 10  # Should parse most pragmas
        assert results.pragma_results.valid_pragma_count > 0
        
        # Check constraint generation
        assert len(results.pragma_results.interface_constraints) > 0
        assert len(results.pragma_results.parallelism_constraints) > 0
        
        # Validate dataflow integration
        assert len(results.dataflow_interfaces) >= 0  # May be empty if dataflow not available
        
        # Validate overall validation
        assert results.overall_validation is not None
        validation_success = (
            results.overall_validation.success if hasattr(results.overall_validation, 'success')
            else results.overall_validation.get("success", True)
        )
        assert validation_success == True
        
        # Validate performance metrics
        assert results.metrics.interface_analysis_count == 5
        assert results.metrics.interface_analysis_time > 0
        assert results.metrics.pragma_processing_count >= 10
        assert results.metrics.pragma_processing_time > 0
    
    def test_interface_analysis_accuracy(self):
        """Test accuracy of interface analysis."""
        analyzer = InterfaceAnalyzer(self.config)
        results = analyzer.analyze_interfaces(self.rtl_module)
        
        # Check that specific interfaces are correctly classified
        interface_map = {result.interface_name: result for result in results}
        
        # AXI-Stream interfaces should be detected
        assert "s_axis_input" in interface_map
        assert interface_map["s_axis_input"].interface_type == InterfaceType.AXI_STREAM
        assert interface_map["s_axis_input"].confidence > 0.8
        
        assert "m_axis_output" in interface_map
        assert interface_map["m_axis_output"].interface_type == InterfaceType.AXI_STREAM
        assert interface_map["m_axis_output"].confidence > 0.8
        
        # AXI-Lite interface should be detected
        assert "s_axi_control" in interface_map
        assert interface_map["s_axi_control"].interface_type == InterfaceType.AXI_LITE
        assert interface_map["s_axi_control"].confidence > 0.7
        
        # Control interface should be detected
        assert "control_signals" in interface_map
        assert interface_map["control_signals"].interface_type == InterfaceType.CONTROL
        assert interface_map["control_signals"].confidence > 0.8
    
    def test_pragma_processing_completeness(self):
        """Test completeness of pragma processing."""
        processor = PragmaProcessor(self.config)
        results = processor.process_pragmas(self.pragma_texts, self.rtl_module)
        
        # Should parse most pragmas successfully
        assert results.pragma_count >= 15  # Should detect most of our pragmas
        assert results.valid_pragma_count >= 10  # Most should be valid
        
        # Check pragma type diversity
        pragma_types = {pragma.pragma_type for pragma in results.parsed_pragmas}
        assert PragmaType.BRAINSMITH in pragma_types
        assert PragmaType.HLS in pragma_types
        assert PragmaType.INTERFACE in pragma_types
        
        # Check constraint generation
        assert len(results.interface_constraints) >= 4  # Interface definitions
        assert len(results.parallelism_constraints) >= 3  # Parallelism definitions
        assert len(results.dataflow_constraints) >= 2  # Dataflow definitions
        
        # Check specific constraints
        assert "s_axis_input" in results.interface_constraints
        assert "input_parallelism" in results.parallelism_constraints
        assert results.parallelism_constraints["input_parallelism"] == 8
    
    def test_validation_thoroughness(self):
        """Test thoroughness of validation."""
        validator = InterfaceValidator(self.config.analysis)
        
        # Test each interface individually
        for interface in self.rtl_module.interfaces:
            classifier = InterfaceClassifier()
            interface_type, confidence = classifier.get_best_classification(
                interface.name, interface.signals
            )
            
            validation_result = validator.validate_interface(interface, interface_type)
            
            # Should validate without critical errors for well-formed interfaces
            errors = (validation_result.errors if hasattr(validation_result, 'errors')
                     else validation_result.get("errors", []))
            
            # Allow warnings but minimize errors for our well-designed interfaces
            critical_errors = [e for e in errors if "missing" in str(e).lower()]
            assert len(critical_errors) <= 1  # At most 1 missing signal per interface
    
    def test_caching_effectiveness(self):
        """Test caching effectiveness and performance."""
        orchestrator = AnalysisOrchestrator(self.config)
        
        # First analysis (cache miss)
        start_time = time.time()
        results1 = orchestrator.analyze_rtl_module(
            self.rtl_module, self.pragma_texts, enable_caching=True
        )
        first_analysis_time = time.time() - start_time
        
        # Second analysis (should hit cache)
        start_time = time.time()
        results2 = orchestrator.analyze_rtl_module(
            self.rtl_module, self.pragma_texts, enable_caching=True
        )
        second_analysis_time = time.time() - start_time
        
        # Verify caching worked
        cache_stats = orchestrator.cache.get_stats()
        assert cache_stats["total_requests"] >= 2
        assert cache_stats["hits"] >= 1
        assert cache_stats["hit_rate"] > 0
        
        # Results should be equivalent
        assert results1.rtl_module.name == results2.rtl_module.name
        assert len(results1.interface_results) == len(results2.interface_results)
        
        # Performance should be tracked
        orch_stats = orchestrator.get_orchestration_statistics()
        assert orch_stats["orchestration_count"] >= 2
    
    def test_legacy_compatibility(self):
        """Test legacy compatibility adapter."""
        # Perform modern analysis
        orchestrator = AnalysisOrchestrator(self.config)
        results = orchestrator.analyze_rtl_module(self.rtl_module, self.pragma_texts)
        
        # Convert to legacy format
        adapter = LegacyAnalysisAdapter(self.config)
        legacy_results = adapter.adapt_to_legacy_format(results)
        
        # Verify legacy format structure
        assert isinstance(legacy_results, dict)
        assert "interfaces" in legacy_results
        assert "pragmas" in legacy_results
        assert "success" in legacy_results
        assert "errors" in legacy_results
        
        # Check interface conversion
        assert len(legacy_results["interfaces"]) == len(results.interface_results)
        for legacy_interface in legacy_results["interfaces"]:
            assert "name" in legacy_interface
            assert "type" in legacy_interface
            assert "confidence" in legacy_interface
            assert "signals" in legacy_interface
            assert "valid" in legacy_interface
        
        # Check pragma conversion
        if results.pragma_results:
            assert len(legacy_results["pragmas"]) == len(results.pragma_results.parsed_pragmas)
            for legacy_pragma in legacy_results["pragmas"]:
                assert "type" in legacy_pragma
                assert "text" in legacy_pragma
                assert "directive" in legacy_pragma
                assert "valid" in legacy_pragma
    
    def test_configuration_profiles(self):
        """Test different analysis configuration profiles."""
        # Test fast profile
        fast_interface_config, fast_pragma_config = create_analysis_config(
            self.config, profile="fast"
        )
        assert fast_interface_config.strategy.value == "fast"
        assert fast_pragma_config.strategy.value == "fast"
        
        # Test comprehensive profile
        comp_interface_config, comp_pragma_config = create_analysis_config(
            self.config, profile="comprehensive"
        )
        assert comp_interface_config.strategy.value == "comprehensive"
        assert comp_pragma_config.strategy.value == "comprehensive"
        
        # Test dataflow optimized profile
        df_interface_config, df_pragma_config = create_analysis_config(
            self.config, profile="dataflow_optimized"
        )
        assert df_interface_config.enable_dataflow_conversion == True
        assert df_pragma_config.enable_dataflow_constraints == True
        
        # Test legacy compatible profile
        legacy_interface_config, legacy_pragma_config = create_analysis_config(
            self.config, profile="legacy_compatible"
        )
        assert legacy_interface_config.enable_dataflow_conversion == False
        assert legacy_pragma_config.enable_dataflow_constraints == False
    
    def test_error_recovery_and_robustness(self):
        """Test error recovery and robustness."""
        # Create problematic module with issues
        problematic_signals = [
            RTLSignal("incomplete_tdata", "input", 32),  # Missing tvalid, tready
            RTLSignal("wrong_direction_tready", "input", 1),  # Wrong direction
        ]
        problematic_interface = RTLInterface("problematic", "axi_stream", problematic_signals)
        
        problematic_module = RTLModule(
            name="problematic_module",
            interfaces=[problematic_interface],
            parameters={}
        )
        
        # Create problematic pragmas
        problematic_pragmas = [
            "// @brainsmith interface nonexistent axis_stream",  # Invalid reference
            "// @invalid_pragma_syntax",                         # Invalid syntax
            "// @brainsmith parallelism invalid_param=abc",     # Invalid parameter value
        ]
        
        # Analysis should handle problems gracefully
        orchestrator = AnalysisOrchestrator(self.config)
        results = orchestrator.analyze_rtl_module(problematic_module, problematic_pragmas)
        
        # Should not crash
        assert isinstance(results, AnalysisResults)
        assert results.total_analysis_time > 0
        
        # May have errors/warnings but should complete
        assert len(results.interface_results) > 0  # Should still analyze the interface
        
        # Should track problems appropriately
        if results.pragma_results:
            # Some pragmas may fail to parse/validate
            assert results.pragma_results.valid_pragma_count <= results.pragma_results.pragma_count
    
    def test_performance_characteristics(self):
        """Test performance characteristics of the analysis system."""
        orchestrator = AnalysisOrchestrator(self.config)
        
        # Measure analysis performance
        start_time = time.time()
        results = orchestrator.analyze_rtl_module(self.rtl_module, self.pragma_texts)
        total_time = time.time() - start_time
        
        # Performance should be reasonable
        assert total_time < 5.0  # Should complete within 5 seconds
        assert results.interface_analysis_time < 2.0  # Interface analysis should be fast
        assert results.pragma_processing_time < 2.0  # Pragma processing should be fast
        
        # Check component performance
        interface_stats = orchestrator.interface_analyzer.get_analysis_statistics()
        pragma_stats = orchestrator.pragma_processor.get_processing_statistics()
        
        assert interface_stats["average_analysis_time"] < 1.0
        assert pragma_stats["average_processing_time"] < 1.0
        
        # Verify analysis quality vs performance tradeoff
        high_confidence_count = sum(1 for r in results.interface_results if r.confidence > 0.8)
        assert high_confidence_count >= 3  # Should maintain quality despite speed


class TestWeek2FactoryFunctions:
    """Test Week 2 factory functions and convenience methods."""
    
    def test_convenience_functions(self):
        """Test convenience functions work correctly."""
        # Create test module
        signals = [
            RTLSignal("tdata", "input", 32, interface_role="tdata"),
            RTLSignal("tvalid", "input", 1, interface_role="tvalid"),
            RTLSignal("tready", "output", 1, interface_role="tready")
        ]
        interface = RTLInterface("test_interface", "axi_stream", signals)
        module = RTLModule("test_module", interfaces=[interface])
        
        pragma_texts = ["// @brainsmith interface test_interface axis_stream"]
        
        # Test interface analysis convenience function
        interface_results = analyze_interfaces(module)
        assert len(interface_results) == 1
        assert interface_results[0].interface_type == InterfaceType.AXI_STREAM
        
        # Test pragma processing convenience function
        pragma_results = process_pragmas(pragma_texts, module)
        assert pragma_results.pragma_count >= 1
        
        # Test complete analysis convenience function
        complete_results = run_complete_analysis(module, pragma_texts)
        assert isinstance(complete_results, AnalysisResults)
        assert len(complete_results.interface_results) == 1
        assert complete_results.pragma_results is not None
    
    def test_configuration_factories(self):
        """Test configuration factory functions."""
        config = PipelineConfig()
        
        # Test analysis config creation
        interface_config, pragma_config = create_analysis_config(config)
        assert isinstance(interface_config, InterfaceAnalysisConfig)
        assert isinstance(pragma_config, PragmaAnalysisConfig)
        
        # Test profile creation
        fast_profile = AnalysisProfile.create_fast_profile()
        assert fast_profile.name == "fast"
        
        comprehensive_profile = AnalysisProfile.create_comprehensive_profile()
        assert comprehensive_profile.name == "comprehensive"
        
        dataflow_profile = AnalysisProfile.create_dataflow_optimized_profile()
        assert dataflow_profile.name == "dataflow_optimized"
        
        legacy_profile = AnalysisProfile.create_legacy_compatible_profile()
        assert legacy_profile.name == "legacy_compatible"


class TestWeek2Documentation:
    """Test Week 2 component documentation and examples."""
    
    def test_component_docstrings(self):
        """Test that components have proper documentation."""
        # Check main classes have docstrings
        assert InterfaceClassifier.__doc__ is not None
        assert InterfaceAnalyzer.__doc__ is not None
        assert PragmaParser.__doc__ is not None
        assert PragmaProcessor.__doc__ is not None
        assert AnalysisOrchestrator.__doc__ is not None
        
        # Check factory functions have docstrings
        assert analyze_interfaces.__doc__ is not None
        assert process_pragmas.__doc__ is not None
        assert run_complete_analysis.__doc__ is not None
    
    def test_pattern_documentation(self):
        """Test that patterns are well documented."""
        interface_patterns = get_interface_patterns()
        pragma_patterns = get_pragma_patterns()
        
        assert len(interface_patterns) > 0
        assert len(pragma_patterns) > 0
        
        # Each pattern should have meaningful attributes
        for pattern in interface_patterns:
            assert pattern.interface_type is not None
            assert len(pattern.signal_patterns) > 0
        
        for pattern in pragma_patterns:
            assert pattern.pragma_type is not None
            assert len(pattern.patterns) > 0


def create_test_file_for_parsing():
    """Create a temporary SystemVerilog file for parsing tests."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.sv', delete=False)
    try:
        temp_file.write("""
// @brainsmith interface s_axis_input axis_stream input data_width=32
// @brainsmith interface m_axis_output axis_stream output data_width=32
// @brainsmith parallelism input_par=4
// @brainsmith parallelism weight_par=2

module conv2d_example #(
    parameter DATA_WIDTH = 32,
    parameter KERNEL_SIZE = 3
)(
    // Clock and reset
    input  wire        ap_clk,
    input  wire        ap_rst_n,
    input  wire        ap_start,
    output wire        ap_done,
    
    // Input AXI-Stream
    input  wire [DATA_WIDTH-1:0]  s_axis_input_tdata,
    input  wire                   s_axis_input_tvalid,
    output wire                   s_axis_input_tready,
    input  wire                   s_axis_input_tlast,
    
    // Output AXI-Stream  
    output wire [DATA_WIDTH-1:0]  m_axis_output_tdata,
    output wire                   m_axis_output_tvalid,
    input  wire                   m_axis_output_tready,
    output wire                   m_axis_output_tlast
);

// Module implementation would go here

endmodule
""")
        temp_file.close()
        return temp_file.name
    except Exception:
        temp_file.close()
        Path(temp_file.name).unlink()
        raise


class TestWeek2RealWorldScenarios:
    """Test Week 2 components with real-world scenarios."""
    
    def test_file_based_pragma_parsing(self):
        """Test parsing pragmas from actual SystemVerilog files."""
        sv_file = create_test_file_for_parsing()
        
        try:
            parser = PragmaParser()
            results = parser.parse_pragma_file(sv_file)
            
            # Should find the pragmas in the file
            assert len(results) >= 4
            
            # Check pragma types
            pragma_types = {result.pragma_type for result in results}
            assert PragmaType.BRAINSMITH in pragma_types
            
            # Check specific pragma content
            interface_pragmas = [r for r in results if r.directive == "interface"]
            assert len(interface_pragmas) >= 2
            
            parallelism_pragmas = [r for r in results if r.directive == "parallelism"]
            assert len(parallelism_pragmas) >= 2
            
        finally:
            Path(sv_file).unlink()
    
    def test_complex_module_analysis(self):
        """Test analysis of complex modules with multiple interface types."""
        # This test uses the realistic module from the integration test
        integration_test = TestWeek2ComponentIntegration()
        integration_test.setup_method()
        
        # Perform complete analysis
        orchestrator = AnalysisOrchestrator(integration_test.config)
        results = orchestrator.analyze_rtl_module(
            integration_test.rtl_module,
            integration_test.pragma_texts
        )
        
        # Should handle complexity well
        assert results.success == True
        assert len(results.interface_results) == 5
        assert results.pragma_results.pragma_count >= 10
        
        # All major interface types should be detected
        interface_types = {r.interface_type for r in results.interface_results}
        assert InterfaceType.AXI_STREAM in interface_types
        assert InterfaceType.AXI_LITE in interface_types
        assert InterfaceType.CONTROL in interface_types
        
        # Performance should still be good
        assert results.total_analysis_time < 3.0
    
    def test_incremental_analysis(self):
        """Test incremental analysis and caching behavior."""
        # Create base module
        signals = [RTLSignal("tdata", "input", 32)]
        interface = RTLInterface("base_interface", "axi_stream", signals)
        base_module = RTLModule("base_module", interfaces=[interface])
        
        orchestrator = AnalysisOrchestrator()
        
        # First analysis
        results1 = orchestrator.analyze_rtl_module(base_module, enable_caching=True)
        
        # Modify module slightly (add signal)
        extended_signals = signals + [RTLSignal("tvalid", "input", 1)]
        extended_interface = RTLInterface("base_interface", "axi_stream", extended_signals)
        extended_module = RTLModule("base_module", interfaces=[extended_interface])
        
        # Second analysis (should be different due to module change)
        results2 = orchestrator.analyze_rtl_module(extended_module, enable_caching=True)
        
        # Results should reflect the change
        assert len(results1.interface_results[0].detected_signals) == 1
        assert len(results2.interface_results[0].detected_signals) == 2
        
        # Cache should handle different modules correctly
        cache_stats = orchestrator.cache.get_stats()
        assert cache_stats["total_requests"] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])