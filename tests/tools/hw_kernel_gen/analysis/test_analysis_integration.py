"""
Test suite for Analysis Integration Layer.

Tests the orchestrated analysis workflow, caching, and legacy compatibility.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from brainsmith.tools.hw_kernel_gen.enhanced_data_structures import (
    RTLSignal, RTLInterface, RTLModule, ParsedRTLData
)
from brainsmith.tools.hw_kernel_gen.enhanced_config import PipelineConfig
from brainsmith.tools.hw_kernel_gen.analysis.analysis_integration import (
    AnalysisResults, AnalysisCache, LegacyAnalysisAdapter,
    AnalysisOrchestrator, create_analysis_orchestrator, run_complete_analysis
)
from brainsmith.tools.hw_kernel_gen.analysis.enhanced_interface_analyzer import (
    InterfaceAnalysisResult, InterfaceAnalyzer
)
from brainsmith.tools.hw_kernel_gen.analysis.enhanced_pragma_processor import (
    PragmaProcessingResult, ParsedPragma, PragmaProcessor
)
from brainsmith.tools.hw_kernel_gen.analysis.analysis_patterns import (
    InterfaceType, PragmaType
)


class TestAnalysisResults:
    """Test analysis results container."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test module
        signals = [
            RTLSignal("tdata", "input", 32, interface_role="tdata"),
            RTLSignal("tvalid", "input", 1, interface_role="tvalid")
        ]
        interface = RTLInterface("test_interface", "axi_stream", signals)
        self.test_module = RTLModule("test_module", interfaces=[interface])
        
        self.results = AnalysisResults(rtl_module=self.test_module)
    
    def test_results_initialization(self):
        """Test results container initialization."""
        assert self.results.rtl_module == self.test_module
        assert len(self.results.interface_results) == 0
        assert self.results.pragma_results is None
        assert self.results.success == False
        assert self.results.analysis_start_time > 0
    
    def test_add_error_and_warning(self):
        """Test adding errors and warnings."""
        self.results.add_error("Test error")
        self.results.add_warning("Test warning")
        
        assert len(self.results.errors) == 1
        assert len(self.results.warnings) == 1
        assert self.results.metrics.error_count == 1
        assert self.results.metrics.warning_count == 1
    
    def test_interface_lookup(self):
        """Test interface result lookup."""
        # Add test interface result
        interface_result = InterfaceAnalysisResult(
            interface_name="test_interface",
            interface_type=InterfaceType.AXI_STREAM,
            confidence=0.8
        )
        self.results.interface_results.append(interface_result)
        
        # Test lookup by name
        found = self.results.get_interface_by_name("test_interface")
        assert found is not None
        assert found.interface_name == "test_interface"
        
        # Test lookup by type
        axi_interfaces = self.results.get_interfaces_by_type(InterfaceType.AXI_STREAM)
        assert len(axi_interfaces) == 1
        assert axi_interfaces[0].interface_name == "test_interface"
    
    def test_finalization(self):
        """Test results finalization."""
        # Add some test data
        interface_result = InterfaceAnalysisResult(
            interface_name="test_interface",
            interface_type=InterfaceType.AXI_STREAM,
            confidence=0.8
        )
        self.results.interface_results.append(interface_result)
        self.results.interface_analysis_time = 0.1
        
        # Finalize
        self.results.finalize()
        
        assert self.results.total_analysis_time > 0
        assert self.results.success == True  # No errors, has interfaces
        assert self.results.metrics.interface_analysis_count == 1
        assert self.results.metrics.interface_analysis_time == 0.1
    
    def test_serialization(self):
        """Test results serialization."""
        # Add test data
        interface_result = InterfaceAnalysisResult(
            interface_name="test_interface",
            interface_type=InterfaceType.AXI_STREAM,
            confidence=0.8
        )
        self.results.interface_results.append(interface_result)
        
        pragma_result = PragmaProcessingResult()
        pragma_result.pragma_count = 2
        pragma_result.valid_pragma_count = 2
        self.results.pragma_results = pragma_result
        
        self.results.finalize()
        
        # Serialize
        result_dict = self.results.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["module_name"] == "test_module"
        assert result_dict["success"] == True
        assert result_dict["interface_analysis"]["count"] == 1
        assert result_dict["pragma_processing"]["enabled"] == True


class TestAnalysisCache:
    """Test analysis caching functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = AnalysisCache(max_size=5, ttl=1.0)  # Small cache for testing
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        assert self.cache.max_size == 5
        assert self.cache.ttl == 1.0
        assert self.cache._hits == 0
        assert self.cache._misses == 0
    
    def test_interface_result_caching(self):
        """Test interface result caching."""
        # Create test result
        result = InterfaceAnalysisResult(
            interface_name="test",
            interface_type=InterfaceType.AXI_STREAM,
            confidence=0.8
        )
        
        cache_key = "test_key"
        
        # Should be cache miss initially
        cached = self.cache.get_interface_result(cache_key)
        assert cached is None
        assert self.cache._misses == 1
        
        # Put in cache
        self.cache.put_interface_result(cache_key, result)
        
        # Should be cache hit now
        cached = self.cache.get_interface_result(cache_key)
        assert cached is not None
        assert cached.interface_name == "test"
        assert self.cache._hits == 1
    
    def test_pragma_result_caching(self):
        """Test pragma result caching."""
        # Create test result
        result = PragmaProcessingResult()
        result.pragma_count = 3
        result.valid_pragma_count = 2
        
        cache_key = "pragma_key"
        
        # Cache miss
        cached = self.cache.get_pragma_result(cache_key)
        assert cached is None
        
        # Put and get
        self.cache.put_pragma_result(cache_key, result)
        cached = self.cache.get_pragma_result(cache_key)
        
        assert cached is not None
        assert cached.pragma_count == 3
    
    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        result = InterfaceAnalysisResult(
            interface_name="test",
            interface_type=InterfaceType.AXI_STREAM,
            confidence=0.8
        )
        
        cache_key = "expiring_key"
        
        # Put in cache
        self.cache.put_interface_result(cache_key, result)
        
        # Should be available immediately
        cached = self.cache.get_interface_result(cache_key)
        assert cached is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired now
        cached = self.cache.get_interface_result(cache_key)
        assert cached is None
    
    def test_cache_size_limit(self):
        """Test cache size limitation."""
        # Fill cache beyond limit
        for i in range(10):
            result = InterfaceAnalysisResult(
                interface_name=f"test_{i}",
                interface_type=InterfaceType.AXI_STREAM,
                confidence=0.8
            )
            self.cache.put_interface_result(f"key_{i}", result)
        
        # Cache should not exceed max size significantly
        stats = self.cache.get_stats()
        assert stats["cache_sizes"]["interface"] <= self.cache.max_size
    
    def test_cache_statistics(self):
        """Test cache statistics."""
        # Perform some cache operations
        result = InterfaceAnalysisResult(
            interface_name="test",
            interface_type=InterfaceType.AXI_STREAM,
            confidence=0.8
        )
        
        # Miss, put, hit
        self.cache.get_interface_result("key")
        self.cache.put_interface_result("key", result)
        self.cache.get_interface_result("key")
        
        stats = self.cache.get_stats()
        
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["total_requests"] == 2
        assert stats["hit_rate"] == 0.5
    
    def test_cache_clearing(self):
        """Test cache clearing."""
        # Add some items
        result = InterfaceAnalysisResult(
            interface_name="test",
            interface_type=InterfaceType.AXI_STREAM,
            confidence=0.8
        )
        self.cache.put_interface_result("key", result)
        
        # Clear cache
        self.cache.clear()
        
        # Should be empty
        stats = self.cache.get_stats()
        assert stats["total_size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0


class TestLegacyAnalysisAdapter:
    """Test legacy compatibility adapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PipelineConfig()
        self.adapter = LegacyAnalysisAdapter(self.config)
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        assert self.adapter.config is not None
    
    def test_legacy_interface_adaptation(self):
        """Test adaptation of legacy interface data."""
        legacy_interfaces = [
            {
                "name": "input0",
                "type": "axi_stream",
                "signals": [
                    {"name": "tdata", "direction": "input", "width": 32, "role": "tdata"},
                    {"name": "tvalid", "direction": "input", "width": 1, "role": "tvalid"}
                ]
            }
        ]
        
        rtl_interfaces = self.adapter.adapt_legacy_interface_data(legacy_interfaces)
        
        assert len(rtl_interfaces) == 1
        assert rtl_interfaces[0].name == "input0"
        assert rtl_interfaces[0].interface_type == "axi_stream"
        assert len(rtl_interfaces[0].signals) == 2
    
    def test_legacy_pragma_adaptation(self):
        """Test adaptation of legacy pragma data."""
        legacy_pragmas = [
            "// @brainsmith interface input0 axis_stream",
            "// @brainsmith parallelism input_par=4"
        ]
        
        parsed_pragmas = self.adapter.adapt_legacy_pragma_data(legacy_pragmas)
        
        assert len(parsed_pragmas) >= 1  # At least some should parse
        assert all(isinstance(p, ParsedPragma) for p in parsed_pragmas)
    
    def test_legacy_format_conversion(self):
        """Test conversion to legacy format."""
        # Create analysis results
        signals = [RTLSignal("tdata", "input", 32)]
        interface = RTLInterface("test_interface", "axi_stream", signals)
        module = RTLModule("test_module", interfaces=[interface])
        
        results = AnalysisResults(rtl_module=module)
        
        # Add interface result
        interface_result = InterfaceAnalysisResult(
            interface_name="test_interface",
            interface_type=InterfaceType.AXI_STREAM,
            confidence=0.8,
            detected_signals=signals,
            is_valid=True
        )
        results.interface_results.append(interface_result)
        
        # Add pragma result
        pragma_result = PragmaProcessingResult()
        pragma = ParsedPragma(
            pragma_type=PragmaType.BRAINSMITH,
            raw_text="// @brainsmith interface test axis",
            directive="interface",
            is_valid=True
        )
        pragma_result.parsed_pragmas.append(pragma)
        results.pragma_results = pragma_result
        
        results.success = True
        
        # Convert to legacy format
        legacy_result = self.adapter.adapt_to_legacy_format(results)
        
        assert isinstance(legacy_result, dict)
        assert "interfaces" in legacy_result
        assert "pragmas" in legacy_result
        assert legacy_result["success"] == True
        
        # Check interface conversion
        assert len(legacy_result["interfaces"]) == 1
        legacy_interface = legacy_result["interfaces"][0]
        assert legacy_interface["name"] == "test_interface"
        assert legacy_interface["type"] == "axi_stream"
        assert legacy_interface["confidence"] == 0.8
        
        # Check pragma conversion
        assert len(legacy_result["pragmas"]) == 1
        legacy_pragma = legacy_result["pragmas"][0]
        assert legacy_pragma["type"] == "brainsmith"
        assert legacy_pragma["directive"] == "interface"


class TestAnalysisOrchestrator:
    """Test analysis orchestration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PipelineConfig()
        self.orchestrator = AnalysisOrchestrator(self.config)
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator.config is not None
        assert self.orchestrator.interface_analyzer is not None
        assert self.orchestrator.pragma_processor is not None
        assert self.orchestrator.cache is not None
        assert self.orchestrator.legacy_adapter is not None
    
    def create_comprehensive_test_module(self) -> RTLModule:
        """Create a comprehensive test module."""
        # AXI-Stream input
        axis_signals = [
            RTLSignal("s_axis_tdata", "input", 32, interface_role="tdata"),
            RTLSignal("s_axis_tvalid", "input", 1, interface_role="tvalid"),
            RTLSignal("s_axis_tready", "output", 1, interface_role="tready")
        ]
        axis_interface = RTLInterface("s_axis_input", "axi_stream", axis_signals)
        
        # Control interface
        control_signals = [
            RTLSignal("ap_clk", "input", 1, interface_role="clock"),
            RTLSignal("ap_rst_n", "input", 1, interface_role="reset")
        ]
        control_interface = RTLInterface("control", "control", control_signals)
        
        return RTLModule(
            name="comprehensive_module",
            interfaces=[axis_interface, control_interface],
            parameters={"WIDTH": 32, "DEPTH": 64}
        )
    
    def test_rtl_module_analysis(self):
        """Test complete RTL module analysis."""
        module = self.create_comprehensive_test_module()
        
        results = self.orchestrator.analyze_rtl_module(module)
        
        assert isinstance(results, AnalysisResults)
        assert results.rtl_module == module
        assert len(results.interface_results) == 2  # Two interfaces
        assert results.total_analysis_time > 0
        assert results.interface_analysis_time > 0
    
    def test_analysis_with_pragmas(self):
        """Test analysis with pragma processing."""
        module = self.create_comprehensive_test_module()
        
        pragma_texts = [
            "// @brainsmith interface s_axis_input axis_stream",
            "// @brainsmith parallelism input_par=4"
        ]
        
        results = self.orchestrator.analyze_rtl_module(module, pragma_texts)
        
        assert results.pragma_results is not None
        assert results.pragma_results.pragma_count >= 1
        assert results.pragma_processing_time > 0
    
    def test_analysis_caching(self):
        """Test analysis result caching."""
        module = self.create_comprehensive_test_module()
        
        # First analysis
        results1 = self.orchestrator.analyze_rtl_module(module, enable_caching=True)
        
        # Second analysis (should use cache)
        results2 = self.orchestrator.analyze_rtl_module(module, enable_caching=True)
        
        # Results should be equivalent (cached)
        assert results1.rtl_module.name == results2.rtl_module.name
        assert len(results1.interface_results) == len(results2.interface_results)
        
        # Check cache statistics
        cache_stats = self.orchestrator.cache.get_stats()
        assert cache_stats["hits"] > 0 or cache_stats["total_requests"] > 0
    
    def test_analysis_without_caching(self):
        """Test analysis without caching."""
        module = self.create_comprehensive_test_module()
        
        results = self.orchestrator.analyze_rtl_module(module, enable_caching=False)
        
        assert isinstance(results, AnalysisResults)
        # Should still work without caching
        assert len(results.interface_results) > 0
    
    def test_parsed_rtl_data_analysis(self):
        """Test analysis of parsed RTL data."""
        module1 = self.create_comprehensive_test_module()
        module2 = RTLModule("simple_module", interfaces=[], parameters={})
        
        parsed_rtl = ParsedRTLData(
            modules=[module1, module2],
            top_module="comprehensive_module"
        )
        
        results = self.orchestrator.analyze_parsed_rtl_data(parsed_rtl)
        
        assert isinstance(results, dict)
        assert len(results) == 2
        assert "comprehensive_module" in results
        assert "simple_module" in results
        
        # Check individual results
        comp_results = results["comprehensive_module"]
        assert len(comp_results.interface_results) == 2
        
        simple_results = results["simple_module"]
        assert len(simple_results.interface_results) == 0
    
    @patch('brainsmith.tools.hw_kernel_gen.analysis.analysis_integration.DATAFLOW_AVAILABLE', True)
    def test_dataflow_integration(self):
        """Test dataflow integration in orchestrated analysis."""
        # Enable dataflow mode
        self.orchestrator.config.dataflow.mode = self.orchestrator.config.dataflow.mode.__class__.DATAFLOW_ONLY
        
        module = self.create_comprehensive_test_module()
        
        # Mock dataflow components
        with patch('brainsmith.tools.hw_kernel_gen.analysis.analysis_integration.DataflowModel') as mock_model:
            mock_instance = Mock()
            mock_model.return_value = mock_instance
            
            results = self.orchestrator.analyze_rtl_module(module)
            
            # Should attempt dataflow integration
            assert results is not None
    
    def test_validation_integration(self):
        """Test validation integration in orchestrated analysis."""
        module = self.create_comprehensive_test_module()
        
        results = self.orchestrator.analyze_rtl_module(module)
        
        assert results.overall_validation is not None
        assert results.validation_time > 0
    
    def test_orchestration_statistics(self):
        """Test orchestration statistics tracking."""
        module = self.create_comprehensive_test_module()
        
        initial_stats = self.orchestrator.get_orchestration_statistics()
        initial_count = initial_stats["orchestration_count"]
        
        # Perform analysis
        self.orchestrator.analyze_rtl_module(module)
        
        final_stats = self.orchestrator.get_orchestration_statistics()
        assert final_stats["orchestration_count"] == initial_count + 1
        assert final_stats["total_orchestration_time"] > 0
        
        # Check sub-component statistics
        assert "interface_analyzer_stats" in final_stats
        assert "pragma_processor_stats" in final_stats
        assert "cache_stats" in final_stats
    
    def test_cache_management(self):
        """Test cache management operations."""
        module = self.create_comprehensive_test_module()
        
        # Perform analysis to populate cache
        self.orchestrator.analyze_rtl_module(module, enable_caching=True)
        
        # Check cache has content
        initial_stats = self.orchestrator.cache.get_stats()
        assert initial_stats["total_size"] > 0
        
        # Clear cache
        self.orchestrator.clear_cache()
        
        # Check cache is cleared
        final_stats = self.orchestrator.cache.get_stats()
        assert final_stats["total_size"] == 0
    
    def test_error_handling(self):
        """Test error handling in orchestrated analysis."""
        # Create problematic module
        empty_module = RTLModule("empty_module", interfaces=[], parameters={})
        
        # Analysis should handle empty modules gracefully
        results = self.orchestrator.analyze_rtl_module(empty_module)
        
        assert isinstance(results, AnalysisResults)
        assert len(results.interface_results) == 0
        # Should not fail completely
        assert results.total_analysis_time > 0


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_analysis_orchestrator(self):
        """Test orchestrator factory."""
        config = PipelineConfig()
        orchestrator = create_analysis_orchestrator(config)
        
        assert isinstance(orchestrator, AnalysisOrchestrator)
        assert orchestrator.config is config
    
    def test_run_complete_analysis_convenience(self):
        """Test convenience function for complete analysis."""
        # Create test module
        signals = [
            RTLSignal("tdata", "input", 32, interface_role="tdata"),
            RTLSignal("tvalid", "input", 1, interface_role="tvalid")
        ]
        interface = RTLInterface("test_interface", "axi_stream", signals)
        module = RTLModule("test_module", interfaces=[interface])
        
        pragma_texts = ["// @brainsmith interface test_interface axis_stream"]
        
        # Use convenience function
        results = run_complete_analysis(module, pragma_texts)
        
        assert isinstance(results, AnalysisResults)
        assert len(results.interface_results) == 1
        assert results.pragma_results is not None


class TestIntegration:
    """Integration tests for complete analysis workflow."""
    
    def test_comprehensive_analysis_workflow(self):
        """Test complete analysis workflow from RTL to results."""
        # Create comprehensive RTL module
        axis_input_signals = [
            RTLSignal("s_axis_tdata", "input", 32, interface_role="tdata"),
            RTLSignal("s_axis_tvalid", "input", 1, interface_role="tvalid"),
            RTLSignal("s_axis_tready", "output", 1, interface_role="tready")
        ]
        axis_input = RTLInterface("s_axis_input", "axi_stream", axis_input_signals)
        
        axis_output_signals = [
            RTLSignal("m_axis_tdata", "output", 32, interface_role="tdata"),
            RTLSignal("m_axis_tvalid", "output", 1, interface_role="tvalid"),
            RTLSignal("m_axis_tready", "input", 1, interface_role="tready")
        ]
        axis_output = RTLInterface("m_axis_output", "axi_stream", axis_output_signals)
        
        control_signals = [
            RTLSignal("ap_clk", "input", 1, interface_role="clock"),
            RTLSignal("ap_rst_n", "input", 1, interface_role="reset"),
            RTLSignal("ap_start", "input", 1, interface_role="enable"),
            RTLSignal("ap_done", "output", 1, interface_role="interrupt")
        ]
        control = RTLInterface("control", "control", control_signals)
        
        module = RTLModule(
            name="comprehensive_test_module",
            interfaces=[axis_input, axis_output, control],
            parameters={"WIDTH": 32, "DEPTH": 64, "PARALLELISM": 4}
        )
        
        # Create comprehensive pragma set
        pragma_texts = [
            "// @brainsmith interface s_axis_input axis_stream",
            "// @brainsmith interface m_axis_output axis_stream",
            "// @brainsmith parallelism input_par=4",
            "// @brainsmith parallelism weight_par=2",
            "// @interface control type=control direction=slave",
            "#pragma HLS PIPELINE II=1"
        ]
        
        # Create comprehensive configuration
        config = PipelineConfig()
        config.analysis.analyze_interfaces = True
        config.analysis.analyze_dataflow_interfaces = True
        config.analysis.validate_pragma_compatibility = True
        config.validation.validate_interface_constraints = True
        config.validation.validate_dataflow_model = True
        
        # Perform complete analysis
        orchestrator = AnalysisOrchestrator(config)
        results = orchestrator.analyze_rtl_module(module, pragma_texts, enable_caching=True)
        
        # Verify comprehensive results
        assert results.success == True
        assert len(results.interface_results) == 3
        assert results.pragma_results is not None
        assert results.pragma_results.pragma_count >= 4
        
        # Check interface classification
        interface_types = {result.interface_type for result in results.interface_results}
        assert InterfaceType.AXI_STREAM in interface_types
        assert InterfaceType.CONTROL in interface_types
        
        # Check pragma processing
        assert len(results.pragma_results.interface_constraints) > 0
        assert len(results.pragma_results.parallelism_constraints) > 0
        
        # Check validation
        assert results.overall_validation is not None
        
        # Check performance metrics
        assert results.total_analysis_time > 0
        assert results.interface_analysis_time > 0
        assert results.pragma_processing_time > 0
        assert results.validation_time > 0
        
        # Check metrics
        assert results.metrics.interface_analysis_count == 3
        assert results.metrics.pragma_processing_count >= 4
        
        # Verify serialization works
        result_dict = results.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["success"] == True
        assert result_dict["module_name"] == "comprehensive_test_module"
    
    def test_legacy_compatibility_workflow(self):
        """Test legacy compatibility in complete workflow."""
        # Create module
        signals = [RTLSignal("tdata", "input", 32)]
        interface = RTLInterface("test_interface", "axi_stream", signals)
        module = RTLModule("legacy_module", interfaces=[interface])
        
        # Create legacy configuration
        config = PipelineConfig()
        config.legacy_mode = True
        config.dataflow.mode = config.dataflow.mode.__class__.DISABLED
        
        orchestrator = AnalysisOrchestrator(config)
        results = orchestrator.analyze_rtl_module(module)
        
        # Should work in legacy mode
        assert isinstance(results, AnalysisResults)
        assert len(results.interface_results) > 0
        
        # Convert to legacy format
        legacy_results = orchestrator.legacy_adapter.adapt_to_legacy_format(results)
        assert isinstance(legacy_results, dict)
        assert "interfaces" in legacy_results
    
    def test_performance_and_caching(self):
        """Test performance optimization and caching."""
        # Create module
        signals = [RTLSignal("tdata", "input", 32)]
        interface = RTLInterface("perf_test", "axi_stream", signals)
        module = RTLModule("performance_module", interfaces=[interface])
        
        orchestrator = AnalysisOrchestrator()
        
        # First run (cache miss)
        start_time = time.time()
        results1 = orchestrator.analyze_rtl_module(module, enable_caching=True)
        first_run_time = time.time() - start_time
        
        # Second run (cache hit)
        start_time = time.time()
        results2 = orchestrator.analyze_rtl_module(module, enable_caching=True)
        second_run_time = time.time() - start_time
        
        # Verify caching effectiveness
        assert results1.rtl_module.name == results2.rtl_module.name
        assert len(results1.interface_results) == len(results2.interface_results)
        
        # Check cache statistics
        cache_stats = orchestrator.cache.get_stats()
        assert cache_stats["total_requests"] >= 2
        assert cache_stats["hits"] >= 1
        
        # Verify orchestration statistics
        orch_stats = orchestrator.get_orchestration_statistics()
        assert orch_stats["orchestration_count"] >= 2
        assert orch_stats["total_orchestration_time"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])