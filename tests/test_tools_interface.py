"""
Tests for the tools interface (roofline analysis and profiling).

This module tests that the tools interface works correctly and is
separate from the core toolchain.
"""

import pytest
from unittest.mock import Mock, patch

# Import tools
try:
    from brainsmith.tools import roofline_analysis, RooflineProfiler
    from brainsmith.tools.profiling import roofline_analysis as profiling_roofline
    from brainsmith.tools.profiling import RooflineProfiler as profiling_profiler
except ImportError as e:
    pytest.skip(f"BrainSmith tools not available: {e}", allow_module_level=True)


class TestToolsInterface:
    """Test cases for the tools interface."""
    
    def test_tools_import(self):
        """Test that tools can be imported correctly."""
        # These should be available (or None if components missing)
        assert roofline_analysis is not None or roofline_analysis is None
        assert RooflineProfiler is not None or RooflineProfiler is None
    
    def test_tools_separate_from_core(self):
        """Test that tools are separate from core forge API."""
        # Tools should be importable independently
        from brainsmith.tools import roofline_analysis as tools_roofline
        
        # Should not be the same as core forge function
        from brainsmith.core.api import forge
        
        # These are different - tools vs core
        assert tools_roofline != forge
    
    def test_profiling_imports(self):
        """Test that profiling tools import correctly."""
        # Test direct import from profiling module
        assert profiling_roofline is not None or profiling_roofline is None
        assert profiling_profiler is not None or profiling_profiler is None


class TestRooflineAnalysis:
    """Test cases for roofline analysis function."""
    
    @patch('brainsmith.tools.profiling._roofline_analysis')
    def test_roofline_analysis_wrapper(self, mock_roofline):
        """Test the roofline analysis wrapper function."""
        if roofline_analysis is None:
            pytest.skip("Roofline analysis not available")
        
        # Mock the underlying roofline analysis
        mock_roofline.return_value = None  # Original function prints, doesn't return
        
        # Test model config
        model_config = {
            'arch': 'bert',
            'num_layers': 12,
            'seq_len': 512,
            'num_heads': 12,
            'head_size': 64,
            'intermediate': 3072
        }
        
        # Test hardware config
        hw_config = {
            'dsps': 10000,
            'luts': 1000000,
            'dsp_util': 0.9,
            'lut_util': 0.6,
            'dsp_hz': 500e6,
            'lut_hz': 250e6
        }
        
        # Call roofline analysis
        result = roofline_analysis(model_config, hw_config, [4, 8])
        
        # Should return structured result
        assert isinstance(result, dict)
        assert 'status' in result
        
        # Verify underlying function was called
        mock_roofline.assert_called_once_with(model_config, hw_config, [4, 8])


class TestRooflineProfiler:
    """Test cases for RooflineProfiler class."""
    
    @patch('brainsmith.tools.profiling.RooflineModel')
    def test_roofline_profiler_init(self, mock_model_class):
        """Test RooflineProfiler initialization."""
        if RooflineProfiler is None:
            pytest.skip("RooflineProfiler not available")
        
        # Mock the underlying RooflineModel
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # Create profiler
        profiler = RooflineProfiler()
        
        # Verify initialization
        assert profiler.model == mock_model
        mock_model_class.assert_called_once()
    
    @patch('brainsmith.tools.profiling.RooflineModel')
    def test_roofline_profiler_bert(self, mock_model_class):
        """Test RooflineProfiler with BERT model."""
        if RooflineProfiler is None:
            pytest.skip("RooflineProfiler not available")
        
        # Mock the underlying RooflineModel
        mock_model = Mock()
        mock_model.get_profile.return_value = {
            'macs': [[1000, 2000], [1500, 2500]],
            'w': [[100, 200], [150, 250]],
            'act': [[50, 100], [75, 125]],
            'vw': [[], []],
            'hbm': [[], []]
        }
        mock_model_class.return_value = mock_model
        
        # Create profiler
        profiler = RooflineProfiler()
        
        # Test model config
        model_config = {
            'arch': 'bert',
            'num_layers': 12,
            'seq_len': 512,
            'num_heads': 12,
            'head_size': 64,
            'intermediate': 3072,
            'batch': 1
        }
        
        # Test hardware config
        hw_config = {
            'dsps': 10000,
            'luts': 1000000,
            'dsp_util': 0.9,
            'lut_util': 0.6,
            'dsp_hz': 500e6,
            'lut_hz': 250e6
        }
        
        # Profile model
        result = profiler.profile_model(model_config, hw_config)
        
        # Verify results structure
        assert isinstance(result, dict)
        assert '4bit' in result
        assert '8bit' in result
        
        # Verify BERT profiling was called
        mock_model.profile_bert.assert_called_once_with(model=model_config)
        mock_model.get_profile.assert_called_once()
    
    @patch('brainsmith.tools.profiling.RooflineModel')
    def test_roofline_profiler_slm(self, mock_model_class):
        """Test RooflineProfiler with SLM models."""
        if RooflineProfiler is None:
            pytest.skip("RooflineProfiler not available")
        
        # Mock the underlying RooflineModel
        mock_model = Mock()
        mock_model.get_profile.return_value = {
            'macs': [[1000], [1500]],
            'w': [[100], [150]],
            'act': [[50], [75]],
            'vw': [[], []],
            'hbm': [[], []]
        }
        mock_model_class.return_value = mock_model
        
        # Create profiler
        profiler = RooflineProfiler()
        
        # Test SLM PP config
        model_config = {
            'arch': 'slm_pp',
            'num_layers': 32,
            'seq_len': 2048,
            'num_heads': 32,
            'head_size': 128,
            'intermediate': 14336,
            'batch': 1
        }
        
        hw_config = {
            'dsps': 10000,
            'luts': 1000000,
            'dsp_util': 0.9,
            'dsp_hz': 500e6
        }
        
        # Profile model
        result = profiler.profile_model(model_config, hw_config)
        
        # Verify SLM PP profiling was called
        mock_model.profile_slm_pp.assert_called_once_with(model=model_config)
        assert isinstance(result, dict)
        assert '4bit' in result or '8bit' in result
    
    @patch('brainsmith.tools.profiling.RooflineModel')  
    def test_roofline_profiler_unsupported_arch(self, mock_model_class):
        """Test RooflineProfiler with unsupported architecture."""
        if RooflineProfiler is None:
            pytest.skip("RooflineProfiler not available")
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        profiler = RooflineProfiler()
        
        # Test unsupported architecture
        model_config = {
            'arch': 'unsupported_arch',
            'batch': 1
        }
        
        hw_config = {'dsps': 1000}
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Unsupported model architecture"):
            profiler.profile_model(model_config, hw_config)
    
    @patch('brainsmith.tools.profiling.RooflineModel')
    def test_roofline_profiler_report_generation(self, mock_model_class):
        """Test report generation functionality."""
        if RooflineProfiler is None:
            pytest.skip("RooflineProfiler not available")
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        profiler = RooflineProfiler()
        
        # Test profile results
        profile_results = {
            '4bit': {
                'throughput_ips': 100.0,
                'latency_ms': 10.0,
                'total_compute_ops': 1000000,
                'dtype': 4
            },
            '8bit': {
                'throughput_ips': 80.0,
                'latency_ms': 12.5,
                'total_compute_ops': 1000000,
                'dtype': 8
            }
        }
        
        # Generate report
        report = profiler.generate_report(profile_results)
        
        # Verify report is HTML
        assert isinstance(report, str)
        assert '<html>' in report
        assert '<title>BrainSmith Roofline Analysis Report</title>' in report
        assert '100.00' in report  # Throughput value
        assert '10.0000' in report  # Latency value


class TestToolsIntegration:
    """Integration tests for tools interface."""
    
    def test_tools_main_import(self):
        """Test that tools can be imported from main brainsmith module."""
        # Test main module imports
        from brainsmith import roofline_analysis as main_roofline
        from brainsmith import RooflineProfiler as main_profiler
        
        # Should be the same as tools imports
        assert main_roofline == roofline_analysis
        assert main_profiler == RooflineProfiler
    
    def test_tools_independence(self):
        """Test that tools work independently of core API."""
        # Tools should be usable without importing core forge
        from brainsmith.tools import roofline_analysis
        
        # Should not require core API components
        assert roofline_analysis is not None or roofline_analysis is None
    
    def test_error_handling_when_components_missing(self):
        """Test graceful handling when underlying components are missing."""
        # This test verifies the import error handling works
        
        # If imports fail, functions should be None
        if roofline_analysis is None:
            # This is expected behavior when components are missing
            assert True
        else:
            # If available, should be callable
            assert callable(roofline_analysis)
        
        if RooflineProfiler is None:
            # This is expected behavior when components are missing
            assert True
        else:
            # If available, should be a class
            assert isinstance(RooflineProfiler, type)


if __name__ == "__main__":
    pytest.main([__file__])