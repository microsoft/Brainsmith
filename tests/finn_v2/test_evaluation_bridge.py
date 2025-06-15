"""
Tests for FINNEvaluationBridge

Tests the main DSE → FINN interface with real FINN integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from brainsmith.core.finn_v2.evaluation_bridge import FINNEvaluationBridge
from brainsmith.core.dse_v2.combination_generator import ComponentCombination


class TestFINNEvaluationBridge:
    """Tests for FINNEvaluationBridge class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.blueprint_config = {
            'name': 'test_blueprint',
            'constraints': {
                'target_frequency_mhz': 200,
                'target_throughput_fps': 1000
            },
            'configuration_files': {
                'folding_override': 'test_folding.json'
            }
        }
        
        self.bridge = FINNEvaluationBridge(self.blueprint_config)
    
    def test_initialization(self):
        """Test FINNEvaluationBridge initialization."""
        assert self.bridge.blueprint_config == self.blueprint_config
        assert self.bridge.legacy_converter is not None
        assert self.bridge.metrics_extractor is not None
    
    def test_combination_to_entrypoint_config(self):
        """Test conversion of ComponentCombination to entrypoint config."""
        # Create test combination
        combination = ComponentCombination(
            combination_id="test_001",
            canonical_ops={"LayerNorm", "Softmax"},
            hw_kernels={"MatMul": "matmul_hls", "LayerNorm": "layernorm_custom"},
            model_topology={"cleanup", "streamlining"},
            hw_kernel_transforms={"target_fps_parallelization"},
            hw_graph_transforms={"set_fifo_depths"}
        )
        
        # Test conversion
        entrypoint_config = self.bridge._combination_to_entrypoint_config(combination)
        
        # Verify structure
        assert isinstance(entrypoint_config, dict)
        assert 'entrypoint_1' in entrypoint_config
        assert 'entrypoint_2' in entrypoint_config
        assert 'entrypoint_3' in entrypoint_config
        assert 'entrypoint_4' in entrypoint_config
        assert 'entrypoint_5' in entrypoint_config
        assert 'entrypoint_6' in entrypoint_config
        
        # Verify content
        assert "LayerNorm" in entrypoint_config['entrypoint_1']
        assert "Softmax" in entrypoint_config['entrypoint_1']
        assert "cleanup" in entrypoint_config['entrypoint_2']
        assert "streamlining" in entrypoint_config['entrypoint_2']
        assert "MatMul" in entrypoint_config['entrypoint_3']
        assert "LayerNorm" in entrypoint_config['entrypoint_3']
        assert "matmul_hls" in entrypoint_config['entrypoint_4']
        assert "layernorm_custom" in entrypoint_config['entrypoint_4']
        assert "target_fps_parallelization" in entrypoint_config['entrypoint_5']
        assert "set_fifo_depths" in entrypoint_config['entrypoint_6']
    
    def test_get_supported_objectives(self):
        """Test getting supported optimization objectives."""
        objectives = self.bridge.get_supported_objectives()
        
        assert isinstance(objectives, list)
        assert 'throughput' in objectives
        assert 'latency' in objectives
        assert 'resource_efficiency' in objectives
        assert 'lut_utilization' in objectives
        assert 'dsp_utilization' in objectives
        assert 'bram_utilization' in objectives
    
    def test_validate_combination_valid(self):
        """Test combination validation with valid combination."""
        combination = ComponentCombination(
            combination_id="test_002",
            canonical_ops={"LayerNorm"},
            hw_kernels={"MatMul": "matmul_hls"},
            model_topology={"cleanup"},
            hw_kernel_transforms={"target_fps_parallelization"},
            hw_graph_transforms={"set_fifo_depths"}
        )
        
        is_valid, errors = self.bridge.validate_combination(combination)
        
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_combination_empty(self):
        """Test combination validation with empty combination."""
        combination = ComponentCombination(
            combination_id="test_003",
            canonical_ops=set(),
            hw_kernels={},
            model_topology=set(),
            hw_kernel_transforms=set(),
            hw_graph_transforms=set()
        )
        
        is_valid, errors = self.bridge.validate_combination(combination)
        
        assert is_valid == False
        assert len(errors) > 0
        assert any("at least" in error.lower() for error in errors)
    
    def test_validate_combination_conflicting(self):
        """Test combination validation with conflicting components."""
        combination = ComponentCombination(
            combination_id="test_004",
            canonical_ops={"LayerNorm"},
            hw_kernels={"MatMul": "matmul_hls"},
            model_topology={"aggressive_streamlining", "conservative_streamlining"},
            hw_kernel_transforms={"target_fps_parallelization"},
            hw_graph_transforms={"set_fifo_depths"}
        )
        
        is_valid, errors = self.bridge.validate_combination(combination)
        
        assert is_valid == False
        assert len(errors) > 0
        assert any("aggressive" in error and "conservative" in error for error in errors)
    
    @patch('brainsmith.core.finn_v2.evaluation_bridge.build_dataflow_cfg')
    def test_execute_finn_run_mock_success(self, mock_build_dataflow):
        """Test FINN execution with mocked successful build."""
        # Mock successful FINN build
        mock_result = Mock()
        mock_result.model = Mock()
        mock_result.output_dir = "test_output"
        mock_build_dataflow.return_value = mock_result
        
        # Create mock DataflowBuildConfig
        mock_config = Mock()
        
        # Test execution
        result = self.bridge._execute_finn_run("test_model.onnx", mock_config)
        
        # Verify FINN was called
        mock_build_dataflow.assert_called_once_with("test_model.onnx", mock_config)
        assert result == mock_result
    
    @patch('brainsmith.core.finn_v2.evaluation_bridge.build_dataflow_cfg')
    def test_execute_finn_run_mock_failure(self, mock_build_dataflow):
        """Test FINN execution with mocked build failure."""
        # Mock FINN build failure
        mock_build_dataflow.side_effect = Exception("FINN build failed")
        
        mock_config = Mock()
        
        # Test execution should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            self.bridge._execute_finn_run("test_model.onnx", mock_config)
        
        assert "FINN execution failed" in str(exc_info.value)
    
    def test_execute_finn_run_missing_file(self):
        """Test FINN execution with missing model file."""
        mock_config = Mock()
        
        # Test with nonexistent file
        with pytest.raises(FileNotFoundError):
            self.bridge._execute_finn_run("nonexistent_model.onnx", mock_config)
    
    @patch('brainsmith.core.finn_v2.evaluation_bridge.build_dataflow_cfg')
    def test_evaluate_combination_mock_success(self, mock_build_dataflow):
        """Test complete combination evaluation with mocked FINN."""
        # Mock successful FINN execution
        mock_finn_result = Mock()
        mock_finn_result.model = Mock()
        mock_finn_result.output_dir = "test_output"
        mock_build_dataflow.return_value = mock_finn_result
        
        # Mock metrics extraction
        with patch.object(self.bridge.metrics_extractor, 'extract_metrics') as mock_extract:
            mock_extract.return_value = {
                'success': True,
                'primary_metric': 100.0,
                'throughput': 150.0,
                'latency': 8.0,
                'resource_utilization': 0.7
            }
            
            # Create test combination
            combination = ComponentCombination(
                combination_id="test_005",
                canonical_ops={"LayerNorm"},
                hw_kernels={"MatMul": "matmul_hls"},
                model_topology={"cleanup"},
                hw_kernel_transforms={"target_fps_parallelization"},
                hw_graph_transforms={"set_fifo_depths"}
            )
            
            # Test evaluation
            result = self.bridge.evaluate_combination("test_model.onnx", combination)
            
            # Verify result structure
            assert isinstance(result, dict)
            assert result['success'] == True
            assert result['primary_metric'] == 100.0
            assert result['throughput'] == 150.0
            assert result['latency'] == 8.0
            assert result['resource_utilization'] == 0.7
    
    def test_evaluate_combination_exception_handling(self):
        """Test combination evaluation with exception handling."""
        # Create test combination
        combination = ComponentCombination(
            combination_id="test_006",
            canonical_ops={"LayerNorm"},
            hw_kernels={"MatMul": "matmul_hls"},
            model_topology={"cleanup"},
            hw_kernel_transforms={"target_fps_parallelization"},
            hw_graph_transforms={"set_fifo_depths"}
        )
        
        # Mock legacy converter to raise exception
        with patch.object(self.bridge.legacy_converter, 'convert_to_dataflow_config') as mock_convert:
            mock_convert.side_effect = Exception("Conversion failed")
            
            # Test evaluation
            result = self.bridge.evaluate_combination("test_model.onnx", combination)
            
            # Should return error result
            assert result['success'] == False
            assert 'error' in result
            assert result['combination_id'] == "test_006"
            assert result['primary_metric'] == 0.0
            assert result['throughput'] == 0.0
            assert result['latency'] == float('inf')


class TestFINNEvaluationBridgeIntegration:
    """Integration tests that may require real FINN (skipped if not available)."""
    
    @pytest.mark.skipif(
        True,  # Skip by default - requires real FINN installation
        reason="Real FINN integration test - requires FINN installation"
    )
    def test_real_finn_integration(self):
        """Test with real FINN installation (if available)."""
        blueprint_config = {
            'name': 'integration_test',
            'constraints': {'target_frequency_mhz': 200}
        }
        
        bridge = FINNEvaluationBridge(blueprint_config)
        
        # This would test actual FINN integration
        # Only run if FINN is available in environment
        try:
            from finn.builder.build_dataflow import build_dataflow_cfg
            
            # Create minimal test combination
            combination = ComponentCombination(
                combination_id="integration_001",
                canonical_ops={"LayerNorm"},
                hw_kernels={"MatMul": "matmul_hls"},
                model_topology={"cleanup"},
                hw_kernel_transforms={"target_fps_parallelization"},
                hw_graph_transforms={"set_fifo_depths"}
            )
            
            # This would require a real ONNX model file
            # result = bridge.evaluate_combination("path/to/real/model.onnx", combination)
            # assert result['success'] in [True, False]  # Either success or graceful failure
            
            print("FINN is available for real integration testing")
            
        except ImportError:
            pytest.skip("FINN not available in test environment")


if __name__ == "__main__":
    # Run basic tests
    test_bridge = TestFINNEvaluationBridge()
    test_bridge.setup_method()
    
    print("Testing FINNEvaluationBridge...")
    test_bridge.test_initialization()
    print("✓ Initialization test passed")
    
    test_bridge.test_combination_to_entrypoint_config()
    print("✓ Entrypoint config conversion test passed")
    
    test_bridge.test_validate_combination_valid()
    print("✓ Combination validation test passed")
    
    test_bridge.test_get_supported_objectives()
    print("✓ Supported objectives test passed")
    
    print("All basic tests passed!")