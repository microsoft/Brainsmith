"""
Tests for LegacyConversionLayer

Tests the 6-entrypoint → DataflowBuildConfig translation bridge.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from brainsmith.core.finn_v2.legacy_conversion import LegacyConversionLayer


class TestLegacyConversionLayer:
    """Tests for LegacyConversionLayer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.converter = LegacyConversionLayer()
        
        self.test_blueprint_config = {
            'name': 'test_blueprint',
            'constraints': {
                'target_frequency_mhz': 200,
                'target_throughput_fps': 1000,
                'max_luts': 50000,
                'max_power': 25.0
            },
            'configuration_files': {
                'folding_override': 'test_folding.json',
                'platform_config': 'zynq_ultrascale.yaml'
            },
            'output_dir': './test_output'
        }
        
        self.test_entrypoint_config = {
            'entrypoint_1': ['LayerNorm', 'Softmax'],
            'entrypoint_2': ['cleanup', 'streamlining'],
            'entrypoint_3': ['MatMul', 'LayerNorm'],
            'entrypoint_4': ['matmul_hls', 'layernorm_custom'],
            'entrypoint_5': ['target_fps_parallelization', 'apply_folding_config'],
            'entrypoint_6': ['set_fifo_depths', 'create_stitched_ip']
        }
    
    def test_initialization(self):
        """Test LegacyConversionLayer initialization."""
        assert self.converter.entrypoint_mappings is not None
        assert isinstance(self.converter.entrypoint_mappings, dict)
        
        # Check that all 6 entrypoints are mapped
        for i in range(1, 7):
            assert i in self.converter.entrypoint_mappings
            assert isinstance(self.converter.entrypoint_mappings[i], dict)
    
    def test_entrypoint_mappings_structure(self):
        """Test entrypoint mappings structure and content."""
        mappings = self.converter.entrypoint_mappings
        
        # Entrypoint 1: canonical ops
        assert 'LayerNorm' in mappings[1]
        assert 'Softmax' in mappings[1]
        assert 'GELU' in mappings[1]
        
        # Entrypoint 2: topology transforms
        assert 'cleanup' in mappings[2]
        assert 'streamlining' in mappings[2]
        assert 'aggressive_streamlining' in mappings[2]
        
        # Entrypoint 3: hw kernels
        assert 'MatMul' in mappings[3]
        assert 'Conv2D' in mappings[3]
        assert 'LayerNorm' in mappings[3]
        
        # Entrypoint 4: hw specializations
        assert 'matmul_rtl' in mappings[4]
        assert 'matmul_hls' in mappings[4]
        assert 'layernorm_custom' in mappings[4]
        
        # Entrypoint 5: hw kernel transforms
        assert 'target_fps_parallelization' in mappings[5]
        assert 'apply_folding_config' in mappings[5]
        assert 'minimize_bit_width' in mappings[5]
        
        # Entrypoint 6: hw graph transforms
        assert 'set_fifo_depths' in mappings[6]
        assert 'create_stitched_ip' in mappings[6]
    
    def test_map_entrypoint_to_steps(self):
        """Test mapping individual entrypoint components to FINN steps."""
        # Test entrypoint 1 (canonical ops)
        steps = self.converter._map_entrypoint_to_steps(1, ['LayerNorm', 'Softmax'])
        assert 'custom_step_register_layernorm' in steps
        assert 'custom_step_register_softmax' in steps
        
        # Test entrypoint 2 (topology transforms)
        steps = self.converter._map_entrypoint_to_steps(2, ['cleanup', 'streamlining'])
        assert 'custom_step_cleanup' in steps
        assert 'step_streamline' in steps
        
        # Test entrypoint 5 (hw kernel transforms)
        steps = self.converter._map_entrypoint_to_steps(5, ['target_fps_parallelization'])
        assert 'step_target_fps_parallelization' in steps
    
    def test_map_entrypoint_to_steps_unknown_component(self):
        """Test mapping with unknown component."""
        steps = self.converter._map_entrypoint_to_steps(1, ['unknown_component'])
        assert len(steps) == 0  # Should return empty list for unknown components
    
    def test_build_step_sequence(self):
        """Test building complete FINN step sequence."""
        steps = self.converter._build_step_sequence(self.test_entrypoint_config)
        
        # Check that sequence contains expected steps
        assert isinstance(steps, list)
        assert len(steps) > 0
        
        # Should start with entrypoint 1 custom steps
        assert any('custom_step_register_layernorm' in step for step in steps[:5])
        
        # Should contain standard FINN initialization steps
        assert 'step_qonnx_to_finn' in steps
        assert 'step_tidy_up' in steps
        
        # Should contain entrypoint 2 transforms
        assert 'custom_step_cleanup' in steps
        assert 'step_streamline' in steps
        
        # Should contain standard FINN transformation steps
        assert 'step_convert_to_hw' in steps
        assert 'step_create_dataflow_partition' in steps
        assert 'step_specialize_layers' in steps
        
        # Should contain entrypoint 5 optimization steps
        assert 'step_target_fps_parallelization' in steps
        assert 'step_apply_folding_config' in steps
        
        # Should contain standard build steps
        assert 'step_generate_estimate_reports' in steps
        assert 'step_hw_codegen' in steps
        assert 'step_hw_ipgen' in steps
        
        # Should contain entrypoint 6 graph optimization steps
        assert 'step_set_fifo_depths' in steps
        assert 'step_create_stitched_ip' in steps
        
        # Should end with performance measurement
        assert 'step_measure_rtlsim_performance' in steps
    
    def test_build_step_sequence_empty_entrypoints(self):
        """Test building step sequence with empty entrypoints."""
        empty_config = {f'entrypoint_{i}': [] for i in range(1, 7)}
        
        steps = self.converter._build_step_sequence(empty_config)
        
        # Should still contain core FINN steps
        assert 'step_qonnx_to_finn' in steps
        assert 'step_tidy_up' in steps
        assert 'step_streamline' in steps
        assert 'step_convert_to_hw' in steps
        assert 'step_specialize_layers' in steps
        
        # Should contain default optimization steps when entrypoint 5 is empty
        assert 'step_target_fps_parallelization' in steps
        assert 'step_apply_folding_config' in steps
        assert 'step_minimize_bit_width' in steps
        
        # Should contain default graph optimization when entrypoint 6 is empty
        assert 'step_set_fifo_depths' in steps
        assert 'step_create_stitched_ip' in steps
    
    def test_build_finn_config_params(self):
        """Test building FINN configuration parameters from blueprint."""
        params = self.converter._build_finn_config_params(self.test_blueprint_config)
        
        assert isinstance(params, dict)
        
        # Check clock frequency conversion (MHz → ns)
        assert 'synth_clk_period_ns' in params
        assert params['synth_clk_period_ns'] == 5.0  # 1000/200 = 5.0 ns
        
        # Check target FPS
        assert 'target_fps' in params
        assert params['target_fps'] == 1000
        
        # Check configuration files
        assert 'folding_config_file' in params
        assert params['folding_config_file'] == 'test_folding.json'
        
        # Check output directory
        assert 'output_dir' in params
        assert params['output_dir'] == './test_output'
        
        # Check defaults
        assert 'auto_fifo_depths' in params
        assert params['auto_fifo_depths'] == True
        assert 'save_intermediate_models' in params
        assert params['save_intermediate_models'] == True
    
    def test_build_finn_config_params_minimal(self):
        """Test building FINN parameters with minimal blueprint config."""
        minimal_config = {'name': 'minimal_test'}
        
        params = self.converter._build_finn_config_params(minimal_config)
        
        # Should contain defaults
        assert params['synth_clk_period_ns'] == 5.0  # 200 MHz default
        assert params['auto_fifo_depths'] == True
        assert params['save_intermediate_models'] == True
        assert params['output_dir'] == './finn_output'
    
    def test_extract_board_from_platform(self):
        """Test board name extraction from platform config."""
        board = self.converter._extract_board_from_platform('configs/zynq_ultrascale.yaml')
        assert board == 'Pynq-Z1'  # Should map zynq_ultrascale → Pynq-Z1
        
        board = self.converter._extract_board_from_platform('configs/alveo_u250.yaml')
        assert board == 'U250'  # Should map alveo_u250 → U250
        
        board = self.converter._extract_board_from_platform('configs/unknown_platform.yaml')
        assert board == 'unknown_platform'  # Should return platform name if no mapping
    
    @patch('brainsmith.core.finn_v2.legacy_conversion.DataflowBuildConfig')
    @patch('brainsmith.core.finn_v2.legacy_conversion.DataflowOutputType')
    def test_convert_to_dataflow_config_mock(self, mock_output_type, mock_config_class):
        """Test complete conversion to DataflowBuildConfig with mocked FINN."""
        # Mock FINN classes
        mock_config_instance = Mock()
        mock_config_class.return_value = mock_config_instance
        mock_output_type.STITCHED_IP = 'STITCHED_IP'
        
        # Test conversion
        result = self.converter.convert_to_dataflow_config(
            self.test_entrypoint_config, 
            self.test_blueprint_config
        )
        
        # Verify DataflowBuildConfig was called
        mock_config_class.assert_called_once()
        call_args = mock_config_class.call_args[1]  # keyword arguments
        
        # Check required parameters
        assert 'steps' in call_args
        assert 'output_dir' in call_args
        assert 'synth_clk_period_ns' in call_args
        assert 'generate_outputs' in call_args
        
        # Check steps list
        steps = call_args['steps']
        assert isinstance(steps, list)
        assert len(steps) > 0
        
        # Check configuration parameters
        assert call_args['output_dir'] == './test_output'
        assert call_args['synth_clk_period_ns'] == 5.0
        assert call_args['target_fps'] == 1000
        
        assert result == mock_config_instance
    
    def test_convert_to_dataflow_config_import_error(self):
        """Test conversion failure when FINN not available."""
        # Mock ImportError for FINN imports
        with patch('brainsmith.core.finn_v2.legacy_conversion.DataflowBuildConfig', side_effect=ImportError("FINN not found")):
            with pytest.raises(RuntimeError) as exc_info:
                self.converter.convert_to_dataflow_config(
                    self.test_entrypoint_config, 
                    self.test_blueprint_config
                )
            
            assert "FINN not available" in str(exc_info.value)
    
    def test_convert_to_dataflow_config_finn_error(self):
        """Test conversion failure when FINN DataflowBuildConfig creation fails."""
        # Mock FINN classes to raise exception
        with patch('brainsmith.core.finn_v2.legacy_conversion.DataflowBuildConfig', side_effect=Exception("Invalid config")):
            with pytest.raises(RuntimeError) as exc_info:
                self.converter.convert_to_dataflow_config(
                    self.test_entrypoint_config, 
                    self.test_blueprint_config
                )
            
            assert "Failed to create FINN configuration" in str(exc_info.value)


class TestLegacyConversionLayerIntegration:
    """Integration tests for LegacyConversionLayer."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.converter = LegacyConversionLayer()
    
    def test_realistic_bert_conversion(self):
        """Test conversion with realistic BERT configuration."""
        bert_entrypoint_config = {
            'entrypoint_1': ['LayerNorm', 'Softmax', 'GELU', 'MultiHeadAttention'],
            'entrypoint_2': ['cleanup', 'streamlining', 'bert_attention_fusion'],
            'entrypoint_3': ['MatMul', 'LayerNorm', 'Softmax'],
            'entrypoint_4': ['matmul_hls', 'layernorm_custom', 'softmax_hls'],
            'entrypoint_5': ['target_fps_parallelization', 'apply_folding_config', 'minimize_bit_width'],
            'entrypoint_6': ['set_fifo_depths', 'create_stitched_ip']
        }
        
        bert_blueprint_config = {
            'name': 'bert_accelerator_v2',
            'constraints': {
                'target_frequency_mhz': 200,
                'target_throughput_fps': 3000
            },
            'configuration_files': {
                'folding_override': 'configs/bert_folding.json',
                'platform_config': 'configs/zynq_ultrascale.yaml'
            }
        }
        
        # Test step sequence generation
        steps = self.converter._build_step_sequence(bert_entrypoint_config)
        
        # Should contain BERT-specific mappings
        assert len(steps) > 15  # Should have substantial step sequence
        
        # Test parameter extraction
        params = self.converter._build_finn_config_params(bert_blueprint_config)
        
        assert params['synth_clk_period_ns'] == 5.0  # 200 MHz
        assert params['target_fps'] == 3000
        assert 'bert_folding.json' in params['folding_config_file']
    
    @pytest.mark.skipif(
        True,  # Skip by default - requires real FINN installation
        reason="Real FINN integration test - requires FINN installation"
    )
    def test_real_finn_dataflow_config_creation(self):
        """Test creating real FINN DataflowBuildConfig (if FINN available)."""
        try:
            from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType
            
            # Create realistic configuration
            entrypoint_config = {
                'entrypoint_1': ['LayerNorm'],
                'entrypoint_2': ['cleanup'],
                'entrypoint_3': ['MatMul'],
                'entrypoint_4': ['matmul_hls'],
                'entrypoint_5': ['target_fps_parallelization'],
                'entrypoint_6': ['set_fifo_depths']
            }
            
            blueprint_config = {
                'constraints': {'target_frequency_mhz': 200},
                'output_dir': './test_finn_output'
            }
            
            # Test real conversion
            config = self.converter.convert_to_dataflow_config(entrypoint_config, blueprint_config)
            
            # Verify it's a real DataflowBuildConfig
            assert isinstance(config, DataflowBuildConfig)
            assert config.synth_clk_period_ns == 5.0
            assert config.output_dir == './test_finn_output'
            assert len(config.steps) > 0
            
            print("Real FINN DataflowBuildConfig creation successful")
            
        except ImportError:
            pytest.skip("FINN not available in test environment")


if __name__ == "__main__":
    # Run basic tests
    test_converter = TestLegacyConversionLayer()
    test_converter.setup_method()
    
    print("Testing LegacyConversionLayer...")
    test_converter.test_initialization()
    print("✓ Initialization test passed")
    
    test_converter.test_entrypoint_mappings_structure()
    print("✓ Entrypoint mappings test passed")
    
    test_converter.test_build_step_sequence()
    print("✓ Step sequence building test passed")
    
    test_converter.test_build_finn_config_params()
    print("✓ FINN config parameters test passed")
    
    print("All basic tests passed!")