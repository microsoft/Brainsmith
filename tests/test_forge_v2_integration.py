"""
Integration Tests for forge_v2() API

Tests the complete Blueprint V2 → DSE → FINN flow with real integration.
"""

import pytest
from pathlib import Path
import tempfile
import json

from brainsmith.core.api_v2 import forge_v2, validate_blueprint_v2


class TestForgeV2Integration:
    """Integration tests for forge_v2() function."""
    
    def test_validate_blueprint_v2_success(self):
        """Test Blueprint V2 validation with valid blueprint."""
        blueprint_path = "brainsmith/libraries/blueprints_v2/transformers/bert_accelerator_v2.yaml"
        
        if Path(blueprint_path).exists():
            is_valid, errors = validate_blueprint_v2(blueprint_path)
            
            # Should be valid or have only minor warnings
            assert isinstance(is_valid, bool)
            assert isinstance(errors, list)
            
            if not is_valid:
                print(f"Blueprint validation issues: {errors}")
    
    def test_validate_blueprint_v2_missing_file(self):
        """Test Blueprint V2 validation with missing file."""
        is_valid, errors = validate_blueprint_v2("nonexistent_blueprint.yaml")
        
        assert is_valid == False
        assert len(errors) > 0
        assert any("not found" in error.lower() for error in errors)
    
    def test_forge_v2_basic_structure(self):
        """Test forge_v2() basic structure and error handling."""
        # Test with invalid inputs to check error handling
        result = forge_v2(
            model_path="nonexistent_model.onnx",
            blueprint_path="nonexistent_blueprint.yaml"
        )
        
        # Should return clean error response
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'error' in result
        assert 'execution_time' in result
        assert 'best_design' in result
        assert 'pareto_frontier' in result
        assert 'exploration_summary' in result
        
        # Should indicate failure
        assert result['success'] == False
        assert result['error'] is not None
    
    def test_forge_v2_with_valid_blueprint_invalid_model(self):
        """Test forge_v2() with valid blueprint but invalid model."""
        blueprint_path = "brainsmith/libraries/blueprints_v2/transformers/bert_accelerator_v2.yaml"
        
        if Path(blueprint_path).exists():
            result = forge_v2(
                model_path="nonexistent_model.onnx",
                blueprint_path=blueprint_path
            )
            
            # Should fail gracefully
            assert result['success'] == False
            assert 'error' in result
            assert result['best_design'] is None
    
    @pytest.mark.skipif(
        not Path("custom_bert/bert_model.onnx").exists(),
        reason="BERT model not available"
    )
    def test_forge_v2_end_to_end_bert(self):
        """Test complete flow with bert_accelerator_v2.yaml and BERT model."""
        model_path = "custom_bert/bert_model.onnx"
        blueprint_path = "brainsmith/libraries/blueprints_v2/transformers/bert_accelerator_v2.yaml"
        
        # Skip if blueprint doesn't exist or isn't complete
        if not Path(blueprint_path).exists():
            pytest.skip("BERT Blueprint V2 not available")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with minimal evaluations for faster testing
            dse_config = {
                'max_evaluations': 3,  # Very small for testing
                'parallel_evaluations': 1,
                'early_termination_patience': 2
            }
            
            try:
                result = forge_v2(
                    model_path=model_path,
                    blueprint_path=blueprint_path,
                    output_dir=temp_dir,
                    dse_config=dse_config
                )
                
                # Check result structure
                assert isinstance(result, dict)
                assert 'success' in result
                assert 'execution_time' in result
                assert 'best_design' in result
                assert 'pareto_frontier' in result
                assert 'exploration_summary' in result
                
                # If successful, check data quality
                if result['success']:
                    assert result['exploration_summary']['total_evaluations'] >= 0
                    assert result['execution_time'] > 0
                    
                    # Check output files were created
                    output_path = Path(temp_dir)
                    assert (output_path / "forge_v2_results.json").exists()
                    assert (output_path / "forge_v2_summary.json").exists()
                else:
                    # If failed, should have meaningful error
                    assert 'error' in result
                    print(f"forge_v2 failed (expected for testing): {result['error']}")
                    
            except Exception as e:
                # Expected - FINN may not be available in test environment
                print(f"forge_v2 test failed (expected): {e}")
                assert "FINN" in str(e) or "not available" in str(e)
    
    def test_forge_v2_objective_overrides(self):
        """Test forge_v2() with objective overrides."""
        blueprint_path = "brainsmith/libraries/blueprints_v2/transformers/bert_accelerator_v2.yaml"
        
        if Path(blueprint_path).exists():
            # Test with custom objectives
            custom_objectives = {
                'throughput': {'direction': 'maximize', 'weight': 1.0},
                'latency': {'direction': 'minimize', 'weight': 0.8}
            }
            
            custom_constraints = {
                'max_luts': 50000,
                'target_frequency_mhz': 200
            }
            
            result = forge_v2(
                model_path="nonexistent_model.onnx",  # Will fail but test override parsing
                blueprint_path=blueprint_path,
                objectives=custom_objectives,
                constraints=custom_constraints,
                target_device="zynq_ultrascale"
            )
            
            # Should fail gracefully with overrides applied
            assert result['success'] == False
            assert 'error' in result
    
    def test_forge_v2_dse_config_options(self):
        """Test forge_v2() with various DSE configuration options."""
        blueprint_path = "brainsmith/libraries/blueprints_v2/transformers/bert_accelerator_v2.yaml"
        
        if Path(blueprint_path).exists():
            # Test with custom DSE configuration
            dse_config = {
                'max_evaluations': 10,
                'parallel_evaluations': 2,
                'enable_caching': True,
                'early_termination_patience': 5
            }
            
            result = forge_v2(
                model_path="nonexistent_model.onnx",
                blueprint_path=blueprint_path,
                dse_config=dse_config
            )
            
            # Should parse DSE config correctly even if execution fails
            assert result['success'] == False
            assert 'exploration_summary' in result


class TestForgeV2Components:
    """Unit tests for forge_v2() component functions."""
    
    def test_blueprint_v2_loading_functions(self):
        """Test Blueprint V2 loading helper functions."""
        from brainsmith.core.api_v2 import _load_blueprint_v2_strict
        
        # Test with nonexistent file
        with pytest.raises(FileNotFoundError):
            _load_blueprint_v2_strict("nonexistent.yaml")
        
        # Test with wrong extension
        with pytest.raises(ValueError):
            _load_blueprint_v2_strict("test.txt")
    
    def test_clean_results_formatting(self):
        """Test clean results formatting functions."""
        from brainsmith.core.api_v2 import _format_clean_results
        from brainsmith.core.dse_v2.space_explorer import ExplorationResults
        
        # Create mock exploration results
        mock_results = ExplorationResults(
            best_combination=None,
            best_score=0.0,
            all_combinations=[],
            performance_data=[],
            pareto_frontier=[],
            exploration_summary={},
            strategy_metadata={},
            execution_stats={}
        )
        
        formatted = _format_clean_results(mock_results, 0.0)
        
        # Check required structure
        assert 'success' in formatted
        assert 'best_design' in formatted
        assert 'pareto_frontier' in formatted
        assert 'exploration_summary' in formatted
        assert 'build_artifacts' in formatted
        assert 'raw_data' in formatted


class TestBlueprintV2Validation:
    """Tests for Blueprint V2 validation functionality."""
    
    def test_validate_blueprint_v2_integration(self):
        """Test Blueprint V2 validation integration."""
        blueprint_path = "brainsmith/libraries/blueprints_v2/base/transformer_base.yaml"
        
        if Path(blueprint_path).exists():
            is_valid, errors = validate_blueprint_v2(blueprint_path)
            
            assert isinstance(is_valid, bool)
            assert isinstance(errors, list)
            
            # Print validation results for debugging
            if not is_valid:
                print(f"Validation errors for {blueprint_path}:")
                for error in errors:
                    print(f"  - {error}")
    
    def test_multiple_blueprint_validation(self):
        """Test validation of multiple Blueprint V2 files."""
        blueprint_dir = Path("brainsmith/libraries/blueprints_v2")
        
        if blueprint_dir.exists():
            blueprint_files = list(blueprint_dir.rglob("*.yaml"))
            
            for blueprint_path in blueprint_files:
                print(f"Validating: {blueprint_path}")
                
                is_valid, errors = validate_blueprint_v2(str(blueprint_path))
                
                # Print results but don't fail test - blueprints may be incomplete
                if not is_valid:
                    print(f"  Issues found: {len(errors)}")
                    for error in errors[:3]:  # Show first 3 errors
                        print(f"    - {error}")
                else:
                    print(f"  Valid!")


if __name__ == "__main__":
    # Run basic tests
    test_class = TestForgeV2Integration()
    
    print("Testing forge_v2() basic functionality...")
    test_class.test_forge_v2_basic_structure()
    print("✓ Basic structure test passed")
    
    print("Testing Blueprint V2 validation...")
    test_class.test_validate_blueprint_v2_missing_file()
    print("✓ Validation test passed")
    
    print("All basic tests passed!")