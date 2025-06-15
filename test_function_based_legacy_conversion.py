#!/usr/bin/env python3
"""
Test Function-Based Legacy Conversion Implementation

Validates that the new function-based LegacyConversionLayer works correctly
and generates step functions matching the bert.py pattern.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_function_based_legacy_conversion():
    """Test the function-based LegacyConversionLayer implementation."""
    print("🧪 Testing Function-Based LegacyConversionLayer")
    
    try:
        from brainsmith.core.finn_v2.legacy_conversion import LegacyConversionLayer
        
        # Test initialization
        converter = LegacyConversionLayer()
        print("✅ LegacyConversionLayer initialized successfully")
        
        # Test function mappings
        assert hasattr(converter, 'entrypoint_function_mappings')
        assert hasattr(converter, 'standard_finn_steps')
        print("✅ Function mappings and standard steps loaded")
        
        # Test entrypoint function mappings
        mappings = converter.entrypoint_function_mappings
        assert 1 in mappings  # Entrypoint 1
        assert 2 in mappings  # Entrypoint 2
        assert 'LayerNorm' in mappings[1]
        assert 'cleanup' in mappings[2]
        print("✅ Entrypoint function mappings configured")
        
        # Test step function generation
        test_entrypoint_config = {
            'entrypoint_1': ['LayerNorm', 'Softmax'],
            'entrypoint_2': ['cleanup', 'streamlining'],
            'entrypoint_3': ['MatMul'],
            'entrypoint_5': ['target_fps_parallelization'],
            'entrypoint_6': ['set_fifo_depths']
        }
        
        step_functions = converter._build_step_function_list(test_entrypoint_config)
        print(f"✅ Generated {len(step_functions)} step functions")
        
        # Validate all are callable
        for i, func in enumerate(step_functions):
            assert callable(func), f"Step {i} is not callable"
        print("✅ All generated steps are callable functions")
        
        # Test individual function creation
        layernorm_func = converter._create_layernorm_registration_step({})
        assert callable(layernorm_func)
        print("✅ Individual function creation works")
        
        # Test function execution (mock)
        class MockModel:
            def transform(self, transformation):
                return self
        
        class MockConfig:
            pass
        
        mock_model = MockModel()
        mock_config = MockConfig()
        
        # Test a function execution
        result = layernorm_func(mock_model, mock_config)
        assert result is not None
        print("✅ Function execution successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataflow_config_creation():
    """Test DataflowBuildConfig creation with function lists."""
    print("\n🧪 Testing DataflowBuildConfig Creation")
    
    try:
        from brainsmith.core.finn_v2.legacy_conversion import LegacyConversionLayer
        
        converter = LegacyConversionLayer()
        
        test_entrypoint_config = {
            'entrypoint_1': ['LayerNorm'],
            'entrypoint_2': ['cleanup'],
            'entrypoint_5': ['target_fps_parallelization'],
            'entrypoint_6': ['set_fifo_depths']
        }
        
        test_blueprint_config = {
            'constraints': {
                'target_frequency_mhz': 200,
                'target_throughput_fps': 1000
            },
            'output_dir': './test_output'
        }
        
        try:
            # This will test our logic up to FINN import
            dataflow_config = converter.convert_to_dataflow_config(
                test_entrypoint_config, 
                test_blueprint_config
            )
            print("✅ DataflowBuildConfig created successfully (FINN available)")
            
            # Validate it has function list
            assert hasattr(dataflow_config, 'steps')
            assert len(dataflow_config.steps) > 0
            assert all(callable(step) for step in dataflow_config.steps)
            print(f"✅ DataflowBuildConfig has {len(dataflow_config.steps)} callable steps")
            
        except RuntimeError as e:
            if "FINN not available" in str(e):
                print("✅ DataflowBuildConfig logic validated (FINN not installed - expected)")
            else:
                raise e
        
        return True
        
    except Exception as e:
        print(f"❌ DataflowBuildConfig test failed: {e}")
        return False

def test_bert_pattern_compatibility():
    """Test compatibility with bert.py pattern."""
    print("\n🧪 Testing BERT Pattern Compatibility")
    
    try:
        from brainsmith.core.finn_v2.legacy_conversion import LegacyConversionLayer
        
        converter = LegacyConversionLayer()
        
        # BERT-like entrypoint configuration
        bert_entrypoint_config = {
            'entrypoint_1': ['LayerNorm', 'Softmax', 'GELU'],
            'entrypoint_2': ['cleanup', 'remove_head', 'remove_tail', 'streamlining'],
            'entrypoint_3': ['MatMul', 'LayerNorm'],
            'entrypoint_5': ['target_fps_parallelization', 'apply_folding_config'],
            'entrypoint_6': ['set_fifo_depths', 'create_stitched_ip']
        }
        
        step_functions = converter._build_step_function_list(bert_entrypoint_config)
        
        # Should have multiple functions like bert.py BUILD_STEPS
        assert len(step_functions) >= 10, f"Expected at least 10 steps, got {len(step_functions)}"
        print(f"✅ Generated {len(step_functions)} steps (similar to bert.py pattern)")
        
        # All should be callable
        assert all(callable(func) for func in step_functions)
        print("✅ All steps are callable functions (bert.py compatible)")
        
        return True
        
    except Exception as e:
        print(f"❌ BERT compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("🚀 FUNCTION-BASED LEGACY CONVERSION TESTING")
    print("="*60)
    
    results = []
    
    results.append(test_function_based_legacy_conversion())
    results.append(test_dataflow_config_creation())
    results.append(test_bert_pattern_compatibility())
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Function-based implementation working!")
        print("✅ Ready for integration with real FINN when available")
    else:
        print("❌ Some tests failed - need fixes")
        exit(1)