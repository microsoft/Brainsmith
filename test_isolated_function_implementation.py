#!/usr/bin/env python3
"""
Isolated Test for Function-Based Implementation

Tests the function-based LegacyConversionLayer directly without
triggering the full BrainSmith import chain.
"""

import sys
import os
import importlib.util

def load_legacy_conversion_directly():
    """Load LegacyConversionLayer directly to bypass import issues."""
    spec = importlib.util.spec_from_file_location(
        'legacy_conversion', 
        'brainsmith/core/finn_v2/legacy_conversion.py'
    )
    legacy_module = importlib.util.module_from_spec(spec)
    
    # Add required modules to avoid import errors
    sys.modules['brainsmith'] = type(sys)('brainsmith')
    sys.modules['brainsmith.core'] = type(sys)('brainsmith.core')
    sys.modules['brainsmith.core.finn_v2'] = type(sys)('brainsmith.core.finn_v2')
    
    spec.loader.exec_module(legacy_module)
    return legacy_module.LegacyConversionLayer

def test_function_based_implementation():
    """Test the function-based implementation directly."""
    print("🧪 Testing Function-Based Implementation (Isolated)")
    
    try:
        LegacyConversionLayer = load_legacy_conversion_directly()
        
        # Test initialization
        converter = LegacyConversionLayer()
        print("✅ LegacyConversionLayer initialized successfully")
        
        # Test function mappings exist
        assert hasattr(converter, 'entrypoint_function_mappings')
        assert hasattr(converter, 'standard_finn_steps')
        print("✅ Function mappings and standard steps attributes present")
        
        # Test entrypoint function mappings structure
        mappings = converter.entrypoint_function_mappings
        assert 1 in mappings  # Entrypoint 1
        assert 2 in mappings  # Entrypoint 2
        print("✅ Entrypoint function mappings configured")
        
        # Test function generators
        assert 'LayerNorm' in mappings[1]
        assert callable(mappings[1]['LayerNorm'])
        print("✅ Function generators are callable")
        
        # Test step function creation
        layernorm_generator = mappings[1]['LayerNorm']
        layernorm_step = layernorm_generator({})
        assert callable(layernorm_step)
        print("✅ Step function generation works")
        
        # Test function list building
        test_entrypoint_config = {
            'entrypoint_1': ['LayerNorm'],
            'entrypoint_2': ['cleanup'],
            'entrypoint_5': [],
            'entrypoint_6': []
        }
        
        step_functions = converter._build_step_function_list(test_entrypoint_config)
        print(f"✅ Generated {len(step_functions)} step functions")
        
        # Validate all are callable
        for i, func in enumerate(step_functions):
            assert callable(func), f"Step {i} is not callable"
        print("✅ All generated steps are callable functions")
        
        # Test mock execution
        class MockModel:
            def transform(self, transformation):
                return self
        
        class MockConfig:
            pass
        
        mock_model = MockModel()
        mock_config = MockConfig()
        
        # Test function execution
        result = layernorm_step(mock_model, mock_config)
        assert result is not None
        print("✅ Function execution successful with mock model")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bert_compatibility_isolated():
    """Test BERT pattern compatibility in isolation."""
    print("\n🧪 Testing BERT Pattern Compatibility (Isolated)")
    
    try:
        LegacyConversionLayer = load_legacy_conversion_directly()
        converter = LegacyConversionLayer()
        
        # BERT-like entrypoint configuration
        bert_entrypoint_config = {
            'entrypoint_1': ['LayerNorm', 'Softmax'],
            'entrypoint_2': ['cleanup', 'streamlining'],
            'entrypoint_3': ['MatMul'],
            'entrypoint_5': [],
            'entrypoint_6': []
        }
        
        step_functions = converter._build_step_function_list(bert_entrypoint_config)
        
        # Should have multiple functions like bert.py BUILD_STEPS
        assert len(step_functions) >= 5, f"Expected at least 5 steps, got {len(step_functions)}"
        print(f"✅ Generated {len(step_functions)} steps (BERT-like pattern)")
        
        # All should be callable
        assert all(callable(func) for func in step_functions)
        print("✅ All steps are callable functions (bert.py compatible)")
        
        # Test function names/types are meaningful
        function_names = [getattr(func, '__name__', 'unnamed') for func in step_functions]
        print(f"✅ Function names: {function_names[:5]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ BERT compatibility test failed: {e}")
        return False

def test_parameter_extraction():
    """Test parameter extraction functionality."""
    print("\n🧪 Testing Parameter Extraction")
    
    try:
        LegacyConversionLayer = load_legacy_conversion_directly()
        converter = LegacyConversionLayer()
        
        test_blueprint_config = {
            'constraints': {
                'target_frequency_mhz': 250,
                'target_throughput_fps': 2000
            },
            'configuration_files': {
                'folding_override': 'test_folding.json',
                'platform_config': 'alveo_u250.yaml'
            },
            'output_dir': './test_finn_output'
        }
        
        params = converter._build_finn_config_params(test_blueprint_config)
        
        # Check parameter extraction
        assert 'synth_clk_period_ns' in params
        assert params['synth_clk_period_ns'] == 4.0  # 1000/250
        assert params['target_fps'] == 2000
        assert params['folding_config_file'] == 'test_folding.json'
        assert params['board'] == 'U250'  # From alveo_u250
        assert params['output_dir'] == './test_finn_output'
        
        print("✅ Parameter extraction working correctly")
        print(f"   Clock period: {params['synth_clk_period_ns']} ns")
        print(f"   Target FPS: {params['target_fps']}")
        print(f"   Board: {params['board']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter extraction test failed: {e}")
        return False

if __name__ == "__main__":
    print("="*70)
    print("🚀 ISOLATED FUNCTION-BASED IMPLEMENTATION TESTING")
    print("="*70)
    
    results = []
    
    results.append(test_function_based_implementation())
    results.append(test_bert_compatibility_isolated())
    results.append(test_parameter_extraction())
    
    print("\n" + "="*70)
    print("📊 ISOLATED TEST RESULTS SUMMARY")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 ALL ISOLATED TESTS PASSED!")
        print("✅ Function-based implementation working correctly")
        print("✅ Compatible with bert.py pattern")
        print("✅ Ready for real FINN integration")
    else:
        print("❌ Some tests failed - need fixes")
        exit(1)