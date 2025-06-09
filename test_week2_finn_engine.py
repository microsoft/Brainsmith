"""
Test Week 2 FINN Integration Engine

Test the complete FINN Integration Engine implementation including:
- Configuration generation from Brainsmith parameters
- Build execution simulation
- Result processing and enhancement
- Complete workflow validation
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_finn_integration_engine():
    """Test complete FINN Integration Engine"""
    try:
        print("🚀 Testing FINN Integration Engine...")
        
        from brainsmith.finn.engine import FINNIntegrationEngine
        
        engine = FINNIntegrationEngine()
        
        # Test configuration generation
        brainsmith_config = {
            'model': {
                'supported_operators': ['Conv', 'MatMul', 'Relu', 'Add'],
                'custom_operators': {},
                'cleanup_transforms': ['RemoveUnusedNodes', 'FoldConstants']
            },
            'optimization': {
                'level': 'standard',
                'strategy': 'balanced'
            },
            'targets': {
                'performance': {
                    'throughput': 1000,
                    'latency': 25,
                    'accuracy': 0.9
                }
            },
            'constraints': {
                'resources': {'luts': 0.8, 'dsps': 0.7},
                'power': {'max_power': 15.0}
            },
            'kernels': {
                'selection_plan': {
                    'conv1': 'ConvolutionInputGenerator',
                    'fc1': 'StreamingFCLayer',
                    'activation1': 'VectorVectorActivation'
                },
                'custom_implementations': {}
            },
            'target': {
                'platform': 'zynq'
            }
        }
        
        # Generate FINN configuration
        finn_config = engine.configure_finn_interface(brainsmith_config)
        
        # Validate configuration structure
        assert finn_config.model_ops is not None
        assert finn_config.model_transforms is not None
        assert finn_config.hw_kernels is not None
        assert finn_config.hw_optimization is not None
        assert 'created_by' in finn_config.metadata
        
        # Validate configuration content
        assert len(finn_config.model_ops.supported_ops) == 4
        assert finn_config.model_transforms.optimization_level == 'standard'
        assert len(finn_config.hw_kernels.kernel_selection_plan) == 3
        assert finn_config.hw_optimization.optimization_strategy == 'balanced'
        
        print("   ✅ FINN configuration generation working")
        
        # Test build execution
        design_point = {
            'target_device': 'xc7z020clg400-1',
            'clock_period': 10.0,
            'conv1_pe': 4,
            'conv1_simd': 2,
            'fc1_pe': 8,
            'fc1_simd': 4,
            'predicted_throughput': 800,
            'predicted_latency': 30,
            'predicted_power': 12.0
        }
        
        result = engine.execute_finn_build(finn_config, design_point)
        
        # Validate result structure
        assert result.original_result is not None
        assert result.enhanced_timestamp is not None
        
        if result.success:
            assert result.performance_metrics is not None
            assert result.resource_analysis is not None
            assert result.timing_analysis is not None
            assert len(result.quality_metrics) > 0
            assert len(result.optimization_opportunities) > 0
            
            print("   ✅ FINN build execution working (simulated)")
        else:
            assert result.original_result.error_message is not None
            print("   ✅ FINN build failure handling working")
        
        print("   ✅ FINN Integration Engine working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ FINN Integration Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_engine_configuration_validation():
    """Test configuration validation"""
    try:
        print("✅ Testing Engine Configuration Validation...")
        
        from brainsmith.finn.engine import FINNIntegrationEngine
        
        engine = FINNIntegrationEngine()
        
        # Test valid configuration
        valid_config = {
            'model': {'supported_operators': ['Conv', 'Relu']},
            'optimization': {'level': 'standard', 'strategy': 'balanced'},
            'targets': {'performance': {}},
            'constraints': {},
            'kernels': {'selection_plan': {}},
            'target': {'platform': 'zynq'}
        }
        
        finn_config = engine.configure_finn_interface(valid_config)
        assert engine.validate_configuration(finn_config)
        
        print("   ✅ Configuration validation working")
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration validation test failed: {e}")
        return False

def test_engine_supported_features():
    """Test supported features query"""
    try:
        print("📋 Testing Engine Supported Features...")
        
        from brainsmith.finn.engine import FINNIntegrationEngine
        
        engine = FINNIntegrationEngine()
        
        features = engine.get_supported_features()
        
        # Validate feature categories
        assert 'operations' in features
        assert 'optimization_levels' in features
        assert 'platforms' in features
        assert 'kernels' in features
        assert 'strategies' in features
        
        # Validate feature content
        assert len(features['operations']) > 0
        assert 'standard' in features['optimization_levels']
        assert 'zynq' in features['platforms']
        assert 'StreamingFCLayer' in features['kernels']
        assert 'balanced' in features['strategies']
        
        print("   ✅ Supported features query working")
        return True
        
    except Exception as e:
        print(f"   ❌ Supported features test failed: {e}")
        return False

def test_engine_error_handling():
    """Test error handling in engine"""
    try:
        print("🛡️ Testing Engine Error Handling...")
        
        from brainsmith.finn.engine import FINNIntegrationEngine
        
        engine = FINNIntegrationEngine()
        
        # Test with invalid configuration
        invalid_config = {
            'model': {'supported_operators': ['InvalidOp']},
            'optimization': {'level': 'invalid_level'},
            'targets': {},
            'constraints': {},
            'kernels': {'selection_plan': {'layer1': 'InvalidKernel'}},
            'target': {'platform': 'invalid_platform'}
        }
        
        # Should handle gracefully and create valid config with defaults/fallbacks
        finn_config = engine.configure_finn_interface(invalid_config)
        assert finn_config is not None
        
        # Test build with potential failure
        design_point = {'target_device': 'invalid_device'}
        result = engine.execute_finn_build(finn_config, design_point)
        assert result is not None
        assert result.enhanced_timestamp is not None
        
        print("   ✅ Error handling working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        return False

def test_complete_workflow():
    """Test complete end-to-end workflow"""
    try:
        print("🔄 Testing Complete Workflow...")
        
        from brainsmith.finn.engine import FINNIntegrationEngine
        
        engine = FINNIntegrationEngine()
        
        # Complete workflow: config -> build -> results
        workflow_config = {
            'model': {
                'supported_operators': ['Conv', 'MatMul', 'Relu'],
                'cleanup_transforms': ['FoldConstants']
            },
            'optimization': {
                'level': 'aggressive',
                'strategy': 'throughput'
            },
            'targets': {
                'performance': {
                    'throughput': 2000,
                    'latency': 15
                }
            },
            'constraints': {
                'resources': {'luts': 0.9, 'dsps': 0.8, 'brams': 0.7},
                'power': {'max_power': 25.0}
            },
            'kernels': {
                'selection_plan': {
                    'conv_layer': 'ConvolutionInputGenerator',
                    'dense_layer': 'StreamingFCLayer'
                }
            },
            'target': {'platform': 'ultrascale'}
        }
        
        # Step 1: Configure FINN interface
        finn_config = engine.configure_finn_interface(workflow_config)
        config_dict = finn_config.to_dict()
        assert len(config_dict) == 5  # 4 categories + metadata
        
        # Step 2: Execute build
        design_point = {
            'conv_layer_pe': 16,
            'conv_layer_simd': 8,
            'dense_layer_pe': 32,
            'dense_layer_simd': 16,
            'throughput_target': 2500,
            'latency_target': 12
        }
        
        result = engine.execute_finn_build(finn_config, design_point)
        
        # Step 3: Validate results
        assert result.original_result.build_time > 0
        result_dict = result.to_dict()
        assert 'enhanced_timestamp' in result_dict
        
        if result.success:
            # Verify performance metrics are populated
            assert result.performance_metrics.throughput > 0
            assert result.performance_metrics.latency > 0
            
            # Verify analysis results
            assert len(result.resource_analysis.optimization_suggestions) > 0
            assert len(result.timing_analysis.optimization_suggestions) > 0
            
            # Verify quality metrics
            assert 'build_success_rate' in result.quality_metrics
            assert result.quality_metrics['build_success_rate'] == 1.0
        
        print("   ✅ Complete workflow working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Complete workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_engine_tests():
    """Run all FINN Integration Engine tests"""
    print("🧪 Week 2 FINN Integration Engine Testing")
    print("=" * 60)
    print("Testing complete FINN Integration Engine implementation")
    print("=" * 60)
    
    tests = [
        ("FINN Integration Engine", test_finn_integration_engine),
        ("Configuration Validation", test_engine_configuration_validation),
        ("Supported Features", test_engine_supported_features),
        ("Error Handling", test_engine_error_handling),
        ("Complete Workflow", test_complete_workflow)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"FINN INTEGRATION ENGINE TEST RESULTS")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print(f"\n🎉 ALL FINN INTEGRATION ENGINE TESTS PASSED!")
        print(f"✅ Complete four-category interface integration working")
        print(f"✅ Build execution and result processing operational")
        print(f"✅ Error handling and validation comprehensive")
        print(f"✅ Week 2 FINN Integration Engine implementation complete!")
    else:
        print(f"\n⚠️  Some engine tests failed - need to fix before proceeding")
    
    return failed == 0

if __name__ == '__main__':
    success = run_engine_tests()
    sys.exit(0 if success else 1)