"""
Test Week 2 FINN Integration Engine Managers

Test the four manager components that handle FINN's four-category interface:
1. ModelOpsManager
2. ModelTransformsManager  
3. HwKernelsManager
4. HwOptimizationManager
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_ops_manager():
    """Test ModelOpsManager functionality"""
    try:
        print("🔧 Testing ModelOpsManager...")
        
        from brainsmith.finn.model_ops_manager import ModelOpsManager
        
        manager = ModelOpsManager()
        
        # Test basic configuration
        config = manager.configure(
            supported_ops=['Conv', 'MatMul', 'Relu', 'Add'],
            custom_ops={
                'CustomThreshold': {
                    'backend': 'rtl',
                    'implementation_path': 'custom.threshold'
                }
            },
            frontend_cleanup=['RemoveUnusedNodes', 'FoldConstants']
        )
        
        # Validate configuration
        assert len(config.supported_ops) == 4
        assert 'Conv' in config.supported_ops
        assert 'CustomThreshold' in config.custom_ops
        assert len(config.frontend_cleanup) == 2
        assert len(config.preprocessing_steps) > 0
        
        # Test registry queries
        available_ops = manager.get_supported_operations()
        assert len(available_ops) > 0
        
        custom_ops = manager.get_custom_operations()
        assert len(custom_ops) > 0
        
        print("   ✅ ModelOpsManager working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ ModelOpsManager test failed: {e}")
        return False

def test_model_transforms_manager():
    """Test ModelTransformsManager functionality"""
    try:
        print("🔄 Testing ModelTransformsManager...")
        
        from brainsmith.finn.model_transforms_manager import ModelTransformsManager
        
        manager = ModelTransformsManager()
        
        # Test basic configuration
        config = manager.configure(
            optimization_level='standard',
            target_platform='zynq',
            performance_targets={'throughput': 500, 'accuracy': 0.85}
        )
        
        # Validate configuration
        assert config.optimization_level == 'standard'
        assert config.target_platform == 'zynq'
        assert len(config.transforms_sequence) > 0
        assert 'weights_bitwidth' in config.quantization_config
        assert len(config.graph_optimizations) > 0
        
        # Test aggressive optimization
        aggressive_config = manager.configure(
            optimization_level='aggressive',
            target_platform='ultrascale',
            performance_targets={'throughput': 2000}
        )
        assert len(aggressive_config.transforms_sequence) > len(config.transforms_sequence)
        
        # Test registry queries
        levels = manager.get_available_optimization_levels()
        assert 'standard' in levels
        assert 'aggressive' in levels
        
        platforms = manager.get_supported_platforms()
        assert 'zynq' in platforms
        assert 'ultrascale' in platforms
        
        print("   ✅ ModelTransformsManager working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ ModelTransformsManager test failed: {e}")
        return False

def test_hw_kernels_manager():
    """Test HwKernelsManager functionality"""
    try:
        print("⚙️ Testing HwKernelsManager...")
        
        from brainsmith.finn.hw_kernels_manager import HwKernelsManager
        
        manager = HwKernelsManager()
        
        # Test basic configuration
        kernel_plan = {
            'conv1': 'ConvolutionInputGenerator',
            'fc1': 'StreamingFCLayer',
            'activation1': 'VectorVectorActivation'
        }
        
        config = manager.configure(
            kernel_selection_plan=kernel_plan,
            resource_constraints={'luts': 0.7, 'dsps': 0.8},
            custom_kernels={
                'CustomKernel': {
                    'backend': 'hls',
                    'implementation_path': 'custom.kernel'
                }
            }
        )
        
        # Validate configuration
        assert len(config.kernel_selection_plan) == 3
        assert 'conv1' in config.kernel_selection_plan
        assert 'CustomKernel' in config.custom_kernels
        assert len(config.folding_config) == 3  # One for each layer
        
        # Check folding configuration has required fields
        for layer_name, folding in config.folding_config.items():
            assert 'PE' in folding or 'SIMD' in folding
            assert 'mem_mode' in folding
        
        # Test registry queries
        available_kernels = manager.get_available_kernels()
        assert 'StreamingFCLayer' in available_kernels
        assert 'ConvolutionInputGenerator' in available_kernels
        
        # Test kernel info
        kernel_info = manager.get_kernel_info('StreamingFCLayer')
        assert 'backend' in kernel_info
        assert 'parameters' in kernel_info
        
        print("   ✅ HwKernelsManager working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ HwKernelsManager test failed: {e}")
        return False

def test_hw_optimization_manager():
    """Test HwOptimizationManager functionality"""
    try:
        print("📊 Testing HwOptimizationManager...")
        
        from brainsmith.finn.hw_optimization_manager import HwOptimizationManager
        
        manager = HwOptimizationManager()
        
        # Test basic configuration
        config = manager.configure(
            optimization_strategy='balanced',
            performance_targets={'throughput': 1000, 'latency': 25},
            power_constraints={'max_power': 15.0}
        )
        
        # Validate configuration
        assert config.optimization_strategy == 'balanced'
        assert 'throughput' in config.performance_targets
        assert 'max_power' in config.power_constraints
        assert len(config.timing_constraints) > 0
        assert len(config.resource_constraints) > 0
        
        # Test different strategies
        throughput_config = manager.configure(optimization_strategy='throughput')
        assert throughput_config.optimization_strategy == 'throughput'
        
        area_config = manager.configure(optimization_strategy='area')
        assert area_config.optimization_strategy == 'area'
        
        # Test registry queries
        strategies = manager.get_available_strategies()
        assert 'balanced' in strategies
        assert 'throughput' in strategies
        assert 'area' in strategies
        
        # Test strategy info
        strategy_info = manager.get_strategy_info('throughput')
        assert 'focus' in strategy_info
        assert 'techniques' in strategy_info
        
        # Test recommendations
        recommended = manager.get_recommended_strategy(
            'edge_inference',
            {'max_power': 3.0}
        )
        assert recommended in strategies
        
        print("   ✅ HwOptimizationManager working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ HwOptimizationManager test failed: {e}")
        return False

def test_managers_integration():
    """Test integration between all managers"""
    try:
        print("🔗 Testing Managers Integration...")
        
        from brainsmith.finn.model_ops_manager import ModelOpsManager
        from brainsmith.finn.model_transforms_manager import ModelTransformsManager
        from brainsmith.finn.hw_kernels_manager import HwKernelsManager
        from brainsmith.finn.hw_optimization_manager import HwOptimizationManager
        from brainsmith.finn.types import FINNInterfaceConfig
        
        # Create all managers
        ops_manager = ModelOpsManager()
        transforms_manager = ModelTransformsManager()
        kernels_manager = HwKernelsManager()
        optimization_manager = HwOptimizationManager()
        
        # Configure each component
        ops_config = ops_manager.configure(
            supported_ops=['Conv', 'MatMul', 'Relu'],
            frontend_cleanup=['RemoveUnusedNodes']
        )
        
        transforms_config = transforms_manager.configure(
            optimization_level='standard',
            target_platform='zynq'
        )
        
        kernels_config = kernels_manager.configure(
            kernel_selection_plan={
                'conv1': 'ConvolutionInputGenerator',
                'fc1': 'StreamingFCLayer'
            },
            resource_constraints={'luts': 0.8}
        )
        
        optimization_config = optimization_manager.configure(
            optimization_strategy='balanced',
            performance_targets={'throughput': 500}
        )
        
        # Create integrated FINN interface config
        finn_config = FINNInterfaceConfig(
            model_ops=ops_config,
            model_transforms=transforms_config,
            hw_kernels=kernels_config,
            hw_optimization=optimization_config,
            metadata={'created_by': 'test', 'version': '1.0'}
        )
        
        # Test serialization
        config_dict = finn_config.to_dict()
        assert 'model_ops' in config_dict
        assert 'model_transforms' in config_dict
        assert 'hw_kernels' in config_dict
        assert 'hw_optimization' in config_dict
        assert 'metadata' in config_dict
        
        # Test copy functionality
        finn_config_copy = finn_config.copy()
        assert finn_config_copy.model_ops.supported_ops == finn_config.model_ops.supported_ops
        
        print("   ✅ Managers integration working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Managers integration test failed: {e}")
        return False

def run_manager_tests():
    """Run all manager tests"""
    print("🧪 Week 2 FINN Integration Engine Managers Testing")
    print("=" * 60)
    print("Testing the four-category interface managers")
    print("=" * 60)
    
    tests = [
        ("ModelOpsManager", test_model_ops_manager),
        ("ModelTransformsManager", test_model_transforms_manager),
        ("HwKernelsManager", test_hw_kernels_manager),
        ("HwOptimizationManager", test_hw_optimization_manager),
        ("Managers Integration", test_managers_integration)
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
    print(f"FINN MANAGERS TEST RESULTS")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print(f"\n🎉 ALL FINN MANAGER TESTS PASSED!")
        print(f"✅ Four-category interface managers are working correctly")
        print(f"✅ Ready for FINN Integration Engine implementation")
    else:
        print(f"\n⚠️  Some manager tests failed - need to fix before proceeding")
    
    return failed == 0

if __name__ == '__main__':
    success = run_manager_tests()
    sys.exit(0 if success else 1)