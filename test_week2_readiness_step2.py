"""
Week 2 Readiness Validation - Step 2: Performance Model Testing
Test performance model accuracy and interfaces for build result validation.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_performance_model_accuracy():
    """Test performance model accuracy and scaling behavior."""
    try:
        print("📊 Step 2a: Testing Performance Model Accuracy...")
        
        from brainsmith.kernels.database import PerformanceModel, ResourceRequirements
        
        # Create analytical performance model
        perf_model = PerformanceModel(
            model_type="analytical",
            throughput_model={
                'cycles_per_op': 2,
                'base_throughput': 1000,
                'efficiency': 0.85
            },
            latency_model={
                'base_latency': 10,
                'pipeline_depth': 3,
                'setup_cycles': 5
            },
            power_model={
                'base_power': 1.5,
                'dynamic_factor': 1.2,
                'frequency_scaling': 0.8
            }
        )
        
        # Test different parameter configurations
        test_configs = [
            {'pe': 1, 'simd': 1},
            {'pe': 4, 'simd': 2},
            {'pe': 16, 'simd': 8},
            {'pe': 32, 'simd': 16}
        ]
        
        platform_configs = [
            {'clock_frequency': 100e6, 'device': 'zynq'},
            {'clock_frequency': 200e6, 'device': 'ultrascale'}
        ]
        
        performance_results = []
        
        for params in test_configs:
            for platform in platform_configs:
                # Estimate performance
                throughput = perf_model.estimate_throughput(params, platform)
                latency = perf_model.estimate_latency(params, platform)
                
                # Validate results are reasonable
                assert throughput > 0, f"Invalid throughput: {throughput}"
                assert latency > 0, f"Invalid latency: {latency}"
                assert throughput < 1e8, f"Unrealistic throughput: {throughput}"
                assert latency < 10000, f"Unrealistic latency: {latency}"
                
                performance_results.append({
                    'params': params,
                    'platform': platform,
                    'throughput': throughput,
                    'latency': latency
                })
        
        print(f"   ✅ Performance model validation passed")
        print(f"   ✅ Tested {len(test_configs)} parameter configurations")
        print(f"   ✅ Generated {len(performance_results)} performance estimates")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance model accuracy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_resource_estimation():
    """Test resource estimation accuracy and scaling."""
    try:
        print("🔧 Step 2b: Testing Resource Estimation...")
        
        from brainsmith.kernels.database import ResourceRequirements
        
        # Test resource estimation with scaling
        resource_model = ResourceRequirements(
            lut_count=1000,
            ff_count=2000,
            dsp_count=4,
            bram_count=2,
            lut_scaling={'pe': 1.1, 'simd': 0.9},
            dsp_scaling={'pe': 1.0, 'simd': 0.1},
            bram_scaling={'pe': 0.3, 'simd': 0.2}
        )
        
        test_configs = [
            {'pe': 1, 'simd': 1},
            {'pe': 4, 'simd': 2},
            {'pe': 8, 'simd': 4},
            {'pe': 16, 'simd': 8}
        ]
        
        for params in test_configs:
            estimated = resource_model.estimate_resources(params)
            
            # Validate resource estimates
            assert estimated.lut_count >= resource_model.lut_count, "LUT count decreased"
            assert estimated.dsp_count >= resource_model.dsp_count, "DSP count decreased"
            
            # Check scaling makes sense
            pe_factor = params['pe']
            simd_factor = params['simd']
            
            expected_luts = resource_model.lut_count * pe_factor * simd_factor
            assert estimated.lut_count <= expected_luts * 2, f"LUT scaling too aggressive: {estimated.lut_count}"
            
            expected_dsps = resource_model.dsp_count * pe_factor
            assert estimated.dsp_count <= expected_dsps * 1.5, f"DSP scaling too aggressive: {estimated.dsp_count}"
        
        print(f"   ✅ Resource scaling validation passed")
        print(f"   ✅ Tested {len(test_configs)} scaling configurations")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Resource estimation test failed: {e}")
        return False

def test_kernel_performance_integration():
    """Test kernel performance model integration."""
    try:
        print("⚡ Step 2c: Testing Kernel Performance Integration...")
        
        from brainsmith.kernels.database import FINNKernelInfo, OperatorType, BackendType
        from brainsmith.kernels.database import ParameterSchema, ResourceRequirements, PerformanceModel
        
        # Create test kernel with performance model
        test_kernel = FINNKernelInfo(
            name='perf_test_kernel',
            operator_type=OperatorType.MATMUL,
            backend_type=BackendType.HLS,
            implementation_files={'hls': '/test/matmul.cpp'},
            parameterization=ParameterSchema(
                pe_range=(1, 32),
                simd_range=(1, 16),
                supported_datatypes=['int8'],
                memory_modes=['internal'],
                folding_factors={}
            ),
            performance_model=PerformanceModel(
                model_type="analytical",
                throughput_model={'cycles_per_op': 1, 'efficiency': 0.8},
                latency_model={'base_latency': 3, 'pipeline_depth': 2},
                power_model={'base_power': 2.0}
            ),
            resource_requirements=ResourceRequirements(
                lut_count=1500,
                ff_count=3000,
                dsp_count=4,
                bram_count=2
            ),
            finn_version_compatibility=['0.8+']
        )
        
        # Test performance estimation through kernel interface
        test_params = [
            {'pe': 4, 'simd': 2},
            {'pe': 8, 'simd': 4},
            {'pe': 16, 'simd': 8}
        ]
        
        platform = {'clock_frequency': 100e6}
        
        for params in test_params:
            perf_estimate = test_kernel.estimate_performance(params, platform)
            
            # Validate estimates
            assert 'throughput' in perf_estimate, "Missing throughput estimate"
            assert 'latency' in perf_estimate, "Missing latency estimate"
            assert perf_estimate['throughput'] > 0, "Invalid throughput"
            assert perf_estimate['latency'] > 0, "Invalid latency"
        
        print(f"   ✅ Kernel performance integration working")
        print(f"   ✅ Tested {len(test_params)} parameter combinations")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Kernel performance integration test failed: {e}")
        return False

def run_step2_tests():
    """Run Step 2 validation tests."""
    print("🧪 Week 2 Readiness Validation - Step 2")
    print("=" * 60)
    print("Testing performance model interfaces for build validation")
    print("=" * 60)
    
    tests = [
        ("Performance Model Accuracy", test_performance_model_accuracy),
        ("Resource Estimation", test_resource_estimation),
        ("Kernel Performance Integration", test_kernel_performance_integration)
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
    print(f"STEP 2 RESULTS")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print(f"\n🎉 STEP 2 PASSED - Performance models are ready!")
        print(f"✅ Ready to proceed to Step 3: Build Orchestration Testing")
    else:
        print(f"\n⚠️  Step 2 issues detected - need to fix before proceeding")
    
    return failed == 0

if __name__ == '__main__':
    success = run_step2_tests()
    sys.exit(0 if success else 1)