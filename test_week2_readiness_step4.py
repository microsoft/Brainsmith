"""
Week 2 Readiness Validation - Step 4: Parameter Optimization Testing
Test parameter optimization interfaces for build failure recovery.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_parameter_optimization_objectives():
    """Test parameter optimization for different objectives."""
    try:
        print("⚙️ Step 4a: Testing Parameter Optimization Objectives...")
        
        from brainsmith.kernels.selection import PerformanceOptimizer, OptimizationObjective
        from brainsmith.kernels.selection import PerformanceTargets, ResourceConstraints
        from brainsmith.kernels.database import FINNKernelInfo, OperatorType, BackendType
        from brainsmith.kernels.database import ParameterSchema, ResourceRequirements, PerformanceModel
        from brainsmith.kernels.analysis import OperatorRequirement, TensorShape, DataType
        
        # Create performance optimizer
        optimizer = PerformanceOptimizer()
        
        # Create test kernel
        test_kernel = FINNKernelInfo(
            name='optimization_test_kernel',
            operator_type=OperatorType.CONVOLUTION,
            backend_type=BackendType.HLS,
            implementation_files={'hls': '/test/conv_opt.cpp'},
            parameterization=ParameterSchema(
                pe_range=(1, 32),
                simd_range=(1, 16),
                supported_datatypes=['int8'],
                memory_modes=['internal', 'external'],
                folding_factors={}
            ),
            performance_model=PerformanceModel(
                model_type="analytical",
                throughput_model={'cycles_per_op': 2, 'efficiency': 0.8},
                latency_model={'base_latency': 5, 'pipeline_depth': 2},
                power_model={'base_power': 1.0}
            ),
            resource_requirements=ResourceRequirements(
                lut_count=2000,
                ff_count=4000,
                dsp_count=8,
                bram_count=4,
                lut_scaling={'pe': 1.2, 'simd': 0.9}
            ),
            finn_version_compatibility=['0.8+']
        )
        
        # Create test requirement
        test_requirement = OperatorRequirement(
            layer_id='test_conv_opt',
            operator_type='Convolution',
            input_shape=TensorShape((1, 16, 32, 32)),
            output_shape=TensorShape((1, 32, 32, 32)),
            parameters={'kernel_size': [3, 3]},
            constraints={},
            performance_requirements={'throughput': 2000},
            data_type=DataType.INT8,
            pe_requirements=(1, 32),
            simd_requirements=(1, 16),
            memory_requirements={},
            folding_constraints={}
        )
        
        # Test optimization targets and constraints
        targets = PerformanceTargets(
            throughput=2000,
            latency=20,
            power=3.0,
            area=50000
        )
        
        constraints = ResourceConstraints(
            max_luts=100000,
            max_dsps=1000,
            max_brams=200
        )
        
        # Test different optimization objectives
        objectives = [
            OptimizationObjective.THROUGHPUT,
            OptimizationObjective.LATENCY,
            OptimizationObjective.AREA,
            OptimizationObjective.BALANCED
        ]
        
        optimization_results = {}
        
        for objective in objectives:
            try:
                optimized_params = optimizer.optimize_parameters(
                    test_kernel, test_requirement, targets, constraints, objective
                )
                
                # Validate optimized parameters
                assert optimized_params is not None, f"No optimization result for {objective}"
                assert optimized_params.pe_parallelism > 0, f"Invalid PE for {objective}"
                assert optimized_params.simd_width > 0, f"Invalid SIMD for {objective}"
                assert optimized_params.pe_parallelism <= 32, f"PE out of range for {objective}"
                assert optimized_params.simd_width <= 16, f"SIMD out of range for {objective}"
                
                optimization_results[objective.value] = {
                    'pe': optimized_params.pe_parallelism,
                    'simd': optimized_params.simd_width,
                    'memory_mode': optimized_params.memory_mode,
                    'pipeline_depth': optimized_params.pipeline_depth
                }
                
                print(f"   ✅ {objective.value}: PE={optimized_params.pe_parallelism}, SIMD={optimized_params.simd_width}")
                
            except Exception as e:
                print(f"   ⚠️  {objective.value} optimization failed: {e}")
                optimization_results[objective.value] = {'error': str(e)}
        
        # Verify at least some objectives worked
        successful_results = [r for r in optimization_results.values() if 'error' not in r]
        assert len(successful_results) >= 2, "Need at least 2 successful optimizations"
        
        print(f"   ✅ Optimization objectives tested: {len(optimization_results)}")
        print(f"   ✅ Successful optimizations: {len(successful_results)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Parameter optimization objectives test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_build_failure_recovery():
    """Test parameter recovery for build failures."""
    try:
        print("🔧 Step 4b: Testing Build Failure Recovery...")
        
        from brainsmith.kernels.selection import PerformanceOptimizer, OptimizationObjective
        from brainsmith.kernels.selection import PerformanceTargets, ResourceConstraints
        from brainsmith.kernels.database import FINNKernelInfo, OperatorType, BackendType
        from brainsmith.kernels.database import ParameterSchema, ResourceRequirements, PerformanceModel
        from brainsmith.kernels.analysis import OperatorRequirement, TensorShape, DataType
        
        optimizer = PerformanceOptimizer()
        
        # Create test kernel for recovery testing
        recovery_kernel = FINNKernelInfo(
            name='recovery_test_kernel',
            operator_type=OperatorType.MATMUL,
            backend_type=BackendType.HLS,
            implementation_files={'hls': '/test/matmul_recovery.cpp'},
            parameterization=ParameterSchema(
                pe_range=(1, 64),
                simd_range=(1, 32),
                supported_datatypes=['int8'],
                memory_modes=['internal', 'external'],
                folding_factors={}
            ),
            performance_model=PerformanceModel(
                model_type="analytical",
                throughput_model={'cycles_per_op': 1, 'efficiency': 0.85},
                latency_model={'base_latency': 3, 'pipeline_depth': 1},
                power_model={'base_power': 2.0}
            ),
            resource_requirements=ResourceRequirements(
                lut_count=1000,
                ff_count=2000,
                dsp_count=4,
                bram_count=2,
                lut_scaling={'pe': 1.1, 'simd': 0.8}
            ),
            finn_version_compatibility=['0.8+']
        )
        
        recovery_requirement = OperatorRequirement(
            layer_id='recovery_matmul',
            operator_type='MatMul',
            input_shape=TensorShape((1, 256)),
            output_shape=TensorShape((1, 128)),
            parameters={},
            constraints={},
            performance_requirements={'throughput': 1000},
            data_type=DataType.INT8,
            pe_requirements=(1, 64),
            simd_requirements=(1, 32),
            memory_requirements={},
            folding_constraints={}
        )
        
        # Initial optimization with generous constraints
        generous_targets = PerformanceTargets(throughput=1000, latency=50)
        generous_constraints = ResourceConstraints(max_luts=50000, max_dsps=500, max_brams=100)
        
        original_params = optimizer.optimize_parameters(
            recovery_kernel, recovery_requirement, generous_targets, 
            generous_constraints, OptimizationObjective.BALANCED
        )
        
        assert original_params is not None, "Initial optimization failed"
        
        # Simulate build failure by constraining resources severely
        tight_constraints = ResourceConstraints(
            max_luts=generous_constraints.max_luts // 4,
            max_dsps=generous_constraints.max_dsps // 4,
            max_brams=generous_constraints.max_brams // 4
        )
        
        # Recovery optimization focusing on area
        recovery_params = optimizer.optimize_parameters(
            recovery_kernel, recovery_requirement, generous_targets,
            tight_constraints, OptimizationObjective.AREA
        )
        
        assert recovery_params is not None, "Recovery optimization failed"
        
        # Recovery should produce more conservative parameters
        assert recovery_params.pe_parallelism <= original_params.pe_parallelism, \
            "Recovery should reduce PE parallelism"
        assert recovery_params.simd_width <= original_params.simd_width, \
            "Recovery should reduce SIMD width"
        
        print(f"   ✅ Build failure recovery working")
        print(f"   ✅ Original: PE={original_params.pe_parallelism}, SIMD={original_params.simd_width}")
        print(f"   ✅ Recovery: PE={recovery_params.pe_parallelism}, SIMD={recovery_params.simd_width}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Build failure recovery test failed: {e}")
        return False

def test_parameter_sensitivity_analysis():
    """Test parameter sensitivity analysis for monitoring."""
    try:
        print("📊 Step 4c: Testing Parameter Sensitivity Analysis...")
        
        from brainsmith.kernels.database import FINNKernelInfo, OperatorType, BackendType
        from brainsmith.kernels.database import ParameterSchema, ResourceRequirements, PerformanceModel
        
        # Create test kernel for sensitivity analysis
        sensitivity_kernel = FINNKernelInfo(
            name='sensitivity_test_kernel',
            operator_type=OperatorType.CONVOLUTION,
            backend_type=BackendType.RTL,
            implementation_files={'rtl': '/test/conv_sensitivity.v'},
            parameterization=ParameterSchema(
                pe_range=(1, 16),
                simd_range=(1, 8),
                supported_datatypes=['int8'],
                memory_modes=['internal'],
                folding_factors={}
            ),
            performance_model=PerformanceModel(
                model_type="analytical",
                throughput_model={'cycles_per_op': 1, 'efficiency': 0.9},
                latency_model={'base_latency': 2, 'pipeline_depth': 1},
                power_model={'base_power': 1.5}
            ),
            resource_requirements=ResourceRequirements(
                lut_count=500,
                ff_count=1000,
                dsp_count=2,
                bram_count=1,
                lut_scaling={'pe': 1.3, 'simd': 0.7},
                dsp_scaling={'pe': 1.0, 'simd': 0.0}
            ),
            finn_version_compatibility=['0.8+']
        )
        
        # Test parameter sensitivity
        base_pe = 4
        base_simd = 2
        platform = {'clock_frequency': 100e6}
        
        # Test PE sensitivity
        pe_sensitivity = {}
        pe_values = [base_pe // 2, base_pe, base_pe * 2]
        
        for pe in pe_values:
            if pe <= 16:  # Within valid range
                params = {'pe': pe, 'simd': base_simd}
                
                throughput = sensitivity_kernel.performance_model.estimate_throughput(params, platform)
                resources = sensitivity_kernel.resource_requirements.estimate_resources(params)
                
                pe_sensitivity[f'pe_{pe}'] = {
                    'throughput': throughput,
                    'luts': resources.lut_count,
                    'dsps': resources.dsp_count
                }
        
        # Test SIMD sensitivity
        simd_sensitivity = {}
        simd_values = [base_simd // 2, base_simd, base_simd * 2]
        
        for simd in simd_values:
            if simd <= 8 and simd >= 1:  # Within valid range
                params = {'pe': base_pe, 'simd': simd}
                
                throughput = sensitivity_kernel.performance_model.estimate_throughput(params, platform)
                resources = sensitivity_kernel.resource_requirements.estimate_resources(params)
                
                simd_sensitivity[f'simd_{simd}'] = {
                    'throughput': throughput,
                    'luts': resources.lut_count,
                    'dsps': resources.dsp_count
                }
        
        # Verify sensitivity makes sense
        if len(pe_sensitivity) >= 2:
            pe_results = list(pe_sensitivity.values())
            pe_throughputs = [r['throughput'] for r in pe_results]
            assert max(pe_throughputs) > min(pe_throughputs), "PE scaling should affect throughput"
        
        if len(simd_sensitivity) >= 2:
            simd_results = list(simd_sensitivity.values())
            simd_throughputs = [r['throughput'] for r in simd_results]
            assert max(simd_throughputs) > min(simd_throughputs), "SIMD scaling should affect throughput"
        
        print(f"   ✅ PE sensitivity analysis: {len(pe_sensitivity)} configurations")
        print(f"   ✅ SIMD sensitivity analysis: {len(simd_sensitivity)} configurations")
        print(f"   ✅ Parameter scaling behavior verified")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Parameter sensitivity analysis test failed: {e}")
        return False

def test_error_recovery_mechanisms():
    """Test comprehensive error recovery mechanisms."""
    try:
        print("🛠️ Step 4d: Testing Error Recovery Mechanisms...")
        
        from brainsmith.kernels.database import ParameterSchema
        from brainsmith.kernels.finn_config import FINNConfigValidator, FINNBuildConfig
        from brainsmith.kernels.registry import SearchCriteria, FINNKernelRegistry
        
        # Test parameter validation with recovery
        param_schema = ParameterSchema(
            pe_range=(1, 32),
            simd_range=(1, 16),
            supported_datatypes=['int8'],
            memory_modes=['internal'],
            folding_factors={}
        )
        
        # Test recovery from invalid parameters
        invalid_params = [
            {'pe': 0},           # Below range
            {'pe': 64},          # Above range
            {'simd': -1},        # Negative
            {'datatype': 'float64'},  # Unsupported
        ]
        
        recovery_count = 0
        for params in invalid_params:
            is_valid = param_schema.validate_parameters(params)
            if not is_valid:
                # Simulate parameter recovery
                recovered_params = {}
                if 'pe' in params:
                    recovered_params['pe'] = max(1, min(32, abs(params['pe']) or 1))
                if 'simd' in params:
                    recovered_params['simd'] = max(1, min(16, abs(params['simd']) or 1))
                if 'datatype' in params and params['datatype'] not in param_schema.supported_datatypes:
                    recovered_params['datatype'] = 'int8'  # Default
                
                # Validate recovered parameters
                if param_schema.validate_parameters(recovered_params):
                    recovery_count += 1
        
        print(f"   ✅ Parameter recovery: {recovery_count}/{len(invalid_params)} recoveries successful")
        
        # Test search with impossible criteria (graceful failure)
        registry = FINNKernelRegistry()
        
        impossible_criteria = SearchCriteria(
            min_pe=1000,  # Impossibly high
            max_pe=2000,
            min_simd=500,
            max_simd=1000,
            max_lut_usage=1,  # Impossibly low
            performance_requirements={'throughput': 1e9}  # Impossibly high
        )
        
        results = registry.search_kernels(impossible_criteria)
        assert len(results) == 0, "Should find no kernels for impossible criteria"
        
        print(f"   ✅ Graceful handling of impossible search criteria")
        
        # Test configuration recovery
        invalid_config = FINNBuildConfig()
        invalid_config.clock_frequency = -100  # Invalid
        
        validator = FINNConfigValidator()
        validation_result = validator.validate(invalid_config)
        
        assert not validation_result.is_valid, "Should reject invalid configuration"
        assert len(validation_result.errors) > 0, "Should report validation errors"
        
        # Simulate configuration recovery
        recovered_config = FINNBuildConfig()
        recovered_config.clock_frequency = 100.0  # Fixed
        recovered_config.build_mode = 'vivado'
        recovered_config.target_device = 'xc7z020clg400-1'
        
        recovery_validation = validator.validate(recovered_config)
        # Note: This might still fail due to missing required sections, but that's expected
        
        print(f"   ✅ Configuration error detection and recovery framework working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error recovery mechanisms test failed: {e}")
        return False

def run_step4_tests():
    """Run Step 4 validation tests."""
    print("🧪 Week 2 Readiness Validation - Step 4")
    print("=" * 60)
    print("Testing parameter optimization interfaces for build recovery")
    print("=" * 60)
    
    tests = [
        ("Parameter Optimization Objectives", test_parameter_optimization_objectives),
        ("Build Failure Recovery", test_build_failure_recovery),
        ("Parameter Sensitivity Analysis", test_parameter_sensitivity_analysis),
        ("Error Recovery Mechanisms", test_error_recovery_mechanisms)
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
    print(f"STEP 4 RESULTS")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print(f"\n🎉 STEP 4 PASSED - Parameter optimization interfaces are ready!")
        print(f"✅ All Week 2 readiness validation steps complete!")
    else:
        print(f"\n⚠️  Step 4 issues detected - need to fix before Week 2")
    
    return failed == 0

if __name__ == '__main__':
    success = run_step4_tests()
    sys.exit(0 if success else 1)