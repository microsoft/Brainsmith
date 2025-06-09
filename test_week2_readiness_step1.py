"""
Week 2 Readiness Validation - Step 1: Basic Interface Testing
Test fundamental interfaces needed for FINN Integration Engine.
"""

import os
import sys
import json
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_finn_config_generation():
    """Test basic FINN configuration generation capability."""
    try:
        print("🔧 Step 1: Testing Basic FINN Configuration Generation...")
        
        from brainsmith.kernels import create_kernel_registry, generate_finn_config_for_model
        
        # Create minimal registry
        registry = create_kernel_registry()
        
        # Add a simple test kernel
        from brainsmith.kernels.database import (
            FINNKernelInfo, OperatorType, BackendType, ParameterSchema,
            ResourceRequirements, PerformanceModel
        )
        
        test_kernel = FINNKernelInfo(
            name='simple_test_kernel',
            operator_type=OperatorType.CONVOLUTION,
            backend_type=BackendType.HLS,
            implementation_files={'hls': '/test/simple_conv.cpp'},
            parameterization=ParameterSchema(
                pe_range=(1, 16),
                simd_range=(1, 8),
                supported_datatypes=['int8'],
                memory_modes=['internal'],
                folding_factors={}
            ),
            performance_model=PerformanceModel(
                model_type="analytical",
                throughput_model={'cycles_per_op': 1},
                latency_model={'base_latency': 5},
                power_model={'base_power': 1.0}
            ),
            resource_requirements=ResourceRequirements(
                lut_count=1000,
                ff_count=2000,
                dsp_count=2,
                bram_count=1
            ),
            finn_version_compatibility=['0.8+']
        )
        
        result = registry.register_kernel(test_kernel)
        assert result.success, f"Failed to register test kernel: {result.message}"
        print(f"   ✅ Test kernel registered: {test_kernel.name}")
        
        # Create simple test model
        simple_model = {
            'layers': [
                {
                    'name': 'test_conv',
                    'type': 'Conv2d',
                    'input_shape': [1, 3, 8, 8],
                    'output_shape': [1, 8, 8, 8],
                    'parameters': {
                        'kernel_size': [3, 3],
                        'weight_shape': [8, 3, 3, 3]
                    },
                    'data_type': 'int8'
                }
            ],
            'connections': []
        }
        
        # Generate configuration
        finn_config = generate_finn_config_for_model(
            simple_model,
            registry,
            performance_targets={'throughput': 1000},
            resource_constraints={'luts': 50000}
        )
        
        # Basic validation
        assert finn_config is not None, "No FINN configuration generated"
        
        config_dict = finn_config.to_dict()
        assert 'model_ops' in config_dict, "Missing model_ops section"
        assert 'hw_kernels' in config_dict, "Missing hw_kernels section"
        assert 'build_settings' in config_dict, "Missing build_settings section"
        
        print(f"   ✅ FINN configuration generated successfully")
        print(f"   ✅ Configuration sections: {list(config_dict.keys())}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic FINN config generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_export():
    """Test configuration export for external build systems."""
    try:
        print("📄 Step 1b: Testing Configuration Export...")
        
        from brainsmith.kernels.finn_config import FINNBuildConfig, ModelOpsConfig, HwKernelsConfig
        
        # Create minimal config
        config = FINNBuildConfig()
        config.model_ops = ModelOpsConfig(
            supported_ops=['Conv', 'MatMul'],
            frontend_cleanup=['RemoveUnusedNodes']
        )
        config.hw_kernels = HwKernelsConfig(
            kernel_selection_plan={'test_layer': 'test_kernel'},
            kernel_options={'test_layer': {'PE': 4, 'SIMD': 2}}
        )
        
        # Test JSON export
        config_dict = config.to_dict()
        config_json = json.dumps(config_dict, indent=2)
        
        # Validate JSON is parseable
        parsed_config = json.loads(config_json)
        assert parsed_config == config_dict, "JSON serialization mismatch"
        
        print(f"   ✅ JSON serialization working: {len(config_json)} characters")
        
        # Test file export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            config_path = tmp_file.name
        
        try:
            config.save_to_file(config_path)
            
            # Verify file was created
            assert os.path.exists(config_path), "Config file not created"
            
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            assert loaded_config == config_dict, "File export/import mismatch"
            print(f"   ✅ File export working")
            
        finally:
            os.unlink(config_path)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration export test failed: {e}")
        return False

def test_basic_error_handling():
    """Test basic error handling capabilities."""
    try:
        print("🛠️ Step 1c: Testing Basic Error Handling...")
        
        from brainsmith.kernels.database import ParameterSchema
        from brainsmith.kernels.finn_config import FINNConfigValidator, FINNBuildConfig
        
        # Test parameter validation
        param_schema = ParameterSchema(
            pe_range=(1, 32),
            simd_range=(1, 16),
            supported_datatypes=['int8'],
            memory_modes=['internal'],
            folding_factors={}
        )
        
        # Test valid parameters
        valid_params = {'pe': 8, 'simd': 4, 'datatype': 'int8'}
        assert param_schema.validate_parameters(valid_params) == True, "Valid params rejected"
        
        # Test invalid parameters
        invalid_params = [
            {'pe': 0},           # Below range
            {'pe': 64},          # Above range
            {'datatype': 'float64'}  # Unsupported
        ]
        
        for params in invalid_params:
            assert param_schema.validate_parameters(params) == False, f"Invalid params accepted: {params}"
        
        print(f"   ✅ Parameter validation working")
        
        # Test configuration validation
        invalid_config = FINNBuildConfig()
        invalid_config.clock_frequency = -100  # Invalid
        
        validator = FINNConfigValidator()
        validation_result = validator.validate(invalid_config)
        
        assert not validation_result.is_valid, "Should reject invalid configuration"
        assert len(validation_result.errors) > 0, "Should report validation errors"
        
        print(f"   ✅ Configuration validation working: {len(validation_result.errors)} errors detected")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic error handling test failed: {e}")
        return False

def run_step1_tests():
    """Run Step 1 validation tests."""
    print("🧪 Week 2 Readiness Validation - Step 1")
    print("=" * 60)
    print("Testing basic interfaces for FINN Integration Engine")
    print("=" * 60)
    
    tests = [
        ("Basic FINN Config Generation", test_basic_finn_config_generation),
        ("Configuration Export", test_configuration_export),
        ("Basic Error Handling", test_basic_error_handling)
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
    print(f"STEP 1 RESULTS")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print(f"\n🎉 STEP 1 PASSED - Basic interfaces are ready!")
        print(f"✅ Ready to proceed to Step 2: Performance Model Testing")
    else:
        print(f"\n⚠️  Step 1 issues detected - need to fix before proceeding")
    
    return failed == 0

if __name__ == '__main__':
    success = run_step1_tests()
    sys.exit(0 if success else 1)