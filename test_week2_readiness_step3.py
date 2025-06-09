"""
Week 2 Readiness Validation - Step 3: Build Orchestration Testing
Test build orchestration interfaces needed for FINN Integration Engine.
"""

import os
import sys
import json
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_build_configuration_export():
    """Test build configuration export for external build systems."""
    try:
        print("🔧 Step 3a: Testing Build Configuration Export...")
        
        from brainsmith.kernels import create_kernel_registry, generate_finn_config_for_model
        
        # Create minimal setup
        registry = create_kernel_registry()
        
        test_model = {
            'layers': [
                {
                    'name': 'conv1',
                    'type': 'Conv2d',
                    'input_shape': [1, 3, 32, 32],
                    'output_shape': [1, 16, 32, 32],
                    'parameters': {'kernel_size': [3, 3], 'weight_shape': [16, 3, 3, 3]},
                    'data_type': 'int8'
                }
            ],
            'connections': []
        }
        
        config = generate_finn_config_for_model(
            test_model,
            registry,
            performance_targets={'throughput': 1000},
            resource_constraints={'luts': 50000}
        )
        
        # Test JSON serialization for build systems
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
            
            # Verify file was created and is valid
            assert os.path.exists(config_path), "Config file not created"
            
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            assert loaded_config == config_dict, "File export/import mismatch"
            print(f"   ✅ File export working")
            
        finally:
            os.unlink(config_path)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Build configuration export test failed: {e}")
        return False

def test_build_command_interface():
    """Test build command interface preparation."""
    try:
        print("⚙️ Step 3b: Testing Build Command Interface...")
        
        from brainsmith.kernels import generate_finn_config_for_model, create_kernel_registry
        
        registry = create_kernel_registry()
        
        test_model = {
            'layers': [
                {
                    'name': 'simple_op',
                    'type': 'Conv2d',
                    'input_shape': [1, 1, 8, 8],
                    'output_shape': [1, 1, 8, 8],
                    'parameters': {'kernel_size': [3, 3], 'weight_shape': [1, 1, 3, 3]},
                    'data_type': 'int8'
                }
            ],
            'connections': []
        }
        
        config = generate_finn_config_for_model(
            test_model,
            registry,
            performance_targets={'throughput': 500},
            resource_constraints={'luts': 25000}
        )
        
        # Test that configuration can be converted to build commands
        build_settings = config.to_dict()['build_settings']
        
        required_build_fields = [
            'output_dir', 'build_mode', 'target_device', 'clock_frequency'
        ]
        
        for field in required_build_fields:
            assert field in build_settings, f"Missing build field: {field}"
        
        # Test build command construction
        build_command_parts = []
        build_command_parts.append(f"--output_dir {build_settings['output_dir']}")
        build_command_parts.append(f"--build_mode {build_settings['build_mode']}")
        build_command_parts.append(f"--part {build_settings['target_device']}")
        build_command_parts.append(f"--clk_ns {1000 / build_settings['clock_frequency']}")
        
        build_command = "finn_build " + " ".join(build_command_parts)
        assert len(build_command) > 0, "Failed to construct build command"
        
        print(f"   ✅ Build command construction: {len(build_command)} characters")
        print(f"   ✅ Required build fields present: {len(required_build_fields)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Build command interface test failed: {e}")
        return False

def test_folding_configuration_export():
    """Test folding configuration export for FINN."""
    try:
        print("📊 Step 3c: Testing Folding Configuration Export...")
        
        from brainsmith.kernels import generate_finn_config_for_model, create_kernel_registry
        
        registry = create_kernel_registry()
        
        # Create model with multiple layers
        multi_layer_model = {
            'layers': [
                {
                    'name': 'conv1',
                    'type': 'Conv2d',
                    'input_shape': [1, 3, 16, 16],
                    'output_shape': [1, 8, 16, 16],
                    'parameters': {'kernel_size': [3, 3], 'weight_shape': [8, 3, 3, 3]},
                    'data_type': 'int8'
                },
                {
                    'name': 'relu1',
                    'type': 'Relu',
                    'input_shape': [1, 8, 16, 16],
                    'output_shape': [1, 8, 16, 16],
                    'parameters': {},
                    'data_type': 'int8'
                }
            ],
            'connections': [('conv1', 'relu1')]
        }
        
        config = generate_finn_config_for_model(
            multi_layer_model,
            registry,
            performance_targets={'throughput': 1500},
            resource_constraints={'luts': 75000}
        )
        
        # Test folding configuration export
        folding_configs = config.hw_kernels.kernel_options
        assert len(folding_configs) > 0, "No folding configurations generated"
        
        # Export folding as separate file (FINN requirement)
        folding_dict = {}
        for layer_name, layer_config in folding_configs.items():
            folding_dict[layer_name] = {
                'PE': layer_config['PE'],
                'SIMD': layer_config['SIMD'],
                'mem_mode': layer_config.get('mem_mode', 'internal'),
                'ram_style': layer_config.get('ram_style', 'auto')
            }
        
        folding_json = json.dumps(folding_dict, indent=2)
        assert len(folding_json) > 0, "Failed to export folding configuration"
        
        print(f"   ✅ Folding configuration export: {len(folding_dict)} layers")
        
        # Validate folding parameters
        for layer_name, layer_config in folding_dict.items():
            assert 'PE' in layer_config, f"Missing PE for {layer_name}"
            assert 'SIMD' in layer_config, f"Missing SIMD for {layer_name}"
            assert isinstance(layer_config['PE'], int), f"PE not integer for {layer_name}"
            assert isinstance(layer_config['SIMD'], int), f"SIMD not integer for {layer_name}"
            assert layer_config['PE'] > 0, f"Invalid PE for {layer_name}"
            assert layer_config['SIMD'] > 0, f"Invalid SIMD for {layer_name}"
        
        print(f"   ✅ Folding parameter validation passed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Folding configuration export test failed: {e}")
        return False

def test_configuration_validation():
    """Test configuration validation for build readiness."""
    try:
        print("✅ Step 3d: Testing Configuration Validation...")
        
        from brainsmith.kernels.finn_config import FINNBuildConfig, FINNConfigValidator
        from brainsmith.kernels.finn_config import ModelOpsConfig, HwKernelsConfig
        
        # Create valid configuration
        valid_config = FINNBuildConfig()
        valid_config.model_ops = ModelOpsConfig(
            supported_ops=['Conv', 'MatMul', 'Threshold'],
            frontend_cleanup=['RemoveUnusedNodes']
        )
        valid_config.hw_kernels = HwKernelsConfig(
            kernel_selection_plan={'layer1': 'kernel1'},
            kernel_options={'layer1': {'PE': 4, 'SIMD': 2}}
        )
        valid_config.clock_frequency = 100.0
        valid_config.target_device = 'xc7z020clg400-1'
        valid_config.build_mode = 'vivado'
        
        validator = FINNConfigValidator()
        validation_result = validator.validate(valid_config)
        
        assert validation_result.is_valid, f"Valid configuration rejected: {validation_result.errors}"
        
        print(f"   ✅ Valid configuration accepted")
        
        # Test invalid configuration
        invalid_config = FINNBuildConfig()
        invalid_config.clock_frequency = -100  # Invalid
        invalid_config.build_mode = "invalid_mode"  # Invalid
        
        validation_result = validator.validate(invalid_config)
        
        assert not validation_result.is_valid, "Should reject invalid configuration"
        assert len(validation_result.errors) > 0, "Should report validation errors"
        
        print(f"   ✅ Invalid configuration rejected: {len(validation_result.errors)} errors")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration validation test failed: {e}")
        return False

def run_step3_tests():
    """Run Step 3 validation tests."""
    print("🧪 Week 2 Readiness Validation - Step 3")
    print("=" * 60)
    print("Testing build orchestration interfaces")
    print("=" * 60)
    
    tests = [
        ("Build Configuration Export", test_build_configuration_export),
        ("Build Command Interface", test_build_command_interface),
        ("Folding Configuration Export", test_folding_configuration_export),
        ("Configuration Validation", test_configuration_validation)
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
    print(f"STEP 3 RESULTS")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print(f"\n🎉 STEP 3 PASSED - Build orchestration interfaces are ready!")
        print(f"✅ Ready to proceed to Step 4: Parameter Optimization Testing")
    else:
        print(f"\n⚠️  Step 3 issues detected - need to fix before proceeding")
    
    return failed == 0

if __name__ == '__main__':
    success = run_step3_tests()
    sys.exit(0 if success else 1)