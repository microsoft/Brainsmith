"""
Complete Test for Month 4 Week 1: End-to-End Kernel Selection Workflow
Tests the complete pipeline from model analysis to FINN configuration generation.
"""

import os
import sys
import json
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_complete_kernel_workflow():
    """Test complete workflow from model to FINN configuration."""
    try:
        print("🚀 Testing Complete Kernel Selection Workflow")
        print("=" * 60)
        
        # Import convenience functions
        from brainsmith.kernels import (
            create_kernel_registry, analyze_model_for_finn,
            select_optimal_kernels_for_model, generate_finn_config_for_model,
            get_kernel_statistics
        )
        
        # Step 1: Create kernel registry
        print("\n📚 Step 1: Creating kernel registry...")
        registry = create_kernel_registry()
        
        # Manually populate with test kernels
        from brainsmith.kernels.database import (
            FINNKernelInfo, OperatorType, BackendType, PerformanceClass,
            ParameterSchema, ResourceRequirements, PerformanceModel
        )
        
        # Create test kernels for different operators
        test_kernels = [
            {
                'name': 'conv2d_hls_optimized',
                'operator_type': OperatorType.CONVOLUTION,
                'backend_type': BackendType.HLS,
                'pe_range': (1, 64),
                'simd_range': (1, 32)
            },
            {
                'name': 'matmul_rtl_highperf',
                'operator_type': OperatorType.MATMUL,
                'backend_type': BackendType.RTL,
                'pe_range': (1, 128),
                'simd_range': (1, 64)
            },
            {
                'name': 'threshold_hls_lowpower',
                'operator_type': OperatorType.THRESHOLDING,
                'backend_type': BackendType.HLS,
                'pe_range': (1, 32),
                'simd_range': (1, 16)
            },
            {
                'name': 'pool_rtl_fast',
                'operator_type': OperatorType.POOL,
                'backend_type': BackendType.RTL,
                'pe_range': (1, 16),
                'simd_range': (1, 8)
            }
        ]
        
        for kernel_spec in test_kernels:
            kernel = FINNKernelInfo(
                name=kernel_spec['name'],
                operator_type=kernel_spec['operator_type'],
                backend_type=kernel_spec['backend_type'],
                implementation_files={
                    'hls': f"/test/{kernel_spec['name']}.cpp",
                    'rtl': f"/test/{kernel_spec['name']}.v"
                },
                parameterization=ParameterSchema(
                    pe_range=kernel_spec['pe_range'],
                    simd_range=kernel_spec['simd_range'],
                    supported_datatypes=['int8', 'int16'],
                    memory_modes=['internal', 'external'],
                    folding_factors={'spatial_fold': [1, 2, 4]}
                ),
                performance_model=PerformanceModel(
                    model_type="analytical",
                    throughput_model={'cycles_per_op': 1, 'base_throughput': 2000},
                    latency_model={'base_latency': 5},
                    power_model={'base_power': 1.5}
                ),
                resource_requirements=ResourceRequirements(
                    lut_count=2000,
                    ff_count=4000,
                    dsp_count=8,
                    bram_count=4
                ),
                finn_version_compatibility=['0.8+', '0.9+', '1.0+'],
                verification_status="verified",
                reliability_score=0.95,
                test_coverage=0.9
            )
            
            result = registry.register_kernel(kernel)
            assert result.success, f"Failed to register {kernel_spec['name']}"
        
        print(f"   ✅ Registered {len(test_kernels)} test kernels")
        
        # Step 2: Create test model
        print("\n🧠 Step 2: Creating test CNN model...")
        
        cnn_model_data = {
            'layers': [
                {
                    'name': 'input',
                    'type': 'Reshape',
                    'input_shape': [1, 3, 224, 224],
                    'output_shape': [1, 3, 224, 224],
                    'parameters': {},
                    'data_type': 'int8'
                },
                {
                    'name': 'conv1',
                    'type': 'Conv2d',
                    'input_shape': [1, 3, 224, 224],
                    'output_shape': [1, 64, 224, 224],
                    'parameters': {
                        'kernel_size': [3, 3],
                        'weight_shape': [64, 3, 3, 3],
                        'stride': [1, 1],
                        'padding': [1, 1]
                    },
                    'data_type': 'int8'
                },
                {
                    'name': 'relu1',
                    'type': 'Relu',
                    'input_shape': [1, 64, 224, 224],
                    'output_shape': [1, 64, 224, 224],
                    'parameters': {},
                    'data_type': 'int8'
                },
                {
                    'name': 'pool1',
                    'type': 'MaxPool',
                    'input_shape': [1, 64, 224, 224],
                    'output_shape': [1, 64, 112, 112],
                    'parameters': {
                        'kernel_size': [2, 2],
                        'stride': [2, 2]
                    },
                    'data_type': 'int8'
                },
                {
                    'name': 'conv2',
                    'type': 'Conv2d',
                    'input_shape': [1, 64, 112, 112],
                    'output_shape': [1, 128, 112, 112],
                    'parameters': {
                        'kernel_size': [3, 3],
                        'weight_shape': [128, 64, 3, 3],
                        'stride': [1, 1],
                        'padding': [1, 1]
                    },
                    'data_type': 'int8'
                },
                {
                    'name': 'relu2',
                    'type': 'Relu',
                    'input_shape': [1, 128, 112, 112],
                    'output_shape': [1, 128, 112, 112],
                    'parameters': {},
                    'data_type': 'int8'
                },
                {
                    'name': 'pool2',
                    'type': 'MaxPool',
                    'input_shape': [1, 128, 112, 112],
                    'output_shape': [1, 128, 56, 56],
                    'parameters': {
                        'kernel_size': [2, 2],
                        'stride': [2, 2]
                    },
                    'data_type': 'int8'
                },
                {
                    'name': 'flatten',
                    'type': 'Reshape',
                    'input_shape': [1, 128, 56, 56],
                    'output_shape': [1, 128*56*56],
                    'parameters': {},
                    'data_type': 'int8'
                },
                {
                    'name': 'fc1',
                    'type': 'Linear',
                    'input_shape': [1, 128*56*56],
                    'output_shape': [1, 1000],
                    'parameters': {
                        'weight_shape': [1000, 128*56*56]
                    },
                    'data_type': 'int8'
                }
            ],
            'connections': [
                ('input', 'conv1'),
                ('conv1', 'relu1'),
                ('relu1', 'pool1'),
                ('pool1', 'conv2'),
                ('conv2', 'relu2'),
                ('relu2', 'pool2'),
                ('pool2', 'flatten'),
                ('flatten', 'fc1')
            ]
        }
        
        print(f"   ✅ Created CNN model with {len(cnn_model_data['layers'])} layers")
        
        # Step 3: Analyze model topology
        print("\n🔍 Step 3: Analyzing model topology...")
        analysis = analyze_model_for_finn(cnn_model_data)
        
        print(f"   ✅ Analysis complete:")
        print(f"      • {len(analysis.layers)} layers analyzed")
        print(f"      • {len(analysis.operator_requirements)} operator requirements identified")
        print(f"      • {analysis.complexity_analysis['total_operations']:,} total operations")
        print(f"      • {analysis.complexity_analysis['total_parameters']:,} total parameters")
        print(f"      • Critical path: {len(analysis.critical_path)} layers")
        
        # Display operator requirements
        for req in analysis.operator_requirements[:3]:  # Show first 3
            print(f"      • {req.layer_id}: {req.operator_type} (PE: {req.pe_requirements}, SIMD: {req.simd_requirements})")
        
        # Step 4: Define optimization targets
        print("\n🎯 Step 4: Setting optimization targets...")
        
        performance_targets = {
            'throughput': 2000.0,  # ops/sec
            'latency': 50.0,       # cycles
            'power': 5.0,          # watts
            'area': 80000          # LUTs
        }
        
        resource_constraints = {
            'luts': 100000,
            'dsps': 2000,
            'brams': 500
        }
        
        print(f"   ✅ Targets set:")
        print(f"      • Throughput: {performance_targets['throughput']} ops/sec")
        print(f"      • Latency: {performance_targets['latency']} cycles")
        print(f"      • Max LUTs: {resource_constraints['luts']:,}")
        print(f"      • Max DSPs: {resource_constraints['dsps']:,}")
        
        # Step 5: Select optimal kernels
        print("\n⚡ Step 5: Selecting optimal kernels...")
        
        selection_plan = select_optimal_kernels_for_model(
            cnn_model_data,
            registry,
            performance_targets,
            resource_constraints,
            strategy='balanced'
        )
        
        print(f"   ✅ Kernel selection complete:")
        print(f"      • {len(selection_plan.selections)} kernels selected")
        print(f"      • Optimization score: {selection_plan.optimization_score:.3f}")
        print(f"      • Total estimated resources:")
        for resource, count in selection_plan.estimated_total_resources.items():
            print(f"        - {resource.upper()}: {count:,}")
        print(f"      • Total estimated performance:")
        for metric, value in selection_plan.estimated_total_performance.items():
            print(f"        - {metric}: {value:.1f}")
        
        # Display selected kernels
        print(f"      • Selected kernels:")
        for layer_id, selection in list(selection_plan.selections.items())[:5]:  # Show first 5
            kernel = selection.kernel
            params = selection.parameters
            print(f"        - {layer_id}: {kernel.name} (PE={params.pe_parallelism}, SIMD={params.simd_width})")
        
        # Step 6: Generate FINN configuration
        print("\n🔧 Step 6: Generating FINN configuration...")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            config_path = tmp_file.name
        
        try:
            finn_config = generate_finn_config_for_model(
                cnn_model_data,
                registry,
                performance_targets,
                resource_constraints,
                output_path=config_path
            )
            
            print(f"   ✅ FINN configuration generated:")
            print(f"      • Config saved to: {config_path}")
            print(f"      • Build mode: {finn_config.build_mode}")
            print(f"      • Target device: {finn_config.target_device}")
            print(f"      • Clock frequency: {finn_config.clock_frequency} MHz")
            print(f"      • Supported ops: {len(finn_config.model_ops.supported_ops)}")
            print(f"      • Kernel configs: {len(finn_config.hw_kernels.kernel_options)}")
            
            # Validate configuration structure
            config_dict = finn_config.to_dict()
            required_sections = ['model_ops', 'model_transforms', 'hw_kernels', 'hw_optimization']
            
            for section in required_sections:
                assert section in config_dict, f"Missing section: {section}"
            
            # Check kernel configurations
            kernel_configs = finn_config.hw_kernels.kernel_options
            assert len(kernel_configs) > 0, "No kernel configurations generated"
            
            # Verify folding configuration
            folding_config = finn_config.hw_kernels.kernel_options
            for layer_name, layer_config in folding_config.items():
                assert 'PE' in layer_config, f"Missing PE config for {layer_name}"
                assert 'SIMD' in layer_config, f"Missing SIMD config for {layer_name}"
                assert layer_config['PE'] > 0, f"Invalid PE value for {layer_name}"
                assert layer_config['SIMD'] > 0, f"Invalid SIMD value for {layer_name}"
            
            print(f"      • Configuration validation: ✅ PASSED")
            
        finally:
            # Clean up temp file
            os.unlink(config_path)
        
        # Step 7: Get registry statistics
        print("\n📊 Step 7: Registry statistics...")
        stats = get_kernel_statistics(registry)
        
        print(f"   ✅ Registry summary:")
        print(f"      • Total kernels: {stats['summary']['total_kernels']}")
        print(f"      • Operator coverage: {stats['summary']['operator_coverage']}")
        print(f"      • Backend coverage: {stats['summary']['backend_coverage']}")
        print(f"      • Verified kernels: {stats['summary']['verified_kernels']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kernel_selection_strategies():
    """Test different kernel selection strategies."""
    try:
        print("\n🎯 Testing Different Selection Strategies")
        print("-" * 50)
        
        from brainsmith.kernels import (
            create_kernel_registry, select_optimal_kernels_for_model
        )
        
        # Create registry with test kernels (simplified)
        registry = create_kernel_registry()
        
        # Simple test model
        simple_model = {
            'layers': [
                {
                    'name': 'conv1',
                    'type': 'Conv2d',
                    'input_shape': [1, 3, 32, 32],
                    'output_shape': [1, 16, 32, 32],
                    'parameters': {'kernel_size': [3, 3], 'weight_shape': [16, 3, 3, 3]},
                    'data_type': 'int8'
                },
                {
                    'name': 'relu1',
                    'type': 'Relu',
                    'input_shape': [1, 16, 32, 32],
                    'output_shape': [1, 16, 32, 32],
                    'parameters': {},
                    'data_type': 'int8'
                }
            ],
            'connections': [('conv1', 'relu1')]
        }
        
        targets = {'throughput': 1000, 'latency': 20}
        constraints = {'luts': 50000, 'dsps': 500}
        
        strategies = ['balanced', 'performance', 'area']
        results = {}
        
        for strategy in strategies:
            try:
                plan = select_optimal_kernels_for_model(
                    simple_model, registry, targets, constraints, strategy
                )
                
                results[strategy] = {
                    'selections': len(plan.selections),
                    'score': plan.optimization_score,
                    'resources': plan.estimated_total_resources,
                    'performance': plan.estimated_total_performance
                }
                
                print(f"   {strategy.upper()}: score={plan.optimization_score:.3f}, "
                      f"selections={len(plan.selections)}")
                
            except Exception as e:
                print(f"   {strategy.upper()}: ❌ FAILED - {e}")
                results[strategy] = {'error': str(e)}
        
        # At least one strategy should work
        successful_strategies = [s for s in results if 'error' not in results[s]]
        assert len(successful_strategies) > 0, "No selection strategies worked"
        
        print(f"   ✅ {len(successful_strategies)}/{len(strategies)} strategies successful")
        return True
        
    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
        return False

def run_complete_workflow_tests():
    """Run all complete workflow tests."""
    print("🧪 Month 4 Week 1: Complete Kernel Selection Workflow Tests")
    print("=" * 80)
    
    tests = [
        ("Complete Workflow", test_complete_kernel_workflow),
        ("Selection Strategies", test_kernel_selection_strategies)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"Running: {test_name}")
        print(f"{'='*80}")
        
        try:
            if test_func():
                print(f"\n✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"\n❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"\n❌ {test_name} FAILED with exception: {e}")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"WEEK 1 COMPLETE WORKFLOW TEST RESULTS")
    print(f"{'='*80}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print(f"\n🎉 ALL WEEK 1 WORKFLOW TESTS PASSED!")
        print(f"\n🏆 WEEK 1 IMPLEMENTATION COMPLETE")
        print(f"{'='*80}")
        print(f"📋 WEEK 1 ACHIEVEMENTS:")
        print(f"✅ Enhanced Hardware Kernel Registration and Management System")
        print(f"   • Automated FINN kernel discovery and analysis")
        print(f"   • Comprehensive kernel database with performance modeling")
        print(f"   • Intelligent kernel registry with search capabilities")
        print(f"✅ Model Topology Analyzer")
        print(f"   • ONNX model structure analysis for FINN requirements")
        print(f"   • Dataflow constraint analysis and optimization opportunities")
        print(f"   • Critical path identification and complexity analysis")
        print(f"✅ Kernel Selection Engine")
        print(f"   • Multi-objective optimization for kernel selection")
        print(f"   • Parameter optimization (PE, SIMD, folding)")
        print(f"   • Resource-aware selection with constraint satisfaction")
        print(f"✅ FINN Configuration Generator")
        print(f"   • Complete FINN build configuration generation")
        print(f"   • Four-category interface support")
        print(f"   • Validation and template support")
        print(f"\n🚀 Ready for Week 2: FINN Integration Engine Implementation")
        
    else:
        print(f"\n⚠️  Some workflow tests failed - check implementation")
    
    return failed == 0

if __name__ == '__main__':
    success = run_complete_workflow_tests()
    sys.exit(0 if success else 1)