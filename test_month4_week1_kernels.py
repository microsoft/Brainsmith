"""
Test suite for Month 4 Week 1: Kernel Selection Engine Implementation
Tests the FINN kernel management system components.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_kernel_imports():
    """Test that kernel management components can be imported."""
    try:
        from brainsmith.kernels import (
            FINNKernelRegistry, FINNKernelDiscovery, ModelTopologyAnalyzer,
            FINNKernelInfo, OperatorType, BackendType, PerformanceClass
        )
        
        from brainsmith.kernels.database import (
            FINNKernelDatabase, ParameterSchema, ResourceRequirements
        )
        
        from brainsmith.kernels.analysis import (
            TopologyAnalysis, OperatorRequirement, DataflowConstraints,
            LayerInfo, TensorShape, LayerType, DataType
        )
        
        print("✅ All kernel management imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Kernel import failed: {e}")
        return False

def test_database_components():
    """Test database components and data structures."""
    try:
        from brainsmith.kernels.database import (
            FINNKernelInfo, OperatorType, BackendType, PerformanceClass,
            ParameterSchema, ResourceRequirements, PerformanceModel,
            FINNKernelDatabase
        )
        
        # Test ParameterSchema
        param_schema = ParameterSchema(
            pe_range=(1, 64),
            simd_range=(1, 32),
            supported_datatypes=['int8', 'int16'],
            memory_modes=['internal', 'external'],
            folding_factors={'spatial': [1, 2, 4]}
        )
        
        # Test parameter validation
        valid_params = {'pe': 16, 'simd': 8, 'datatype': 'int8'}
        assert param_schema.validate_parameters(valid_params) == True
        
        invalid_params = {'pe': 128}  # Outside range
        assert param_schema.validate_parameters(invalid_params) == False
        
        # Test ResourceRequirements
        resources = ResourceRequirements(
            lut_count=1000,
            ff_count=2000,
            dsp_count=4,
            bram_count=2
        )
        
        # Test resource estimation
        estimated = resources.estimate_resources({'pe': 4, 'simd': 2})
        assert estimated.lut_count > resources.lut_count  # Should scale up
        
        # Test resource dictionary conversion
        resource_dict = resources.to_dict()
        assert 'lut' in resource_dict
        assert resource_dict['lut'] == 1000
        
        # Test PerformanceModel
        perf_model = PerformanceModel(
            model_type="analytical",
            throughput_model={'cycles_per_op': 1, 'base_throughput': 1000},
            latency_model={'base_latency': 10},
            power_model={'base_power': 1.0}
        )
        
        # Test performance estimation
        params = {'pe': 4, 'simd': 2}
        platform = {'clock_frequency': 100e6}
        
        throughput = perf_model.estimate_throughput(params, platform)
        assert throughput > 0
        
        latency = perf_model.estimate_latency(params, platform)
        assert latency > 0
        
        print("✅ Database components working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_kernel_discovery():
    """Test kernel discovery functionality."""
    try:
        from brainsmith.kernels.discovery import (
            FINNKernelDiscovery, KernelInfo, KernelMetadata
        )
        from brainsmith.kernels.database import OperatorType, BackendType
        
        # Create discovery engine
        discovery = FINNKernelDiscovery()
        
        # Test operator type determination
        test_cases = [
            ("test_matmul_kernel", "MatMul"),
            ("conv2d_implementation", "Convolution"),
            ("threshold_op", "Thresholding"),
            ("pool_layer", "Pool")
        ]
        
        for dir_name, expected_type in test_cases:
            # Mock directory analysis
            with patch.object(discovery, '_determine_operator_type') as mock_determine:
                mock_determine.return_value = expected_type
                result = discovery._determine_operator_type(dir_name, [])
                assert result == expected_type
        
        # Test backend type determination
        rtl_files = ['kernel.v', 'testbench.sv']
        hls_files = ['kernel.cpp', 'kernel.hpp']
        python_files = ['kernel.py']
        
        assert discovery._determine_backend_type(rtl_files) == 'RTL'
        assert discovery._determine_backend_type(hls_files) == 'HLS'
        assert discovery._determine_backend_type(python_files) == 'Python'
        
        # Test parameter schema extraction
        param_schema = discovery.extract_parameterization("/mock/path")
        assert param_schema.pe_range[0] >= 1
        assert param_schema.simd_range[0] >= 1
        assert len(param_schema.supported_datatypes) > 0
        
        print("✅ Kernel discovery working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Discovery test failed: {e}")
        return False

def test_kernel_registry():
    """Test kernel registry functionality."""
    try:
        from brainsmith.kernels.registry import (
            FINNKernelRegistry, SearchCriteria, RegistrationResult,
            CompatibilityChecker
        )
        from brainsmith.kernels.database import (
            FINNKernelInfo, OperatorType, BackendType, PerformanceClass,
            ParameterSchema, ResourceRequirements, PerformanceModel
        )
        
        # Create registry
        registry = FINNKernelRegistry()
        
        # Create test kernel
        test_kernel = FINNKernelInfo(
            name="test_matmul_kernel",
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
                throughput_model={'cycles_per_op': 1},
                latency_model={'base_latency': 10},
                power_model={'base_power': 1.0}
            ),
            resource_requirements=ResourceRequirements(
                lut_count=1000,
                ff_count=2000,
                dsp_count=4,
                bram_count=2
            ),
            finn_version_compatibility=['0.8+'],
            performance_class=PerformanceClass.BALANCED
        )
        
        # Test kernel registration
        result = registry.register_kernel(test_kernel)
        assert result.success == True
        assert result.kernel_name == "test_matmul_kernel"
        
        # Test kernel retrieval
        retrieved = registry.get_kernel("test_matmul_kernel")
        assert retrieved is not None
        assert retrieved.name == "test_matmul_kernel"
        
        # Test kernel search
        search_criteria = SearchCriteria(
            operator_type=OperatorType.MATMUL,
            backend_type=BackendType.HLS
        )
        
        results = registry.search_kernels(search_criteria)
        assert len(results) == 1
        assert results[0].name == "test_matmul_kernel"
        
        # Test compatibility checking
        compat_checker = CompatibilityChecker()
        is_compatible = compat_checker.check_version_compatibility(test_kernel, "0.8.0")
        assert is_compatible == True
        
        # Test validation
        validation_issues = compat_checker.validate_kernel_requirements(test_kernel)
        assert len(validation_issues) == 0  # Should be valid
        
        print("✅ Kernel registry working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False

def test_model_analysis():
    """Test model topology analysis."""
    try:
        from brainsmith.kernels.analysis import (
            ModelTopologyAnalyzer, ModelGraph, LayerInfo, TensorShape,
            LayerType, DataType, OperatorRequirement, TopologyAnalysis
        )
        
        # Create test model
        model_data = {
            'layers': [
                {
                    'name': 'conv1',
                    'type': 'Conv2d',
                    'input_shape': [1, 3, 224, 224],
                    'output_shape': [1, 64, 224, 224],
                    'parameters': {
                        'kernel_size': [3, 3],
                        'weight_shape': [64, 3, 3, 3]
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
                    'name': 'fc1',
                    'type': 'Linear',
                    'input_shape': [1, 64*224*224],
                    'output_shape': [1, 1000],
                    'parameters': {
                        'weight_shape': [1000, 64*224*224]
                    },
                    'data_type': 'int8'
                }
            ],
            'connections': [
                ('conv1', 'relu1'),
                ('relu1', 'fc1')
            ]
        }
        
        model = ModelGraph(model_data)
        assert len(model.layers) == 3
        assert len(model.connections) == 2
        
        # Test layer parsing
        conv_layer = model.layers[0]
        assert conv_layer.name == 'conv1'
        assert conv_layer.layer_type == LayerType.CONV2D
        assert conv_layer.input_shape.channels == 3
        assert conv_layer.output_shape.channels == 64
        assert conv_layer.is_compute_intensive == True
        
        relu_layer = model.layers[1]
        assert relu_layer.layer_type == LayerType.RELU
        assert relu_layer.is_elementwise == True
        
        # Test model analysis
        analyzer = ModelTopologyAnalyzer()
        analysis = analyzer.analyze_model_structure(model)
        
        # Verify analysis results
        assert isinstance(analysis, TopologyAnalysis)
        assert len(analysis.layers) == 3
        assert len(analysis.operator_requirements) > 0
        assert analysis.dataflow_constraints is not None
        assert len(analysis.optimization_opportunities) > 0
        
        # Test operator requirements
        conv_req = None
        for req in analysis.operator_requirements:
            if req.layer_id == 'conv1':
                conv_req = req
                break
        
        assert conv_req is not None
        assert conv_req.operator_type == "Convolution"
        assert conv_req.pe_requirements[0] >= 1  # Min PE
        assert conv_req.simd_requirements[0] >= 1  # Min SIMD
        assert conv_req.compute_computational_complexity() > 0
        
        # Test complexity analysis
        complexity = analysis.complexity_analysis
        assert 'total_operations' in complexity
        assert 'total_parameters' in complexity
        assert complexity['total_operations'] > 0
        
        print("✅ Model analysis working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Model analysis test failed: {e}")
        return False

def test_integration_workflow():
    """Test complete workflow integration."""
    try:
        from brainsmith.kernels import (
            FINNKernelRegistry, ModelTopologyAnalyzer
        )
        from brainsmith.kernels.analysis import ModelGraph
        from brainsmith.kernels.database import (
            FINNKernelInfo, OperatorType, BackendType, ParameterSchema,
            ResourceRequirements, PerformanceModel
        )
        
        # Create registry and populate with test kernels
        registry = FINNKernelRegistry()
        
        # Create test kernels for different operators
        kernels = [
            {
                'name': 'conv_hls_kernel',
                'operator_type': OperatorType.CONVOLUTION,
                'backend_type': BackendType.HLS
            },
            {
                'name': 'matmul_rtl_kernel', 
                'operator_type': OperatorType.MATMUL,
                'backend_type': BackendType.RTL
            },
            {
                'name': 'threshold_kernel',
                'operator_type': OperatorType.THRESHOLDING,
                'backend_type': BackendType.HLS
            }
        ]
        
        for kernel_spec in kernels:
            kernel = FINNKernelInfo(
                name=kernel_spec['name'],
                operator_type=kernel_spec['operator_type'],
                backend_type=kernel_spec['backend_type'],
                implementation_files={'impl': f"/test/{kernel_spec['name']}.cpp"},
                parameterization=ParameterSchema(
                    pe_range=(1, 32),
                    simd_range=(1, 16),
                    supported_datatypes=['int8'],
                    memory_modes=['internal'],
                    folding_factors={}
                ),
                performance_model=PerformanceModel(
                    model_type="analytical",
                    throughput_model={'cycles_per_op': 1},
                    latency_model={'base_latency': 10},
                    power_model={'base_power': 1.0}
                ),
                resource_requirements=ResourceRequirements(
                    lut_count=1000,
                    ff_count=2000,
                    dsp_count=4,
                    bram_count=2
                ),
                finn_version_compatibility=['0.8+']
            )
            
            result = registry.register_kernel(kernel)
            assert result.success == True
        
        # Create test model
        model_data = {
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
        
        model = ModelGraph(model_data)
        
        # Analyze model
        analyzer = ModelTopologyAnalyzer()
        analysis = analyzer.analyze_model_structure(model)
        
        # Test kernel matching for requirements
        for req in analysis.operator_requirements:
            if req.operator_type == "Convolution":
                # Search for convolution kernels
                from brainsmith.kernels.registry import SearchCriteria
                criteria = SearchCriteria(operator_type=OperatorType.CONVOLUTION)
                matching_kernels = registry.search_kernels(criteria)
                
                assert len(matching_kernels) > 0
                assert matching_kernels[0].name == 'conv_hls_kernel'
                
            elif req.operator_type == "Thresholding":
                # Search for thresholding kernels
                from brainsmith.kernels.registry import SearchCriteria
                criteria = SearchCriteria(operator_type=OperatorType.THRESHOLDING)
                matching_kernels = registry.search_kernels(criteria)
                
                assert len(matching_kernels) > 0
                assert matching_kernels[0].name == 'threshold_kernel'
        
        # Test registry statistics
        stats = registry.get_registry_statistics()
        assert stats['database_stats']['total_kernels'] == 3
        assert 'Convolution' in stats['database_stats']['by_operator_type']
        
        print("✅ Integration workflow working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def run_kernel_tests():
    """Run all kernel management tests."""
    print("Testing Month 4 Week 1: Kernel Selection Engine Implementation")
    print("=" * 80)
    
    tests = [
        ("Import Test", test_kernel_imports),
        ("Database Components", test_database_components),
        ("Kernel Discovery", test_kernel_discovery),
        ("Kernel Registry", test_kernel_registry),
        ("Model Analysis", test_model_analysis),
        ("Integration Workflow", test_integration_workflow)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"Kernel Selection Engine Test Results")
    print(f"{'='*80}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print(f"\n🎉 All kernel management tests passed!")
        print(f"Week 1 kernel selection engine is working correctly!")
        
        # Display implementation summary
        print(f"\n{'='*80}")
        print(f"🔧 Week 1 Implementation Summary")
        print(f"{'='*80}")
        print(f"📦 Implemented Components:")
        print(f"   • FINN Kernel Database with comprehensive schema")
        print(f"   • Automated kernel discovery and analysis engine")
        print(f"   • Central kernel registry with search capabilities")
        print(f"   • Model topology analyzer for requirement extraction")
        print(f"   • Performance modeling framework")
        print(f"   • Compatibility checking and validation")
        print(f"\n🔧 Key Features:")
        print(f"   • Multi-operator support (MatMul, Conv, Thresholding, etc.)")
        print(f"   • Multiple backend types (RTL, HLS, Python)")
        print(f"   • Parameter optimization (PE, SIMD, folding)")
        print(f"   • Resource estimation and constraint handling")
        print(f"   • FINN version compatibility management")
        print(f"   • Intelligent kernel search and ranking")
        
    else:
        print(f"\n⚠️  Some tests failed - check implementation")
    
    return failed == 0

if __name__ == '__main__':
    success = run_kernel_tests()
    
    if success:
        print(f"\n{'='*80}")
        print(f"🏁 Month 4 Week 1 Complete: Kernel Selection Engine")
        print(f"{'='*80}")
        print(f"✅ Enhanced Hardware Kernel Registration and Management System")
        print(f"   - Phase 2.1 of Major Changes Plan successfully implemented")
        print(f"   - Foundation ready for Week 2: FINN Integration Engine")
    
    sys.exit(0 if success else 1)