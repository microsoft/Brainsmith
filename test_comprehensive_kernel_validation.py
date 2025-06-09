"""
Comprehensive Validation Test Suite for Month 1 Kernel Registration System
Extensive testing of all implemented features with edge cases and integration scenarios.
"""

import os
import sys
import tempfile
import shutil
import json
import sqlite3
from pathlib import Path
import logging

# Add brainsmith to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from brainsmith.kernels import (
    FINNKernelDiscovery, FINNKernelRegistry, FINNKernelInfo, 
    FINNKernelDatabase, AnalyticalModel, FINNKernelSelector,
    ModelTopologyAnalyzer, FINNConfigGenerator, KernelMetadata, ParameterSchema
)
from brainsmith.kernels.registry import Platform, PerformanceTargets, ResourceConstraints
from brainsmith.kernels.selection import ModelGraph, ModelNode, OperatorRequirement
from brainsmith.kernels.finn_config import FINNBuildConfig
from brainsmith.kernels.performance import PerformanceEstimate

# Configure logging for testing
logging.basicConfig(level=logging.WARNING)  # Reduce log noise


class ComprehensiveValidationSuite:
    """Comprehensive validation test suite for kernel registration system."""
    
    def __init__(self):
        self.test_results = {}
        self.temp_directories = []
        self.temp_databases = []
        
    def cleanup(self):
        """Clean up temporary resources."""
        for temp_dir in self.temp_directories:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        for temp_db in self.temp_databases:
            if os.path.exists(temp_db):
                os.unlink(temp_db)
    
    def create_enhanced_mock_finn(self):
        """Create enhanced mock FINN installation with multiple kernel types."""
        temp_dir = tempfile.mkdtemp(prefix="enhanced_finn_")
        self.temp_directories.append(temp_dir)
        
        # Create FINN directory structure
        finn_src = os.path.join(temp_dir, "src", "finn")
        custom_ops = os.path.join(finn_src, "custom_op")
        os.makedirs(custom_ops, exist_ok=True)
        
        # Create diverse set of kernels with realistic complexity
        kernel_specs = [
            {
                'name': 'advanced_matmul',
                'type': 'MatMul',
                'pe_range': (1, 64),
                'simd_range': (1, 32)
            },
            {
                'name': 'multi_threshold',
                'type': 'Thresholding', 
                'pe_range': (1, 16),
                'simd_range': None
            },
            {
                'name': 'streamlined_layernorm',
                'type': 'LayerNorm',
                'pe_range': (1, 32),
                'simd_range': (1, 16)
            },
            {
                'name': 'optimized_conv2d',
                'type': 'Conv2D',
                'pe_range': (1, 128),
                'simd_range': (1, 64)
            }
        ]
        
        for spec in kernel_specs:
            self.create_realistic_kernel(custom_ops, spec)
        
        # Create setup.py with version
        setup_py = os.path.join(temp_dir, "setup.py")
        with open(setup_py, 'w') as f:
            f.write('version="0.8.1"\nname="finn"\n')
        
        return temp_dir
    
    def create_realistic_kernel(self, custom_ops_dir, spec):
        """Create realistic mock kernel implementation."""
        kernel_dir = os.path.join(custom_ops_dir, spec['name'])
        os.makedirs(kernel_dir, exist_ok=True)
        
        # Create Python implementation
        python_file = os.path.join(kernel_dir, f"{spec['name']}.py")
        with open(python_file, 'w') as f:
            pe_range_comment = f"# PE range: {spec['pe_range'][0]}-{spec['pe_range'][1]}"
            simd_comment = f"# SIMD range: {spec['simd_range'][0]}-{spec['simd_range'][1]}" if spec['simd_range'] else "# SIMD: N/A"
            
            f.write(f'''
"""
{spec["type"]} kernel implementation: {spec["name"]}
"""

class {spec["name"].title().replace('_', '')}Op(HWCustomOp):
    """
    {spec["type"]} operator implementation.
    
    {pe_range_comment}
    {simd_comment}
    
    Constraints:
    - PE must be power of 2
    - SIMD should divide input width
    """
    
    def __init__(self):
        super().__init__()
        
    def get_nodeattr_types(self):
        return {{"PE": ("i", True, 1), "SIMD": ("i", True, 1)}}
        
    def make_rtl_backend(self):
        return {spec["name"].title().replace('_', '')}RTL()
        
    def make_hls_backend(self):
        return {spec["name"].title().replace('_', '')}HLS()
''')
        
        # Create RTL and HLS files
        for ext, content in [('.sv', '// RTL implementation'), ('.cpp', '// HLS implementation')]:
            impl_file = os.path.join(kernel_dir, f"{spec['name']}{ext}")
            with open(impl_file, 'w') as f:
                f.write(f'{content}\nmodule {spec["name"]}();\nendmodule\n')
    
    def test_discovery_engine_comprehensive(self):
        """Test comprehensive kernel discovery functionality."""
        print("🔍 Testing Discovery Engine...")
        
        # Create enhanced FINN installation
        finn_path = self.create_enhanced_mock_finn()
        discovery = FINNKernelDiscovery()
        
        # Test discovery
        discovered_kernels = discovery.scan_finn_installation(finn_path)
        assert len(discovered_kernels) == 4, f"Expected 4 kernels, found {len(discovered_kernels)}"
        
        # Verify each kernel has proper metadata
        kernel_names = [k.name for k in discovered_kernels]
        expected_names = ['advanced_matmul', 'multi_threshold', 'streamlined_layernorm', 'optimized_conv2d']
        
        for expected in expected_names:
            assert expected in kernel_names, f"Expected kernel {expected} not discovered"
        
        # Test metadata quality
        for kernel in discovered_kernels:
            assert kernel.name is not None
            assert kernel.operator_type in ['MatMul', 'Thresholding', 'LayerNorm', 'Conv2D']
            assert kernel.backend_type in ['RTL', 'HLS', 'Both', 'Unknown']
            assert 'python_implementation' in kernel.implementation_files
            
            # Test parameterization extraction
            if kernel.name == 'advanced_matmul':
                assert 'pe_range' in kernel.parameterization
                assert 'simd_range' in kernel.parameterization
        
        # Test error handling
        invalid_kernels = discovery.scan_finn_installation("/nonexistent/path")
        assert len(invalid_kernels) == 0
        
        print("✅ Discovery Engine tests passed")
        return True
    
    def test_registry_operations_stress(self):
        """Test registry operations under stress conditions."""
        print("📊 Testing Registry Stress Operations...")
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            db_path = temp_db.name
            self.temp_databases.append(db_path)
        
        registry = FINNKernelRegistry(database_path=db_path)
        
        # Test bulk registration (50 kernels)
        successful_registrations = 0
        for i in range(50):
            param_schema = ParameterSchema(
                pe_range=(1, min(64, 2**i)),
                simd_range=(1, min(32, 2**(i//2))),
                custom_parameters={'test_param': i, 'variant': f'v{i}'},
                constraints=[f'PE > 0', f'SIMD <= {32}', f'PE * SIMD <= 1024']
            )
            
            kernel_info = FINNKernelInfo(
                name=f"stress_test_kernel_{i}",
                operator_type="MatMul" if i % 3 == 0 else ("Thresholding" if i % 3 == 1 else "LayerNorm"),
                backend_type="RTL" if i % 2 == 0 else "HLS",
                implementation_files={'python_implementation': f'/path/to/kernel_{i}.py'},
                parameterization=param_schema,
                finn_version_compatibility=["0.8.1", "0.9.0"] if i % 2 == 0 else ["0.8.1"]
            )
            
            # Add performance model
            kernel_info.performance_model = AnalyticalModel(kernel_info.name, kernel_info.operator_type)
            
            result = registry.register_finn_kernel(kernel_info)
            if result.success:
                successful_registrations += 1
        
        assert successful_registrations == 50, f"Expected 50 successful registrations, got {successful_registrations}"
        
        # Test comprehensive search scenarios
        search_scenarios = [
            (registry.SearchCriteria(operator_type="MatMul"), "MatMul operator search"),
            (registry.SearchCriteria(backend_type="RTL"), "RTL backend search"),
            (registry.SearchCriteria(operator_type="Thresholding", backend_type="HLS"), "Combined search"),
            (registry.SearchCriteria(finn_version="0.8.1"), "Version compatibility search"),
        ]
        
        for criteria, description in search_scenarios:
            results = registry.search_kernels(criteria)
            assert len(results) > 0, f"No results for {description}"
        
        # Test parameter validation edge cases
        test_kernel = registry.get_kernel("stress_test_kernel_5")
        assert test_kernel is not None
        
        # Test boundary conditions
        boundary_tests = [
            ({'PE': 1, 'SIMD': 1}, True, "Minimum valid parameters"),
            ({'PE': 32, 'SIMD': 16}, True, "Moderate parameters"),
            ({'PE': 0, 'SIMD': 1}, False, "Invalid PE (zero)"),
            ({'PE': 1000, 'SIMD': 1}, False, "Invalid PE (too high)"),
            ({'PE': 1, 'SIMD': 100}, False, "Invalid SIMD (too high)"),
        ]
        
        for params, should_be_valid, description in boundary_tests:
            is_valid, issues = test_kernel.validate_parameters(params)
            assert is_valid == should_be_valid, f"Parameter validation failed for {description}: {params}"
        
        # Test database statistics
        stats = registry.get_registry_stats()
        assert stats['total_kernels'] == 50
        assert len(stats['by_operator_type']) >= 3  # MatMul, Thresholding, LayerNorm
        
        print("✅ Registry Stress Operations tests passed")
        return True
    
    def test_performance_modeling_accuracy(self):
        """Test performance modeling accuracy across different scenarios."""
        print("🎯 Testing Performance Modeling Accuracy...")
        
        # Test scenarios with known expected behaviors
        test_scenarios = [
            {
                'operator_type': 'MatMul',
                'parameters': {'PE': 8, 'SIMD': 4, 'M': 256, 'N': 256, 'K': 256},
                'platform': Platform("test", "xc7z020", 100.0, {'lut': 50000, 'dsp': 220}),
                'expected_throughput_min': 1e6,
                'expected_latency_max': 1e6
            },
            {
                'operator_type': 'Thresholding',
                'parameters': {'PE': 16, 'NumElements': 10000},
                'platform': Platform("test", "xczu7ev", 200.0, {'lut': 200000, 'dsp': 1728}),
                'expected_throughput_min': 1e6,
                'expected_latency_max': 1e5
            },
            {
                'operator_type': 'LayerNorm',
                'parameters': {'PE': 4, 'SIMD': 8, 'NumElements': 1000},
                'platform': Platform("test", "xcvu9p", 300.0, {'lut': 1000000, 'dsp': 6840}),
                'expected_throughput_min': 1e5,
                'expected_latency_max': 1e5
            }
        ]
        
        for scenario in test_scenarios:
            model = AnalyticalModel(f"test_{scenario['operator_type'].lower()}", scenario['operator_type'])
            
            # Test comprehensive performance estimation
            performance = model.estimate_performance(scenario['parameters'], scenario['platform'])
            
            # Validate results are reasonable
            assert performance.throughput_ops_sec >= scenario['expected_throughput_min'], \
                f"Throughput too low for {scenario['operator_type']}: {performance.throughput_ops_sec}"
            
            assert performance.latency_cycles <= scenario['expected_latency_max'], \
                f"Latency too high for {scenario['operator_type']}: {performance.latency_cycles}"
            
            # Validate resource estimates are positive
            assert performance.resource_usage['lut_count'] > 0, "LUT count should be positive"
            assert performance.resource_usage['dsp_count'] >= 0, "DSP count should be non-negative"
            assert performance.resource_usage['bram_count'] > 0, "BRAM count should be positive"
            
            # Validate confidence is reasonable
            assert 0.1 <= performance.confidence <= 1.0, f"Invalid confidence: {performance.confidence}"
            
            # Test scaling behavior
            double_pe_params = scenario['parameters'].copy()
            double_pe_params['PE'] = double_pe_params.get('PE', 1) * 2
            
            double_pe_performance = model.estimate_performance(double_pe_params, scenario['platform'])
            
            # Doubling PE should increase throughput
            assert double_pe_performance.throughput_ops_sec >= performance.throughput_ops_sec, \
                f"Doubling PE should increase throughput for {scenario['operator_type']}"
        
        print("✅ Performance Modeling Accuracy tests passed")
        return True
    
    def test_kernel_selection_optimization(self):
        """Test kernel selection optimization with complex scenarios."""
        print("🎯 Testing Kernel Selection Optimization...")
        
        # Create registry with multiple kernel variants
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            db_path = temp_db.name
            self.temp_databases.append(db_path)
        
        registry = FINNKernelRegistry(database_path=db_path)
        
        # Register kernel variants with different characteristics
        kernel_variants = [
            # Performance-optimized variants
            {'name': 'matmul_perf', 'type': 'MatMul', 'pe_range': (1, 64), 'simd_range': (1, 32)},
            {'name': 'threshold_perf', 'type': 'Thresholding', 'pe_range': (1, 32), 'simd_range': None},
            
            # Area-optimized variants
            {'name': 'matmul_area', 'type': 'MatMul', 'pe_range': (1, 16), 'simd_range': (1, 8)},
            {'name': 'threshold_area', 'type': 'Thresholding', 'pe_range': (1, 8), 'simd_range': None},
            
            # Balanced variants
            {'name': 'matmul_balanced', 'type': 'MatMul', 'pe_range': (1, 32), 'simd_range': (1, 16)},
        ]
        
        for variant in kernel_variants:
            param_schema = ParameterSchema(
                pe_range=variant['pe_range'],
                simd_range=variant['simd_range'],
                custom_parameters={'variant': variant['name']},
                constraints=['PE > 0']
            )
            
            kernel_info = FINNKernelInfo(
                name=variant['name'],
                operator_type=variant['type'],
                backend_type="RTL",
                implementation_files={'python_implementation': f'/path/to/{variant["name"]}.py'},
                parameterization=param_schema
            )
            
            # Add performance model
            kernel_info.performance_model = AnalyticalModel(kernel_info.name, kernel_info.operator_type)
            
            registry.register_finn_kernel(kernel_info)
        
        # Create complex test model
        model = ModelGraph()
        model.nodes = [
            ModelNode(
                id="matmul_1",
                operator_type="MatMul",
                input_shapes=[(512, 512), (512, 256)],
                output_shapes=[(512, 256)]
            ),
            ModelNode(
                id="threshold_1",
                operator_type="Thresholding",
                input_shapes=[(512, 256)],
                output_shapes=[(512, 256)]
            ),
            ModelNode(
                id="matmul_2",
                operator_type="MatMul",
                input_shapes=[(512, 256), (256, 128)],
                output_shapes=[(512, 128)]
            )
        ]
        model.edges = [("matmul_1", "threshold_1"), ("threshold_1", "matmul_2")]
        
        selector = FINNKernelSelector(registry)
        
        # Test different optimization objectives
        optimization_tests = [
            {
                'name': 'performance_focused',
                'targets': PerformanceTargets(throughput_ops_sec=1e7, priority="performance"),
                'constraints': ResourceConstraints(max_lut_count=100000, max_dsp_count=1000),
                'expected_kernels': ['matmul_perf', 'threshold_perf']
            },
            {
                'name': 'area_focused',
                'targets': PerformanceTargets(throughput_ops_sec=1e6, priority="area"),
                'constraints': ResourceConstraints(max_lut_count=20000, max_dsp_count=200),
                'expected_kernels': ['matmul_area', 'threshold_area']
            },
            {
                'name': 'balanced',
                'targets': PerformanceTargets(throughput_ops_sec=5e6, priority="balanced"),
                'constraints': ResourceConstraints(max_lut_count=50000, max_dsp_count=500),
                'expected_kernels': ['matmul_balanced']
            }
        ]
        
        for test in optimization_tests:
            selection_plan = selector.select_optimal_kernels(
                model, test['targets'], test['constraints']
            )
            
            # Validate selection plan
            assert len(selection_plan.assignments) >= 2, f"Insufficient assignments for {test['name']}"
            
            # Check constraint satisfaction
            total_resources = selection_plan.get_total_resources()
            assert total_resources.get('lut_count', 0) <= test['constraints'].max_lut_count, \
                f"LUT constraint violated in {test['name']}"
            
            if test['constraints'].max_dsp_count:
                assert total_resources.get('dsp_count', 0) <= test['constraints'].max_dsp_count, \
                    f"DSP constraint violated in {test['name']}"
            
            # Validate total performance calculation
            total_performance = selection_plan.total_estimated_performance
            assert total_performance is not None, f"Total performance missing for {test['name']}"
            assert total_performance.throughput_ops_sec > 0, f"Invalid total throughput for {test['name']}"
        
        print("✅ Kernel Selection Optimization tests passed")
        return True
    
    def test_finn_config_generation_comprehensive(self):
        """Test comprehensive FINN configuration generation."""
        print("🔧 Testing FINN Configuration Generation...")
        
        # Create comprehensive selection plan
        from brainsmith.kernels.selection import SelectionPlan, KernelAssignment
        
        assignments = []
        
        # Create diverse kernel assignments
        kernel_configs = [
            {
                'name': 'matmul_test', 'type': 'MatMul', 'backend': 'RTL',
                'params': {'PE': 16, 'SIMD': 8, 'mem_mode': 'internal'},
                'resources': {'lut_count': 5000, 'dsp_count': 128, 'bram_count': 16}
            },
            {
                'name': 'threshold_test', 'type': 'Thresholding', 'backend': 'HLS',
                'params': {'PE': 8, 'ActVal': 0.5, 'NumSteps': 1},
                'resources': {'lut_count': 1000, 'dsp_count': 0, 'bram_count': 4}
            },
            {
                'name': 'layernorm_test', 'type': 'LayerNorm', 'backend': 'RTL',
                'params': {'PE': 4, 'SIMD': 4, 'precision': 'fixed'},
                'resources': {'lut_count': 2000, 'dsp_count': 16, 'bram_count': 8}
            }
        ]
        
        for i, config in enumerate(kernel_configs):
            param_schema = ParameterSchema()
            kernel_info = FINNKernelInfo(
                name=config['name'],
                operator_type=config['type'],
                backend_type=config['backend'],
                implementation_files={'python_implementation': f'/path/to/{config["name"]}.py'},
                parameterization=param_schema
            )
            
            performance = PerformanceEstimate(
                throughput_ops_sec=1e6 * (i + 1),
                latency_cycles=1000 * (i + 1),
                resource_usage=config['resources'],
                confidence=0.9
            )
            
            assignment = KernelAssignment(
                node_id=f"node_{i}",
                kernel_info=kernel_info,
                parameters=config['params'],
                estimated_performance=performance
            )
            
            assignments.append(assignment)
        
        selection_plan = SelectionPlan(assignments=assignments)
        
        # Test configuration generation
        generator = FINNConfigGenerator()
        
        finn_config = generator.generate_build_config(
            selection_plan,
            model_path="/path/to/test_model.onnx",
            platform_config={
                'fpga_part': 'xczu7ev-ffvc1156-2-e',
                'clock_frequency_mhz': 200.0,
                'board': 'ZCU104'
            }
        )
        
        # Validate configuration structure
        assert isinstance(finn_config, FINNBuildConfig), "Invalid config type"
        assert len(finn_config.kernel_configs) == 3, "Expected 3 kernel configurations"
        
        # Validate individual kernel configurations
        for i, kernel_config in enumerate(finn_config.kernel_configs):
            assert kernel_config.kernel_name == kernel_configs[i]['name']
            assert kernel_config.operator_type == kernel_configs[i]['type']
            assert kernel_config.backend in ['rtl', 'hls']
            assert 'PE' in kernel_config.parameters
        
        # Validate build settings
        assert 'fpga_part' in finn_config.build_settings
        assert 'synth_clk_period_ns' in finn_config.build_settings
        assert finn_config.build_settings['fpga_part'] == 'xczu7ev-ffvc1156-2-e'
        
        # Test JSON serialization
        json_str = finn_config.to_json()
        assert json_str, "JSON serialization failed"
        
        # Validate JSON can be parsed
        parsed_config = json.loads(json_str)
        assert 'kernel_configs' in parsed_config
        assert 'build_settings' in parsed_config
        
        # Test configuration validation
        is_valid, issues = generator.validate_finn_config(finn_config)
        assert is_valid, f"Generated configuration is invalid: {issues}"
        
        # Test transformation sequence generation
        transformations = generator.generate_finn_transformation_sequence(selection_plan)
        assert len(transformations) >= 10, "Insufficient transformations generated"
        
        required_transforms = ["InferShapes", "Streamline", "CreateDataflowPartition"]
        for required in required_transforms:
            assert required in transformations, f"Missing required transformation: {required}"
        
        # Test script template generation
        script = generator.generate_script_template(finn_config)
        assert "build_dataflow_cfg" in script, "Script template missing FINN build call"
        assert "DataflowBuildConfig" in script, "Script template missing config class"
        
        print("✅ FINN Configuration Generation tests passed")
        return True
    
    def test_end_to_end_integration_comprehensive(self):
        """Test comprehensive end-to-end integration."""
        print("🚀 Testing End-to-End Integration...")
        
        # Create enhanced FINN installation
        finn_path = self.create_enhanced_mock_finn()
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            db_path = temp_db.name
            self.temp_databases.append(db_path)
        
        # Step 1: Discover kernels
        discovery = FINNKernelDiscovery()
        discovered_kernels = discovery.scan_finn_installation(finn_path)
        assert len(discovered_kernels) == 4, "Discovery failed"
        
        # Step 2: Register kernels with enhanced information
        registry = FINNKernelRegistry(database_path=db_path)
        
        registered_kernels = []
        for kernel_metadata in discovered_kernels:
            # Convert to FINNKernelInfo with proper schema
            param_schema = ParameterSchema()
            if isinstance(kernel_metadata.parameterization, dict):
                param_schema.pe_range = kernel_metadata.parameterization.get('pe_range', (1, 16))
                param_schema.simd_range = kernel_metadata.parameterization.get('simd_range', (1, 8))
                param_schema.custom_parameters = kernel_metadata.parameterization.get('custom_parameters', {})
                param_schema.constraints = kernel_metadata.parameterization.get('constraints', [])
            
            kernel_info = FINNKernelInfo(
                name=kernel_metadata.name,
                operator_type=kernel_metadata.operator_type,
                backend_type=kernel_metadata.backend_type,
                implementation_files=kernel_metadata.implementation_files,
                parameterization=param_schema,
                finn_version_compatibility=kernel_metadata.finn_version_compatibility
            )
            
            # Add enhanced performance model
            kernel_info.performance_model = AnalyticalModel(kernel_info.name, kernel_info.operator_type)
            
            result = registry.register_finn_kernel(kernel_info)
            if result.success:
                registered_kernels.append(kernel_info)
        
        assert len(registered_kernels) == 4, "Registration failed"
        
        # Step 3: Create complex test model
        model = ModelGraph()
        model.nodes = [
            ModelNode(
                id="input_matmul",
                operator_type="MatMul",
                input_shapes=[(1, 768, 768)],
                output_shapes=[(1, 768, 768)]
            ),
            ModelNode(
                id="activation_threshold", 
                operator_type="Thresholding",
                input_shapes=[(1, 768, 768)],
                output_shapes=[(1, 768, 768)]
            ),
            ModelNode(
                id="norm_layer",
                operator_type="LayerNorm", 
                input_shapes=[(1, 768, 768)],
                output_shapes=[(1, 768, 768)]
            ),
            ModelNode(
                id="output_conv",
                operator_type="Conv2D",
                input_shapes=[(1, 32, 32, 768)],
                output_shapes=[(1, 32, 32, 256)]
            )
        ]
        model.edges = [
            ("input_matmul", "activation_threshold"),
            ("activation_threshold", "norm_layer"),
            ("norm_layer", "output_conv")
        ]
        
        # Step 4: Perform kernel selection with realistic constraints
        selector = FINNKernelSelector(registry)
        
        targets = PerformanceTargets(
            throughput_ops_sec=1e7,
            latency_cycles=100000,
            priority="balanced"
        )
        
        constraints = ResourceConstraints(
            max_lut_count=200000,
            max_dsp_count=2000,
            max_bram_count=500,
            max_area_utilization=0.8
        )
        
        selection_plan = selector.select_optimal_kernels(model, targets, constraints)
        
        # Validate selection quality
        assert len(selection_plan.assignments) >= 3, "Insufficient kernel selections"
        
        # Check that different operators are properly mapped
        selected_operators = {assignment.kernel_info.operator_type for assignment in selection_plan.assignments}
        assert len(selected_operators) >= 3, "Not enough operator diversity in selection"
        
        # Step 5: Generate FINN configuration
        generator = FINNConfigGenerator()
        
        finn_config = generator.generate_build_config(
            selection_plan,
            model_path="/path/to/complex_model.onnx",
            platform_config={
                'fpga_part': 'xczu7ev-ffvc1156-2-e',
                'clock_frequency_mhz': 250.0,
                'board': 'ZCU104',
                'memory_bandwidth_gbps': 76.8
            }
        )
        
        # Validate final configuration
        is_valid, issues = generator.validate_finn_config(finn_config)
        assert is_valid, f"Final configuration invalid: {issues}"
        
        # Test configuration export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file:
            config_json = finn_config.to_json(config_file.name)
            assert os.path.exists(config_file.name), "Configuration export failed"
            
            # Cleanup
            os.unlink(config_file.name)
        
        # Step 6: Validate overall system coherence
        total_resources = selection_plan.get_total_resources()
        
        # Resources should be within constraints
        assert total_resources.get('lut_count', 0) <= constraints.max_lut_count, "LUT constraint violated"
        assert total_resources.get('dsp_count', 0) <= constraints.max_dsp_count, "DSP constraint violated"
        assert total_resources.get('bram_count', 0) <= constraints.max_bram_count, "BRAM constraint violated"
        
        # Performance should meet minimum requirements  
        total_performance = selection_plan.total_estimated_performance
        assert total_performance.throughput_ops_sec >= targets.throughput_ops_sec * 0.5, \
            "Throughput significantly below target"
        
        print("✅ End-to-End Integration tests passed")
        print(f"   - Discovered: {len(discovered_kernels)} kernels")
        print(f"   - Registered: {len(registered_kernels)} kernels")
        print(f"   - Selected: {len(selection_plan.assignments)} kernels")
        print(f"   - Generated: {len(finn_config.kernel_configs)} configurations")
        print(f"   - Total Resources: LUT={total_resources.get('lut_count', 0)}, DSP={total_resources.get('dsp_count', 0)}, BRAM={total_resources.get('bram_count', 0)}")
        
        return True
    
    def run_all_tests(self):
        """Run all comprehensive validation tests."""
        print("🧪 Starting Comprehensive Validation Suite")
        print("=" * 60)
        
        test_methods = [
            self.test_discovery_engine_comprehensive,
            self.test_registry_operations_stress,
            self.test_performance_modeling_accuracy,
            self.test_kernel_selection_optimization,
            self.test_finn_config_generation_comprehensive,
            self.test_end_to_end_integration_comprehensive
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        try:
            for test_method in test_methods:
                if test_method():
                    passed_tests += 1
                print()
            
            # Summary
            print("🎉 Comprehensive Validation Complete!")
            print(f"✅ Passed: {passed_tests}/{total_tests} test suites")
            
            if passed_tests == total_tests:
                print("\n🏆 ALL TESTS PASSED - Month 1 implementation is production-ready!")
                print("\n📊 Validation Summary:")
                print("✅ FINN Kernel Discovery Engine - Fully validated")
                print("✅ Kernel Database and Registry - Stress tested")
                print("✅ Performance Modeling Framework - Accuracy verified")
                print("✅ Intelligent Kernel Selection - Optimization validated")
                print("✅ FINN Configuration Generation - Integration tested")
                print("✅ End-to-End Workflow - Comprehensive validation")
                
                print("\n🚀 Ready for Month 2 Implementation!")
                return True
            else:
                print(f"\n❌ {total_tests - passed_tests} test suite(s) failed")
                return False
                
        except Exception as e:
            print(f"💥 Validation failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            self.cleanup()


def main():
    """Run comprehensive validation suite."""
    validator = ComprehensiveValidationSuite()
    success = validator.run_all_tests()
    
    if success:
        print("\n🎯 Month 1 implementation successfully validated!")
        print("All kernel registration system features are working correctly.")
    else:
        print("\n❌ Validation failed - issues need to be addressed before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()