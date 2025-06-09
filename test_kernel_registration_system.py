"""
Test Suite for Kernel Registration System
Comprehensive testing of Month 1 implementation: Enhanced Hardware Kernel Registration and Management System
"""

import os
import sys
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Add brainsmith to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from brainsmith.kernels import (
    FINNKernelDiscovery, FINNKernelRegistry, FINNKernelInfo, 
    FINNKernelDatabase, AnalyticalModel, FINNKernelSelector,
    ModelTopologyAnalyzer, FINNConfigGenerator
)
from brainsmith.kernels.registry import Platform, PerformanceTargets, ResourceConstraints, ParameterSchema
from brainsmith.kernels.selection import ModelGraph, ModelNode, OperatorRequirement
from brainsmith.kernels.finn_config import FINNBuildConfig


def create_mock_finn_installation():
    """Create mock FINN installation for testing."""
    temp_dir = tempfile.mkdtemp(prefix="mock_finn_")
    
    # Create FINN directory structure
    finn_src = os.path.join(temp_dir, "src", "finn")
    custom_ops = os.path.join(finn_src, "custom_op")
    os.makedirs(custom_ops, exist_ok=True)
    
    # Create mock kernel implementations
    create_mock_kernel(custom_ops, "thresholding")
    create_mock_kernel(custom_ops, "matmul")
    create_mock_kernel(custom_ops, "layernorm")
    
    # Create setup.py with version
    setup_py = os.path.join(temp_dir, "setup.py")
    with open(setup_py, 'w') as f:
        f.write('version="0.8.1"\n')
    
    return temp_dir


def create_mock_kernel(custom_ops_dir, kernel_name):
    """Create mock kernel implementation files."""
    kernel_dir = os.path.join(custom_ops_dir, kernel_name)
    os.makedirs(kernel_dir, exist_ok=True)
    
    # Create Python implementation
    python_file = os.path.join(kernel_dir, f"{kernel_name}.py")
    with open(python_file, 'w') as f:
        if kernel_name == "thresholding":
            f.write('''
"""
Thresholding activation function kernel.
"""

class ThresholdingOp(HWCustomOp):
    """Thresholding operator implementation."""
    
    def __init__(self):
        super().__init__()
        # PE range: 1-16
        # SIMD range: 1-8
        
    def get_nodeattr_types(self):
        return {"PE": ("i", True, 1), "ActVal": ("f", True, 0.0)}
        
    def make_rtl_backend(self):
        return ThresholdingRTL()
        
    def make_hls_backend(self):
        return ThresholdingHLS()
''')
        elif kernel_name == "matmul":
            f.write('''
"""
Matrix multiplication kernel.
"""

class MatMulOp(HWCustomOp):
    """Matrix multiplication operator implementation."""
    
    def __init__(self):
        super().__init__()
        # PE range: 1-64
        # SIMD range: 1-32
        
    def get_nodeattr_types(self):
        return {"PE": ("i", True, 1), "SIMD": ("i", True, 1), "mem_mode": ("s", False, "internal")}
        
    def make_rtl_backend(self):
        return MatMulRTL()
        
    def make_hls_backend(self):
        return MatMulHLS()
''')
        elif kernel_name == "layernorm":
            f.write('''
"""
Layer normalization kernel.
"""

class LayerNormOp(HWCustomOp):
    """Layer normalization operator implementation."""
    
    def __init__(self):
        super().__init__()
        # PE range: 1-32
        # SIMD range: 1-16
        
    def get_nodeattr_types(self):
        return {"PE": ("i", True, 1), "SIMD": ("i", True, 1), "precision": ("s", False, "fixed")}
        
    def make_rtl_backend(self):
        return LayerNormRTL()
        
    def make_hls_backend(self):
        return LayerNormHLS()
''')
    
    # Create RTL files
    rtl_file = os.path.join(kernel_dir, f"{kernel_name}.sv")
    with open(rtl_file, 'w') as f:
        f.write(f'''
// {kernel_name.capitalize()} RTL implementation
module {kernel_name}_rtl #(
    parameter PE = 1,
    parameter SIMD = 1
) (
    input clk,
    input rst,
    // AXI interfaces would be here
);

endmodule
''')
    
    # Create HLS files
    hls_file = os.path.join(kernel_dir, f"{kernel_name}.cpp")
    with open(hls_file, 'w') as f:
        f.write(f'''
// {kernel_name.capitalize()} HLS implementation
#include "{kernel_name}.hpp"

void {kernel_name}_hls(/* parameters */) {{
    // Implementation would be here
}}
''')


def test_finn_kernel_discovery():
    """Test FINN kernel discovery functionality."""
    print("Testing FINN Kernel Discovery...")
    
    # Create mock FINN installation
    mock_finn_path = create_mock_finn_installation()
    
    try:
        # Test discovery
        discovery = FINNKernelDiscovery()
        discovered_kernels = discovery.scan_finn_installation(mock_finn_path)
        
        print(f"Discovered {len(discovered_kernels)} kernels")
        
        # Verify discovered kernels
        assert len(discovered_kernels) == 3, f"Expected 3 kernels, found {len(discovered_kernels)}"
        
        kernel_names = [k.name for k in discovered_kernels]
        expected_names = ["thresholding", "matmul", "layernorm"]
        
        for expected in expected_names:
            assert expected in kernel_names, f"Expected kernel {expected} not discovered"
        
        # Test kernel structure analysis
        for kernel in discovered_kernels:
            assert kernel.name is not None
            assert kernel.operator_type is not None
            assert kernel.backend_type in ["RTL", "HLS", "Both", "Unknown"]
            assert 'python_implementation' in kernel.implementation_files
            
        print("✅ FINN Kernel Discovery tests passed")
        
    finally:
        # Cleanup
        shutil.rmtree(mock_finn_path)


def test_kernel_registry():
    """Test kernel registry functionality."""
    print("Testing Kernel Registry...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
        db_path = temp_db.name
    
    try:
        # Create registry
        registry = FINNKernelRegistry(database_path=db_path)
        
        # Create test kernel info
        param_schema = ParameterSchema(
            pe_range=(1, 16),
            simd_range=(1, 8),
            custom_parameters={'ActVal': 0.0},
            constraints=['PE > 0', 'SIMD > 0']
        )
        
        test_kernel = FINNKernelInfo(
            name="test_thresholding",
            operator_type="Thresholding",
            backend_type="RTL",
            implementation_files={'python_implementation': '/path/to/test.py'},
            parameterization=param_schema,
            finn_version_compatibility=["0.8.1"]
        )
        
        # Test registration
        result = registry.register_finn_kernel(test_kernel)
        assert result.success, f"Kernel registration failed: {result.message}"
        
        # Test retrieval
        retrieved_kernel = registry.get_kernel("test_thresholding")
        assert retrieved_kernel is not None, "Failed to retrieve registered kernel"
        assert retrieved_kernel.name == "test_thresholding"
        assert retrieved_kernel.operator_type == "Thresholding"
        
        # Test search
        search_criteria = registry.SearchCriteria(operator_type="Thresholding")
        matching_kernels = registry.search_kernels(search_criteria)
        assert len(matching_kernels) >= 1, "Search failed to find registered kernel"
        
        # Test parameter validation
        valid_params = {'PE': 8, 'SIMD': 4, 'ActVal': 0.5}
        is_valid, issues = test_kernel.validate_parameters(valid_params)
        assert is_valid, f"Valid parameters rejected: {issues}"
        
        invalid_params = {'PE': 0, 'SIMD': 10}  # PE too low, SIMD too high
        is_valid, issues = test_kernel.validate_parameters(invalid_params)
        assert not is_valid, "Invalid parameters accepted"
        
        # Test registry stats
        stats = registry.get_registry_stats()
        assert stats['total_kernels'] >= 1
        assert 'Thresholding' in stats['by_operator_type']
        
        print("✅ Kernel Registry tests passed")
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_performance_modeling():
    """Test performance modeling functionality."""
    print("Testing Performance Modeling...")
    
    # Create analytical model
    model = AnalyticalModel("test_matmul", "MatMul")
    
    # Create test platform
    platform = Platform(
        name="test_platform",
        fpga_part="xc7z020clg400-1",
        clock_frequency_mhz=100.0,
        resource_limits={'lut': 50000, 'dsp': 220, 'bram': 140}
    )
    
    # Test performance estimation
    parameters = {'PE': 8, 'SIMD': 4, 'M': 256, 'N': 256, 'K': 256}
    
    # Test individual estimations
    throughput = model.estimate_throughput(parameters, platform)
    assert throughput > 0, "Throughput estimation failed"
    
    latency = model.estimate_latency(parameters, platform)
    assert latency > 0, "Latency estimation failed"
    
    resources = model.estimate_resource_usage(parameters)
    assert 'lut_count' in resources, "Resource estimation missing LUT count"
    assert 'dsp_count' in resources, "Resource estimation missing DSP count"
    
    # Test comprehensive performance estimation
    performance = model.estimate_performance(parameters, platform)
    assert performance.throughput_ops_sec > 0
    assert performance.latency_cycles > 0
    assert performance.confidence > 0
    assert performance.resource_usage
    
    print(f"Performance estimate: {performance.throughput_ops_sec:.0f} ops/sec, "
          f"{performance.latency_cycles} cycles, "
          f"{performance.efficiency_ratio:.2f} efficiency")
    
    print("✅ Performance Modeling tests passed")


def test_kernel_selection():
    """Test kernel selection functionality."""
    print("Testing Kernel Selection...")
    
    # Create mock registry with test kernels
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
        db_path = temp_db.name
    
    try:
        registry = FINNKernelRegistry(database_path=db_path)
        
        # Register test kernels
        for op_type in ["MatMul", "Thresholding"]:
            param_schema = ParameterSchema(
                pe_range=(1, 16),
                simd_range=(1, 8)
            )
            
            kernel = FINNKernelInfo(
                name=f"test_{op_type.lower()}",
                operator_type=op_type,
                backend_type="RTL",
                implementation_files={'python_implementation': '/path/to/test.py'},
                parameterization=param_schema
            )
            
            # Add performance model
            kernel.performance_model = AnalyticalModel(kernel.name, kernel.operator_type)
            
            registry.register_finn_kernel(kernel)
        
        # Create test model
        model = ModelGraph()
        model.nodes = [
            ModelNode(
                id="node1",
                operator_type="MatMul",
                input_shapes=[(256, 256), (256, 256)],
                output_shapes=[(256, 256)]
            ),
            ModelNode(
                id="node2", 
                operator_type="Thresholding",
                input_shapes=[(256, 256)],
                output_shapes=[(256, 256)]
            )
        ]
        model.edges = [("node1", "node2")]
        
        # Create selector
        selector = FINNKernelSelector(registry)
        
        # Define targets and constraints
        targets = PerformanceTargets(
            throughput_ops_sec=1e6,
            priority="performance"
        )
        
        constraints = ResourceConstraints(
            max_lut_count=10000,
            max_dsp_count=100
        )
        
        # Test kernel selection
        selection_plan = selector.select_optimal_kernels(model, targets, constraints)
        
        assert len(selection_plan.assignments) == 2, "Selection plan should have 2 assignments"
        
        # Verify assignments
        node1_assignment = selection_plan.get_assignment("node1")
        assert node1_assignment is not None, "No assignment for node1"
        assert node1_assignment.kernel_info.operator_type == "MatMul"
        
        node2_assignment = selection_plan.get_assignment("node2")
        assert node2_assignment is not None, "No assignment for node2"
        assert node2_assignment.kernel_info.operator_type == "Thresholding"
        
        # Test parameter optimization
        kernels = [assignment.kernel_info for assignment in selection_plan.assignments]
        platform = Platform("test", "xc7z020", 100.0)
        optimized_params = selector.optimize_kernel_parameters(kernels, targets, platform)
        
        assert len(optimized_params) == len(kernels), "Parameter optimization failed"
        
        print(f"Selection plan: {len(selection_plan.assignments)} assignments")
        print(f"Total resources: {selection_plan.get_total_resources()}")
        
        print("✅ Kernel Selection tests passed")
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_finn_config_generation():
    """Test FINN configuration generation."""
    print("Testing FINN Configuration Generation...")
    
    # Create mock selection plan
    from brainsmith.kernels.selection import SelectionPlan, KernelAssignment
    from brainsmith.kernels.performance import PerformanceEstimate
    
    # Create test kernel info
    param_schema = ParameterSchema(pe_range=(1, 16), simd_range=(1, 8))
    kernel_info = FINNKernelInfo(
        name="test_matmul",
        operator_type="MatMul",
        backend_type="RTL",
        implementation_files={'python_implementation': '/path/to/test.py'},
        parameterization=param_schema
    )
    
    # Create test assignment
    performance = PerformanceEstimate(
        throughput_ops_sec=1e6,
        latency_cycles=1000,
        resource_usage={'lut_count': 1000, 'dsp_count': 32}
    )
    
    assignment = KernelAssignment(
        node_id="test_node",
        kernel_info=kernel_info,
        parameters={'PE': 8, 'SIMD': 4},
        estimated_performance=performance
    )
    
    selection_plan = SelectionPlan(assignments=[assignment])
    
    # Test configuration generation
    generator = FINNConfigGenerator()
    
    # Generate build configuration
    config = generator.generate_build_config(
        selection_plan,
        model_path="/path/to/model.onnx",
        platform_config={'fpga_part': 'xc7z020clg400-1', 'clock_frequency_mhz': 100.0}
    )
    
    assert isinstance(config, FINNBuildConfig), "Failed to generate FINN build config"
    assert len(config.kernel_configs) == 1, "Expected 1 kernel configuration"
    assert config.kernel_configs[0].kernel_name == "test_matmul"
    assert config.kernel_configs[0].parameters['PE'] == 8
    
    # Test JSON export
    json_str = config.to_json()
    assert json_str, "Failed to export configuration to JSON"
    
    # Validate JSON can be parsed
    parsed_config = json.loads(json_str)
    assert 'kernel_configs' in parsed_config
    assert 'build_settings' in parsed_config
    
    # Test configuration validation
    is_valid, issues = generator.validate_finn_config(config)
    assert is_valid, f"Generated configuration is invalid: {issues}"
    
    # Test transformation sequence generation
    transformations = generator.generate_finn_transformation_sequence(selection_plan)
    assert len(transformations) > 0, "No transformations generated"
    assert "InferShapes" in transformations, "Missing basic transformation"
    
    # Test script template generation
    script = generator.generate_script_template(config)
    assert "build_dataflow_cfg" in script, "Script template missing FINN build call"
    
    print(f"Generated {len(config.kernel_configs)} kernel configurations")
    print(f"Generated {len(transformations)} transformations")
    
    print("✅ FINN Configuration Generation tests passed")


def test_integration():
    """Test end-to-end integration of kernel registration system."""
    print("Testing End-to-End Integration...")
    
    # Create mock FINN installation
    mock_finn_path = create_mock_finn_installation()
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
        db_path = temp_db.name
    
    try:
        # 1. Discover kernels from FINN installation
        discovery = FINNKernelDiscovery()
        discovered_kernels = discovery.scan_finn_installation(mock_finn_path)
        
        # 2. Register discovered kernels
        registry = FINNKernelRegistry(database_path=db_path)
        
        registered_count = 0
        for kernel_metadata in discovered_kernels:
            # Convert to FINNKernelInfo
            param_schema = ParameterSchema()
            if isinstance(kernel_metadata.parameterization, dict):
                param_schema.pe_range = kernel_metadata.parameterization.get('pe_range', (1, 16))
                param_schema.simd_range = kernel_metadata.parameterization.get('simd_range', (1, 8))
            
            kernel_info = FINNKernelInfo(
                name=kernel_metadata.name,
                operator_type=kernel_metadata.operator_type,
                backend_type=kernel_metadata.backend_type,
                implementation_files=kernel_metadata.implementation_files,
                parameterization=param_schema
            )
            
            # Add performance model
            kernel_info.performance_model = AnalyticalModel(kernel_info.name, kernel_info.operator_type)
            
            result = registry.register_finn_kernel(kernel_info)
            if result.success:
                registered_count += 1
        
        print(f"Registered {registered_count}/{len(discovered_kernels)} kernels")
        
        # 3. Create test model requiring discovered kernels
        model = ModelGraph()
        model.nodes = [
            ModelNode(id="matmul_node", operator_type="MatMul"),
            ModelNode(id="threshold_node", operator_type="Thresholding")
        ]
        
        # 4. Select kernels for model
        selector = FINNKernelSelector(registry)
        targets = PerformanceTargets(throughput_ops_sec=1e6)
        constraints = ResourceConstraints(max_lut_count=50000)
        
        selection_plan = selector.select_optimal_kernels(model, targets, constraints)
        
        assert len(selection_plan.assignments) >= 1, "No kernels selected for model"
        
        # 5. Generate FINN configuration
        generator = FINNConfigGenerator()
        finn_config = generator.generate_build_config(selection_plan)
        
        assert len(finn_config.kernel_configs) >= 1, "No kernel configurations generated"
        
        # 6. Validate final configuration
        is_valid, issues = generator.validate_finn_config(finn_config)
        assert is_valid, f"Final configuration invalid: {issues}"
        
        print(f"✅ End-to-End Integration test passed")
        print(f"   - Discovered: {len(discovered_kernels)} kernels")
        print(f"   - Registered: {registered_count} kernels")
        print(f"   - Selected: {len(selection_plan.assignments)} kernels")
        print(f"   - Generated: {len(finn_config.kernel_configs)} configurations")
        
    finally:
        # Cleanup
        shutil.rmtree(mock_finn_path)
        if os.path.exists(db_path):
            os.unlink(db_path)


def main():
    """Run comprehensive test suite for Month 1 implementation."""
    print("🚀 Starting Kernel Registration System Test Suite")
    print("=" * 60)
    
    try:
        # Run individual component tests
        test_finn_kernel_discovery()
        print()
        
        test_kernel_registry()
        print()
        
        test_performance_modeling()
        print()
        
        test_kernel_selection()
        print()
        
        test_finn_config_generation()
        print()
        
        # Run integration test
        test_integration()
        print()
        
        print("🎉 All tests passed! Month 1 implementation is working correctly.")
        print("\n📊 Month 1 Implementation Status:")
        print("✅ FINN Kernel Discovery Engine - COMPLETE")
        print("✅ Kernel Database with Performance Models - COMPLETE")
        print("✅ Intelligent Kernel Selection Algorithm - COMPLETE")
        print("✅ FINN Configuration Generation - COMPLETE")
        print("\n🎯 Ready to proceed to Month 2: Core Infrastructure")
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()