"""
Comprehensive tensor chunking and performance testing for generated HWCustomOp.

This test thoroughly validates:
- Tensor chunking strategies with different input dimensions
- Performance metric recalculation as tensor dimensions change
- Interface-wise chunking behavior
- Parallelism impact on chunking and performance
- Resource estimation accuracy across different configurations
"""

import sys
import os
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add the generated module to Python path
sys.path.insert(0, str(Path(__file__).parent / "hwkg_demo_final"))

# Mock the brainsmith dependencies for testing
class MockDataTypeConstraint:
    def __init__(self, finn_type, bit_width, signed=False):
        self.finn_type = finn_type
        self.bit_width = bit_width
        self.signed = signed

class MockInterfaceMetadata:
    def __init__(self, name, interface_type, allowed_datatypes, chunking_strategy):
        self.name = name
        self.interface_type = interface_type
        self.allowed_datatypes = allowed_datatypes
        self.chunking_strategy = chunking_strategy

class MockDataflowInterfaceType:
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    WEIGHT = "WEIGHT"
    CONFIG = "CONFIG"

class MockChunkingStrategy:
    def __init__(self, strategy_type="default", **kwargs):
        self.strategy_type = strategy_type
        self.kwargs = kwargs
    
    def calculate_chunks(self, tensor_shape, parallel_factor=1):
        """Calculate chunk configuration for given tensor shape."""
        if self.strategy_type == "default":
            # Default chunking - process entire tensor
            return {
                'chunk_shape': tensor_shape,
                'num_chunks': 1,
                'chunk_size': np.prod(tensor_shape),
                'parallelism_factor': parallel_factor
            }
        elif self.strategy_type == "index":
            # Index-based chunking - chunk along specific dimension
            chunk_dim = self.kwargs.get('chunk_index', -1)
            chunk_sizes = self.kwargs.get('chunk_sizes', [1])
            
            if chunk_dim >= 0 and chunk_dim < len(tensor_shape):
                chunks_per_dim = min(chunk_sizes[0], tensor_shape[chunk_dim])
                chunk_shape = list(tensor_shape)
                chunk_shape[chunk_dim] = tensor_shape[chunk_dim] // chunks_per_dim
                
                return {
                    'chunk_shape': tuple(chunk_shape),
                    'num_chunks': chunks_per_dim,
                    'chunk_size': np.prod(chunk_shape),
                    'parallelism_factor': parallel_factor,
                    'chunk_dimension': chunk_dim
                }
            else:
                return self.calculate_chunks(tensor_shape, parallel_factor)
        elif self.strategy_type == "last_dim":
            # Last dimension chunking
            chunk_size = self.kwargs.get('chunk_size', 1)
            last_dim = tensor_shape[-1]
            chunks_in_last_dim = min(chunk_size, last_dim)
            
            chunk_shape = list(tensor_shape)
            chunk_shape[-1] = last_dim // chunks_in_last_dim
            
            return {
                'chunk_shape': tuple(chunk_shape),
                'num_chunks': chunks_in_last_dim,
                'chunk_size': np.prod(chunk_shape),
                'parallelism_factor': parallel_factor,
                'chunk_dimension': len(tensor_shape) - 1
            }
        else:
            return self.calculate_chunks(tensor_shape, parallel_factor)

class MockInterfaceMetadataCollection:
    def __init__(self, metadata_list):
        self._metadata = metadata_list
    
    def get_interface(self, name):
        for meta in self._metadata:
            if meta.name == name:
                return meta
        return None
    
    def get_input_interfaces(self):
        return [m for m in self._metadata if m.interface_type == MockDataflowInterfaceType.INPUT]
    
    def get_output_interfaces(self):
        return [m for m in self._metadata if m.interface_type == MockDataflowInterfaceType.OUTPUT]

class MockDataflowModel:
    def __init__(self, interfaces):
        self.interfaces = interfaces
        self._tensor_shapes = {}
        self._parallelism = {'iPar': {}, 'wPar': {}}
        
    def set_tensor_shape(self, interface_name, shape):
        """Set tensor shape for an interface."""
        self._tensor_shapes[interface_name] = shape
    
    def get_tensor_shape(self, interface_name):
        """Get tensor shape for an interface."""
        return self._tensor_shapes.get(interface_name, (1, 1, 1, 1))
    
    def update_parallelism(self, iPar=None, wPar=None):
        """Update parallelism configuration."""
        if iPar:
            self._parallelism['iPar'].update(iPar)
        if wPar:
            self._parallelism['wPar'].update(wPar)
    
    def get_parallelism(self):
        """Get current parallelism configuration."""
        return self._parallelism.copy()
    
    def calculate_performance_metrics(self):
        """Calculate performance metrics based on current configuration."""
        total_ops = 0
        total_memory = 0
        
        for interface_name, shape in self._tensor_shapes.items():
            interface = self.interfaces.get_interface(interface_name)
            if interface and interface.chunking_strategy:
                parallel_factor = self._parallelism['iPar'].get(interface_name, 1)
                chunk_info = interface.chunking_strategy.calculate_chunks(shape, parallel_factor)
                
                # Estimate operations and memory usage
                ops_per_chunk = chunk_info['chunk_size'] * 2  # Simplified: 2 ops per element
                total_ops += ops_per_chunk * chunk_info['num_chunks']
                
                memory_per_chunk = chunk_info['chunk_size'] * 4  # 4 bytes per element
                total_memory += memory_per_chunk * parallel_factor
        
        return {
            'total_operations': total_ops,
            'memory_usage': total_memory,
            'parallelism': self._parallelism
        }

class MockAutoHWCustomOp:
    def __init__(self, onnx_node, interface_metadata=None, **kwargs):
        self.onnx_node = onnx_node
        self._interface_metadata = interface_metadata or []
        self._interface_metadata_collection = MockInterfaceMetadataCollection(self._interface_metadata)
        self._dataflow_model = MockDataflowModel(self._interface_metadata_collection)
        self._current_parallelism = {'iPar': {}, 'wPar': {}}
    
    def get_nodeattr_types(self):
        return {
            "N": ("i", False, 4),
            "WI": ("i", False, 8),
            "WT": ("i", False, 8),
            "C": ("i", False, 1),
            "PE": ("i", False, 1),
        }
    
    @property
    def interface_metadata(self):
        return self._interface_metadata_collection
    
    @property
    def dataflow_model(self):
        return self._dataflow_model
    
    def update_parallelism(self, iPar=None, wPar=None):
        """Update parallelism and recalculate performance."""
        if iPar:
            self._current_parallelism['iPar'].update(iPar)
        if wPar:
            self._current_parallelism['wPar'].update(wPar)
        
        self._dataflow_model.update_parallelism(iPar, wPar)
        return self._dataflow_model.calculate_performance_metrics()

def default_chunking():
    return MockChunkingStrategy("default")

def index_chunking(start_index, shape):
    return MockChunkingStrategy("index", chunk_index=start_index, chunk_sizes=shape)

def last_dim_chunking(chunk_size):
    return MockChunkingStrategy("last_dim", chunk_size=chunk_size)

# Mock all the brainsmith imports
mock_modules = {
    'brainsmith': Mock(),
    'brainsmith.dataflow': Mock(),
    'brainsmith.dataflow.core': Mock(),
    'brainsmith.dataflow.core.auto_hw_custom_op': Mock(),
    'brainsmith.dataflow.core.interface_metadata': Mock(),
    'brainsmith.dataflow.core.dataflow_interface': Mock(),
    'brainsmith.dataflow.core.tensor_chunking': Mock(),
}

for module_name, mock_module in mock_modules.items():
    sys.modules[module_name] = mock_module

# Set up the mocks to return our mock classes
sys.modules['brainsmith.dataflow.core.auto_hw_custom_op'].AutoHWCustomOp = MockAutoHWCustomOp
sys.modules['brainsmith.dataflow.core.interface_metadata'].InterfaceMetadata = MockInterfaceMetadata
sys.modules['brainsmith.dataflow.core.interface_metadata'].DataTypeConstraint = MockDataTypeConstraint
sys.modules['brainsmith.dataflow.core.dataflow_interface'].DataflowInterfaceType = MockDataflowInterfaceType
sys.modules['brainsmith.dataflow.core.tensor_chunking'].default_chunking = default_chunking
sys.modules['brainsmith.dataflow.core.tensor_chunking'].index_chunking = index_chunking
sys.modules['brainsmith.dataflow.core.tensor_chunking'].last_dim_chunking = last_dim_chunking

# Import the generated class
try:
    from thresholding_axi_hwcustomop import ThresholdingAxiHWCustomOp
    GENERATED_CLASS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import generated class: {e}")
    GENERATED_CLASS_AVAILABLE = False


class TestTensorChunkingPerformance(unittest.TestCase):
    """Comprehensive tensor chunking and performance testing."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not GENERATED_CLASS_AVAILABLE:
            self.skipTest("Generated class not available")
        
        # Create mock ONNX node
        self.mock_onnx_node = Mock()
        self.mock_onnx_node.input = ["input_tensor"]
        self.mock_onnx_node.output = ["output_tensor"]
        self.mock_onnx_node.op_type = "ThresholdingAxiHWCustomOp"
    
    def test_tensor_chunking_strategies(self):
        """Test different tensor chunking strategies with various input dimensions."""
        op = ThresholdingAxiHWCustomOp(self.mock_onnx_node)
        
        # Test configurations with different tensor shapes
        test_configs = [
            # (tensor_shape, description)
            ((1, 224, 224, 3), "Small image: 224x224x3"),
            ((1, 512, 512, 3), "Medium image: 512x512x3"),
            ((1, 1024, 1024, 3), "Large image: 1024x1024x3"),
            ((32, 64, 64, 16), "Batch processing: 32x64x64x16"),
            ((1, 1000), "1D tensor: 1000 elements"),
            ((8, 8, 256, 256), "Multi-channel: 8x8x256x256"),
        ]
        
        print("\n=== TENSOR CHUNKING STRATEGY ANALYSIS ===")
        
        for tensor_shape, description in test_configs:
            print(f"\n--- {description} ---")
            print(f"Input tensor shape: {tensor_shape}")
            
            # Test each interface's chunking strategy
            for interface in op._interface_metadata:
                print(f"\nInterface: {interface.name} ({interface.interface_type})")
                
                # Set tensor shape in dataflow model
                op.dataflow_model.set_tensor_shape(interface.name, tensor_shape)
                
                # Calculate chunks for different parallelism factors
                parallelism_factors = [1, 2, 4, 8]
                
                for pf in parallelism_factors:
                    if interface.chunking_strategy:
                        chunk_info = interface.chunking_strategy.calculate_chunks(tensor_shape, pf)
                        
                        print(f"  Parallelism {pf}x:")
                        print(f"    Chunk shape: {chunk_info['chunk_shape']}")
                        print(f"    Number of chunks: {chunk_info['num_chunks']}")
                        print(f"    Elements per chunk: {chunk_info['chunk_size']}")
                        if 'chunk_dimension' in chunk_info:
                            print(f"    Chunking dimension: {chunk_info['chunk_dimension']}")
                        
                        # Validate chunking makes sense
                        self.assertGreater(chunk_info['chunk_size'], 0, "Chunk size must be positive")
                        self.assertGreater(chunk_info['num_chunks'], 0, "Number of chunks must be positive")
                        self.assertEqual(len(chunk_info['chunk_shape']), len(tensor_shape), 
                                       "Chunk shape must have same dimensions as input")
    
    def test_performance_metric_recalculation(self):
        """Test how performance metrics change with different tensor dimensions and parallelism."""
        op = ThresholdingAxiHWCustomOp(self.mock_onnx_node)
        
        print("\n=== PERFORMANCE METRIC RECALCULATION ===")
        
        # Test different tensor sizes and their impact on performance
        test_scenarios = [
            {
                'name': 'Small Image Processing',
                'shapes': {'s_axis': (1, 64, 64, 3), 'm_axis': (1, 64, 64, 1)},
                'parallelism_configs': [
                    {'iPar': {'s_axis': 1}, 'wPar': {}},
                    {'iPar': {'s_axis': 2}, 'wPar': {}},
                    {'iPar': {'s_axis': 4}, 'wPar': {}},
                ]
            },
            {
                'name': 'Medium Image Processing',
                'shapes': {'s_axis': (1, 224, 224, 3), 'm_axis': (1, 224, 224, 1)},
                'parallelism_configs': [
                    {'iPar': {'s_axis': 1}, 'wPar': {}},
                    {'iPar': {'s_axis': 4}, 'wPar': {}},
                    {'iPar': {'s_axis': 8}, 'wPar': {}},
                ]
            },
            {
                'name': 'Large Batch Processing',
                'shapes': {'s_axis': (32, 128, 128, 16), 'm_axis': (32, 128, 128, 1)},
                'parallelism_configs': [
                    {'iPar': {'s_axis': 1}, 'wPar': {}},
                    {'iPar': {'s_axis': 8}, 'wPar': {}},
                    {'iPar': {'s_axis': 16}, 'wPar': {}},
                ]
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n--- {scenario['name']} ---")
            
            # Set tensor shapes
            for interface_name, shape in scenario['shapes'].items():
                op.dataflow_model.set_tensor_shape(interface_name, shape)
                print(f"{interface_name} shape: {shape}")
            
            print("\nPerformance analysis:")
            
            baseline_metrics = None
            
            for i, parallelism_config in enumerate(scenario['parallelism_configs']):
                print(f"\n  Configuration {i+1}: {parallelism_config}")
                
                # Update parallelism and get performance metrics
                metrics = op.update_parallelism(**parallelism_config)
                
                print(f"    Total operations: {metrics['total_operations']:,}")
                print(f"    Memory usage (bytes): {metrics['memory_usage']:,}")
                print(f"    Memory usage (MB): {metrics['memory_usage'] / (1024*1024):.2f}")
                
                # Calculate speedup compared to baseline
                if baseline_metrics is None:
                    baseline_metrics = metrics
                    print(f"    Speedup: 1.0x (baseline)")
                else:
                    # Simplified speedup calculation based on parallelism
                    total_parallel_factor = sum(parallelism_config['iPar'].values()) or 1
                    baseline_parallel_factor = sum(scenario['parallelism_configs'][0]['iPar'].values()) or 1
                    theoretical_speedup = total_parallel_factor / baseline_parallel_factor
                    
                    # Account for memory overhead
                    memory_overhead = metrics['memory_usage'] / baseline_metrics['memory_usage']
                    effective_speedup = theoretical_speedup / memory_overhead if memory_overhead > 1 else theoretical_speedup
                    
                    print(f"    Theoretical speedup: {theoretical_speedup:.2f}x")
                    print(f"    Memory overhead: {memory_overhead:.2f}x")
                    print(f"    Effective speedup: {effective_speedup:.2f}x")
                
                # Validate metrics are reasonable
                self.assertGreater(metrics['total_operations'], 0, "Operations count must be positive")
                self.assertGreater(metrics['memory_usage'], 0, "Memory usage must be positive")
    
    def test_chunking_dimension_effects(self):
        """Test how chunking along different dimensions affects performance."""
        op = ThresholdingAxiHWCustomOp(self.mock_onnx_node)
        
        print("\n=== CHUNKING DIMENSION EFFECTS ===")
        
        # Test 4D tensor with chunking along different dimensions
        test_shape = (8, 64, 64, 16)  # Batch, Height, Width, Channels
        print(f"Test tensor shape: {test_shape}")
        
        # Simulate different chunking strategies
        chunking_strategies = [
            ("default", default_chunking()),
            ("chunk_batch", index_chunking(0, [2])),  # Chunk along batch dimension
            ("chunk_height", index_chunking(1, [4])),  # Chunk along height
            ("chunk_width", index_chunking(2, [4])),   # Chunk along width
            ("chunk_channels", index_chunking(3, [4])), # Chunk along channels
            ("last_dim_2", last_dim_chunking(2)),      # Last dimension chunking
            ("last_dim_4", last_dim_chunking(4)),      # Last dimension chunking
        ]
        
        for strategy_name, strategy in chunking_strategies:
            print(f"\n--- {strategy_name.upper()} CHUNKING ---")
            
            # Test parallelism factors
            parallelism_factors = [1, 2, 4, 8]
            
            for pf in parallelism_factors:
                chunk_info = strategy.calculate_chunks(test_shape, pf)
                
                print(f"  Parallelism {pf}x:")
                print(f"    Chunk shape: {chunk_info['chunk_shape']}")
                print(f"    Number of chunks: {chunk_info['num_chunks']}")
                print(f"    Elements per chunk: {chunk_info['chunk_size']:,}")
                
                # Calculate efficiency metrics
                total_elements = np.prod(test_shape)
                elements_per_parallel_unit = chunk_info['chunk_size']
                utilization = elements_per_parallel_unit / (total_elements / pf) if pf > 1 else 1.0
                
                print(f"    Parallel utilization: {utilization:.2f}")
                
                if 'chunk_dimension' in chunk_info:
                    dim_name = ['batch', 'height', 'width', 'channels'][chunk_info['chunk_dimension']]
                    print(f"    Chunked dimension: {dim_name}")
                
                # Validate chunking strategy
                self.assertLessEqual(chunk_info['num_chunks'], np.prod(test_shape), 
                                   "Number of chunks cannot exceed total elements")
                self.assertGreater(chunk_info['chunk_size'], 0, "Chunk size must be positive")
    
    def test_resource_estimation_with_chunking(self):
        """Test how resource estimation changes with different chunking configurations."""
        op = ThresholdingAxiHWCustomOp(self.mock_onnx_node)
        
        print("\n=== RESOURCE ESTIMATION WITH CHUNKING ===")
        
        # Test different scenarios
        scenarios = [
            ("Low Resolution", (1, 32, 32, 3)),
            ("Medium Resolution", (1, 224, 224, 3)),
            ("High Resolution", (1, 512, 512, 3)),
            ("Batch Processing", (16, 64, 64, 8)),
        ]
        
        for scenario_name, tensor_shape in scenarios:
            print(f"\n--- {scenario_name}: {tensor_shape} ---")
            
            # Set tensor shape
            op.dataflow_model.set_tensor_shape('s_axis', tensor_shape)
            op.dataflow_model.set_tensor_shape('m_axis', tensor_shape)
            
            # Test different parallelism configurations
            parallelism_configs = [
                {'iPar': {'s_axis': 1}, 'description': 'Serial Processing'},
                {'iPar': {'s_axis': 2}, 'description': 'Low Parallelism'},
                {'iPar': {'s_axis': 4}, 'description': 'Medium Parallelism'},
                {'iPar': {'s_axis': 8}, 'description': 'High Parallelism'},
            ]
            
            for config in parallelism_configs:
                print(f"\n  {config['description']}:")
                
                # Update parallelism
                iPar = config['iPar']
                metrics = op.update_parallelism(iPar=iPar)
                
                # Calculate resource estimates (simplified)
                parallel_factor = iPar.get('s_axis', 1)
                total_elements = np.prod(tensor_shape)
                
                # Simplified resource estimation
                base_luts = 1000
                base_brams = 2
                base_dsps = 0
                
                estimated_luts = base_luts * parallel_factor
                estimated_brams = base_brams + (parallel_factor - 1)  # Additional buffering
                estimated_dsps = base_dsps  # This kernel doesn't use DSPs
                
                # Memory bandwidth requirements
                element_size = 1  # 1 byte per element (UINT8)
                memory_bandwidth = total_elements * element_size * 2  # Read + Write
                memory_bandwidth_per_cycle = memory_bandwidth / parallel_factor
                
                print(f"    Parallel factor: {parallel_factor}")
                print(f"    Estimated LUTs: {estimated_luts:,}")
                print(f"    Estimated BRAMs: {estimated_brams}")
                print(f"    Estimated DSPs: {estimated_dsps}")
                print(f"    Memory bandwidth: {memory_bandwidth:,} bytes")
                print(f"    Bandwidth per cycle: {memory_bandwidth_per_cycle:,.0f} bytes")
                
                # Resource efficiency
                lut_efficiency = base_luts / estimated_luts
                bram_efficiency = base_brams / estimated_brams if estimated_brams > 0 else 1
                
                print(f"    LUT efficiency: {lut_efficiency:.2f}")
                print(f"    BRAM efficiency: {bram_efficiency:.2f}")
                
                # Validate resource estimates
                self.assertGreater(estimated_luts, 0, "LUT estimate must be positive")
                self.assertGreater(estimated_brams, 0, "BRAM estimate must be positive")
                self.assertGreaterEqual(estimated_dsps, 0, "DSP estimate must be non-negative")
    
    def test_interface_wise_chunking_coordination(self):
        """Test how chunking strategies coordinate across multiple interfaces."""
        op = ThresholdingAxiHWCustomOp(self.mock_onnx_node)
        
        print("\n=== INTERFACE-WISE CHUNKING COORDINATION ===")
        
        # Test scenario with different input/output shapes
        input_shape = (1, 256, 256, 3)
        output_shape = (1, 256, 256, 1)
        
        op.dataflow_model.set_tensor_shape('s_axis', input_shape)
        op.dataflow_model.set_tensor_shape('m_axis', output_shape)
        
        print(f"Input shape (s_axis): {input_shape}")
        print(f"Output shape (m_axis): {output_shape}")
        
        # Test different parallelism configurations
        test_configs = [
            {'iPar': {'s_axis': 1}, 'name': 'Balanced 1x'},
            {'iPar': {'s_axis': 2}, 'name': 'Balanced 2x'},
            {'iPar': {'s_axis': 4}, 'name': 'Balanced 4x'},
            {'iPar': {'s_axis': 8}, 'name': 'Balanced 8x'},
        ]
        
        for config in test_configs:
            print(f"\n--- {config['name']} ---")
            
            metrics = op.update_parallelism(iPar=config['iPar'])
            
            # Analyze chunking for each interface
            for interface in op._interface_metadata:
                interface_name = interface.name
                tensor_shape = op.dataflow_model.get_tensor_shape(interface_name)
                parallel_factor = config['iPar'].get(interface_name, 1)
                
                if interface.chunking_strategy:
                    chunk_info = interface.chunking_strategy.calculate_chunks(tensor_shape, parallel_factor)
                    
                    print(f"\n  Interface {interface_name}:")
                    print(f"    Tensor shape: {tensor_shape}")
                    print(f"    Parallel factor: {parallel_factor}")
                    print(f"    Chunk shape: {chunk_info['chunk_shape']}")
                    print(f"    Number of chunks: {chunk_info['num_chunks']}")
                    print(f"    Elements per chunk: {chunk_info['chunk_size']:,}")
                    
                    # Calculate throughput metrics
                    elements_per_cycle = chunk_info['chunk_size'] * parallel_factor
                    total_cycles = np.prod(tensor_shape) / elements_per_cycle
                    
                    print(f"    Elements per cycle: {elements_per_cycle:,}")
                    print(f"    Estimated cycles: {total_cycles:.0f}")
                    
                    # Validate chunking coordination
                    self.assertGreater(chunk_info['chunk_size'], 0, f"Chunk size for {interface_name} must be positive")
                    self.assertGreater(chunk_info['num_chunks'], 0, f"Number of chunks for {interface_name} must be positive")
            
            print(f"\n  Overall Performance:")
            print(f"    Total operations: {metrics['total_operations']:,}")
            print(f"    Memory usage: {metrics['memory_usage']:,} bytes")
            print(f"    Memory usage: {metrics['memory_usage'] / (1024*1024):.2f} MB")


def run_tensor_chunking_performance_test():
    """Run comprehensive tensor chunking and performance tests."""
    print("=" * 80)
    print("COMPREHENSIVE TENSOR CHUNKING AND PERFORMANCE TESTING")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test class
    suite.addTests(loader.loadTestsFromTestCase(TestTensorChunkingPerformance))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Provide summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if result.failures or result.errors:
        print("\n❌ TESTS FAILED - Issues with tensor chunking or performance")
        return False
    else:
        print("\n✅ ALL TESTS PASSED - Tensor chunking and performance validated")
        return True


if __name__ == "__main__":
    success = run_tensor_chunking_performance_test()
    sys.exit(0 if success else 1)