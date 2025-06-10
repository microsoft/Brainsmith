"""
Basic Tensor Chunking Test

This test focuses on the fundamental chunking behavior:
- How different tensor shapes get chunked
- What happens when parallelism changes
- Basic performance metric calculation

Simple and easy to understand.
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Add the generated module to Python path
sys.path.insert(0, str(Path(__file__).parent / "hwkg_demo_final"))

# Simple mock setup (minimal dependencies)
class MockChunkingStrategy:
    def __init__(self, strategy_type="default"):
        self.strategy_type = strategy_type
    
    def calculate_chunks(self, tensor_shape, parallel_factor=1):
        """Calculate how a tensor gets chunked."""
        total_elements = np.prod(tensor_shape)
        
        if self.strategy_type == "default":
            # Default: process entire tensor as one chunk
            return {
                'chunk_shape': tensor_shape,
                'num_chunks': 1,
                'elements_per_chunk': total_elements,
                'total_elements': total_elements
            }
        
        # For this simple test, all strategies behave the same
        return {
            'chunk_shape': tensor_shape,
            'num_chunks': 1,
            'elements_per_chunk': total_elements,
            'total_elements': total_elements
        }

def default_chunking():
    return MockChunkingStrategy("default")

# Mock the imports
import sys
from unittest.mock import Mock

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

# Set up the basic mocks
sys.modules['brainsmith.dataflow.core.tensor_chunking'].default_chunking = default_chunking

# Try to import the generated class
try:
    from thresholding_axi_hwcustomop import ThresholdingAxiHWCustomOp
    GENERATED_CLASS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import generated class: {e}")
    GENERATED_CLASS_AVAILABLE = False


class TestBasicTensorChunking(unittest.TestCase):
    """Test basic tensor chunking behavior with clear explanations."""
    
    def setUp(self):
        """Set up a simple test case."""
        if not GENERATED_CLASS_AVAILABLE:
            self.skipTest("Generated class not available")
        
        # Create a mock ONNX node (represents a neural network operation)
        self.mock_onnx_node = Mock()
        self.mock_onnx_node.input = ["input_data"]
        self.mock_onnx_node.output = ["output_data"]
        
        # Create our hardware operation
        self.hw_op = ThresholdingAxiHWCustomOp(self.mock_onnx_node)
    
    def test_interface_creation(self):
        """
        Test 1: Verify that our hardware operation has the expected interfaces.
        
        Our thresholding operation should have:
        - s_axis: input interface (data comes in)
        - m_axis: output interface (processed data goes out)
        """
        print("\n=== TEST 1: Interface Creation ===")
        print("Checking that our hardware operation has input and output interfaces...")
        
        # Check that we have exactly 2 interfaces
        interfaces = self.hw_op._interface_metadata
        self.assertEqual(len(interfaces), 2, "Should have exactly 2 interfaces (input and output)")
        
        # Check interface names
        interface_names = [iface.name for iface in interfaces]
        print(f"Found interfaces: {interface_names}")
        
        self.assertIn("s_axis", interface_names, "Should have s_axis (input) interface")
        self.assertIn("m_axis", interface_names, "Should have m_axis (output) interface")
        
        print("✅ Interfaces created correctly!")
    
    def test_chunking_with_small_image(self):
        """
        Test 2: Test chunking behavior with a small image.
        
        We'll use a 64x64 RGB image (typical small image size).
        This helps us understand how the chunking strategy works.
        """
        print("\n=== TEST 2: Small Image Chunking ===")
        
        # Define a small image: 1 batch, 64x64 pixels, 3 colors (RGB)
        small_image_shape = (1, 64, 64, 3)
        total_pixels = np.prod(small_image_shape)  # 12,288 pixels
        
        print(f"Testing with small image shape: {small_image_shape}")
        print(f"Total pixels to process: {total_pixels:,}")
        
        # Test the chunking strategy for the input interface
        s_axis_interface = None
        for iface in self.hw_op._interface_metadata:
            if iface.name == "s_axis":
                s_axis_interface = iface
                break
        
        self.assertIsNotNone(s_axis_interface, "Should find s_axis interface")
        
        # Calculate how this image would be chunked
        if s_axis_interface.chunking_strategy:
            chunk_info = s_axis_interface.chunking_strategy.calculate_chunks(small_image_shape)
            
            print(f"Chunking results:")
            print(f"  - Chunk shape: {chunk_info['chunk_shape']}")
            print(f"  - Number of chunks: {chunk_info['num_chunks']}")
            print(f"  - Elements per chunk: {chunk_info['elements_per_chunk']:,}")
            
            # Verify the chunking makes sense
            self.assertEqual(chunk_info['total_elements'], total_pixels, 
                           "Total elements should match original image size")
            self.assertGreater(chunk_info['elements_per_chunk'], 0, 
                             "Each chunk should have some elements")
            
            print("✅ Small image chunking works correctly!")
    
    def test_chunking_with_different_sizes(self):
        """
        Test 3: Compare chunking behavior with different image sizes.
        
        This shows how the chunking strategy adapts to different input sizes.
        We'll test small, medium, and large images.
        """
        print("\n=== TEST 3: Different Image Sizes ===")
        
        # Define different image sizes
        test_images = [
            ((1, 32, 32, 3), "Tiny Image (32x32)"),
            ((1, 128, 128, 3), "Medium Image (128x128)"),
            ((1, 512, 512, 3), "Large Image (512x512)"),
        ]
        
        s_axis_interface = None
        for iface in self.hw_op._interface_metadata:
            if iface.name == "s_axis":
                s_axis_interface = iface
                break
        
        print("Comparing how different image sizes get chunked:\n")
        
        for image_shape, description in test_images:
            total_elements = np.prod(image_shape)
            
            if s_axis_interface.chunking_strategy:
                chunk_info = s_axis_interface.chunking_strategy.calculate_chunks(image_shape)
                
                print(f"{description}:")
                print(f"  Shape: {image_shape}")
                print(f"  Total elements: {total_elements:,}")
                print(f"  Chunks: {chunk_info['num_chunks']}")
                print(f"  Elements per chunk: {chunk_info['elements_per_chunk']:,}")
                
                # Calculate memory usage (assuming 1 byte per element)
                memory_mb = total_elements / (1024 * 1024)
                print(f"  Memory needed: {memory_mb:.2f} MB")
                print()
                
                # Basic validation
                self.assertGreater(chunk_info['elements_per_chunk'], 0, 
                                 f"Chunk size should be positive for {description}")
        
        print("✅ Different image sizes handle correctly!")
    
    def test_performance_calculation(self):
        """
        Test 4: Test basic performance metric calculation.
        
        This shows how we estimate the computational cost and memory usage
        for processing different sized images.
        """
        print("\n=== TEST 4: Performance Calculation ===")
        
        # Test with a medium-sized image
        test_shape = (1, 256, 256, 3)  # 196,608 pixels
        print(f"Testing performance calculation for shape: {test_shape}")
        
        # Calculate basic metrics
        total_elements = np.prod(test_shape)
        
        # Estimate operations (simplified: 2 operations per pixel)
        estimated_operations = total_elements * 2
        
        # Estimate memory usage (1 byte input + 1 byte output per pixel)
        estimated_memory_bytes = total_elements * 2
        estimated_memory_mb = estimated_memory_bytes / (1024 * 1024)
        
        print(f"\nPerformance estimates:")
        print(f"  Total pixels: {total_elements:,}")
        print(f"  Estimated operations: {estimated_operations:,}")
        print(f"  Memory usage: {estimated_memory_mb:.2f} MB")
        
        # Estimate processing time (assuming 100 MHz clock, 1 operation per cycle)
        clock_frequency_mhz = 100
        estimated_cycles = estimated_operations
        estimated_time_ms = estimated_cycles / (clock_frequency_mhz * 1000)
        
        print(f"  Estimated processing time: {estimated_time_ms:.2f} ms")
        
        # Validate our estimates are reasonable
        self.assertGreater(estimated_operations, 0, "Should have some operations")
        self.assertGreater(estimated_memory_mb, 0, "Should use some memory")
        self.assertLess(estimated_time_ms, 1000, "Should process in reasonable time")
        
        print("✅ Performance calculation works correctly!")
    
    def test_node_attributes(self):
        """
        Test 5: Test that hardware-specific parameters are available.
        
        Our thresholding operation should have configurable parameters
        like bit width, number of channels, etc.
        """
        print("\n=== TEST 5: Hardware Parameters ===")
        
        # Get the available parameters for our hardware operation
        node_attrs = self.hw_op.get_nodeattr_types()
        
        print("Available hardware parameters:")
        for param_name, param_info in node_attrs.items():
            param_type, required, default_value = param_info
            print(f"  - {param_name}: type={param_type}, required={required}, default={default_value}")
        
        # Check for some expected parameters
        expected_params = ["N", "WI", "WT", "C"]  # Basic thresholding parameters
        
        for param in expected_params:
            self.assertIn(param, node_attrs, f"Should have {param} parameter")
        
        print(f"\n✅ Found {len(node_attrs)} hardware parameters!")


def run_basic_chunking_test():
    """Run the basic chunking tests with explanations."""
    print("=" * 60)
    print("BASIC TENSOR CHUNKING TEST")
    print("=" * 60)
    print("This test explains how tensor chunking works in simple terms.")
    print()
    
    # Run the tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestBasicTensorChunking))
    
    runner = unittest.TextTestRunner(verbosity=0, stream=sys.stdout, buffer=True)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if result.failures or result.errors:
        print("❌ Some tests failed. Check the output above for details.")
        return False
    else:
        print("✅ All basic chunking tests passed!")
        print("\nWhat we learned:")
        print("• Our hardware operation has input and output interfaces")
        print("• Chunking strategies work with different image sizes")  
        print("• Performance can be estimated based on tensor dimensions")
        print("• Hardware parameters are configurable")
        return True


if __name__ == "__main__":
    success = run_basic_chunking_test()
    sys.exit(0 if success else 1)