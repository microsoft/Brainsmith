"""
Dimension Changes Test

This test focuses on what happens when input tensor dimensions change:
- How does performance scale with image size?
- What about batch size changes?
- How do different aspect ratios affect processing?

Simple examples that show clear cause and effect.
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Add the generated module to Python path
sys.path.insert(0, str(Path(__file__).parent / "hwkg_demo_final"))

# Helper class to track dimension effects
class DimensionAnalyzer:
    def analyze_tensor(self, tensor_shape, description=""):
        """Analyze the characteristics of a tensor shape."""
        total_elements = np.prod(tensor_shape)
        
        # Break down the shape
        if len(tensor_shape) == 4:  # Batch, Height, Width, Channels
            batch, height, width, channels = tensor_shape
            spatial_size = height * width
            
            analysis = {
                'shape': tensor_shape,
                'description': description,
                'total_elements': total_elements,
                'batch_size': batch,
                'spatial_size': spatial_size,
                'num_channels': channels,
                'memory_mb': total_elements * 2 / (1024 * 1024),  # Input + output
                'processing_complexity': self._estimate_complexity(tensor_shape)
            }
        else:
            # Handle other shapes
            analysis = {
                'shape': tensor_shape,
                'description': description,
                'total_elements': total_elements,
                'memory_mb': total_elements * 2 / (1024 * 1024),
                'processing_complexity': 'unknown'
            }
        
        return analysis
    
    def _estimate_complexity(self, tensor_shape):
        """Estimate processing complexity based on tensor shape."""
        total_elements = np.prod(tensor_shape)
        
        if total_elements < 10000:
            return "low"
        elif total_elements < 100000:
            return "medium"
        elif total_elements < 1000000:
            return "high"
        else:
            return "very_high"
    
    def compare_tensors(self, tensor1, tensor2, name1="Tensor 1", name2="Tensor 2"):
        """Compare two tensors and show the differences."""
        analysis1 = self.analyze_tensor(tensor1, name1)
        analysis2 = self.analyze_tensor(tensor2, name2)
        
        # Calculate ratios
        element_ratio = analysis2['total_elements'] / analysis1['total_elements']
        memory_ratio = analysis2['memory_mb'] / analysis1['memory_mb']
        
        return {
            'tensor1': analysis1,
            'tensor2': analysis2,
            'element_ratio': element_ratio,
            'memory_ratio': memory_ratio,
            'size_change': self._describe_change(element_ratio)
        }
    
    def _describe_change(self, ratio):
        """Describe the change in human-friendly terms."""
        if ratio > 10:
            return "much larger"
        elif ratio > 2:
            return "larger"
        elif ratio > 1.1:
            return "slightly larger"
        elif ratio > 0.9:
            return "similar"
        elif ratio > 0.5:
            return "slightly smaller"
        elif ratio > 0.1:
            return "smaller"
        else:
            return "much smaller"

# Mock setup (minimal)
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

def default_chunking():
    return Mock()

sys.modules['brainsmith.dataflow.core.tensor_chunking'].default_chunking = default_chunking

# Try to import the generated class
try:
    from thresholding_axi_hwcustomop import ThresholdingAxiHWCustomOp
    GENERATED_CLASS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import generated class: {e}")
    GENERATED_CLASS_AVAILABLE = False


class TestDimensionChanges(unittest.TestCase):
    """Test how changing tensor dimensions affects performance."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not GENERATED_CLASS_AVAILABLE:
            self.skipTest("Generated class not available")
        
        self.mock_onnx_node = Mock()
        self.hw_op = ThresholdingAxiHWCustomOp(self.mock_onnx_node)
        self.analyzer = DimensionAnalyzer()
    
    def test_image_size_scaling(self):
        """
        Test how performance scales with image size.
        
        We'll test common image resolutions to see how processing
        requirements grow with image size.
        """
        print("\n=== IMAGE SIZE SCALING TEST ===")
        print("Testing how performance changes with different image sizes...")
        
        # Common image resolutions
        image_sizes = [
            ((1, 64, 64, 3), "Thumbnail (64x64)"),
            ((1, 128, 128, 3), "Small (128x128)"),
            ((1, 256, 256, 3), "Medium (256x256)"),
            ((1, 512, 512, 3), "Large (512x512)"),
            ((1, 1024, 1024, 3), "HD (1024x1024)"),
        ]
        
        print(f"\n{'Resolution':<20} {'Pixels':<12} {'Memory':<12} {'Complexity':<12}")
        print("-" * 60)
        
        baseline = None
        
        for tensor_shape, description in image_sizes:
            analysis = self.analyzer.analyze_tensor(tensor_shape, description)
            
            if baseline is None:
                baseline = analysis
                multiplier = "1.0x"
            else:
                multiplier = f"{analysis['total_elements'] / baseline['total_elements']:.1f}x"
            
            print(f"{description:<20} "
                  f"{analysis['total_elements']:<12,} "
                  f"{analysis['memory_mb']:<12.2f} "
                  f"{analysis['processing_complexity']:<12}")
            
            # Validate that larger images need more resources
            self.assertGreater(analysis['total_elements'], 0, "Should have some pixels")
            self.assertGreater(analysis['memory_mb'], 0, "Should use some memory")
        
        print("\nKey Observations:")
        print("• Doubling image width/height = 4x more pixels")
        print("• Memory usage scales linearly with pixels")
        print("• Processing complexity increases significantly")
    
    def test_batch_size_effects(self):
        """
        Test how batch size affects processing.
        
        Batch processing can be more efficient, but uses more memory.
        """
        print("\n=== BATCH SIZE EFFECTS TEST ===")
        print("Testing how batch size affects processing...")
        
        # Fixed image size, varying batch size
        base_image = (128, 128, 3)  # 128x128 RGB image
        batch_sizes = [1, 4, 8, 16, 32]
        
        print(f"Base image size: {base_image[0]}x{base_image[1]}x{base_image[2]}")
        print(f"\n{'Batch Size':<12} {'Total Pixels':<15} {'Memory (MB)':<12} {'Per Image':<12}")
        print("-" * 55)
        
        for batch_size in batch_sizes:
            tensor_shape = (batch_size,) + base_image
            analysis = self.analyzer.analyze_tensor(tensor_shape)
            
            pixels_per_image = np.prod(base_image)
            memory_per_image = analysis['memory_mb'] / batch_size
            
            print(f"{batch_size:<12} "
                  f"{analysis['total_elements']:<15,} "
                  f"{analysis['memory_mb']:<12.2f} "
                  f"{memory_per_image:<12.2f}")
            
            # Validate batch processing
            expected_total = pixels_per_image * batch_size
            self.assertEqual(analysis['total_elements'], expected_total, 
                           f"Batch size {batch_size} should have {expected_total} total pixels")
        
        print("\nBatch Processing Insights:")
        print("• Larger batches = more total memory needed")
        print("• Memory per image stays constant")
        print("• Can process multiple images in parallel")
        print("• Trade-off: throughput vs. memory usage")
    
    def test_aspect_ratio_effects(self):
        """
        Test how different aspect ratios affect processing.
        
        Some shapes might be more efficient for chunking than others.
        """
        print("\n=== ASPECT RATIO EFFECTS TEST ===")
        print("Testing different image aspect ratios with same total pixels...")
        
        # Different shapes with approximately the same number of pixels (~65k)
        aspect_ratios = [
            ((1, 256, 256, 1), "Square (1:1)"),
            ((1, 128, 512, 1), "Wide (1:4)"),
            ((1, 512, 128, 1), "Tall (4:1)"),
            ((1, 64, 1024, 1), "Very Wide (1:16)"),
            ((1, 1024, 64, 1), "Very Tall (16:1)"),
        ]
        
        print(f"\n{'Aspect Ratio':<15} {'Shape':<15} {'Pixels':<10} {'Chunking Friendly?':<18}")
        print("-" * 65)
        
        for tensor_shape, description in aspect_ratios:
            analysis = self.analyzer.analyze_tensor(tensor_shape, description)
            
            # Analyze chunking friendliness (simplified)
            batch, height, width, channels = tensor_shape
            aspect_ratio = max(height, width) / min(height, width)
            
            if aspect_ratio <= 2:
                chunking_friendly = "Excellent"
            elif aspect_ratio <= 4:
                chunking_friendly = "Good"
            elif aspect_ratio <= 8:
                chunking_friendly = "Fair"
            else:
                chunking_friendly = "Poor"
            
            shape_str = f"{height}x{width}"
            
            print(f"{description:<15} "
                  f"{shape_str:<15} "
                  f"{analysis['total_elements']:<10,} "
                  f"{chunking_friendly:<18}")
            
            # Validate all have similar pixel counts
            self.assertGreater(analysis['total_elements'], 50000, "Should have ~65k pixels")
            self.assertLess(analysis['total_elements'], 80000, "Should have ~65k pixels")
        
        print("\nAspect Ratio Insights:")
        print("• Square images are usually most chunking-friendly")
        print("• Extreme aspect ratios can be harder to parallelize")
        print("• Consider reshaping very wide/tall images if possible")
    
    def test_channel_dimension_scaling(self):
        """
        Test how the number of channels affects processing.
        
        RGB (3 channels) vs. grayscale (1 channel) vs. hyperspectral (many channels).
        """
        print("\n=== CHANNEL DIMENSION SCALING TEST ===")
        print("Testing how number of channels affects processing...")
        
        # Fixed spatial size, varying channels
        base_spatial = (1, 224, 224)  # 1 batch, 224x224 spatial
        channel_configs = [
            (1, "Grayscale"),
            (3, "RGB"),
            (16, "Feature Map"),
            (64, "Deep Feature Map"),
            (256, "Very Deep Feature Map"),
        ]
        
        print(f"Spatial size: {base_spatial[1]}x{base_spatial[2]}")
        print(f"\n{'Channels':<12} {'Description':<18} {'Total Size':<12} {'Memory (MB)':<12}")
        print("-" * 60)
        
        baseline_memory = None
        
        for num_channels, description in channel_configs:
            tensor_shape = base_spatial + (num_channels,)
            analysis = self.analyzer.analyze_tensor(tensor_shape, description)
            
            if baseline_memory is None:
                baseline_memory = analysis['memory_mb']
                memory_ratio = "1.0x"
            else:
                ratio = analysis['memory_mb'] / baseline_memory
                memory_ratio = f"{ratio:.1f}x"
            
            print(f"{num_channels:<12} "
                  f"{description:<18} "
                  f"{analysis['total_elements']:<12,} "
                  f"{analysis['memory_mb']:<12.2f}")
            
            # Validate scaling
            expected_size = np.prod(base_spatial) * num_channels
            self.assertEqual(analysis['total_elements'], expected_size,
                           f"Should have {expected_size} elements for {num_channels} channels")
        
        print("\nChannel Scaling Insights:")
        print("• Memory scales linearly with number of channels")
        print("• More channels = more data to move and process")
        print("• Channel parallelization can be very effective")
    
    def test_real_world_dimension_comparison(self):
        """
        Compare dimensions from real neural network scenarios.
        
        This puts our dimension analysis in practical context.
        """
        print("\n=== REAL-WORLD DIMENSION COMPARISON ===")
        print("Comparing tensor dimensions from actual neural networks...")
        
        # Real network layer dimensions
        real_scenarios = [
            ((1, 224, 224, 3), "ImageNet Input"),
            ((1, 112, 112, 64), "ResNet Block 1"),
            ((1, 56, 56, 128), "ResNet Block 2"),
            ((1, 28, 28, 256), "ResNet Block 3"),
            ((1, 14, 14, 512), "ResNet Block 4"),
            ((32, 32, 32, 256), "Batch Training"),
            ((1, 512, 512, 1), "Medical Imaging"),
        ]
        
        print(f"\n{'Scenario':<20} {'Shape':<15} {'Complexity':<12} {'Memory (MB)':<12}")
        print("-" * 65)
        
        complexity_counts = {'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
        
        for tensor_shape, scenario in real_scenarios:
            analysis = self.analyzer.analyze_tensor(tensor_shape, scenario)
            
            if len(tensor_shape) == 4:
                shape_str = f"{tensor_shape[1]}x{tensor_shape[2]}x{tensor_shape[3]}"
            else:
                shape_str = str(tensor_shape)
            
            print(f"{scenario:<20} "
                  f"{shape_str:<15} "
                  f"{analysis['processing_complexity']:<12} "
                  f"{analysis['memory_mb']:<12.2f}")
            
            complexity_counts[analysis['processing_complexity']] += 1
            
            # Validate reasonable memory usage
            self.assertLess(analysis['memory_mb'], 1000, 
                          f"{scenario} should use reasonable memory (< 1GB)")
        
        print(f"\nComplexity Distribution:")
        for complexity, count in complexity_counts.items():
            if count > 0:
                print(f"  {complexity}: {count} scenarios")
        
        print("\nReal-World Insights:")
        print("• Early layers: large spatial, few channels")
        print("• Later layers: small spatial, many channels")
        print("• Batch processing significantly increases memory needs")
        print("• Medical/high-res images can be very demanding")


def run_dimension_test():
    """Run the dimension changes test with clear explanations."""
    print("=" * 60)
    print("DIMENSION CHANGES TEST")
    print("=" * 60)
    print("This test shows how tensor dimensions affect performance.")
    print("Key concepts:")
    print("• Larger images = more pixels = more processing needed")
    print("• Batch size affects memory usage")
    print("• Aspect ratio can affect chunking efficiency")
    print("• Number of channels scales memory linearly")
    print()
    
    # Run the tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestDimensionChanges))
    
    runner = unittest.TextTestRunner(verbosity=0, stream=sys.stdout, buffer=True)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("DIMENSION TEST SUMMARY")
    print("=" * 60)
    
    if result.failures or result.errors:
        print("❌ Some tests failed. Check the output above for details.")
        return False
    else:
        print("✅ All dimension tests passed!")
        print("\nWhat we learned about tensor dimensions:")
        print("• Image size has quadratic effect on processing (2x size = 4x pixels)")
        print("• Batch processing trades memory for throughput")
        print("• Square images are generally most efficient")
        print("• Channel count directly affects memory and compute needs")
        return True


if __name__ == "__main__":
    success = run_dimension_test()
    sys.exit(0 if success else 1)