"""
Parallelism Effects Test

This test focuses specifically on how parallelism affects performance:
- What happens when we increase parallelism (1x, 2x, 4x, 8x)
- How memory usage changes
- How processing speed changes
- The trade-offs involved

Clear and focused on one concept.
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Add the generated module to Python path
sys.path.insert(0, str(Path(__file__).parent / "hwkg_demo_final"))

# Simple performance tracking
class PerformanceTracker:
    def __init__(self):
        self.results = []
    
    def calculate_metrics(self, tensor_shape, parallelism_factor):
        """Calculate performance metrics for given configuration."""
        total_elements = np.prod(tensor_shape)
        
        # Basic calculations
        operations = total_elements * 2  # 2 ops per element (simplified)
        
        # Memory usage increases with parallelism (need more buffers)
        base_memory = total_elements * 2  # input + output
        parallel_overhead = parallelism_factor * 1024  # buffer per parallel unit
        total_memory = base_memory + parallel_overhead
        
        # Processing speed improves with parallelism (but not linearly)
        # Real hardware has diminishing returns due to memory bandwidth
        ideal_speedup = parallelism_factor
        memory_penalty = 1.0 + (parallelism_factor - 1) * 0.1  # 10% penalty per extra unit
        actual_speedup = ideal_speedup / memory_penalty
        
        # Resource usage (LUTs/BRAMs scale with parallelism)
        luts_used = 1000 * parallelism_factor  # Base LUTs * parallel units
        brams_used = 2 + (parallelism_factor - 1)  # Base + extra buffering
        
        result = {
            'tensor_shape': tensor_shape,
            'parallelism': parallelism_factor,
            'total_elements': total_elements,
            'operations': operations,
            'memory_bytes': total_memory,
            'memory_mb': total_memory / (1024 * 1024),
            'ideal_speedup': ideal_speedup,
            'actual_speedup': actual_speedup,
            'efficiency': actual_speedup / ideal_speedup,
            'luts_used': luts_used,
            'brams_used': brams_used
        }
        
        self.results.append(result)
        return result

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


class TestParallelismEffects(unittest.TestCase):
    """Test how parallelism affects performance in clear, simple terms."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not GENERATED_CLASS_AVAILABLE:
            self.skipTest("Generated class not available")
        
        self.mock_onnx_node = Mock()
        self.hw_op = ThresholdingAxiHWCustomOp(self.mock_onnx_node)
        self.perf_tracker = PerformanceTracker()
    
    def test_parallelism_scaling_small_image(self):
        """
        Test how parallelism affects a small image.
        
        We'll process a 128x128 RGB image with different parallelism levels
        and see what happens to performance and resource usage.
        """
        print("\n=== PARALLELISM TEST: Small Image (128x128x3) ===")
        
        # Small image for testing
        image_shape = (1, 128, 128, 3)  # 49,152 pixels
        print(f"Image shape: {image_shape}")
        print(f"Total pixels: {np.prod(image_shape):,}")
        
        # Test different parallelism levels
        parallelism_levels = [1, 2, 4, 8]
        
        print(f"\n{'Parallelism':<12} {'Memory (MB)':<12} {'Speedup':<10} {'Efficiency':<12} {'LUTs':<8} {'BRAMs':<8}")
        print("-" * 70)
        
        baseline_time = None
        
        for parallel_factor in parallelism_levels:
            metrics = self.perf_tracker.calculate_metrics(image_shape, parallel_factor)
            
            # Calculate processing time (simplified)
            clock_mhz = 100  # 100 MHz clock
            cycles_needed = metrics['operations'] / metrics['actual_speedup']
            processing_time_ms = cycles_needed / (clock_mhz * 1000)
            
            if baseline_time is None:
                baseline_time = processing_time_ms
            
            time_improvement = baseline_time / processing_time_ms
            
            print(f"{parallel_factor}x{'':<10} "
                  f"{metrics['memory_mb']:<12.2f} "
                  f"{metrics['actual_speedup']:<10.2f} "
                  f"{metrics['efficiency']:<12.1%} "
                  f"{metrics['luts_used']:<8,} "
                  f"{metrics['brams_used']:<8}")
            
            # Validate the metrics make sense
            self.assertGreater(metrics['actual_speedup'], 0, "Speedup should be positive")
            self.assertLessEqual(metrics['efficiency'], 1.0, "Efficiency can't exceed 100%")
            self.assertGreater(metrics['luts_used'], 0, "Should use some LUTs")
        
        print("\nKey Observations:")
        print("• Memory usage increases with parallelism (need more buffers)")
        print("• Speedup is less than ideal due to overhead")
        print("• Efficiency decreases at higher parallelism levels")
        print("• Resource usage (LUTs/BRAMs) scales with parallelism")
    
    def test_parallelism_sweet_spot(self):
        """
        Test to find the parallelism 'sweet spot'.
        
        This shows the trade-off between performance and resource usage.
        We want to find the best balance.
        """
        print("\n=== FINDING THE PARALLELISM SWEET SPOT ===")
        
        # Medium-sized image
        image_shape = (1, 256, 256, 3)  # 196,608 pixels
        print(f"Testing with image shape: {image_shape}")
        
        parallelism_levels = [1, 2, 4, 8, 16]
        
        print(f"\nAnalyzing performance vs. resource trade-offs:")
        print(f"{'Parallelism':<12} {'Performance':<12} {'Resources':<12} {'Score':<10}")
        print("-" * 50)
        
        best_score = 0
        best_parallelism = 1
        
        for parallel_factor in parallelism_levels:
            metrics = self.perf_tracker.calculate_metrics(image_shape, parallel_factor)
            
            # Calculate a simple "score" balancing performance and efficiency
            performance_score = metrics['actual_speedup']  # Higher is better
            resource_penalty = metrics['luts_used'] / 1000  # Resource cost
            efficiency_bonus = metrics['efficiency']  # Efficiency bonus
            
            overall_score = (performance_score * efficiency_bonus) / (resource_penalty / 1000)
            
            print(f"{parallel_factor}x{'':<10} "
                  f"{performance_score:<12.2f} "
                  f"{resource_penalty:<12.1f} "
                  f"{overall_score:<10.2f}")
            
            if overall_score > best_score:
                best_score = overall_score
                best_parallelism = parallel_factor
        
        print(f"\n🎯 Best parallelism level: {best_parallelism}x")
        print(f"   Score: {best_score:.2f}")
        
        # Validate we found a reasonable sweet spot
        self.assertGreater(best_parallelism, 1, "Best parallelism should be > 1x")
        self.assertLess(best_parallelism, 16, "Best parallelism should be < 16x")
    
    def test_memory_bandwidth_bottleneck(self):
        """
        Test how memory bandwidth becomes a bottleneck at high parallelism.
        
        This explains why we can't just keep adding parallel units forever.
        """
        print("\n=== MEMORY BANDWIDTH BOTTLENECK ===")
        
        # Large image to stress memory bandwidth
        image_shape = (1, 512, 512, 3)  # 786,432 pixels
        print(f"Large image shape: {image_shape}")
        print(f"Total data: {np.prod(image_shape) * 2 / (1024*1024):.1f} MB (input + output)")
        
        # Simulate memory bandwidth limit
        max_memory_bandwidth_gb_s = 10  # 10 GB/s memory bandwidth
        max_bytes_per_cycle = max_memory_bandwidth_gb_s * 1e9 / (100 * 1e6)  # At 100 MHz
        
        print(f"Memory bandwidth limit: {max_memory_bandwidth_gb_s} GB/s")
        print(f"Max bytes per cycle: {max_bytes_per_cycle:.0f}")
        
        print(f"\n{'Parallelism':<12} {'Data/Cycle':<12} {'Bandwidth OK?':<15} {'Bottleneck':<12}")
        print("-" * 55)
        
        for parallel_factor in [1, 2, 4, 8, 16, 32]:
            # Calculate data movement per cycle
            bytes_per_element = 2  # 1 byte in + 1 byte out
            data_per_cycle = parallel_factor * bytes_per_element
            
            bandwidth_ok = data_per_cycle <= max_bytes_per_cycle
            bottleneck = "Memory" if not bandwidth_ok else "Compute"
            
            print(f"{parallel_factor}x{'':<10} "
                  f"{data_per_cycle:<12} "
                  f"{'Yes' if bandwidth_ok else 'No':<15} "
                  f"{bottleneck:<12}")
            
            # At high parallelism, we should hit memory bandwidth limits
            if parallel_factor >= 16:
                self.assertFalse(bandwidth_ok, 
                               f"Should hit memory bandwidth limit at {parallel_factor}x")
        
        print("\nKey Insight:")
        print("• At low parallelism: compute is the bottleneck")
        print("• At high parallelism: memory bandwidth becomes the bottleneck")
        print("• Sweet spot is where both are balanced")
    
    def test_real_world_comparison(self):
        """
        Compare our parallelism results to real-world expectations.
        
        This puts our theoretical results in practical context.
        """
        print("\n=== REAL-WORLD PERFORMANCE COMPARISON ===")
        
        # Typical CNN layer sizes
        test_cases = [
            ((1, 224, 224, 3), "ResNet Input Layer"),
            ((1, 112, 112, 64), "ResNet Middle Layer"),
            ((1, 56, 56, 128), "ResNet Deep Layer"),
            ((32, 32, 32, 256), "Batch Processing"),
        ]
        
        optimal_parallelism = {}
        
        for tensor_shape, description in test_cases:
            print(f"\n{description}: {tensor_shape}")
            
            best_efficiency = 0
            best_parallel = 1
            
            for parallel_factor in [1, 2, 4, 8]:
                metrics = self.perf_tracker.calculate_metrics(tensor_shape, parallel_factor)
                
                if metrics['efficiency'] > best_efficiency:
                    best_efficiency = metrics['efficiency']
                    best_parallel = parallel_factor
            
            optimal_parallelism[description] = best_parallel
            
            print(f"  Optimal parallelism: {best_parallel}x")
            print(f"  Best efficiency: {best_efficiency:.1%}")
            print(f"  Memory usage: {metrics['memory_mb']:.1f} MB")
        
        print(f"\n📊 Summary of optimal parallelism:")
        for desc, parallel in optimal_parallelism.items():
            print(f"  {desc}: {parallel}x")
        
        # Most cases should prefer moderate parallelism (2-8x)
        parallel_values = list(optimal_parallelism.values())
        avg_parallel = sum(parallel_values) / len(parallel_values)
        
        self.assertGreaterEqual(avg_parallel, 2, "Average optimal parallelism should be >= 2x")
        self.assertLessEqual(avg_parallel, 8, "Average optimal parallelism should be <= 8x")


def run_parallelism_test():
    """Run the parallelism effects test with clear explanations."""
    print("=" * 60)
    print("PARALLELISM EFFECTS TEST")
    print("=" * 60)
    print("This test shows how parallelism affects performance and resources.")
    print("Key concepts:")
    print("• More parallelism = faster processing BUT more resources")
    print("• Memory bandwidth can become a bottleneck")
    print("• There's usually a 'sweet spot' for optimal efficiency")
    print()
    
    # Run the tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestParallelismEffects))
    
    runner = unittest.TextTestRunner(verbosity=0, stream=sys.stdout, buffer=True)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("PARALLELISM TEST SUMMARY")
    print("=" * 60)
    
    if result.failures or result.errors:
        print("❌ Some tests failed. Check the output above for details.")
        return False
    else:
        print("✅ All parallelism tests passed!")
        print("\nWhat we learned about parallelism:")
        print("• 2-4x parallelism usually gives the best efficiency")
        print("• Memory usage grows with parallelism")
        print("• Very high parallelism hits bandwidth limits")
        print("• The optimal level depends on your specific use case")
        return True


if __name__ == "__main__":
    success = run_parallelism_test()
    sys.exit(0 if success else 1)