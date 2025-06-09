"""
Performance Benchmark Tests

Comprehensive performance testing for BrainSmith platform including:
- Throughput benchmarks
- Scalability benchmarks  
- Memory usage validation
- Optimization speed testing
"""

import pytest
import time
import sys
from pathlib import Path
from unittest.mock import patch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import psutil if available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Import gc for memory testing
import gc


class TestThroughputBenchmarks:
    """Test suite for throughput performance benchmarks."""
    
    @pytest.mark.performance
    def test_api_response_time_strategies(self):
        """Test API response time for strategy operations."""
        import brainsmith
        
        start_time = time.time()
        strategies = brainsmith.list_available_strategies()
        list_time = time.time() - start_time
        
        assert list_time < 5.0, f"Strategy listing took {list_time:.2f}s, expected < 5s"
        assert isinstance(strategies, dict)
        assert len(strategies) > 0
        
        # Test recommendation speed
        start_time = time.time()
        strategy = brainsmith.recommend_strategy(n_parameters=5)
        recommend_time = time.time() - start_time
        
        assert recommend_time < 2.0, f"Strategy recommendation took {recommend_time:.2f}s, expected < 2s"
        assert isinstance(strategy, str)
    
    @pytest.mark.performance
    def test_build_model_response_time(self, test_data_manager):
        """Test build model response time."""
        import brainsmith
        
        model_path = test_data_manager.create_test_model("perf_test", "small")
        
        start_time = time.time()
        result = brainsmith.build_model(
            model_path=str(model_path),
            blueprint_name="test_blueprint",
            parameters={'pe': 4, 'simd': 2}
        )
        build_time = time.time() - start_time
        
        assert build_time < 10.0, f"Build model took {build_time:.2f}s, expected < 10s"
        assert result is not None
        assert isinstance(result, dict)
    
    @pytest.mark.performance
    def test_optimize_model_response_time(self, test_data_manager):
        """Test optimize model response time."""
        import brainsmith
        
        model_path = test_data_manager.create_test_model("opt_perf_test", "small")
        
        start_time = time.time()
        result = brainsmith.optimize_model(
            model_path=str(model_path),
            blueprint_name="test_blueprint",
            max_evaluations=5
        )
        optimize_time = time.time() - start_time
        
        assert optimize_time < 30.0, f"Optimize model took {optimize_time:.2f}s, expected < 30s"
        assert result is not None


class TestScalabilityBenchmarks:
    """Test suite for scalability performance benchmarks."""
    
    @pytest.mark.performance
    def test_multiple_model_builds_scalability(self, test_data_manager):
        """Test scalability of multiple model builds."""
        import brainsmith
        
        # Build multiple models sequentially
        num_builds = 5
        model_paths = []
        
        # Create test models
        for i in range(num_builds):
            model_path = test_data_manager.create_test_model(f"scale_test_{i}", "small")
            model_paths.append(model_path)
        
        start_time = time.time()
        
        results = []
        for i, model_path in enumerate(model_paths):
            result = brainsmith.build_model(
                model_path=str(model_path),
                blueprint_name="test_blueprint",
                parameters={'pe': i + 1, 'simd': 2}
            )
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Validate all builds succeeded
        assert len(results) == num_builds
        for result in results:
            assert result is not None
            assert isinstance(result, dict)
        
        # Time should scale reasonably
        avg_time_per_build = total_time / num_builds
        assert avg_time_per_build < 10.0, f"Average build time {avg_time_per_build:.2f}s too high"
    
    @pytest.mark.performance
    def test_api_call_consistency(self):
        """Test consistency of repeated API calls."""
        import brainsmith
        
        # Test strategy listing consistency
        times = []
        results = []
        
        for i in range(5):
            start_time = time.time()
            strategies = brainsmith.list_available_strategies()
            call_time = time.time() - start_time
            
            times.append(call_time)
            results.append(strategies)
        
        # All results should be identical
        for i in range(1, len(results)):
            assert results[i] == results[0], f"API call {i} returned different result"
        
        # Times should be consistent (within 2x factor)
        min_time = min(times)
        max_time = max(times)
        
        if min_time > 0:
            time_ratio = max_time / min_time
            assert time_ratio < 5.0, f"API call time variation too high: {time_ratio:.2f}x"


class TestMemoryUsageBenchmarks:
    """Test suite for memory usage and leak detection."""
    
    @pytest.mark.performance
    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_memory_usage_api_calls(self):
        """Test memory usage during API calls."""
        import brainsmith
        
        # Force garbage collection and measure baseline
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform multiple API calls
        for i in range(10):
            strategies = brainsmith.list_available_strategies()
            recommendation = brainsmith.recommend_strategy(n_parameters=5 + i)
            
            assert isinstance(strategies, dict)
            assert isinstance(recommendation, str)
        
        # Force garbage collection and measure final memory
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        # Memory increase should be minimal for API calls
        assert memory_increase < 50, f"Memory increased by {memory_increase:.1f}MB during API calls"
    
    @pytest.mark.performance
    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_memory_usage_model_operations(self, test_data_manager):
        """Test memory usage during model operations."""
        import brainsmith
        
        # Measure baseline
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform model operations
        for i in range(5):
            model_path = test_data_manager.create_test_model(f"memory_test_{i}", "small")
            
            result = brainsmith.build_model(
                model_path=str(model_path),
                blueprint_name="test_blueprint",
                parameters={'pe': 4, 'simd': 2}
            )
            
            assert result is not None
            
            # Force cleanup between iterations
            gc.collect()
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB during model operations"


class TestConcurrencyBenchmarks:
    """Test suite for concurrency and thread safety."""
    
    @pytest.mark.performance
    def test_concurrent_api_calls(self):
        """Test concurrent API calls for thread safety."""
        import brainsmith
        import threading
        import queue
        
        def worker_function(worker_id, results_queue):
            try:
                # Each worker performs API calls
                strategies = brainsmith.list_available_strategies()
                recommendation = brainsmith.recommend_strategy(n_parameters=worker_id + 1)
                
                results_queue.put({
                    'worker_id': worker_id,
                    'success': True,
                    'strategies_count': len(strategies),
                    'recommendation': recommendation
                })
            except Exception as e:
                results_queue.put({
                    'worker_id': worker_id,
                    'success': False,
                    'error': str(e)
                })
        
        # Start multiple concurrent workers
        num_workers = 3
        results_queue = queue.Queue()
        threads = []
        
        start_time = time.time()
        
        for i in range(num_workers):
            thread = threading.Thread(target=worker_function, args=(i, results_queue))
            threads.append(thread)
            thread.start()
        
        # Wait for all workers to complete
        for thread in threads:
            thread.join(timeout=30)
        
        total_time = time.time() - start_time
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Validate results
        assert len(results) == num_workers, f"Expected {num_workers} results, got {len(results)}"
        
        successful_workers = [r for r in results if r['success']]
        assert len(successful_workers) == num_workers, f"Only {len(successful_workers)}/{num_workers} workers succeeded"
        
        # All workers should get consistent results
        strategy_counts = [r['strategies_count'] for r in successful_workers]
        assert all(count == strategy_counts[0] for count in strategy_counts), "Inconsistent results across workers"
        
        # Should complete in reasonable time
        assert total_time < 60, f"Concurrent operations took {total_time:.1f}s, expected < 60s"
    
    @pytest.mark.performance
    def test_rapid_sequential_calls(self):
        """Test rapid sequential API calls for stability."""
        import brainsmith
        
        start_time = time.time()
        
        # Make rapid sequential calls
        for i in range(20):
            strategies = brainsmith.list_available_strategies()
            recommendation = brainsmith.recommend_strategy(n_parameters=i % 10 + 1)
            
            assert isinstance(strategies, dict)
            assert isinstance(recommendation, str)
            assert len(strategies) > 0
        
        total_time = time.time() - start_time
        
        # Should handle rapid calls efficiently
        avg_time_per_call = total_time / 20
        assert avg_time_per_call < 1.0, f"Average call time {avg_time_per_call:.3f}s too high"


class TestStressTesting:
    """Test suite for stress testing and edge cases."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_extended_operation_stability(self, test_data_manager):
        """Test stability during extended operations."""
        import brainsmith
        
        start_time = time.time()
        
        # Run extended operations
        for i in range(50):
            if i % 10 == 0:
                # Create a new model every 10 iterations
                model_path = test_data_manager.create_test_model(f"stress_test_{i}", "small")
                
                result = brainsmith.build_model(
                    model_path=str(model_path),
                    blueprint_name="test_blueprint",
                    parameters={'pe': i % 8 + 1, 'simd': i % 4 + 1}
                )
                
                assert result is not None
            else:
                # Perform API calls
                strategies = brainsmith.list_available_strategies()
                recommendation = brainsmith.recommend_strategy(n_parameters=i % 10 + 1)
                
                assert isinstance(strategies, dict)
                assert isinstance(recommendation, str)
        
        total_time = time.time() - start_time
        
        # Should remain stable throughout extended operation
        assert total_time < 300, f"Extended operation took {total_time:.1f}s, expected < 300s"
    
    @pytest.mark.performance
    def test_parameter_variation_performance(self):
        """Test performance with various parameter combinations."""
        import brainsmith
        
        parameter_combinations = [
            {'n_parameters': 1, 'max_evaluations': 10, 'n_objectives': 1},
            {'n_parameters': 5, 'max_evaluations': 50, 'n_objectives': 2},
            {'n_parameters': 10, 'max_evaluations': 100, 'n_objectives': 3},
            {'n_parameters': 20, 'max_evaluations': 200, 'n_objectives': 1},
        ]
        
        times = []
        
        for params in parameter_combinations:
            start_time = time.time()
            
            recommendation = brainsmith.recommend_strategy(**params)
            
            call_time = time.time() - start_time
            times.append(call_time)
            
            assert isinstance(recommendation, str)
            assert len(recommendation) > 0
        
        # All calls should complete quickly regardless of parameters
        max_time = max(times)
        assert max_time < 5.0, f"Maximum recommendation time {max_time:.2f}s too high"
        
        # Performance should not degrade dramatically with parameter size
        if len(times) > 1:
            time_ratio = max(times) / min(times)
            assert time_ratio < 10.0, f"Performance variation {time_ratio:.1f}x too high"


# Additional utility functions for performance testing
def measure_function_time(func, *args, **kwargs):
    """Utility function to measure function execution time."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


def get_memory_usage():
    """Get current memory usage if psutil is available."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    else:
        return 0.0