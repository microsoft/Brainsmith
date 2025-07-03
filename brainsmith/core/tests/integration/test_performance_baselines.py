"""
Performance baseline tests for integration testing.
Establishes performance expectations for key operations.
"""

import pytest
import time
from pathlib import Path

from brainsmith.core.phase1 import forge, ForgeAPI
from brainsmith.core.plugins import get_registry, reset_plugin_system
from brainsmith.core.plugins.decorators import transform, kernel, backend


class TestPerformanceBaselines:
    """Test performance baselines for integration operations."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        reset_plugin_system()
        yield
        reset_plugin_system()
    
    @pytest.fixture
    def performance_plugins(self):
        """Register a realistic set of plugins for performance testing."""
        plugins = []
        
        # Register transforms across different stages
        stages = ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt", "post_proc"]
        for i in range(15):  # Moderate number of transforms
            stage = stages[i % len(stages)]
            
            @transform(name=f"PerfTransform{i}", stage=stage)
            class PerfTransform:
                def apply(self, model):
                    # Simulate minimal processing time
                    return model, False
            
            PerfTransform.__name__ = f"PerfTransform{i}Class"
            plugins.append(PerfTransform)
        
        # Register kernels with backends (realistic for FPGA workflow)
        kernel_types = ["MatMul", "Conv2D", "LayerNorm", "Attention", "Pooling"]
        for kernel_type in kernel_types:
            @kernel(name=f"Perf{kernel_type}")
            class PerfKernel:
                def compile(self, node):
                    return {"kernel": kernel_type, "node": node}
            
            PerfKernel.__name__ = f"Perf{kernel_type}Class"
            plugins.append(PerfKernel)
            
            # Add typical backends for each kernel
            languages = ["rtl", "hls"]
            for lang in languages:
                @backend(name=f"Perf{kernel_type}{lang.upper()}", kernel=f"Perf{kernel_type}", language=lang)
                class PerfBackend:
                    def generate(self, kernel_instance):
                        return f"// {kernel_type} {lang} implementation"
                
                PerfBackend.__name__ = f"Perf{kernel_type}{lang.upper()}Class"
                plugins.append(PerfBackend)
        
        return plugins
    
    def test_plugin_registration_performance(self, performance_plugins):
        """Test that plugin registration completes within acceptable time."""
        # All plugins should be registered by the fixture setup
        registry = get_registry()
        
        # Baseline: Registration of ~25 plugins should be very fast
        start_time = time.time()
        
        # Check all plugins are registered (accessing registry is the performance test)
        kernels = registry.list_available_kernels()
        transforms = registry.list_available_transforms()
        
        end_time = time.time()
        registration_time = end_time - start_time
        
        # Performance baseline: Should be under 10ms for registry access
        assert registration_time < 0.01, f"Plugin registry access took {registration_time:.3f}s, expected < 0.01s"
        
        # Verify correct plugin counts
        assert len(kernels) == 5, f"Expected 5 kernels, got {len(kernels)}"
        assert len(transforms) == 15, f"Expected 15 transforms, got {len(transforms)}"
    
    def test_blueprint_parsing_performance(self, performance_plugins, tmp_path):
        """Test that blueprint parsing completes within acceptable time."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake onnx content")
        
        # Create medium complexity blueprint
        blueprint_path = tmp_path / "perf_test.yaml"
        blueprint_path.write_text("""
version: "3.0"
hw_compiler:
  kernels:
    - "PerfMatMul"
    - "PerfConv2D"
    - ["PerfLayerNorm", ["PerfLayerNormHLS"]]
    - "~PerfAttention"
  
  transforms:
    cleanup:
      - "PerfTransform0"
      - "PerfTransform5"
    topology_opt:
      - "PerfTransform1"
      - "~PerfTransform6"
    kernel_opt:
      - "PerfTransform2"
  
  build_steps: ["ConvertToHW", "PrepareIP"]

search:
  strategy: "exhaustive"
  constraints:
    - {metric: "lut_utilization", operator: "<=", value: 0.85}

global:
  output_stage: "rtl"
  working_directory: "./perf_builds"
""")
        
        # Performance test: Blueprint parsing
        start_time = time.time()
        design_space = forge(str(model_path), str(blueprint_path))
        end_time = time.time()
        
        parsing_time = end_time - start_time
        
        # Performance baseline: Should parse within 100ms
        assert parsing_time < 0.1, f"Blueprint parsing took {parsing_time:.3f}s, expected < 0.1s"
        
        # Verify correctness
        assert len(design_space.hw_compiler_space.kernels) == 4
        assert isinstance(design_space.hw_compiler_space.transforms, dict)
        assert design_space.get_total_combinations() > 0
    
    def test_design_space_construction_performance(self, performance_plugins, tmp_path):
        """Test that design space construction scales well with plugin count."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake")
        
        # Test small design space
        small_blueprint = tmp_path / "small.yaml"
        small_blueprint.write_text("""
version: "3.0"
hw_compiler:
  kernels: ["PerfMatMul"]
  transforms: ["PerfTransform0"]
  build_steps: ["ConvertToHW"]
search:
  strategy: "exhaustive"
global:
  output_stage: "rtl"
  working_directory: "./builds"
""")
        
        start_time = time.time()
        small_ds = forge(str(model_path), str(small_blueprint))
        small_time = time.time() - start_time
        
        # Test larger design space
        large_blueprint = tmp_path / "large.yaml"
        large_blueprint.write_text("""
version: "3.0"
hw_compiler:
  kernels:
    - "PerfMatMul"
    - "PerfConv2D"
    - "PerfLayerNorm"
    - "PerfAttention"
    - "PerfPooling"
  transforms:
    cleanup: ["PerfTransform0", "PerfTransform5", "PerfTransform10"]
    topology_opt: ["PerfTransform1", "PerfTransform6", "PerfTransform11"]
    kernel_opt: ["PerfTransform2", "PerfTransform7", "PerfTransform12"]
  build_steps: ["ConvertToHW"]
search:
  strategy: "exhaustive"
global:
  output_stage: "rtl"
  working_directory: "./builds"
""")
        
        start_time = time.time()
        large_ds = forge(str(model_path), str(large_blueprint))
        large_time = time.time() - start_time
        
        # Performance baselines
        assert small_time < 0.05, f"Small design space took {small_time:.3f}s, expected < 0.05s"
        assert large_time < 0.2, f"Large design space took {large_time:.3f}s, expected < 0.2s"
        
        # Scaling should be reasonable (not exponential)
        scaling_factor = large_time / small_time if small_time > 0 else float('inf')
        assert scaling_factor < 10, f"Performance scaling factor {scaling_factor:.1f}x is too high"
        
        # Verify correctness
        assert small_ds.get_total_combinations() > 0
        assert large_ds.get_total_combinations() > small_ds.get_total_combinations()
    
    def test_plugin_discovery_performance(self, performance_plugins):
        """Test that plugin discovery operations are fast."""
        registry = get_registry()
        
        # Test kernel discovery performance
        start_time = time.time()
        for _ in range(100):  # Simulate multiple discovery calls
            kernels = registry.list_available_kernels()
        kernel_time = time.time() - start_time
        
        # Test backend discovery performance
        start_time = time.time()
        for _ in range(100):
            for kernel in ["PerfMatMul", "PerfConv2D", "PerfLayerNorm"]:
                backends = registry.list_backends_by_kernel(kernel)
        backend_time = time.time() - start_time
        
        # Test transform discovery performance
        start_time = time.time()
        for _ in range(100):
            transforms = registry.list_available_transforms()
        transform_time = time.time() - start_time
        
        # Performance baselines: Should handle 100 operations quickly
        assert kernel_time < 0.01, f"100 kernel discoveries took {kernel_time:.3f}s, expected < 0.01s"
        assert backend_time < 0.05, f"300 backend discoveries took {backend_time:.3f}s, expected < 0.05s"
        assert transform_time < 0.01, f"100 transform discoveries took {transform_time:.3f}s, expected < 0.01s"
    
    def test_optimization_overhead_baseline(self, performance_plugins, tmp_path):
        """Test that forge_optimized overhead is acceptable."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake")
        
        blueprint_path = tmp_path / "opt_test.yaml"
        blueprint_path.write_text("""
version: "3.0"
hw_compiler:
  kernels: ["PerfMatMul", "PerfConv2D"]
  transforms:
    cleanup: ["PerfTransform0"]
    topology_opt: ["PerfTransform1"]
  build_steps: ["ConvertToHW"]
search:
  strategy: "exhaustive"
global:
  output_stage: "rtl"
  working_directory: "./builds"
""")
        
        api = ForgeAPI()
        
        # Time regular forge
        start_time = time.time()
        ds_normal = api.forge(str(model_path), str(blueprint_path))
        normal_time = time.time() - start_time
        
        # Time optimized forge
        start_time = time.time()
        ds_optimized = api.forge_optimized(str(model_path), str(blueprint_path))
        optimized_time = time.time() - start_time
        
        # Performance baseline: Optimization overhead should be minimal
        overhead = optimized_time - normal_time
        assert overhead < 0.1, f"Optimization overhead of {overhead:.3f}s is too high"
        
        # Both should be reasonably fast
        assert normal_time < 0.1, f"Normal forge took {normal_time:.3f}s, expected < 0.1s"
        assert optimized_time < 0.2, f"Optimized forge took {optimized_time:.3f}s, expected < 0.2s"
        
        # Results should be equivalent
        assert ds_normal.get_total_combinations() == ds_optimized.get_total_combinations()
    
    def test_memory_efficiency_baseline(self, performance_plugins):
        """Test that plugin registry memory usage is reasonable."""
        import sys
        registry = get_registry()
        
        # Get baseline memory usage
        initial_refs = sys.getrefcount(registry)
        
        # Perform many registry operations
        for _ in range(1000):
            kernels = registry.list_available_kernels()
            transforms = registry.list_available_transforms()
            
            # Simulate lookup operations
            if kernels:
                backends = registry.list_backends_by_kernel(kernels[0])
        
        # Check that memory usage doesn't grow unreasonably
        final_refs = sys.getrefcount(registry)
        ref_growth = final_refs - initial_refs
        
        # Memory baseline: Reference count shouldn't grow significantly
        assert ref_growth < 10, f"Registry reference count grew by {ref_growth}, expected < 10"
        
        # Registry should still function correctly
        assert len(registry.list_available_kernels()) == 5
        assert len(registry.list_available_transforms()) == 15