"""
Performance tests for Phase 1 Design Space Constructor.
"""

import pytest
import time
import psutil
import os
from pathlib import Path
import yaml

from brainsmith.core.phase1 import forge, ForgeAPI
from brainsmith.core.phase1.parser import BlueprintParser, load_blueprint
from brainsmith.core.phase1.validator import DesignSpaceValidator
from brainsmith.core.plugins import get_registry, reset_plugin_system
from brainsmith.core.plugins.decorators import transform, kernel, backend


class TestPerformanceMetrics:
    """Test performance characteristics of Phase 1."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Reset plugin system for each test."""
        reset_plugin_system()
        yield
        reset_plugin_system()
    
    @pytest.fixture
    def large_plugin_set(self):
        """Register a large number of plugins for stress testing."""
        # Register 50 transforms
        for i in range(50):
            stage = ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt", "post_proc"][i % 5]
            
            # Create unique class for each transform
            transform_class = type(
                f"PerfTransform{i}",
                (object,),
                {"apply": lambda self, model: (model, False)}
            )
            
            # Register it
            transform(name=f"PerfTransform{i}", stage=stage)(transform_class)
        
        # Register 20 kernels with 3 backends each
        for i in range(20):
            kernel_class = type(
                f"PerfKernel{i}",
                (object,),
                {"compile": lambda self, node: {"kernel": f"PerfKernel{i}"}}
            )
            kernel(name=f"PerfKernel{i}")(kernel_class)
            
            # Add backends
            for j, lang in enumerate(["rtl", "hls", "dsp"]):
                backend_class = type(
                    f"PerfKernel{i}Backend{j}",
                    (object,),
                    {"generate": lambda self, ki: f"// Backend {j}"}
                )
                backend(name=f"PerfKernel{i}Backend{j}", kernel=f"PerfKernel{i}", language=lang)(backend_class)
        
        # Total: 50 transforms + 20 kernels + 60 backends = 130 plugins
        return 130
    
    @pytest.fixture
    def complex_blueprint_dict(self):
        """Create a complex blueprint for performance testing."""
        return {
            "version": "3.0",
            "name": "Performance Test Blueprint",
            "hw_compiler": {
                "kernels": [
                    "PerfKernel0",
                    ["PerfKernel1", ["rtl", "hls"]],
                    "~PerfKernel2",
                    ["PerfKernel3", "PerfKernel4", "~PerfKernel5", None],
                    [["PerfKernel6", ["dsp"]], "PerfKernel7"],
                ],
                "transforms": {
                    "cleanup": ["PerfTransform0", "~PerfTransform5", ["PerfTransform10", None]],
                    "topology_opt": ["PerfTransform1", "PerfTransform6", "PerfTransform11"],
                    "kernel_opt": ["~PerfTransform2", "PerfTransform7"],
                    "dataflow_opt": ["PerfTransform3", ["PerfTransform8", "PerfTransform13", None]],
                    "post_proc": ["PerfTransform4", "PerfTransform9"]
                },
                "build_steps": ["Step1", "Step2", "Step3", "Step4"],
                "config_flags": {
                    "flag1": "value1",
                    "flag2": 123,
                    "flag3": True
                }
            },
            "processing": {
                "preprocessing": [
                    {
                        "name": "step1",
                        "options": [
                            {"enabled": True, "param1": 1},
                            {"enabled": True, "param1": 2},
                            {"enabled": False}
                        ]
                    },
                    {
                        "name": "step2",
                        "options": [
                            {"enabled": True, "method": "A"},
                            {"enabled": True, "method": "B"}
                        ]
                    }
                ],
                "postprocessing": [
                    {
                        "name": "post1",
                        "options": [{"enabled": True}, {"enabled": False}]
                    }
                ]
            },
            "search": {
                "strategy": "bayesian",
                "constraints": [
                    {"metric": "lut", "operator": "<=", "value": 0.8},
                    {"metric": "bram", "operator": "<=", "value": 0.9},
                    {"metric": "latency", "operator": "<", "value": 1000}
                ],
                "max_evaluations": 1000,
                "parallel_builds": 8
            },
            "global": {
                "output_stage": "rtl",
                "working_directory": "./perf_builds",
                "cache_results": True,
                "max_combinations": 100000
            }
        }
    
    def test_parser_performance(self, large_plugin_set, complex_blueprint_dict, benchmark):
        """Benchmark parser performance with complex blueprint."""
        parser = BlueprintParser()
        
        # Benchmark parsing
        result = benchmark(parser.parse, complex_blueprint_dict, "model.onnx")
        
        # Verify result
        assert result is not None
        assert result.hw_compiler_space.kernels is not None
        
        # Check performance threshold
        assert benchmark.stats['mean'] < 0.1  # Should parse in under 100ms
    
    def test_validator_performance(self, large_plugin_set, complex_blueprint_dict, benchmark):
        """Benchmark validator performance."""
        parser = BlueprintParser()
        validator = DesignSpaceValidator()
        
        # Parse first
        design_space = parser.parse(complex_blueprint_dict, "test_model.onnx")
        
        # Benchmark validation
        result = benchmark(validator.validate, design_space)
        
        # Check performance
        assert benchmark.stats['mean'] < 0.05  # Should validate in under 50ms
    
    def test_forge_performance(self, large_plugin_set, complex_blueprint_dict, tmp_path, benchmark):
        """Benchmark complete forge performance."""
        # Create files
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake onnx")
        
        blueprint_path = tmp_path / "blueprint.yaml"
        blueprint_path.write_text(yaml.dump(complex_blueprint_dict))
        
        # Benchmark forge
        result = benchmark(forge, str(model_path), str(blueprint_path))
        
        # Verify result
        assert result is not None
        assert result.get_total_combinations() > 0
        
        # Check performance threshold
        assert benchmark.stats['mean'] < 0.2  # Should complete in under 200ms
    
    def test_memory_usage(self, large_plugin_set, complex_blueprint_dict, tmp_path):
        """Test memory usage stays reasonable."""
        # Get initial memory
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        
        # Create files
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake onnx")
        
        blueprint_path = tmp_path / "blueprint.yaml"
        blueprint_path.write_text(yaml.dump(complex_blueprint_dict))
        
        # Run forge multiple times
        design_spaces = []
        for i in range(10):
            ds = forge(str(model_path), str(blueprint_path))
            design_spaces.append(ds)
        
        # Check memory after
        mem_after = process.memory_info().rss
        mem_used_mb = (mem_after - mem_before) / 1024 / 1024
        
        # Should use less than 50MB for 10 design spaces
        assert mem_used_mb < 50
    
    def test_optimization_overhead(self, large_plugin_set, tmp_path):
        """Test overhead of forge_optimized vs regular forge."""
        # Create simple blueprint using subset of plugins
        simple_blueprint = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": ["PerfKernel0", "PerfKernel1", "PerfKernel2"],
                "transforms": ["PerfTransform0", "PerfTransform1", "PerfTransform2"],
                "build_steps": ["Step1"]
            },
            "search": {"strategy": "exhaustive"},
            "global": {"output_stage": "rtl", "working_directory": "./builds"}
        }
        
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake")
        
        blueprint_path = tmp_path / "simple.yaml"
        blueprint_path.write_text(yaml.dump(simple_blueprint))
        
        api = ForgeAPI()
        
        # Time regular forge
        times_normal = []
        for _ in range(5):
            start = time.perf_counter()
            api.forge(str(model_path), str(blueprint_path))
            times_normal.append(time.perf_counter() - start)
        
        # Time optimized forge
        times_optimized = []
        for _ in range(5):
            start = time.perf_counter()
            api.forge_optimized(str(model_path), str(blueprint_path))
            times_optimized.append(time.perf_counter() - start)
        
        # Calculate averages
        avg_normal = sum(times_normal) / len(times_normal)
        avg_optimized = sum(times_optimized) / len(times_optimized)
        
        # Optimization overhead should be minimal
        overhead_percent = ((avg_optimized - avg_normal) / avg_normal) * 100
        assert overhead_percent < 20  # Less than 20% overhead
    
    def test_large_combination_count_performance(self, tmp_path):
        """Test performance with blueprints that generate many combinations."""
        # Register minimal plugins for this test
        for i in range(5):
            kernel_class = type(f"LargeKernel{i}", (object,), {})
            kernel(name=f"LargeKernel{i}")(kernel_class)
            
            for j in range(3):
                backend_class = type(f"LargeKernel{i}Backend{j}", (object,), {})
                backend(name=f"LargeKernel{i}Backend{j}", kernel=f"LargeKernel{i}", language="rtl")(backend_class)
        
        # Blueprint with many combinations
        large_combo_blueprint = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": [
                    ["LargeKernel0", "LargeKernel1"],  # 2 choices
                    ["LargeKernel2", "LargeKernel3", "LargeKernel4"],  # 3 choices
                    "LargeKernel0",  # 3 backends
                    "LargeKernel1",  # 3 backends
                ],
                "transforms": [],
                "build_steps": ["Step1"]
            },
            "processing": {
                "preprocessing": [
                    {"name": f"prep{i}", "options": [{"v": j} for j in range(4)]}
                    for i in range(3)  # 3 steps with 4 options each = 64 combinations
                ]
            },
            "search": {"strategy": "exhaustive"},
            "global": {
                "output_stage": "rtl", 
                "working_directory": "./builds",
                "max_combinations": 100000
            }
        }
        
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake")
        
        blueprint_path = tmp_path / "large_combo.yaml"
        blueprint_path.write_text(yaml.dump(large_combo_blueprint))
        
        # Time combination calculation
        start = time.perf_counter()
        design_space = forge(str(model_path), str(blueprint_path))
        total_combinations = design_space.get_total_combinations()
        calc_time = time.perf_counter() - start
        
        # Should calculate even large combination counts quickly
        assert calc_time < 0.5  # Under 500ms
        assert total_combinations > 1000  # Should be many combinations
    
    def test_plugin_lookup_performance(self, large_plugin_set):
        """Test performance of plugin lookups."""
        registry = get_registry()
        
        # Time kernel lookups
        start = time.perf_counter()
        for i in range(1000):
            _ = f"PerfKernel{i % 20}" in registry.kernels
        kernel_lookup_time = time.perf_counter() - start
        
        # Time transform lookups
        start = time.perf_counter()
        for i in range(1000):
            _ = f"PerfTransform{i % 50}" in registry.transforms
        transform_lookup_time = time.perf_counter() - start
        
        # Time backend discovery
        start = time.perf_counter()
        for i in range(100):
            _ = registry.list_backends_by_kernel(f"PerfKernel{i % 20}")
        backend_discovery_time = time.perf_counter() - start
        
        # All lookups should be fast (dict lookups)
        assert kernel_lookup_time < 0.01  # 1000 lookups in 10ms
        assert transform_lookup_time < 0.01
        assert backend_discovery_time < 0.01  # 100 discoveries in 10ms
    
    def test_scalability_with_plugin_count(self, tmp_path):
        """Test how performance scales with number of plugins."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake")
        
        blueprint_dict = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": ["ScaleKernel0"],
                "transforms": ["ScaleTransform0"],
                "build_steps": ["Step1"]
            },
            "search": {"strategy": "exhaustive"},
            "global": {"output_stage": "rtl", "working_directory": "./builds"}
        }
        
        blueprint_path = tmp_path / "scale.yaml"
        blueprint_path.write_text(yaml.dump(blueprint_dict))
        
        times = []
        plugin_counts = []
        
        # Test with increasing plugin counts
        for n in [10, 50, 100, 200]:
            # Reset and register n plugins
            reset_plugin_system()
            
            # Register transforms
            for i in range(n // 2):
                transform_class = type(f"ScaleTransform{i}", (object,), {})
                transform(name=f"ScaleTransform{i}", stage="cleanup")(transform_class)
            
            # Register kernels and backends
            for i in range(n // 4):
                kernel_class = type(f"ScaleKernel{i}", (object,), {})
                kernel(name=f"ScaleKernel{i}")(kernel_class)
                
                backend_class = type(f"ScaleBackend{i}", (object,), {})
                backend(name=f"ScaleBackend{i}", kernel=f"ScaleKernel{i}", language="rtl")(backend_class)
            
            # Ensure our test plugins exist
            if n > 0:
                # Update blueprint to use existing plugin
                blueprint_dict["hw_compiler"]["kernels"] = [f"ScaleKernel0"]
                blueprint_dict["hw_compiler"]["transforms"] = [f"ScaleTransform0"]
                blueprint_path.write_text(yaml.dump(blueprint_dict))
            
            # Time forge
            start = time.perf_counter()
            forge(str(model_path), str(blueprint_path))
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            plugin_counts.append(n)
        
        # Performance should not degrade significantly with more plugins
        # (assuming dict-based lookups)
        # Allow up to 2x slowdown from 10 to 200 plugins
        assert times[-1] < times[0] * 2