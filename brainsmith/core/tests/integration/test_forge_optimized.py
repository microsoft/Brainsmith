"""
Integration tests for forge_optimized functionality.
"""

import pytest
import tempfile
import time
from pathlib import Path

from brainsmith.core.phase1 import ForgeAPI
from brainsmith.core.phase1.data_structures import DesignSpace
from brainsmith.core.plugins import get_registry, reset_plugin_system
from brainsmith.core.plugins.decorators import transform, kernel, backend
from brainsmith.core.plugins.blueprint_loader import BlueprintPluginLoader

# No fake plugins - use real QONNX/FINN plugins only


class TestForgeOptimized:
    """Test the forge_optimized method and plugin optimization."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        reset_plugin_system()
        yield
        reset_plugin_system()
    
    @pytest.fixture
    def many_plugins(self):
        """Register many plugins to test optimization benefits."""
        plugins = []
        
        # Register 20 transforms across different stages
        stages = ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt", "post_proc"]
        for i in range(20):
            stage = stages[i % len(stages)]
            
            @transform(name=f"Transform{i}", stage=stage)
            class TestTransform:
                def apply(self, model):
                    return model, False
            
            # Rename class to avoid conflicts
            TestTransform.__name__ = f"Transform{i}Class"
            plugins.append(TestTransform)
        
        # Register 10 kernels with backends
        for i in range(10):
            @kernel(name=f"Kernel{i}")
            class TestKernel:
                def compile(self, node):
                    return {"kernel": f"Kernel{i}"}
            
            TestKernel.__name__ = f"Kernel{i}Class"
            plugins.append(TestKernel)
            
            # Add 2 backends per kernel
            for j in range(2):
                lang = "rtl" if j == 0 else "hls"
                
                @backend(name=f"Kernel{i}Backend{j}", kernel=f"Kernel{i}", language=lang)
                class TestBackend:
                    def generate(self, kernel_instance):
                        return f"// Backend for Kernel{i}"
                
                TestBackend.__name__ = f"Kernel{i}Backend{j}Class"
                plugins.append(TestBackend)
        
        return plugins
    
    def test_forge_optimized_basic(self, tmp_path):
        """Test basic forge_optimized functionality."""
        # Create simple test setup
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake onnx")
        
        # Register a few plugins
        @kernel(name="OptTestKernel")
        class OptTestKernel:
            pass
        
        @backend(name="OptTestBackend", kernel="OptTestKernel", language="rtl")
        class OptTestBackend:
            pass
        
        @transform(name="OptTestTransform", stage="cleanup")
        class OptTestTransform:
            pass
        
        # Create blueprint using only one plugin
        blueprint_path = tmp_path / "opt_test.yaml"
        blueprint_path.write_text("""
version: "3.0"
hw_compiler:
  kernels: ["OptTestKernel"]
  transforms: ["OptTestTransform"]
  build_steps: ["ConvertToHW"]
search:
  strategy: "exhaustive"
global:
  output_stage: "rtl"
  working_directory: "./builds"
""")
        
        # Test forge_optimized
        api = ForgeAPI()
        design_space = api.forge_optimized(
            str(model_path), 
            str(blueprint_path),
            optimize_plugins=True
        )
        
        # Verify optimization metadata added
        assert hasattr(design_space, '_plugin_optimization_enabled')
        assert design_space._plugin_optimization_enabled is True
        assert hasattr(design_space, '_plugin_stats')
        assert design_space._plugin_stats is not None
        
        # Check stats structure
        stats = design_space._plugin_stats
        assert 'total_available_plugins' in stats
        assert 'total_loaded_plugins' in stats
        assert 'load_percentage' in stats
        assert 'performance_improvement' in stats
    
    def test_forge_optimized_disabled(self, tmp_path):
        """Test forge_optimized with optimization disabled."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake")
        
        blueprint_path = tmp_path / "test.yaml"
        blueprint_path.write_text("""
version: "3.0"
hw_compiler:
  kernels: []
  transforms: []
  build_steps: ["ConvertToHW"]
search:
  strategy: "exhaustive"
global:
  output_stage: "rtl"
  working_directory: "./builds"
""")
        
        api = ForgeAPI()
        design_space = api.forge_optimized(
            str(model_path),
            str(blueprint_path),
            optimize_plugins=False  # Disabled
        )
        
        # Should not have optimization metadata
        assert not hasattr(design_space, '_plugin_optimization_enabled')
        assert not hasattr(design_space, '_plugin_stats')
    
    def test_optimization_stats_accuracy(self, many_plugins, tmp_path):
        """Test that optimization stats are accurate."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake")
        
        # Blueprint using only a few of the many plugins
        blueprint_path = tmp_path / "selective.yaml"
        blueprint_path.write_text("""
version: "3.0"
hw_compiler:
  kernels:
    - "Kernel1"  # Will use 1 kernel + 2 backends = 3 plugins
    - "Kernel5"  # Another 3 plugins
  transforms:
    cleanup:
      - "Transform0"  # 1 plugin
      - "Transform5"  # 1 plugin
    topology_opt:
      - "Transform1"  # 1 plugin
  build_steps: ["ConvertToHW"]
search:
  strategy: "exhaustive"
global:
  output_stage: "rtl"
  working_directory: "./builds"
""")
        
        api = ForgeAPI()
        design_space = api.forge_optimized(str(model_path), str(blueprint_path))
        
        stats = design_space._plugin_stats
        
        # We registered 20 transforms + 10 kernels + 20 backends = 50 plugins
        assert stats['total_available_plugins'] >= 50
        
        # Blueprint uses 2 kernels + 4 backends + 3 transforms = 9 plugins
        assert stats['total_loaded_plugins'] == 9
        
        # Load percentage should be 9/50 = 18%
        expected_percentage = (9 / stats['total_available_plugins']) * 100
        assert abs(stats['load_percentage'] - expected_percentage) < 1  # Allow small float diff
        
        # Should show performance improvement or reduction
        perf_msg = stats['performance_improvement'].lower()
        assert "improvement" in perf_msg or "reduction" in perf_msg
    
    def test_blueprint_plugin_loader_integration(self, tmp_path):
        """Test that BlueprintPluginLoader integration works with real plugins."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake")
        
        # Register test plugins
        @kernel(name="LoaderTestKernel")
        class LoaderTestKernel:
            pass
        
        @backend(name="LoaderTestBackend", kernel="LoaderTestKernel", language="hls")
        class LoaderTestBackend:
            pass
        
        @transform(name="LoaderTestTransform", stage="cleanup")
        class LoaderTestTransform:
            pass
        
        blueprint_path = tmp_path / "loader_test.yaml"
        blueprint_path.write_text("""
version: "3.0"
hw_compiler:
  kernels: ["LoaderTestKernel"]
  transforms: ["LoaderTestTransform"]
  build_steps: ["ConvertToHW"]
search:
  strategy: "exhaustive"
global:
  output_stage: "rtl"
  working_directory: "./builds"
""")
        
        # Test with real BlueprintPluginLoader integration
        api = ForgeAPI()
        design_space = api.forge_optimized(str(model_path), str(blueprint_path))
        
        # Verify optimization worked with real components
        assert hasattr(design_space, '_plugin_optimization_enabled')
        assert design_space._plugin_optimization_enabled is True
        assert hasattr(design_space, '_plugin_stats')
        
        # Verify stats are realistic (should detect the 3 plugins we registered)
        stats = design_space._plugin_stats
        assert stats['total_loaded_plugins'] >= 3  # At least our 3 test plugins
        assert stats['load_percentage'] > 0
        assert stats['total_available_plugins'] >= stats['total_loaded_plugins']
    
    def test_performance_comparison(self, many_plugins, tmp_path):
        """Test performance with and without optimization."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake")
        
        # Blueprint using subset of plugins
        blueprint_path = tmp_path / "perf_test.yaml"
        blueprint_path.write_text("""
version: "3.0"
hw_compiler:
  kernels: ["Kernel0", "Kernel1", "Kernel2"]
  transforms: 
    cleanup: ["Transform0", "Transform5"]
    topology_opt: ["Transform1", "Transform6"]
  build_steps: ["ConvertToHW"]
search:
  strategy: "exhaustive"
global:
  output_stage: "rtl"
  working_directory: "./builds"
""")
        
        api = ForgeAPI()
        
        # Time without optimization (regular forge)
        start = time.time()
        ds_normal = api.forge(str(model_path), str(blueprint_path))
        time_normal = time.time() - start
        
        # Time with optimization
        start = time.time()
        ds_optimized = api.forge_optimized(str(model_path), str(blueprint_path))
        time_optimized = time.time() - start
        
        # Both should produce equivalent design spaces
        assert ds_normal.get_total_combinations() == ds_optimized.get_total_combinations()
        
        # Performance comparison - optimization may be faster or slower depending on implementation
        # Both times should be reasonable (less than 1 second for this simple test)
        assert time_normal < 1.0
        assert time_optimized < 1.0
        
        # Optimized version should have stats
        assert hasattr(ds_optimized, '_plugin_stats')
        assert not hasattr(ds_normal, '_plugin_stats')
    
    def test_forge_optimized_with_complex_blueprint(self, tmp_path):
        """Test forge_optimized with complex blueprint features."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake")
        
        # Register plugins for complex test
        @kernel(name="ComplexKernel1")
        class ComplexKernel1:
            pass
        
        @kernel(name="ComplexKernel2")
        class ComplexKernel2:
            pass
        
        @backend(name="CK1_RTL", kernel="ComplexKernel1", language="rtl")
        class CK1_RTL:
            pass
        
        @backend(name="CK1_HLS", kernel="ComplexKernel1", language="hls")
        class CK1_HLS:
            pass
        
        @backend(name="CK2_HLS", kernel="ComplexKernel2", language="hls")
        class CK2_HLS:
            pass
        
        @transform(name="ComplexTransform1", stage="cleanup")
        class ComplexTransform1:
            pass
        
        @transform(name="ComplexTransform2", stage="topology_opt")
        class ComplexTransform2:
            pass
        
        # Complex blueprint with all features
        blueprint_path = tmp_path / "complex.yaml"
        blueprint_path.write_text("""
version: "3.0"
hw_compiler:
  kernels:
    - "ComplexKernel1"  # Auto-discovery
    - ["ComplexKernel2", ["CK2_HLS"]]  # Explicit with correct backend name
    - ["ComplexKernel1", "~ComplexKernel2", ~]  # Mutually exclusive
  
  transforms:
    cleanup:
      - "ComplexTransform1"
      - ["ComplexTransform2", ~]  # Optional group
    topology_opt:
      - "~ComplexTransform2"  # Optional
  
  build_steps: ["ConvertToHW", "PrepareIP"]
  
  config_flags:
    target_device: "xczu7ev"

processing:
  preprocessing:
    - name: "normalize"
      options:
        - {enabled: true}
        - {enabled: false}

search:
  strategy: "exhaustive"
  constraints:
    - {metric: "lut", operator: "<=", value: 0.8}
  max_evaluations: 100
  parallel_builds: 4

global:
  output_stage: "rtl"
  working_directory: "./complex_builds"
  max_combinations: 5000
""")
        
        api = ForgeAPI()
        design_space = api.forge_optimized(str(model_path), str(blueprint_path))
        
        # Should handle complex blueprint correctly
        assert design_space._plugin_optimization_enabled
        assert design_space.get_total_combinations() > 0
        
        # Verify complex kernel configurations preserved
        kernels = design_space.hw_compiler_space.kernels
        assert len(kernels) == 3
        assert isinstance(kernels[2], list)  # Mutually exclusive group
        
        # Verify stats computed
        stats = design_space._plugin_stats
        assert stats['total_loaded_plugins'] > 0
        assert stats['load_percentage'] > 0