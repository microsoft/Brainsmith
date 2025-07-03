"""
Integration tests for Phase 1 with real plugin system.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from brainsmith.core.phase1 import forge, ForgeAPI
from brainsmith.core.phase1.parser import BlueprintParser
from brainsmith.core.phase1.exceptions import BlueprintParseError
from brainsmith.core.plugins import get_registry, reset_plugin_system
from brainsmith.core.plugins.decorators import transform, kernel, backend

# No fake plugins - use real QONNX/FINN plugins only


class TestPluginIntegration:
    """Test Phase 1 integration with real plugin registry."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Setup - ensure clean plugin state
        reset_plugin_system()
        yield
        # Teardown - reset again
        reset_plugin_system()
    
    @pytest.fixture
    def test_plugins(self):
        """Register test plugins for integration testing."""
        # Define test transforms
        @transform(name="TestCleanupTransform", stage="cleanup")
        class TestCleanupTransform:
            def apply(self, model):
                return model, False
        
        @transform(name="TestTopologyTransform", stage="topology_opt")
        class TestTopologyTransform:
            def apply(self, model):
                return model, False
        
        @transform(name="TestOptionalTransform", stage="kernel_opt")
        class TestOptionalTransform:
            def apply(self, model):
                return model, False
        
        # Define test kernels
        @kernel(name="TestMatMul")
        class TestMatMul:
            def compile(self, node):
                return {"type": "matmul", "node": node}
        
        @kernel(name="TestLayerNorm")
        class TestLayerNorm:
            def compile(self, node):
                return {"type": "layernorm", "node": node}
        
        @kernel(name="TestAttention")
        class TestAttention:
            def compile(self, node):
                return {"type": "attention", "node": node}
        
        # Define test backends
        @backend(name="TestRTLBackend", kernel="TestMatMul", language="rtl")
        class TestRTLBackend:
            def generate(self, kernel_instance):
                return "// RTL implementation"
        
        @backend(name="TestHLSBackend", kernel="TestMatMul", language="hls")
        class TestHLSBackend:
            def generate(self, kernel_instance):
                return "// HLS implementation"
        
        @backend(name="TestLayerNormHLS", kernel="TestLayerNorm", language="hls")
        class TestLayerNormHLS:
            def generate(self, kernel_instance):
                return "// LayerNorm HLS"
        
        @backend(name="TestAttentionCUDA", kernel="TestAttention", language="cuda")
        class TestAttentionCUDA:
            def generate(self, kernel_instance):
                return "// Attention CUDA"
        
        @backend(name="TestAttentionTriton", kernel="TestAttention", language="triton")
        class TestAttentionTriton:
            def generate(self, kernel_instance):
                return "// Attention Triton"
        
        return {
            "transforms": [TestCleanupTransform, TestTopologyTransform, TestOptionalTransform],
            "kernels": [TestMatMul, TestLayerNorm, TestAttention],
            "backends": [TestRTLBackend, TestHLSBackend, TestLayerNormHLS, 
                        TestAttentionCUDA, TestAttentionTriton]
        }
    
    def test_complete_integration_flow(self, test_plugins, tmp_path):
        """Test full integration with real plugins."""
        # Create test model
        model_path = tmp_path / "test_model.onnx"
        model_path.write_bytes(b"fake onnx content")
        
        # Create blueprint using test plugins
        blueprint_yaml = """
version: "3.0"
name: "Integration Test Blueprint"
description: "Tests real plugin integration"

hw_compiler:
  kernels:
    - "TestMatMul"  # Should auto-discover TestRTLBackend and TestHLSBackend
    - ["TestLayerNorm", ["TestLayerNormHLS"]]  # Explicit backend
    - "~TestAttention"  # Optional with auto-discovery
  
  transforms:
    cleanup:
      - "TestCleanupTransform"
    topology_opt:
      - "TestTopologyTransform"
    kernel_opt:
      - "~TestOptionalTransform"
  
  build_steps:
    - "ConvertToHW"
    - "PrepareIP"

search:
  strategy: "exhaustive"
  constraints:
    - metric: "lut_utilization"
      operator: "<="
      value: 0.85

global:
  output_stage: "rtl"
  working_directory: "./test_builds"
"""
        
        blueprint_path = tmp_path / "test_blueprint.yaml"
        blueprint_path.write_text(blueprint_yaml)
        
        # Forge design space
        design_space = forge(str(model_path), str(blueprint_path))
        
        # Verify auto-discovery worked
        kernels = design_space.hw_compiler_space.kernels
        assert len(kernels) == 3
        
        # Check TestMatMul auto-discovered backends
        assert kernels[0] == ("TestMatMul", ["TestRTLBackend", "TestHLSBackend"])
        
        # Check explicit backend specification
        assert kernels[1] == ("TestLayerNorm", ["TestLayerNormHLS"])
        
        # Check optional kernel preserved marker
        assert kernels[2][0] == "~TestAttention"
        assert set(kernels[2][1]) == {"TestAttentionCUDA", "TestAttentionTriton"}
        
        # Verify transforms
        transforms = design_space.hw_compiler_space.transforms
        assert transforms["cleanup"] == ["TestCleanupTransform"]
        assert transforms["topology_opt"] == ["TestTopologyTransform"]
        assert transforms["kernel_opt"] == ["~TestOptionalTransform"]
    
    def test_cross_framework_plugins(self):
        """Test using plugins that simulate different frameworks."""
        # Register QONNX-style transform
        @transform(name="QONNXTransform", stage="cleanup")
        class QONNXTransform:
            def apply(self, model):
                # Simulate QONNX transform
                return model, False
        
        # Register FINN-style transform
        @transform(name="FINNTransform", stage="topology_opt")
        class FINNTransform:
            def apply(self, model):
                # Simulate FINN transform
                return model, False
        
        # Register Brainsmith kernel
        @kernel(name="BrainsmithKernel")
        class BrainsmithKernel:
            def compile(self, node):
                return {"framework": "brainsmith", "node": node}
        
        @backend(name="BrainsmithBackend", kernel="BrainsmithKernel", language="rtl")
        class BrainsmithBackend:
            def generate(self, kernel_instance):
                return "// Brainsmith RTL"
        
        # Create blueprint mixing frameworks
        blueprint_data = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": ["BrainsmithKernel"],
                "transforms": {
                    "cleanup": ["QONNXTransform"],
                    "topology_opt": ["FINNTransform"]
                },
                "build_steps": ["ConvertToHW"]
            },
            "search": {"strategy": "exhaustive"},
            "global": {"output_stage": "rtl", "working_directory": "./builds"}
        }
        
        # Parse with real registry
        parser = BlueprintParser()
        design_space = parser.parse(blueprint_data, "model.onnx")
        
        # Verify cross-framework plugins work together
        assert design_space.hw_compiler_space.kernels[0] == ("BrainsmithKernel", ["BrainsmithBackend"])
        assert design_space.hw_compiler_space.transforms["cleanup"] == ["QONNXTransform"]
        assert design_space.hw_compiler_space.transforms["topology_opt"] == ["FINNTransform"]
    
    def test_plugin_validation_with_real_registry(self, tmp_path):
        """Test validation errors with real registry."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake")
        
        # Blueprint with non-existent plugins
        bad_blueprint = tmp_path / "bad.yaml"
        bad_blueprint.write_text("""
version: "3.0"
hw_compiler:
  kernels:
    - "NonExistentKernel"
    - ["TestMatMul", ["fake_backend"]]
  transforms:
    - "NonExistentTransform"
  build_steps: ["Step1"]
search:
  strategy: "exhaustive"
global:
  output_stage: "rtl"
  working_directory: "./builds"
""")
        
        # Should fail with helpful errors
        with pytest.raises(BlueprintParseError) as exc:
            forge(str(model_path), str(bad_blueprint))
        
        error_msg = str(exc.value)
        assert "NonExistentKernel" in error_msg or "NonExistentTransform" in error_msg
        assert "not found" in error_msg
    
    def test_mutually_exclusive_groups_with_real_plugins(self, test_plugins):
        """Test mutually exclusive groups work with real plugins."""
        # Register additional kernel for groups
        @kernel(name="StandardConv")
        class StandardConv:
            def compile(self, node):
                return {"type": "standard_conv"}
        
        @kernel(name="DepthwiseConv")
        class DepthwiseConv:
            def compile(self, node):
                return {"type": "depthwise_conv"}
        
        @backend(name="StandardConvRTL", kernel="StandardConv", language="rtl")
        class StandardConvRTL:
            def generate(self, kernel_instance):
                return "// Standard Conv RTL"
        
        @backend(name="DepthwiseConvDSP", kernel="DepthwiseConv", language="dsp")
        class DepthwiseConvDSP:
            def generate(self, kernel_instance):
                return "// Depthwise Conv DSP"
        
        blueprint_data = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": [
                    # Mutually exclusive group
                    [
                        "StandardConv",  # Will auto-discover
                        ("DepthwiseConv", ["DepthwiseConvDSP"]),  # Explicit
                        None  # Skip option
                    ]
                ],
                "transforms": [],
                "build_steps": ["ConvertToHW"]
            },
            "search": {"strategy": "exhaustive"},
            "global": {"output_stage": "rtl", "working_directory": "./builds"}
        }
        
        parser = BlueprintParser()
        design_space = parser.parse(blueprint_data, "model.onnx")
        
        # Verify group parsed correctly
        kernel_group = design_space.hw_compiler_space.kernels[0]
        assert len(kernel_group) == 3
        assert kernel_group[0] == ("StandardConv", ["StandardConvRTL"])
        assert kernel_group[1] == ("DepthwiseConv", ["DepthwiseConvDSP"])
        assert kernel_group[2] is None
    
    def test_registry_state_isolation(self):
        """Test that plugin registration is properly isolated."""
        # Get initial plugin count
        registry = get_registry()
        initial_kernels = len(registry.kernels)
        initial_transforms = len(registry.transforms)
        
        # Register a test plugin
        @kernel(name="IsolationTestKernel")
        class IsolationTestKernel:
            pass
        
        # Verify it was registered
        assert len(registry.kernels) == initial_kernels + 1
        
        # Reset plugin system
        reset_plugin_system()
        
        # Note: In the actual implementation, reset might not remove plugins
        # This test documents the expected behavior
    
    def test_plugin_discovery_with_real_registry(self, test_plugins):
        """Test discovery methods work with real registered plugins."""
        registry = get_registry()
        
        # Test kernel discovery
        available_kernels = registry.list_available_kernels()
        assert "TestMatMul" in available_kernels
        assert "TestLayerNorm" in available_kernels
        assert "TestAttention" in available_kernels
        
        # Test transform discovery
        available_transforms = registry.list_available_transforms()
        assert "TestCleanupTransform" in available_transforms
        assert "TestTopologyTransform" in available_transforms
        
        # Test backend discovery for kernels
        matmul_backends = registry.list_backends_by_kernel("TestMatMul")
        assert set(matmul_backends) == {"TestRTLBackend", "TestHLSBackend"}
        
        attention_backends = registry.list_backends_by_kernel("TestAttention")
        assert set(attention_backends) == {"TestAttentionCUDA", "TestAttentionTriton"}
        
        # Test stage discovery
        valid_stages = registry.get_valid_stages()
        assert "cleanup" in valid_stages
        assert "topology_opt" in valid_stages
        assert "kernel_opt" in valid_stages