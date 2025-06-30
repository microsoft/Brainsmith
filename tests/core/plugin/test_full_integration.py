"""Full integration tests for the unified plugin system.

Tests complete workflows and interactions between all components
of the plugin system including BrainSmith, QONNX, and FINN.
"""

import pytest
import tempfile
import json
from pathlib import Path


class TestFullIntegration:
    """Test suite for complete plugin system integration."""
    
    def test_complete_transform_pipeline(self, clean_registry):
        """Test a complete transform pipeline across frameworks."""
        from brainsmith.plugin.core import transform, get_registry
        
        # Register test transforms for each stage
        @transform(name="TestCleanup", stage="cleanup")
        class TestCleanup:
            def apply(self, model):
                model.metadata = getattr(model, 'metadata', {})
                model.metadata['cleanup'] = True
                return model, True
        
        @transform(name="TestTopology", stage="topology_opt")
        class TestTopology:
            def apply(self, model):
                model.metadata['topology'] = True
                return model, True
        
        @transform(name="TestKernel", stage="kernel_opt")
        class TestKernel:
            def apply(self, model):
                model.metadata['kernel'] = True
                return model, True
        
        @transform(name="TestDataflow", stage="dataflow_opt")
        class TestDataflow:
            def apply(self, model):
                model.metadata['dataflow'] = True
                return model, True
        
        # Create a mock model
        class MockModel:
            pass
        
        model = MockModel()
        
        # Run through all stages
        stages = ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt"]
        registry = get_registry()
        
        for stage in stages:
            transforms = registry.query(type="transform", stage=stage, name__startswith="Test")
            assert len(transforms) >= 1
            
            # Get and apply transform
            transform_cls = registry.get("transform", transforms[0]["name"])
            transform_inst = transform_cls()
            model, changed = transform_inst.apply(model)
            assert changed
        
        # Verify all stages were applied
        assert model.metadata['cleanup']
        assert model.metadata['topology']
        assert model.metadata['kernel']
        assert model.metadata['dataflow']
    
    def test_kernel_to_backend_workflow(self, clean_registry):
        """Test complete kernel to backend selection workflow."""
        from brainsmith.plugin.core import kernel, backend, transform
        
        # Register a kernel
        @kernel(name="TestMatMul", op_type="MatMul", description="Test matrix multiplication")
        class TestMatMul:
            def get_nodeattr_types(self):
                return {"M": int, "N": int, "K": int}
        
        # Register multiple backends
        @backend(name="TestMatMulHLS", kernel="TestMatMul", backend_type="hls",
                description="HLS implementation", version="1.0")
        class TestMatMulHLS:
            def compile(self, node):
                return f"HLS code for {node}"
        
        @backend(name="TestMatMulRTL", kernel="TestMatMul", backend_type="rtl",
                description="RTL implementation", version="2.0")
        class TestMatMulRTL:
            def compile(self, node):
                return f"RTL code for {node}"
        
        # Register kernel inference
        @transform(name="InferTestMatMul", kernel="TestMatMul", stage=None)
        class InferTestMatMul:
            def apply(self, model):
                # Mock inference logic
                found_matmul = hasattr(model, 'has_matmul')
                return model, found_matmul
        
        # Test the workflow
        registry = clean_registry
        
        # 1. Query available kernels
        kernels = registry.query(type="kernel", op_type="MatMul")
        assert len(kernels) == 1
        assert kernels[0]["name"] == "TestMatMul"
        
        # 2. Get backends for the kernel
        backends = registry.query(type="backend", kernel="TestMatMul")
        assert len(backends) == 2
        
        # 3. Select backend by type
        hls_backends = registry.query(type="backend", kernel="TestMatMul", backend_type="hls")
        assert len(hls_backends) == 1
        assert hls_backends[0]["name"] == "TestMatMulHLS"
        
        # 4. Get kernel inference
        inferences = registry.query(type="kernel_inference", kernel="TestMatMul")
        assert len(inferences) == 1
        assert inferences[0]["name"] == "InferTestMatMul"
        
        # 5. Use the components
        kernel_cls = registry.get("kernel", "TestMatMul")
        backend_cls = registry.get("backend", "TestMatMulHLS")
        inference_cls = registry.get("kernel_inference", "InferTestMatMul")
        
        assert kernel_cls == TestMatMul
        assert backend_cls == TestMatMulHLS
        assert inference_cls == InferTestMatMul
    
    def test_cross_framework_transform_chain(self, clean_registry):
        """Test chaining transforms from different frameworks."""
        from brainsmith.plugin.core import transform
        
        # Simulate transforms from different frameworks
        @transform(name="BSTransform1", stage="cleanup", framework="brainsmith")
        class BSTransform1:
            def apply(self, model):
                model.history = getattr(model, 'history', [])
                model.history.append("brainsmith:cleanup")
                return model, True
        
        # Simulate QONNX-style transform (would be discovered)
        @transform(name="qonnx:FoldConstants", stage="cleanup", framework="qonnx")
        class QONNXFoldConstants:
            def apply(self, model):
                model.history.append("qonnx:cleanup")
                return model, True
        
        # Simulate FINN-style transform
        @transform(name="finn:StreamlineTransform", stage="topology_opt", framework="finn")
        class FINNStreamline:
            def apply(self, model):
                model.history.append("finn:topology_opt")
                return model, True
        
        # Create workflow
        class MockModel:
            pass
        
        model = MockModel()
        
        # Apply transforms in order
        transforms_to_apply = [
            ("transform", "BSTransform1"),
            ("transform", "qonnx:FoldConstants"),
            ("transform", "finn:StreamlineTransform")
        ]
        
        for plugin_type, name in transforms_to_apply:
            transform_cls = clean_registry.get(plugin_type, name)
            assert transform_cls is not None, f"Failed to get {name}"
            
            inst = transform_cls()
            model, _ = inst.apply(model)
        
        # Verify execution order
        assert model.history == [
            "brainsmith:cleanup",
            "qonnx:cleanup", 
            "finn:topology_opt"
        ]
    
    def test_plugin_discovery_simulation(self, clean_registry):
        """Test simulated plugin discovery from all frameworks."""
        from brainsmith.plugin.core import transform, kernel, backend
        
        # Simulate discovered plugins
        discovered_count = 0
        
        # Discover BrainSmith plugins (already registered)
        bs_transforms = clean_registry.query(framework="brainsmith", type="transform")
        discovered_count += len(bs_transforms)
        
        # Simulate QONNX discovery
        qonnx_transforms = [
            ("qonnx:FoldConstants", "cleanup", "Fold constant expressions"),
            ("qonnx:RemoveIdentityOps", "cleanup", "Remove identity operations"),
            ("qonnx:InferShapes", "cleanup", "Infer tensor shapes"),
            ("qonnx:GiveUniqueNodeNames", "cleanup", "Ensure unique node names")
        ]
        
        for name, stage, desc in qonnx_transforms:
            @transform(name=name, stage=stage, description=desc, framework="qonnx")
            class QONNXTransform:
                pass
            discovered_count += 1
        
        # Simulate FINN discovery
        finn_transforms = [
            ("finn:Streamline", "topology_opt", "FINN streamlining"),
            ("finn:ConvertBipolarToXnor", "kernel_opt", "Convert to XNOR ops"),
            ("finn:InsertDWC", "dataflow_opt", "Insert data width converters")
        ]
        
        for name, stage, desc in finn_transforms:
            @transform(name=name, stage=stage, description=desc, framework="finn")
            class FINNTransform:
                pass
            discovered_count += 1
        
        # Verify discovery results
        all_plugins = clean_registry.query()
        
        # Group by framework
        by_framework = {}
        for plugin in all_plugins:
            fw = plugin.get("framework", "brainsmith")
            by_framework[fw] = by_framework.get(fw, 0) + 1
        
        assert "qonnx" in by_framework
        assert "finn" in by_framework
        assert by_framework["qonnx"] >= len(qonnx_transforms)
        assert by_framework["finn"] >= len(finn_transforms)
    
    def test_stage_based_execution_order(self, clean_registry):
        """Test executing transforms in proper stage order."""
        from brainsmith.plugin.core import transform
        
        # Register transforms with dependencies
        execution_order = []
        
        @transform(name="Clean1", stage="cleanup", priority=1)
        class Clean1:
            def apply(self, model):
                execution_order.append("Clean1")
                return model, True
        
        @transform(name="Clean2", stage="cleanup", priority=2)
        class Clean2:
            def apply(self, model):
                execution_order.append("Clean2")
                return model, True
        
        @transform(name="Topo1", stage="topology_opt")
        class Topo1:
            def apply(self, model):
                execution_order.append("Topo1")
                return model, True
        
        @transform(name="Kernel1", stage="kernel_opt")
        class Kernel1:
            def apply(self, model):
                execution_order.append("Kernel1")
                return model, True
        
        # Execute in stage order
        class MockModel:
            pass
        
        model = MockModel()
        stages = ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt"]
        
        for stage in stages:
            # Get transforms for stage, sorted by priority if available
            transforms = clean_registry.query(type="transform", stage=stage)
            
            # Sort by priority if present
            transforms.sort(key=lambda t: t.get("priority", 999))
            
            for t in transforms:
                if t["name"] in ["Clean1", "Clean2", "Topo1", "Kernel1"]:
                    transform_cls = clean_registry.get("transform", t["name"])
                    inst = transform_cls()
                    model, _ = inst.apply(model)
        
        # Verify stage-based ordering
        assert execution_order.index("Clean1") < execution_order.index("Topo1")
        assert execution_order.index("Clean2") < execution_order.index("Topo1")
        assert execution_order.index("Topo1") < execution_order.index("Kernel1")
    
    def test_metadata_rich_workflow(self, clean_registry):
        """Test workflow using rich metadata for decision making."""
        from brainsmith.plugin.core import transform, kernel, backend
        
        # Register components with rich metadata
        @transform(
            name="OptimizeForSpeed",
            stage="topology_opt",
            description="Optimize graph for speed",
            tags=["performance", "speed", "optimization"],
            target="cpu",
            complexity="high"
        )
        class OptimizeForSpeed:
            def apply(self, model):
                model.optimized_for = "speed"
                return model, True
        
        @transform(
            name="OptimizeForSize", 
            stage="topology_opt",
            description="Optimize graph for size",
            tags=["size", "compression", "optimization"],
            target="embedded",
            complexity="medium"
        )
        class OptimizeForSize:
            def apply(self, model):
                model.optimized_for = "size"
                return model, True
        
        # Query based on metadata
        
        # 1. Find optimizations for embedded targets
        embedded_opts = clean_registry.query(
            type="transform",
            stage="topology_opt", 
            target="embedded"
        )
        assert len(embedded_opts) >= 1
        assert embedded_opts[0]["name"] == "OptimizeForSize"
        
        # 2. Find high complexity transforms
        complex_transforms = clean_registry.query(
            type="transform",
            complexity="high"
        )
        assert any(t["name"] == "OptimizeForSpeed" for t in complex_transforms)
        
        # 3. Find by tags (if supported)
        if hasattr(clean_registry, 'query_by_tags'):
            perf_transforms = clean_registry.query_by_tags(["performance"])
            assert any(t["name"] == "OptimizeForSpeed" for t in perf_transforms)
    
    def test_kernel_inference_pipeline(self, clean_registry):
        """Test complete kernel inference pipeline."""
        from brainsmith.plugin.core import kernel, transform
        
        # Register kernels
        @kernel(name="Conv2D", op_type="Conv")
        class Conv2D:
            pass
        
        @kernel(name="MatMul", op_type="MatMul")
        class MatMul:
            pass
        
        # Register inference transforms
        @transform(name="InferConv2D", kernel="Conv2D", stage=None)
        class InferConv2D:
            def apply(self, model):
                # Check for conv patterns
                model.kernels = getattr(model, 'kernels', [])
                if hasattr(model, 'has_conv'):
                    model.kernels.append("Conv2D")
                    return model, True
                return model, False
        
        @transform(name="InferMatMul", kernel="MatMul", stage=None)
        class InferMatMul:
            def apply(self, model):
                model.kernels = getattr(model, 'kernels', [])
                if hasattr(model, 'has_matmul'):
                    model.kernels.append("MatMul")
                    return model, True
                return model, False
        
        # Run inference pipeline
        class MockModel:
            has_conv = True
            has_matmul = True
        
        model = MockModel()
        
        # Get all inference transforms
        inferences = clean_registry.query(type="kernel_inference")
        
        # Apply all inferences
        changes = []
        for inf in inferences:
            if inf["name"] in ["InferConv2D", "InferMatMul"]:
                inf_cls = clean_registry.get("kernel_inference", inf["name"])
                inst = inf_cls()
                model, changed = inst.apply(model)
                changes.append(changed)
        
        # Verify results
        assert any(changes)  # At least one should have detected something
        assert "Conv2D" in model.kernels
        assert "MatMul" in model.kernels
    
    def test_plugin_versioning_workflow(self, clean_registry):
        """Test plugin versioning and compatibility."""
        from brainsmith.plugin.core import transform
        
        # Register versioned transforms
        @transform(name="VersionedTransform", stage="cleanup", version="1.0.0")
        class VersionedTransformV1:
            def apply(self, model):
                model.version = "1.0.0"
                return model, True
        
        # Try to register newer version (would replace in real system)
        try:
            @transform(name="VersionedTransform", stage="cleanup", version="2.0.0")
            class VersionedTransformV2:
                def apply(self, model):
                    model.version = "2.0.0"
                    return model, True
        except ValueError:
            # Current system prevents duplicates
            pass
        
        # Query by version if supported
        transforms = clean_registry.query(type="transform", name="VersionedTransform")
        if transforms and "version" in transforms[0]:
            version = transforms[0]["version"]
            assert version in ["1.0.0", "2.0.0"]
    
    def test_error_recovery_workflow(self, clean_registry):
        """Test workflow with error handling and recovery."""
        from brainsmith.plugin.core import transform
        
        # Register transforms with potential failures
        @transform(name="SafeTransform", stage="cleanup")
        class SafeTransform:
            def apply(self, model):
                model.safe_applied = True
                return model, True
        
        @transform(name="FailingTransform", stage="cleanup")
        class FailingTransform:
            def apply(self, model):
                if not hasattr(model, 'can_fail'):
                    raise RuntimeError("Expected failure")
                return model, True
        
        @transform(name="RecoveryTransform", stage="cleanup")
        class RecoveryTransform:
            def apply(self, model):
                model.recovered = True
                return model, True
        
        # Run workflow with error handling
        class MockModel:
            pass
        
        model = MockModel()
        applied_transforms = []
        errors = []
        
        transforms = clean_registry.query(type="transform", stage="cleanup")
        
        for t in transforms:
            if t["name"] in ["SafeTransform", "FailingTransform", "RecoveryTransform"]:
                try:
                    transform_cls = clean_registry.get("transform", t["name"])
                    inst = transform_cls()
                    model, changed = inst.apply(model)
                    applied_transforms.append(t["name"])
                except Exception as e:
                    errors.append((t["name"], str(e)))
                    # Continue with next transform
        
        # Verify error handling
        assert "SafeTransform" in applied_transforms
        assert ("FailingTransform", "Expected failure") in errors
        assert "RecoveryTransform" in applied_transforms
        assert hasattr(model, "recovered")
    
    def test_configuration_driven_pipeline(self, clean_registry):
        """Test pipeline execution driven by configuration."""
        from brainsmith.plugin.core import transform
        
        # Register configurable transforms
        @transform(name="ConfigurableCleanup", stage="cleanup")
        class ConfigurableCleanup:
            def __init__(self, config=None):
                self.config = config or {}
            
            def apply(self, model):
                model.cleanup_level = self.config.get("level", "default")
                return model, True
        
        @transform(name="ConfigurableOptimization", stage="topology_opt")
        class ConfigurableOptimization:
            def __init__(self, config=None):
                self.config = config or {}
            
            def apply(self, model):
                model.optimization = self.config.get("target", "balanced")
                return model, True
        
        # Pipeline configuration
        pipeline_config = {
            "transforms": [
                {
                    "name": "ConfigurableCleanup",
                    "config": {"level": "aggressive"}
                },
                {
                    "name": "ConfigurableOptimization",
                    "config": {"target": "speed"}
                }
            ]
        }
        
        # Execute configured pipeline
        class MockModel:
            pass
        
        model = MockModel()
        
        for transform_spec in pipeline_config["transforms"]:
            transform_cls = clean_registry.get("transform", transform_spec["name"])
            if transform_cls:
                inst = transform_cls(config=transform_spec.get("config"))
                model, _ = inst.apply(model)
        
        # Verify configuration was applied
        assert model.cleanup_level == "aggressive"
        assert model.optimization == "speed"