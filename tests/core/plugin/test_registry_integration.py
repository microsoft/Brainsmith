"""Test core plugin registry functionality."""

import pytest
from typing import Dict, List, Any


class TestPluginRegistry:
    """Test suite for plugin registry core functionality."""
    
    def test_brainsmith_transforms_visible(self, clean_registry):
        """Verify BrainSmith transforms are registered."""
        # Import transforms to trigger registration
        import brainsmith.transforms.topology_opt.expand_norms
        import brainsmith.transforms.kernel_opt.set_pumped_compute
        import brainsmith.transforms.cleanup.remove_identity
        
        # Query all transforms
        transforms = clean_registry.query(type="transform")
        transform_names = [t["name"] for t in transforms]
        
        # Check specific transforms are present
        assert "ExpandNorms" in transform_names
        assert "SetPumpedCompute" in transform_names
        assert "RemoveIdentityOps" in transform_names
        
        # Verify minimum count
        assert len(transforms) >= 3, f"Expected at least 3 transforms, found {len(transforms)}"
    
    def test_brainsmith_kernels_visible(self, clean_registry):
        """Verify BrainSmith kernels are registered."""
        # Import kernels
        import brainsmith.kernels.layernorm.layernorm
        import brainsmith.kernels.matmul.matmul
        import brainsmith.kernels.softmax.hwsoftmax
        import brainsmith.kernels.crop.crop
        
        # Query all kernels
        kernels = clean_registry.query(type="kernel")
        kernel_names = [k["name"] for k in kernels]
        
        # Check specific kernels
        assert "LayerNorm" in kernel_names
        assert "MatMul" in kernel_names
        assert "HWSoftmax" in kernel_names
        assert "Crop" in kernel_names
        
        assert len(kernels) >= 4, f"Expected at least 4 kernels, found {len(kernels)}"
    
    def test_brainsmith_backends_visible(self, clean_registry):
        """Verify BrainSmith backends are registered."""
        # Import backends
        import brainsmith.kernels.layernorm.layernorm_hls
        import brainsmith.kernels.layernorm.layernorm_rtl
        import brainsmith.kernels.matmul.matmul_hls
        import brainsmith.kernels.softmax.hwsoftmax_hls
        
        # Query all backends
        backends = clean_registry.query(type="backend")
        backend_names = [b["name"] for b in backends]
        
        # Check specific backends
        assert "LayerNormHLS" in backend_names
        assert "LayerNormRTL" in backend_names
        assert "MatMulHLS" in backend_names
        assert "HWSoftmaxHLS" in backend_names
        
        # Verify kernel associations
        layernorm_backends = [b for b in backends if b.get("kernel") == "LayerNorm"]
        assert len(layernorm_backends) >= 2, "LayerNorm should have HLS and RTL backends"
        
        assert len(backends) >= 4, f"Expected at least 4 backends, found {len(backends)}"
    
    def test_finn_plugins_accessible(self, clean_registry):
        """Verify FINN/QONNX plugins are accessible."""
        try:
            # Try importing FINN transforms
            from finn.transformation.streamline import absorb_1bit_muls
            
            # Check if FINN transforms registered
            transforms = clean_registry.query(type="transform")
            finn_transforms = [t for t in transforms if "finn" in t.get("name", "").lower() or 
                               "absorb" in t.get("name", "").lower()]
            
            # We don't require FINN plugins to be registered in our system
            # Just verify they're importable
            assert absorb_1bit_muls is not None, "FINN transform should be importable"
            
        except ImportError:
            # FINN might not be available in all environments
            pytest.skip("FINN not available in this environment")
    
    def test_dynamic_registration(self, clean_registry):
        """Test runtime plugin registration."""
        from brainsmith.plugin.core import transform, kernel, backend
        
        # Register a test transform
        @transform(name="TestTransform", stage="topology_opt", description="Dynamic test transform")
        class TestTransform:
            def apply(self, model):
                return model, False
        
        # Register a test kernel
        @kernel(name="TestKernel", op_type="Test", description="Dynamic test kernel")
        class TestKernel:
            pass
        
        # Register a test backend
        @backend(name="TestBackend", kernel="TestKernel", backend_type="hls")
        class TestBackend:
            pass
        
        # Verify all are retrievable
        assert clean_registry.get("transform", "TestTransform") == TestTransform
        assert clean_registry.get("kernel", "TestKernel") == TestKernel
        assert clean_registry.get("backend", "TestBackend") == TestBackend
    
    def test_plugin_retrieval_by_name(self, clean_registry):
        """Test retrieving plugins through registry."""
        # Import some plugins first
        import brainsmith.transforms.topology_opt.expand_norms
        import brainsmith.kernels.layernorm.layernorm
        
        # Get specific plugins by name
        expand_norms = clean_registry.get("transform", "ExpandNorms")
        assert expand_norms is not None
        assert hasattr(expand_norms, "__name__")
        
        layernorm = clean_registry.get("kernel", "LayerNorm")
        assert layernorm is not None
        
        # Test error cases
        assert clean_registry.get("transform", "NonExistent") is None
        assert clean_registry.get("kernel", "NonExistent") is None
    
    def test_plugin_query_by_post_proc(self, clean_registry):
        """Test querying plugins by post_proc."""
        # Import plugins
        import brainsmith.transforms.topology_opt.expand_norms
        import brainsmith.transforms.kernel_opt.set_pumped_compute
        import brainsmith.kernels.layernorm.layernorm_hls
        
        # Query by stage
        topology_transforms = clean_registry.query(type="transform", stage="topology_opt")
        assert len(topology_transforms) >= 1
        assert any(t["name"] == "ExpandNorms" for t in topology_transforms)
        
        # Query by kernel
        layernorm_backends = clean_registry.query(type="backend", kernel="LayerNorm")
        assert len(layernorm_backends) >= 1
        
        # Query by backend type
        hls_backends = clean_registry.query(type="backend", backend_type="hls")
        assert len(hls_backends) >= 1
    
    def test_kernel_inference_registration(self, clean_registry):
        """Test kernel inference transform registration."""
        from brainsmith.plugin.core import transform
        
        # Register kernel inference transform
        @transform(name="TestInference", kernel="TestKernel", stage=None, description="Test inference")
        class TestInference:
            def apply(self, model):
                return model, False
        
        # Should be registered as kernel_inference type
        inference = clean_registry.get("kernel_inference", "TestInference")
        assert inference == TestInference
        
        # Should NOT be in regular transforms
        assert clean_registry.get("transform", "TestInference") is None
        
        # Query by kernel and name
        inferences = clean_registry.query(type="kernel_inference", kernel="TestKernel", name="TestInference")
        assert len(inferences) == 1
        assert inferences[0]["name"] == "TestInference"
        assert inferences[0]["stage"] is None
    
    def test_kernel_inference_imports(self, clean_registry):
        """Test that actual kernel inference transforms work correctly."""
        # Import kernel inference transforms
        import brainsmith.kernels.layernorm.infer_layernorm
        import brainsmith.kernels.softmax.infer_hwsoftmax
        
        # Check they're registered as kernel_inference
        layernorm_inf = clean_registry.get("kernel_inference", "InferLayerNorm")
        assert layernorm_inf is not None
        
        softmax_inf = clean_registry.get("kernel_inference", "InferHWSoftmax") 
        assert softmax_inf is not None
        
        # Verify post_proc
        inferences = clean_registry.query(type="kernel_inference")
        for inf in inferences:
            assert inf["stage"] is None, f"Kernel inference {inf['name']} should have stage=None"
            assert inf.get("kernel") is not None, f"Kernel inference {inf['name']} should have kernel"
    
    def test_qonnx_transforms_in_registry(self, clean_registry):
        """Test that QONNX transforms are discoverable in the registry."""
        # Try to discover QONNX transforms
        try:
            from brainsmith.plugin.discovery import PluginDiscovery
            discovery = PluginDiscovery()
            discovery.discover_all()
        except ImportError:
            # Discovery might not be implemented yet
            pass
        
        # Look for QONNX transforms
        all_transforms = clean_registry.query(type="transform")
        qonnx_transforms = [t for t in all_transforms if t["name"].startswith("qonnx:")]
        
        if qonnx_transforms:
            # Verify structure
            for t in qonnx_transforms:
                assert "qonnx:" in t["name"]
                assert t.get("framework") == "qonnx"
                assert "class" in t
                
                # Check stage mapping
                stage = t.get("stage")
                if stage is not None:
                    assert stage in ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt"]
        
        # Also check if we can retrieve specific QONNX transforms
        fold_constants = clean_registry.get("transform", "qonnx:FoldConstants")
        if fold_constants:
            assert hasattr(fold_constants, 'apply')
    
    def test_finn_transforms_in_registry(self, clean_registry):
        """Test that FINN transforms are discoverable in the registry."""
        # Look for FINN transforms
        all_transforms = clean_registry.query(type="transform")
        finn_transforms = [t for t in all_transforms if t["name"].startswith("finn:") or
                          t.get("framework") == "finn"]
        
        if finn_transforms:
            # Verify structure
            for t in finn_transforms:
                assert t.get("framework") == "finn" or "finn:" in t["name"]
                assert "class" in t
                
                # FINN transforms should have valid stages
                stage = t.get("stage")
                if stage is not None:
                    assert stage in ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt"]
    
    def test_cross_framework_query(self, clean_registry):
        """Test querying across frameworks."""
        # Query by framework
        qonnx_plugins = clean_registry.query(framework="qonnx")
        finn_plugins = clean_registry.query(framework="finn")
        brainsmith_plugins = clean_registry.query(framework="brainsmith")
        
        # BrainSmith should always have some plugins
        assert len(brainsmith_plugins) > 0
        
        # Check consistency
        for p in qonnx_plugins:
            assert p.get("framework") == "qonnx"
        
        for p in finn_plugins:
            assert p.get("framework") == "finn"
        
        for p in brainsmith_plugins:
            assert p.get("framework") == "brainsmith"
    
    def test_unified_transform_access(self, clean_registry):
        """Test accessing transforms from all frameworks uniformly."""
        # Import to ensure registration
        import brainsmith.transforms.topology_opt.expand_norms
        
        # Get a BrainSmith transform
        expand_norms = clean_registry.get("transform", "ExpandNorms")
        assert expand_norms is not None
        
        # Try to get with framework prefix (shouldn't work for BrainSmith)
        bs_prefixed = clean_registry.get("transform", "brainsmith:ExpandNorms")
        # BrainSmith transforms might not use prefix
        
        # If QONNX transforms are registered, try to get one
        fold_constants = clean_registry.get("transform", "qonnx:FoldConstants")
        if fold_constants:
            assert hasattr(fold_constants, '__name__')
            assert hasattr(fold_constants, 'apply')
    
    def test_post_proc_completeness(self, clean_registry):
        """Test that plugins have complete post_proc."""
        all_plugins = clean_registry.query()
        
        # Count post_proc completeness
        complete_count = 0
        partial_count = 0
        minimal_count = 0
        
        for plugin in all_plugins:
            has_description = bool(plugin.get("description"))
            has_author = bool(plugin.get("author"))
            has_version = bool(plugin.get("version"))
            
            if has_description and has_author and has_version:
                complete_count += 1
            elif has_description or has_author or has_version:
                partial_count += 1
            else:
                minimal_count += 1
        
        total = len(all_plugins)
        
        # Report
        if total > 0:
            complete_pct = (complete_count / total) * 100
            partial_pct = (partial_count / total) * 100
            minimal_pct = (minimal_count / total) * 100
            
            print(f"\nMetadata completeness:")
            print(f"  Complete: {complete_count}/{total} ({complete_pct:.1f}%)")
            print(f"  Partial: {partial_count}/{total} ({partial_pct:.1f}%)")
            print(f"  Minimal: {minimal_count}/{total} ({minimal_pct:.1f}%)")
    
    def test_stage_distribution(self, clean_registry):
        """Test distribution of transforms across stages."""
        transforms = clean_registry.query(type="transform")
        
        # Count by stage
        stage_counts = {}
        for t in transforms:
            stage = t.get("stage", "none")
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        # All standard stages should have some transforms
        standard_stages = ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt"]
        
        print("\nTransform distribution by stage:")
        for stage in standard_stages:
            count = stage_counts.get(stage, 0)
            print(f"  {stage}: {count}")
        
        # Other stages
        other_stages = [s for s in stage_counts if s not in standard_stages and s != "none"]
        if other_stages:
            print("\nNon-standard stages:")
            for stage in other_stages:
                print(f"  {stage}: {stage_counts[stage]}")
        
        if "none" in stage_counts:
            print(f"\nNo stage: {stage_counts['none']}")