"""Test cross-framework plugin discovery.

Tests the ability to discover and query plugins across
BrainSmith, QONNX, and FINN frameworks.
"""

import pytest
import sys
import os
from collections import defaultdict

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


class TestCrossFramework:
    """Test suite for cross-framework plugin discovery."""
    
    def test_unified_discovery(self):
        """Test discovering plugins from all frameworks."""
        from brainsmith.plugin.core import get_registry
        
        # Check if discovery module exists
        try:
            from brainsmith.plugin.discovery import PluginDiscovery
        except ImportError:
            pytest.skip("Plugin discovery not implemented yet")
        
        # Run discovery
        discovery = PluginDiscovery()
        counts = discovery.discover_all()
        
        print("\n=== Plugin Discovery Results ===")
        for framework, count in counts.items():
            print(f"{framework}: {count} plugins")
        
        # Verify we discovered plugins
        total = sum(counts.values())
        assert total > 0, "Should discover at least some plugins"
        
        # Check registry has plugins from multiple sources
        registry = get_registry()
        all_plugins = registry.query()
        
        # Group by framework
        by_framework = defaultdict(list)
        for plugin in all_plugins:
            framework = plugin.get("framework", "brainsmith")
            by_framework[framework].append(plugin)
        
        print("\n=== Plugins by Framework ===")
        for fw, plugins in by_framework.items():
            print(f"{fw}: {len(plugins)} plugins")
    
    def test_framework_prefixes(self):
        """Test plugin name prefixes for different frameworks."""
        from brainsmith.plugin.core import get_registry
        
        registry = get_registry()
        
        # Check for prefixed names
        all_transforms = registry.query(type="transform")
        
        qonnx_prefixed = [t for t in all_transforms if t["name"].startswith("qonnx:")]
        finn_prefixed = [t for t in all_transforms if t["name"].startswith("finn:")]
        
        print(f"\nQONNX prefixed transforms: {len(qonnx_prefixed)}")
        print(f"FINN prefixed transforms: {len(finn_prefixed)}")
        
        # Verify prefix structure
        for t in qonnx_prefixed:
            assert t["name"].startswith("qonnx:")
            # Should have framework metadata
            if "framework" in t:
                assert t["framework"] == "qonnx"
        
        for t in finn_prefixed:
            assert t["name"].startswith("finn:")
            if "framework" in t:
                assert t["framework"] == "finn"
    
    def test_query_by_framework(self):
        """Test filtering plugins by framework."""
        from brainsmith.plugin.core import get_registry
        
        registry = get_registry()
        
        # Query by framework metadata
        qonnx_plugins = registry.query(framework="qonnx")
        finn_plugins = registry.query(framework="finn")
        brainsmith_plugins = registry.query(framework="brainsmith")
        
        print(f"\n=== Plugins by Framework Query ===")
        print(f"QONNX: {len(qonnx_plugins)}")
        print(f"FINN: {len(finn_plugins)}")
        print(f"BrainSmith: {len(brainsmith_plugins)}")
        
        # Verify framework consistency
        for p in qonnx_plugins:
            assert p.get("framework") == "qonnx"
        
        for p in finn_plugins:
            assert p.get("framework") == "finn"
        
        for p in brainsmith_plugins:
            assert p.get("framework") == "brainsmith"
    
    def test_stage_consistency(self):
        """Verify stage names are consistent across frameworks."""
        from brainsmith.plugin.core import get_registry
        
        registry = get_registry()
        
        valid_stages = ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt"]
        
        # Get all transforms
        all_transforms = registry.query(type="transform")
        
        # Check stage validity
        invalid_stages = []
        stage_counts = defaultdict(int)
        
        for t in all_transforms:
            stage = t.get("stage")
            if stage is not None:  # None is valid for some transforms
                stage_counts[stage] += 1
                if stage not in valid_stages:
                    invalid_stages.append((t["name"], stage))
        
        print("\n=== Stage Distribution ===")
        for stage, count in sorted(stage_counts.items()):
            print(f"{stage}: {count} transforms")
        
        if invalid_stages:
            print("\n⚠️  Transforms with invalid stages:")
            for name, stage in invalid_stages:
                print(f"  - {name}: {stage}")
        
        # All non-None stages should be valid
        for name, stage in invalid_stages:
            assert stage in valid_stages, f"{name} has invalid stage: {stage}"
    
    def test_transform_resolution(self):
        """Test resolving transform names across frameworks."""
        # Check if query interface exists
        try:
            from brainsmith.plugin.query import UnifiedQuery
        except ImportError:
            pytest.skip("UnifiedQuery not implemented yet")
        
        query = UnifiedQuery()
        
        # Test resolving BrainSmith transform
        bs_transform = query.resolve_transform_name("ExpandNorms")
        if bs_transform:
            assert bs_transform.__name__ == "ExpandNorms"
        
        # Test resolving with prefix
        qonnx_transform = query.resolve_transform_name("qonnx:FoldConstants")
        if qonnx_transform:
            assert "FoldConstants" in qonnx_transform.__name__
        
        # Test resolving without prefix (should try prefixes)
        fold_transform = query.resolve_transform_name("FoldConstants")
        if fold_transform:
            assert "FoldConstants" in fold_transform.__name__
    
    def test_cross_framework_query(self):
        """Test querying across multiple framework registries."""
        try:
            from brainsmith.plugin.query import UnifiedQuery
        except ImportError:
            pytest.skip("UnifiedQuery not implemented yet")
        
        query = UnifiedQuery()
        
        if hasattr(query, 'query_cross_framework'):
            # Query all transforms
            all_transforms = query.query_cross_framework(type="transform")
            
            # Group by source
            by_source = defaultdict(list)
            for t in all_transforms:
                source = t.get("framework", "unknown")
                by_source[source].append(t)
            
            print("\n=== Cross-Framework Query Results ===")
            for source, transforms in by_source.items():
                print(f"{source}: {len(transforms)} transforms")
    
    def test_list_transforms_by_stage(self):
        """Test listing transforms organized by stage."""
        try:
            from brainsmith.plugin.query import UnifiedQuery
        except ImportError:
            pytest.skip("UnifiedQuery not implemented yet")
        
        query = UnifiedQuery()
        
        if hasattr(query, 'list_transforms_by_stage'):
            stages = query.list_transforms_by_stage()
            
            print("\n=== Transforms by Stage ===")
            for stage, transforms in stages.items():
                print(f"\n{stage}: {len(transforms)} transforms")
                
                # Show some examples
                examples = transforms[:5]
                for t in examples:
                    print(f"  - {t}")
                if len(transforms) > 5:
                    print(f"  ... and {len(transforms) - 5} more")
    
    def test_list_all_transforms(self):
        """Test listing all transforms grouped by framework."""
        try:
            from brainsmith.plugin.query import UnifiedQuery
        except ImportError:
            pytest.skip("UnifiedQuery not implemented yet")
        
        query = UnifiedQuery()
        
        if hasattr(query, 'list_all_transforms'):
            frameworks = query.list_all_transforms()
            
            print("\n=== All Transforms by Framework ===")
            for fw, transforms in frameworks.items():
                print(f"\n{fw}: {len(transforms)} transforms")
                
                # Show some examples
                examples = transforms[:3]
                for t in examples:
                    print(f"  - {t}")
                if len(transforms) > 3:
                    print(f"  ... and {len(transforms) - 3} more")
    
    def test_kernel_discovery_across_frameworks(self):
        """Test discovering kernels from different sources."""
        from brainsmith.plugin.core import get_registry
        
        registry = get_registry()
        
        # Get all kernels
        all_kernels = registry.query(type="kernel")
        
        # Group by framework
        by_framework = defaultdict(list)
        for kernel in all_kernels:
            fw = kernel.get("framework", "brainsmith")
            by_framework[fw].append(kernel["name"])
        
        print("\n=== Kernels by Framework ===")
        for fw, kernels in by_framework.items():
            print(f"\n{fw}:")
            for k in sorted(kernels):
                print(f"  - {k}")
        
        # BrainSmith should have some kernels
        assert len(by_framework.get("brainsmith", [])) > 0
    
    def test_metadata_consistency(self):
        """Test that metadata is consistent across frameworks."""
        from brainsmith.plugin.core import get_registry
        
        registry = get_registry()
        
        # Check all plugins have required fields
        all_plugins = registry.query()
        
        required_fields = ["type", "name", "class"]
        
        for plugin in all_plugins:
            for field in required_fields:
                assert field in plugin, f"Plugin missing required field '{field}': {plugin.get('name', 'unknown')}"
            
            # Type-specific requirements
            if plugin["type"] == "backend":
                assert "kernel" in plugin
                assert "backend_type" in plugin
                assert plugin["backend_type"] in ["hls", "rtl"]
            
            if plugin["type"] == "kernel_inference":
                assert plugin.get("stage") is None
                assert "kernel" in plugin