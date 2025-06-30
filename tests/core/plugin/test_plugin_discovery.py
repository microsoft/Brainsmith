"""Test plugin discovery capabilities."""

import pytest
from collections import defaultdict


class TestPluginDiscovery:
    """Test suite for plugin discovery and reporting."""
    
    def test_list_all_transforms(self):
        """List all registered transforms."""
        from brainsmith.plugin.core import get_registry
        
        # Import all transforms to ensure registration
        import brainsmith.transforms
        
        registry = get_registry()
        transforms = registry.query(type="transform")
        
        print("\n=== Discovered Transforms ===")
        print(f"Total: {len(transforms)}")
        
        # Group by stage
        by_stage = defaultdict(list)
        for t in transforms:
            stage = t.get("stage", "unknown")
            by_stage[stage].append(t["name"])
        
        for stage, names in sorted(by_stage.items()):
            print(f"\n{stage}:")
            for name in sorted(names):
                print(f"  - {name}")
        
        # No assertions - just discovery and reporting
        assert len(transforms) > 0, "Should discover at least one transform"
    
    def test_list_all_kernels(self):
        """List all registered kernels."""
        from brainsmith.plugin.core import get_registry
        
        # Import kernels
        import brainsmith.kernels
        
        registry = get_registry()
        kernels = registry.query(type="kernel")
        
        print("\n=== Discovered Kernels ===")
        print(f"Total: {len(kernels)}")
        
        for kernel in sorted(kernels, key=lambda k: k["name"]):
            op_type = kernel.get("op_type", "not specified")
            print(f"  - {kernel['name']} (op_type: {op_type})")
        
        assert len(kernels) > 0, "Should discover at least one kernel"
    
    def test_kernel_backend_associations(self):
        """Discover kernel-backend relationships."""
        from brainsmith.plugin.core import get_registry
        
        # Import all kernels and backends
        import brainsmith.kernels
        
        registry = get_registry()
        backends = registry.query(type="backend")
        
        print("\n=== Kernel-Backend Associations ===")
        
        # Group backends by kernel
        kernel_backends = defaultdict(list)
        for backend in backends:
            kernel = backend.get("kernel", "unknown")
            backend_type = backend.get("backend_type", "unknown")
            kernel_backends[kernel].append({
                "name": backend["name"],
                "type": backend_type
            })
        
        for kernel, backend_list in sorted(kernel_backends.items()):
            print(f"\n{kernel}:")
            for backend in backend_list:
                print(f"  - {backend['name']} ({backend['type']})")
        
        # Check for kernels without backends
        all_kernels = registry.query(type="kernel")
        kernel_names = {k["name"] for k in all_kernels}
        kernels_with_backends = set(kernel_backends.keys())
        
        orphan_kernels = kernel_names - kernels_with_backends
        if orphan_kernels:
            print("\n⚠️  Kernels without backends:")
            for kernel in sorted(orphan_kernels):
                print(f"  - {kernel}")
    
    def test_kernel_inference_associations(self):
        """Discover kernel inference relationships."""
        from brainsmith.plugin.core import get_registry
        
        # Import kernel inference transforms
        import brainsmith.kernels
        
        registry = get_registry()
        inferences = registry.query(type="kernel_inference")
        
        print("\n=== Kernel Inference Transforms ===")
        print(f"Total: {len(inferences)}")
        
        # Group by kernel
        by_kernel = defaultdict(list)
        for inf in inferences:
            kernel = inf.get("kernel", "unknown")
            by_kernel[kernel].append(inf["name"])
        
        for kernel, names in sorted(by_kernel.items()):
            print(f"\n{kernel}:")
            for name in sorted(names):
                print(f"  - {name}")
        
        # Verify all have stage=None
        for inf in inferences:
            assert inf.get("stage") is None, f"{inf['name']} should have stage=None"
    
    def test_stage_organization(self):
        """Discover transform organization by stage."""
        from brainsmith.plugin.core import get_registry
        
        # Import transforms
        import brainsmith.transforms
        
        registry = get_registry()
        
        print("\n=== Transform Organization by Stage ===")
        
        stages = ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt"]
        
        for stage in stages:
            transforms = registry.query(type="transform", stage=stage)
            print(f"\n{stage}: {len(transforms)} transforms")
            for t in sorted(transforms, key=lambda x: x["name"]):
                desc = t.get("description", "")
                if desc:
                    print(f"  - {t['name']}: {desc}")
                else:
                    print(f"  - {t['name']}")
        
        # Check for transforms without standard stages
        all_transforms = registry.query(type="transform")
        non_standard = [t for t in all_transforms if t.get("stage") not in stages]
        
        if non_standard:
            print("\n⚠️  Transforms with non-standard stages:")
            for t in non_standard:
                print(f"  - {t['name']} (stage: {t.get('stage', 'None')})")
    
    def test_plugin_summary(self):
        """Generate overall plugin system summary."""
        from brainsmith.plugin.core import get_registry
        
        # Import everything
        import brainsmith.transforms
        import brainsmith.kernels
        
        registry = get_registry()
        
        print("\n" + "=" * 50)
        print("PLUGIN SYSTEM SUMMARY")
        print("=" * 50)
        
        # Count by type
        types = ["transform", "kernel", "backend", "kernel_inference"]
        
        for plugin_type in types:
            items = registry.query(type=plugin_type)
            print(f"\n{plugin_type.replace('_', ' ').title()}s: {len(items)}")
        
        # Check cross-framework access
        print("\n=== Cross-Framework Access ===")
        
        try:
            from finn.transformation.streamline import Streamline
            print("✓ FINN transforms importable")
        except ImportError:
            print("✗ FINN transforms not available")
        
        try:
            from qonnx.transformation.general import GiveUniqueNodeNames
            print("✓ QONNX transforms importable")
        except ImportError:
            print("✗ QONNX transforms not available")
        
        print("\n" + "=" * 50)
    
    def test_qonnx_transform_discovery(self):
        """Discover and report QONNX transforms."""
        from brainsmith.plugin.core import get_registry
        
        registry = get_registry()
        
        # Look for QONNX transforms (prefixed with qonnx:)
        all_transforms = registry.query(type="transform")
        qonnx_transforms = [t for t in all_transforms if t["name"].startswith("qonnx:")]
        
        print("\n=== QONNX Transform Discovery ===")
        print(f"Total QONNX transforms: {len(qonnx_transforms)}")
        
        if qonnx_transforms:
            # Group by stage
            by_stage = defaultdict(list)
            for t in qonnx_transforms:
                stage = t.get("stage", "unassigned")
                by_stage[stage].append(t["name"].replace("qonnx:", ""))
            
            for stage, names in sorted(by_stage.items()):
                print(f"\n{stage}:")
                for name in sorted(names)[:10]:  # Show first 10
                    print(f"  - {name}")
                if len(names) > 10:
                    print(f"  ... and {len(names) - 10} more")
        
        # Check QONNX metadata if available
        try:
            from qonnx.transformation.registry import TRANSFORMATION_METADATA
            
            if TRANSFORMATION_METADATA:
                print("\n✓ QONNX enhanced registry with metadata available")
                
                # Show some metadata examples
                example_count = 0
                for name, metadata in list(TRANSFORMATION_METADATA.items())[:3]:
                    if metadata.get("tags"):
                        print(f"\n{name}:")
                        print(f"  Tags: {', '.join(metadata['tags'])}")
                        if metadata.get("description"):
                            print(f"  Description: {metadata['description']}")
                        example_count += 1
        except ImportError:
            print("\n✗ QONNX enhanced registry not available")
    
    def test_finn_transform_discovery(self):
        """Discover and report FINN transforms."""
        from brainsmith.plugin.core import get_registry
        
        registry = get_registry()
        
        # Look for FINN transforms
        all_transforms = registry.query(type="transform")
        finn_transforms = [t for t in all_transforms if t["name"].startswith("finn:") or 
                          t.get("framework") == "finn"]
        
        print("\n=== FINN Transform Discovery ===")
        print(f"Total FINN transforms: {len(finn_transforms)}")
        
        if finn_transforms:
            # Group by stage
            by_stage = defaultdict(list)
            for t in finn_transforms:
                stage = t.get("stage", "unassigned")
                name = t["name"].replace("finn:", "") if t["name"].startswith("finn:") else t["name"]
                by_stage[stage].append(name)
            
            for stage, names in sorted(by_stage.items()):
                print(f"\n{stage}:")
                for name in sorted(names)[:5]:
                    print(f"  - {name}")
                if len(names) > 5:
                    print(f"  ... and {len(names) - 5} more")
        
        # Check FINN registry
        try:
            from finn.plugin.registry import get_finn_registry
            
            finn_registry = get_finn_registry()
            if hasattr(finn_registry, 'query'):
                finn_direct = finn_registry.query(type="transform")
                print(f"\n✓ FINN enhanced registry available ({len(finn_direct)} transforms)")
            else:
                print("\n✗ FINN enhanced registry not available")
        except ImportError:
            print("\n✗ FINN not available")
    
    def test_framework_metadata_summary(self):
        """Summarize metadata availability across frameworks."""
        from brainsmith.plugin.core import get_registry
        
        registry = get_registry()
        
        print("\n=== Framework Metadata Summary ===")
        
        # Check metadata fields
        all_plugins = registry.query()
        
        metadata_fields = defaultdict(int)
        by_framework = defaultdict(lambda: defaultdict(int))
        
        for plugin in all_plugins:
            fw = plugin.get("framework", "brainsmith")
            
            # Count metadata fields
            for field in ["description", "author", "version", "tags"]:
                if field in plugin and plugin[field]:
                    metadata_fields[field] += 1
                    by_framework[fw][field] += 1
        
        # Report
        print("\nOverall metadata coverage:")
        total = len(all_plugins)
        for field, count in sorted(metadata_fields.items()):
            percent = (count / total * 100) if total > 0 else 0
            print(f"  {field}: {count}/{total} ({percent:.1f}%)")
        
        print("\nBy framework:")
        for fw, fields in sorted(by_framework.items()):
            fw_plugins = len([p for p in all_plugins if p.get("framework", "brainsmith") == fw])
            print(f"\n{fw} ({fw_plugins} plugins):")
            for field, count in sorted(fields.items()):
                percent = (count / fw_plugins * 100) if fw_plugins > 0 else 0
                print(f"  {field}: {count} ({percent:.1f}%)")