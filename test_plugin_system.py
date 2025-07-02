#!/usr/bin/env python3
"""
Test script to validate the BrainSmith plugin system.

This script tests:
1. Plugin registration via decorators
2. Plugin discovery from the new structure
3. Registry functionality
4. Listing and searching plugins
"""

import sys
import logging
from pathlib import Path

# Add brainsmith to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from brainsmith.plugin import PluginRegistry, PluginDiscovery


def test_manual_registration():
    """Test manual plugin registration via decorators."""
    print("\n=== Testing Manual Registration ===")
    
    # Import a transform to trigger decorator registration
    from brainsmith.transforms.cleanup.remove_identity import RemoveIdentityOps
    
    # Check if it was registered
    registry = PluginRegistry()
    transform = registry.get_transform("RemoveIdentityOps")
    
    if transform:
        print(f"✓ Successfully registered transform: {transform.__name__}")
        post_proc = transform._plugin_post_proc
        print(f"  - Stage: {post_proc['stage']}")
        print(f"  - Description: {post_proc['description']}")
    else:
        print("✗ Transform registration failed")


def test_plugin_discovery():
    """Test automatic plugin discovery."""
    print("\n=== Testing Plugin Discovery ===")
    
    # Create a new discovery instance
    discovery = PluginDiscovery()
    
    # Discover all plugins
    discovery.discover_all()
    
    # Get discovered modules
    modules = discovery.get_discovered_modules()
    print(f"Discovered {len(modules)} modules:")
    for module in sorted(modules):
        print(f"  - {module}")


def test_registry_operations():
    """Test registry listing and searching."""
    print("\n=== Testing Registry Operations ===")
    
    registry = PluginRegistry()
    
    # Get statistics
    stats = registry.get_stats()
    print(f"Registry statistics:")
    for plugin_type, count in stats.items():
        print(f"  - {plugin_type}: {count}")
    
    # List transforms by stage
    print("\nTransforms by stage:")
    stages = ["cleanup", "topology_optimization", "kernel_mapping", 
              "kernel_optimization", "graph_optimization"]
    
    for stage in stages:
        transforms = registry.list_transforms(stage=stage)
        if transforms:
            print(f"  {stage}:")
            for name, cls in transforms:
                print(f"    - {name} ({cls.__module__}.{cls.__name__})")
    
    # List kernels
    print("\nKernels:")
    kernels = registry.list_kernels()
    for name, cls in kernels:
        print(f"  - {name} ({cls.__module__}.{cls.__name__})")
    
    # List backends
    print("\nBackends:")
    backends = registry.list_backends()
    for name, cls in backends:
        print(f"  - {name} ({cls.__module__}.{cls.__name__})")
    
    # List hardware transforms
    print("\nHardware Transforms:")
    hw_transforms = registry.list_hw_transforms()
    for name, cls in hw_transforms:
        print(f"  - {name} ({cls.__module__}.{cls.__name__})")


def test_plugin_info():
    """Test getting detailed plugin information."""
    print("\n=== Testing Plugin Info Retrieval ===")
    
    registry = PluginRegistry()
    
    # Get info for a specific transform
    info = registry.get_plugin_info("transform", "ExpandNorms")
    if info:
        print(f"Plugin info for ExpandNorms:")
        for key, value in info.items():
            print(f"  - {key}: {value}")
    
    # Get info for a kernel
    info = registry.get_plugin_info("kernel", "LayerNorm")
    if info:
        print(f"\nPlugin info for LayerNorm kernel:")
        for key, value in info.items():
            print(f"  - {key}: {value}")


def test_search_functionality():
    """Test plugin search functionality."""
    print("\n=== Testing Search Functionality ===")
    
    registry = PluginRegistry()
    
    # Search for norm-related plugins
    results = registry.search_plugins("norm")
    print(f"Search results for 'norm': {len(results)} found")
    for result in results:
        print(f"  - {result['type']}: {result['name']}")
        if result.get('description'):
            print(f"    {result['description']}")
    
    # Search within specific type
    results = registry.search_plugins("layer", plugin_type="kernel")
    print(f"\nSearch results for 'layer' in kernels: {len(results)} found")
    for result in results:
        print(f"  - {result['name']}: {result.get('description', 'No description')}")


def validate_structure():
    """Validate the directory structure is correct."""
    print("\n=== Validating Directory Structure ===")
    
    expected_dirs = [
        "brainsmith/plugin",
        "brainsmith/transforms/cleanup",
        "brainsmith/transforms/topology_optimization",
        "brainsmith/transforms/kernel_mapping",
        "brainsmith/transforms/kernel_optimization",
        "brainsmith/transforms/graph_optimization",
        "brainsmith/kernels/matmul",
        "brainsmith/kernels/layernorm",
        "brainsmith/hw_kernels/hls"
    ]
    
    all_exist = True
    for dir_path in expected_dirs:
        path = Path(dir_path)
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_path}")
        if not exists:
            all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    print("BrainSmith Plugin System Validation")
    print("=" * 50)
    
    # First validate structure
    if not validate_structure():
        print("\nERROR: Directory structure validation failed!")
        return 1
    
    # Run tests
    try:
        test_manual_registration()
        test_plugin_discovery()
        test_registry_operations()
        test_plugin_info()
        test_search_functionality()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())