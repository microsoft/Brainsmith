"""
Debug utilities for the Brainsmith plugin system.

Provides convenient functions for inspecting registered transforms, kernels, and backends.
Can be used programmatically or run as a CLI tool.

Usage:
    # Programmatic
    from brainsmith.core.plugins.debug import list_all_transforms, get_transform_stats
    
    # CLI
    ./smithy exec python -m brainsmith.core.plugins.debug
"""

import sys
from typing import Dict, List, Optional, Any
from .registry import get_registry


def list_all_transforms(framework: Optional[str] = None, stage: Optional[str] = None) -> List[str]:
    """
    List all registered transform names.
    
    Args:
        framework: Filter by framework ('brainsmith', 'qonnx', 'finn')
        stage: Filter by stage ('cleanup', 'topology_opt', etc.)
    
    Returns:
        List of transform names
    """
    registry = get_registry()
    
    if framework:
        transforms = registry.framework_transforms.get(framework, {})
        names = list(transforms.keys())
    elif stage:
        transforms = registry.transforms_by_stage.get(stage, {})
        names = list(transforms.keys())
    else:
        names = list(registry.transforms.keys())
    
    return sorted(names)


def list_transforms_by_framework() -> Dict[str, List[str]]:
    """
    List transforms grouped by framework.
    
    Returns:
        Dict mapping framework name to list of transform names
    """
    registry = get_registry()
    result = {}
    
    for framework, transforms in registry.framework_transforms.items():
        result[framework] = sorted(transforms.keys())
    
    return result


def list_transforms_by_stage() -> Dict[str, List[str]]:
    """
    List transforms grouped by stage.
    
    Returns:
        Dict mapping stage name to list of transform names
    """
    registry = get_registry()
    result = {}
    
    for stage, transforms in registry.transforms_by_stage.items():
        result[stage] = sorted(transforms.keys())
    
    return result


def get_transform_stats() -> Dict[str, Any]:
    """
    Get comprehensive statistics about registered transforms.
    
    Returns:
        Statistics dictionary
    """
    registry = get_registry()
    stats = registry.get_stats()
    
    # Add framework breakdown
    framework_counts = {}
    for framework, transforms in registry.framework_transforms.items():
        framework_counts[framework] = len(transforms)
    
    # Add stage breakdown
    stage_counts = {}
    for stage, transforms in registry.transforms_by_stage.items():
        stage_counts[stage] = len(transforms)
    
    stats['framework_breakdown'] = framework_counts
    stats['stage_breakdown'] = stage_counts
    
    return stats


def show_transform_metadata(name: str) -> Optional[Dict[str, Any]]:
    """
    Show detailed metadata for a specific transform.
    
    Args:
        name: Transform name
    
    Returns:
        Metadata dictionary or None if not found
    """
    registry = get_registry()
    
    if name in registry.transforms:
        metadata = registry.get_plugin_metadata(name)
        metadata['class'] = str(registry.transforms[name])
        return metadata
    
    return None


def search_transforms(pattern: str) -> List[str]:
    """
    Search for transforms by name pattern.
    
    Args:
        pattern: Search pattern (case-insensitive substring match)
    
    Returns:
        List of matching transform names
    """
    registry = get_registry()
    pattern_lower = pattern.lower()
    
    matches = []
    for name in registry.transforms.keys():
        if pattern_lower in name.lower():
            matches.append(name)
    
    return sorted(matches)


def print_transforms_summary():
    """Print a formatted summary of all transforms."""
    print("=" * 60)
    print("BRAINSMITH PLUGIN SYSTEM - TRANSFORM REGISTRY")
    print("=" * 60)
    
    # Overall stats
    stats = get_transform_stats()
    print(f"Total Transforms: {stats['transforms']}")
    print(f"Total Kernels: {stats['kernels']}")
    print(f"Total Backends: {stats['backends']}")
    print(f"Total Plugins: {stats['total_plugins']}")
    print()
    
    # Framework breakdown
    print("BY FRAMEWORK:")
    print("-" * 40)
    by_framework = list_transforms_by_framework()
    for framework, transforms in by_framework.items():
        print(f"{framework.upper()}: {len(transforms)} transforms")
        for name in transforms[:5]:  # Show first 5
            print(f"  • {name}")
        if len(transforms) > 5:
            print(f"  ... and {len(transforms) - 5} more")
        print()
    
    # Stage breakdown
    print("BY STAGE:")
    print("-" * 40)
    by_stage = list_transforms_by_stage()
    for stage, transforms in by_stage.items():
        print(f"{stage}: {len(transforms)} transforms")
        for name in transforms[:3]:  # Show first 3
            print(f"  • {name}")
        if len(transforms) > 3:
            print(f"  ... and {len(transforms) - 3} more")
        print()


def print_all_transforms():
    """Print all transforms in a simple list format."""
    print("ALL REGISTERED TRANSFORMS:")
    print("=" * 50)
    
    by_framework = list_transforms_by_framework()
    for framework, transforms in by_framework.items():
        print(f"\n{framework.upper()} ({len(transforms)}):")
        for name in transforms:
            print(f"  {name}")


def print_transform_details(name: str):
    """Print detailed information about a specific transform."""
    metadata = show_transform_metadata(name)
    
    if metadata is None:
        print(f"Transform '{name}' not found.")
        return
    
    print(f"TRANSFORM: {name}")
    print("=" * (len(name) + 12))
    print(f"Framework: {metadata.get('framework', 'unknown')}")
    print(f"Stage: {metadata.get('stage', 'unknown')}")
    print(f"Type: {metadata.get('type', 'unknown')}")
    print(f"Class: {metadata.get('class', 'unknown')}")
    
    if 'description' in metadata:
        print(f"Description: {metadata['description']}")
    
    if 'original_class' in metadata:
        print(f"Original Class: {metadata['original_class']}")
    
    # Show any additional metadata
    extra_keys = set(metadata.keys()) - {'framework', 'stage', 'type', 'class', 'description', 'original_class'}
    if extra_keys:
        print("\nAdditional Metadata:")
        for key in sorted(extra_keys):
            print(f"  {key}: {metadata[key]}")


def main():
    """CLI interface for the debug utilities."""
    if len(sys.argv) == 1:
        print_transforms_summary()
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        print_all_transforms()
    
    elif command == "stats":
        stats = get_transform_stats()
        print("REGISTRY STATISTICS:")
        print("=" * 30)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: {subvalue}")
            else:
                print(f"{key}: {value}")
    
    elif command == "search":
        if len(sys.argv) < 3:
            print("Usage: python -m brainsmith.core.plugins.debug search <pattern>")
            return
        
        pattern = sys.argv[2]
        matches = search_transforms(pattern)
        print(f"TRANSFORMS MATCHING '{pattern}':")
        print("=" * (len(pattern) + 25))
        for match in matches:
            print(f"  {match}")
        print(f"\nFound {len(matches)} matches.")
    
    elif command == "show":
        if len(sys.argv) < 3:
            print("Usage: python -m brainsmith.core.plugins.debug show <transform_name>")
            return
        
        name = sys.argv[2]
        print_transform_details(name)
    
    elif command == "framework":
        if len(sys.argv) < 3:
            frameworks = list_transforms_by_framework()
            print("AVAILABLE FRAMEWORKS:")
            for fw in frameworks.keys():
                print(f"  {fw}")
            return
        
        framework = sys.argv[2]
        transforms = list_all_transforms(framework=framework)
        print(f"TRANSFORMS IN {framework.upper()}:")
        print("=" * (len(framework) + 15))
        for name in transforms:
            print(f"  {name}")
        print(f"\nTotal: {len(transforms)} transforms")
    
    elif command == "stage":
        if len(sys.argv) < 3:
            stages = list_transforms_by_stage()
            print("AVAILABLE STAGES:")
            for stage in stages.keys():
                print(f"  {stage}")
            return
        
        stage = sys.argv[2]
        transforms = list_all_transforms(stage=stage)
        print(f"TRANSFORMS IN STAGE '{stage}':")
        print("=" * (len(stage) + 20))
        for name in transforms:
            print(f"  {name}")
        print(f"\nTotal: {len(transforms)} transforms")
    
    else:
        print("USAGE:")
        print("  python -m brainsmith.core.plugins.debug                    # Summary")
        print("  python -m brainsmith.core.plugins.debug list               # List all")
        print("  python -m brainsmith.core.plugins.debug stats              # Statistics")
        print("  python -m brainsmith.core.plugins.debug search <pattern>   # Search")
        print("  python -m brainsmith.core.plugins.debug show <name>        # Show details")
        print("  python -m brainsmith.core.plugins.debug framework [name]   # By framework")
        print("  python -m brainsmith.core.plugins.debug stage [name]       # By stage")


if __name__ == "__main__":
    main()