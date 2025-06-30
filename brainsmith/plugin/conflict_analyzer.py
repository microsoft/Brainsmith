"""
Conflict Analysis Tool

Analyzes and reports transform naming conflicts to help users understand
when explicit prefixes are required.
"""

import logging
from typing import Dict, List, Any
from brainsmith.plugin.core import get_registry

logger = logging.getLogger(__name__)


def analyze_naming_conflicts(plugin_type: str = "transform") -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze naming conflicts for a plugin type.
    
    Args:
        plugin_type: Type of plugin to analyze (default: "transform")
        
    Returns:
        Dict mapping conflicted names to lists of conflicting entries
    """
    registry = get_registry()
    return registry.analyze_conflicts(plugin_type)


def print_conflict_report(plugin_type: str = "transform", show_all: bool = False):
    """
    Print a user-friendly conflict report.
    
    Args:
        plugin_type: Type of plugin to analyze
        show_all: If True, show all plugins; if False, show only conflicts
    """
    registry = get_registry()
    conflicts = analyze_naming_conflicts(plugin_type)
    
    print(f"\n🔍 Transform Naming Analysis")
    print("=" * 50)
    
    if not conflicts:
        print("✅ No naming conflicts detected!")
        print("   All transform names are unique across frameworks.")
        
        if show_all:
            _print_all_transforms(registry, plugin_type)
        return
    
    print(f"⚠️  Found {len(conflicts)} naming conflicts:")
    print()
    
    for name, transforms in sorted(conflicts.items()):
        frameworks = [t.get("framework", "unknown") for t in transforms]
        print(f"📛 '{name}' conflicts:")
        
        for transform in transforms:
            framework = transform.get("framework", "unknown")
            full_name = transform.get("name", "")
            description = transform.get("description", "")[:60]
            if len(transform.get("description", "")) > 60:
                description += "..."
            
            print(f"   • {framework}: {full_name}")
            if description:
                print(f"     {description}")
        
        print(f"   💡 Use explicit prefixes: {', '.join(f'{fw}:{name}' for fw in frameworks)}")
        print()
    
    # Summary statistics
    total_transforms = len(registry.query(type=plugin_type))
    unique_transforms = total_transforms - sum(len(transforms) - 1 for transforms in conflicts.values())
    
    print("📊 Summary:")
    print(f"   • Total transforms: {total_transforms}")
    print(f"   • Unique names: {unique_transforms}")
    print(f"   • Conflicted names: {len(conflicts)}")
    print(f"   • Conflicts affect: {sum(len(transforms) for transforms in conflicts.values())} transforms")
    
    if show_all:
        print()
        _print_all_transforms(registry, plugin_type)


def _print_all_transforms(registry, plugin_type: str):
    """Print all transforms grouped by framework."""
    all_transforms = registry.query(type=plugin_type)
    
    # Group by framework
    by_framework = {}
    for t in all_transforms:
        framework = t.get("framework", "unknown")
        if framework not in by_framework:
            by_framework[framework] = []
        by_framework[framework].append(t)
    
    print("📚 All Transforms by Framework:")
    for framework, transforms in sorted(by_framework.items()):
        print(f"\n🔧 {framework.title()} ({len(transforms)} transforms):")
        
        # Group by stage for better organization
        by_stage = {}
        for t in transforms:
            stage = t.get("stage", "general")
            if stage not in by_stage:
                by_stage[stage] = []
            by_stage[stage].append(t)
        
        for stage, stage_transforms in sorted(by_stage.items()):
            print(f"   📁 {stage} ({len(stage_transforms)}):")
            for t in sorted(stage_transforms, key=lambda x: x.get("name", "")):
                name = t.get("name", "")
                if ":" in name:
                    display_name = name.split(":", 1)[1]  # Show unprefixed for readability
                else:
                    display_name = name
                print(f"      • {display_name}")


def suggest_unique_names(plugin_type: str = "transform") -> List[str]:
    """
    Suggest transforms that can be used without prefixes.
    
    Args:
        plugin_type: Type of plugin to analyze
        
    Returns:
        List of transform names that are unique across frameworks
    """
    registry = get_registry()
    conflicts = analyze_naming_conflicts(plugin_type)
    all_transforms = registry.query(type=plugin_type)
    
    unique_names = []
    
    for transform in all_transforms:
        name = transform.get("name", "")
        if ":" in name:
            unprefixed = name.split(":", 1)[1]
        else:
            unprefixed = name
        
        # Only include if not in conflicts and not already added
        if unprefixed not in conflicts and unprefixed not in unique_names:
            unique_names.append(unprefixed)
    
    return sorted(unique_names)


def print_usage_examples():
    """Print examples of how to use transforms with and without prefixes."""
    registry = get_registry()
    conflicts = analyze_naming_conflicts("transform")
    unique_names = suggest_unique_names("transform")
    
    print("\n💡 Usage Examples")
    print("=" * 50)
    
    print("✅ Transforms you can use WITHOUT prefixes:")
    if unique_names:
        for name in unique_names[:5]:  # Show first 5
            print(f'   transforms=["...{name}..."]')
        if len(unique_names) > 5:
            print(f"   ... and {len(unique_names) - 5} more unique transforms")
    else:
        print("   (None found - all transforms have conflicts)")
    
    print()
    print("⚠️  Transforms that REQUIRE prefixes:")
    if conflicts:
        for name, transforms in list(conflicts.items())[:3]:  # Show first 3
            frameworks = [t.get("framework", "unknown") for t in transforms]
            print(f'   # "{name}" exists in: {", ".join(frameworks)}')
            for fw in frameworks:
                print(f'   transforms=["...{fw}:{name}..."]  # Use {fw} version')
        
        if len(conflicts) > 3:
            print(f"   ... and {len(conflicts) - 3} more conflicted names")
    else:
        print("   (None found - all transforms are unique)")
    
    print()
    print("📝 Step Example:")
    print("@finn_step(")
    print("    transforms=[")
    
    # Mix unique and conflicted examples
    if unique_names:
        print(f'        "{unique_names[0]}",  # Unique - no prefix needed')
    if conflicts:
        conflict_name, conflict_transforms = list(conflicts.items())[0]
        fw = conflict_transforms[0].get("framework", "unknown")
        print(f'        "{fw}:{conflict_name}",  # Conflict - prefix required')
    
    print("    ]")
    print(")")
    print("def my_step(model, cfg, transforms):")
    print("    # All transforms resolved automatically")
    print("    pass")


if __name__ == "__main__":
    # Run conflict analysis when called directly
    from brainsmith.plugin.discovery import discover_plugins
    
    print("Discovering plugins...")
    count = discover_plugins()
    print(f"Discovered {count} plugins\n")
    
    # Print comprehensive report
    print_conflict_report(show_all=False)
    print_usage_examples()