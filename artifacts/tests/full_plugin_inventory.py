#!/usr/bin/env python3
"""Generate a complete inventory of all plugins registered in BrainSmith."""

from collections import defaultdict
from brainsmith.core.plugins.registry import (
    get_registry, list_kernels, list_backends, list_transforms, list_steps
)

def analyze_plugins():
    """Analyze all registered plugins by type and framework."""
    registry = get_registry()
    
    # Initialize counters
    stats = {
        'transform': defaultdict(int),
        'kernel': defaultdict(int),
        'backend': defaultdict(int),
        'step': defaultdict(int)
    }
    
    # Detailed lists
    details = {
        'transform': defaultdict(list),
        'kernel': defaultdict(list),
        'backend': defaultdict(list),
        'step': defaultdict(list)
    }
    
    # Analyze each plugin type
    for plugin_type in ['transform', 'kernel', 'backend', 'step']:
        for name, (cls, metadata) in registry._plugins[plugin_type].items():
            framework = metadata.get('framework', 'brainsmith')
            
            # Clean name (remove framework prefix if present)
            clean_name = name.split(':')[-1] if ':' in name else name
            
            stats[plugin_type][framework] += 1
            details[plugin_type][framework].append(clean_name)
    
    return stats, details

def print_inventory(stats, details):
    """Print the full inventory in a structured format."""
    print("=" * 80)
    print("BRAINSMITH PLUGIN INVENTORY")
    print("=" * 80)
    
    # Overall summary
    total_plugins = sum(sum(counts.values()) for counts in stats.values())
    print(f"\nTOTAL PLUGINS: {total_plugins}")
    
    # By type summary
    print("\nBY TYPE:")
    for plugin_type in ['transform', 'kernel', 'backend', 'step']:
        total = sum(stats[plugin_type].values())
        print(f"  - {plugin_type.capitalize()}s: {total}")
    
    # By framework summary
    print("\nBY FRAMEWORK:")
    framework_totals = defaultdict(int)
    for plugin_type in stats:
        for framework, count in stats[plugin_type].items():
            framework_totals[framework] += count
    
    for framework in sorted(framework_totals.keys()):
        print(f"  - {framework}: {framework_totals[framework]}")
    
    # Detailed breakdown
    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN")
    print("=" * 80)
    
    for plugin_type in ['transform', 'kernel', 'backend', 'step']:
        print(f"\n{plugin_type.upper()}S:")
        print("-" * 40)
        
        type_stats = stats[plugin_type]
        type_details = details[plugin_type]
        
        if not type_stats:
            print("  None registered")
            continue
        
        for framework in sorted(type_stats.keys()):
            count = type_stats[framework]
            items = sorted(type_details[framework])
            
            print(f"\n  {framework.upper()} ({count}):")
            
            # For backends, show additional info
            if plugin_type == 'backend':
                registry = get_registry()
                backend_info = defaultdict(list)
                
                for item in items:
                    # Get metadata for this backend
                    for reg_name, (cls, metadata) in registry._plugins['backend'].items():
                        if reg_name.endswith(item) or reg_name == item:
                            kernel = metadata.get('kernel', 'Unknown')
                            language = metadata.get('language', 'Unknown')
                            backend_info[f"{language.upper()}"].append(f"{item} (for {kernel})")
                            break
                
                for lang in sorted(backend_info.keys()):
                    print(f"    {lang}:")
                    for backend in sorted(backend_info[lang]):
                        print(f"      - {backend}")
            else:
                # Regular listing for non-backends
                for item in items:
                    print(f"    - {item}")
    
    # Special sections
    print("\n" + "=" * 80)
    print("SPECIAL CATEGORIES")
    print("=" * 80)
    
    # Kernel inference transforms
    registry = get_registry()
    kernel_inferences = []
    for name, (cls, metadata) in registry._plugins['transform'].items():
        if metadata.get('kernel_inference'):
            framework = metadata.get('framework', 'brainsmith')
            clean_name = name.split(':')[-1] if ':' in name else name
            kernel_inferences.append((framework, clean_name))
    
    print(f"\nKERNEL INFERENCE TRANSFORMS ({len(kernel_inferences)}):")
    by_framework = defaultdict(list)
    for fw, name in kernel_inferences:
        by_framework[fw].append(name)
    
    for fw in sorted(by_framework.keys()):
        print(f"  {fw.upper()} ({len(by_framework[fw])}):")
        for name in sorted(by_framework[fw]):
            print(f"    - {name}")

def main():
    # Ensure all plugins are loaded
    from brainsmith.core.plugins.framework_adapters import ensure_initialized
    ensure_initialized()
    
    # Import BrainSmith kernels to ensure they're registered
    try:
        import brainsmith.kernels
    except:
        pass
    
    stats, details = analyze_plugins()
    print_inventory(stats, details)
    
    # Additional statistics
    print("\n" + "=" * 80)
    print("COVERAGE STATISTICS")
    print("=" * 80)
    
    print("\nFRAMEWORK COVERAGE:")
    print(f"  - QONNX Transforms: {stats['transform']['qonnx']}")
    print(f"  - FINN Transforms: {stats['transform']['finn']}")
    print(f"  - FINN Kernels: {stats['kernel']['finn']}")
    print(f"  - FINN Backends: {stats['backend']['finn']}")
    print(f"  - BrainSmith Kernels: {stats['kernel']['brainsmith']}")
    print(f"  - BrainSmith Backends: {stats['backend']['brainsmith']}")
    print(f"  - BrainSmith Transforms: {stats['transform']['brainsmith']}")
    print(f"  - BrainSmith Steps: {stats['step']['brainsmith']}")

if __name__ == "__main__":
    main()