#!/usr/bin/env python3
"""Debug registry loading issues."""

import sys
sys.path.insert(0, '/home/tafk/dev/brainsmith-4')

# First, let's see what happens with minimal imports
try:
    from brainsmith.core.plugins.registry import get_registry
    print("✓ Registry imported successfully")
    
    registry = get_registry()
    print(f"✓ Registry instance created: {registry}")
    
    # Check what's registered
    print("\nRegistered plugins:")
    for plugin_type in ['step', 'transform', 'kernel', 'backend']:
        plugins = registry._plugins.get(plugin_type, {})
        print(f"\n{plugin_type}: {len(plugins)} registered")
        if plugins:
            for name in list(plugins.keys())[:5]:  # Show first 5
                print(f"  - {name}")
            if len(plugins) > 5:
                print(f"  ... and {len(plugins) - 5} more")
    
except Exception as e:
    print(f"✗ Error during import: {e}")
    import traceback
    traceback.print_exc()

# Now try importing the full plugin system
print("\n" + "="*50)
print("Attempting full plugin system import...")
try:
    from brainsmith.core.plugins import has_step, list_steps
    print("✓ Plugin system imported successfully")
    
    # List steps
    print("\nAvailable steps:")
    steps = list_steps()
    print(f"Total steps: {len(steps)}")
    for step in sorted(steps)[:10]:
        print(f"  - {step}")
    if len(steps) > 10:
        print(f"  ... and {len(steps) - 10} more")
        
except Exception as e:
    print(f"✗ Error during plugin import: {e}")
    import traceback
    traceback.print_exc()