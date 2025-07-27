#!/usr/bin/env python3
"""Test that circular import is resolved."""

import sys
sys.path.insert(0, '/home/tafk/dev/brainsmith-4')

print("Testing circular import resolution...")

try:
    # Test direct imports that should work
    print("1. Testing forge import...")
    from brainsmith.core import forge
    print("✓ forge imported successfully")
    
    print("\n2. Testing explorer import...")
    from brainsmith.core import explorer
    print("✓ explorer imported successfully")
    
    print("\n3. Testing interfaces import...")
    from brainsmith.core.interfaces import run_exploration
    print("✓ interfaces imported successfully")
    
    # Test that forge can be imported from steps (was part of circular chain)
    print("\n4. Testing import chain...")
    from brainsmith.steps import core_steps
    print("✓ core_steps imported successfully")
    
    # Verify forge function exists
    print("\n5. Testing forge function...")
    from brainsmith.core.forge import forge as forge_func
    print("✓ forge function exists")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nDay 2-3 fix complete: Circular imports resolved using interfaces.py")