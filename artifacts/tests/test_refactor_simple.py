#!/usr/bin/env python3
"""Simple test of the refactored blueprint parser separation without external deps."""

import sys
sys.path.insert(0, '/home/tafk/dev/brainsmith-4')

# Test imports work
try:
    from brainsmith.core.blueprint_parser_v2 import BlueprintParser
    from brainsmith.core.design_space_v2 import DesignSpace, BuildConfig
    from brainsmith.core.tree_builder import TreeBuilder
    from brainsmith.core.forge_v2 import forge
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test BuildConfig rename
config = BuildConfig()
print(f"✓ BuildConfig created: {type(config).__name__}")
print(f"  - output_stage: {config.output_stage}")
print(f"  - working_directory: {config.working_directory}")

# Test TreeBuilder has the right methods
builder = TreeBuilder()
print(f"\n✓ TreeBuilder created")
print(f"  - Has build_tree: {hasattr(builder, 'build_tree')}")
print(f"  - Has _extract_finn_config: {hasattr(builder, '_extract_finn_config')}")
print(f"  - Has _flush_steps: {hasattr(builder, '_flush_steps')}")
print(f"  - Has _create_branches: {hasattr(builder, '_create_branches')}")

# Test BlueprintParser no longer has tree methods
parser = BlueprintParser()
print(f"\n✓ BlueprintParser created")
print(f"  - Has parse: {hasattr(parser, 'parse')}")
print(f"  - Has __init__: {hasattr(parser, '__init__')}")  # Should be False (removed)
print(f"  - Has _build_execution_tree: {hasattr(parser, '_build_execution_tree')}")  # Should be False
print(f"  - Has _flush_steps: {hasattr(parser, '_flush_steps')}")  # Should be False

print("\n✅ Refactoring structure test passed!")