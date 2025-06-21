#!/usr/bin/env python3
"""Debug script to check for differences in step execution between demos."""

import os
import sys

print("Checking for potential state pollution or execution differences...")
print("="*80)

# Check environment variables
print("Environment variables:")
print(f"BSMITH_BUILD_DIR: {os.environ.get('BSMITH_BUILD_DIR', 'NOT SET')}")
print(f"FINN_ROOT: {os.environ.get('FINN_ROOT', 'NOT SET')}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'NOT SET')}")
print()

# Check if running in container
print("Container check:")
if os.path.exists("/.dockerenv"):
    print("✓ Running inside Docker container")
else:
    print("✗ NOT running inside Docker container")
print()

# Key observations
print("Key differences between bert_new and bert_direct:")
print("1. Model generation: Identical (checked)")
print("2. Step sequence: Now identical (19 steps)")
print("3. Parameters: Identical (including standalone_thresholds)")
print("4. Output directory: Fixed to use proper path")
print()

print("Remaining differences to investigate:")
print("1. Reference IO generation (6 min vs cached)")
print("2. Execution context (6-entrypoint vs direct)")
print("3. Possible state pollution between steps")
print("4. Import order or module initialization")
print()

print("The issue MUST be in the 6-entrypoint wrapper since bert_direct works!")
print("Next step: Add debug output to track model state between steps")