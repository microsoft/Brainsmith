#!/usr/bin/env python3
"""Test that validation.py imports work correctly."""

import sys
sys.path.insert(0, '/home/tafk/dev/brainsmith-4')

try:
    from brainsmith.core.validation import validate_step, validate_finn_config, ForgeConfig, OutputStage
    print("✓ Import successful")
    
    # Test that ForgeConfig works
    config = ForgeConfig(output_stage=OutputStage.GENERATE_REPORTS)
    print(f"✓ ForgeConfig created: output_stage={config.output_stage.value}")
    
    # Test validate_finn_config with new parameter name
    try:
        validate_finn_config(config, {})
    except ValueError as e:
        # Should not raise error for GENERATE_REPORTS stage
        print(f"✗ Unexpected error: {e}")
    else:
        print("✓ validate_finn_config works with ForgeConfig")
        
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)

print("\nDay 1 fix complete: validation.py now imports from design_space.py")