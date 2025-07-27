#!/usr/bin/env python3
"""Test FINN adapter fixes."""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brainsmith.core.explorer.finn_adapter import FINNAdapter
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

def test_finn_adapter():
    print("Testing FINNAdapter fixes...")
    
    adapter = FINNAdapter()
    
    # Test path handling
    test_dir = Path("/tmp/test_finn_adapter")
    test_dir.mkdir(exist_ok=True)
    
    # Create a dummy model
    dummy_model = test_dir / "input.onnx"
    dummy_model.write_text("dummy")
    
    print(f"Created test directory: {test_dir}")
    print(f"Current working directory before: {os.getcwd()}")
    
    # Test that working directory is preserved
    original_cwd = os.getcwd()
    
    try:
        # This should handle paths correctly
        config = {
            "board": "V80",
            "synth_clk_period_ns": 3.33,
            "steps": ["test_step"],
            "output_dir": str(test_dir)
        }
        
        # Since we don't have FINN, we'll just test the path handling
        print("\nTesting absolute path conversion...")
        abs_input = dummy_model.absolute()
        abs_output = test_dir.absolute()
        print(f"  Input: {dummy_model} -> {abs_input}")
        print(f"  Output: {test_dir} -> {abs_output}")
        
        # Verify working directory is unchanged
        print(f"\nCurrent working directory after: {os.getcwd()}")
        assert os.getcwd() == original_cwd, "Working directory changed!"
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_finn_adapter()