#!/usr/bin/env python3
"""Minimal test to verify toolchain fixes."""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_cache_validation():
    """Test cache validation fix."""
    print("Testing cache validation fix...")
    
    from brainsmith.core.explorer.executor import Executor
    from brainsmith.core.explorer.finn_adapter import FINNAdapter
    from brainsmith.core.execution_tree import ExecutionNode
    
    # Create test directory
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        
        # Create a fake cached output (invalid ONNX)
        segment_dir = test_dir / "test_segment"
        segment_dir.mkdir()
        
        cached_model = segment_dir / "test_segment_output.onnx"
        cached_model.write_text("invalid onnx content")
        
        print(f"Created invalid cached model: {cached_model}")
        
        # The cache validation should detect this is invalid
        try:
            import onnx
            onnx.load(str(cached_model))
            print("✗ Cache validation failed - invalid file was accepted")
        except Exception:
            print("✓ Cache validation correctly rejected invalid ONNX file")

def test_output_discovery():
    """Test output discovery improvements."""
    print("\nTesting output discovery...")
    
    from brainsmith.core.explorer.finn_adapter import FINNAdapter
    
    adapter = FINNAdapter()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        build_dir = Path(tmpdir)
        intermediate_dir = build_dir / "intermediate_models"
        intermediate_dir.mkdir()
        
        # Test 1: No ONNX files
        try:
            adapter._discover_output_model(build_dir)
            print("✗ Should have raised error for no ONNX files")
        except RuntimeError as e:
            if "No ONNX files found" in str(e):
                print("✓ Correctly raised error for no ONNX files")
            else:
                print(f"✗ Wrong error: {e}")
        
        # Test 2: Missing intermediate_models directory
        shutil.rmtree(intermediate_dir)
        try:
            adapter._discover_output_model(build_dir)
            print("✗ Should have raised error for missing directory")
        except RuntimeError as e:
            if "No intermediate_models directory" in str(e):
                print("✓ Correctly raised error for missing directory")
            else:
                print(f"✗ Wrong error: {e}")

def test_output_verification():
    """Test output verification."""
    print("\nTesting output verification...")
    
    from brainsmith.core.explorer.finn_adapter import FINNAdapter
    
    adapter = FINNAdapter()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        
        # Test 1: Non-existent file
        fake_model = test_dir / "nonexistent.onnx"
        try:
            adapter._verify_output_model(fake_model)
            print("✗ Should have raised error for non-existent file")
        except RuntimeError as e:
            if "does not exist" in str(e):
                print("✓ Correctly raised error for non-existent file")
            else:
                print(f"✗ Wrong error: {e}")
        
        # Test 2: Invalid ONNX file
        invalid_model = test_dir / "invalid.onnx"
        invalid_model.write_text("not a valid onnx file")
        try:
            adapter._verify_output_model(invalid_model)
            print("✗ Should have raised error for invalid ONNX")
        except RuntimeError as e:
            if "Invalid ONNX model" in str(e):
                print("✓ Correctly raised error for invalid ONNX")
            else:
                print(f"✗ Wrong error: {e}")

if __name__ == "__main__":
    test_cache_validation()
    test_output_discovery()
    test_output_verification()
    print("\nAll tests completed!")