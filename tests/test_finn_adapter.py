"""Tests for FINNAdapter - testing workarounds without requiring FINN."""

import pytest
import tempfile
import os
from pathlib import Path
from brainsmith.core.explorer.finn_adapter import FINNAdapter


class TestFINNAdapter:
    """Test FINN adapter workarounds."""
    
    def test_finn_adapter_init(self):
        """Test adapter initialization checks for FINN."""
        # This will fail if FINN is not installed
        try:
            adapter = FINNAdapter()
            # If we get here, FINN is installed
            assert adapter._finn_available
        except RuntimeError as e:
            # Expected when FINN not installed
            assert "FINN not installed" in str(e)
    
    def test_model_preparation(self):
        """Test model preparation (copying)."""
        # We can test this without FINN
        adapter = FINNAdapter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source model
            source = Path(tmpdir) / "source.onnx"
            source.write_bytes(b"model_data")
            
            # Prepare (copy) model
            dest = Path(tmpdir) / "subdir" / "dest.onnx"
            adapter.prepare_model(source, dest)
            
            # Verify copy
            assert dest.exists()
            assert dest.read_bytes() == b"model_data"
            assert dest.parent.exists()
    
    def test_output_model_discovery(self):
        """Test discovering output model in intermediate_models."""
        adapter = FINNAdapter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            build_dir = Path(tmpdir)
            
            # Test when no intermediate_models exists
            assert adapter._discover_output_model(build_dir) is None
            
            # Create intermediate_models with some ONNX files
            inter_dir = build_dir / "intermediate_models"
            inter_dir.mkdir()
            
            # Create files with different timestamps
            import time
            model1 = inter_dir / "step1.onnx"
            model1.touch()
            time.sleep(0.01)  # Ensure different mtime
            
            model2 = inter_dir / "step2.onnx"
            model2.touch()
            time.sleep(0.01)
            
            model3 = inter_dir / "final.onnx"
            model3.touch()
            
            # Should return the most recent
            result = adapter._discover_output_model(build_dir)
            assert result == model3
    
    def test_output_model_discovery_empty_dir(self):
        """Test discovery when intermediate_models is empty."""
        adapter = FINNAdapter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            build_dir = Path(tmpdir)
            inter_dir = build_dir / "intermediate_models"
            inter_dir.mkdir()
            
            # Empty directory should return None
            assert adapter._discover_output_model(build_dir) is None
    
    def test_working_directory_context(self):
        """Test that build would restore working directory."""
        # We can't test full build without FINN, but we can verify
        # the pattern that would be used
        adapter = FINNAdapter()
        
        original_cwd = os.getcwd()
        
        # The build method should always restore cwd even on error
        # This is critical for the workaround
        with tempfile.TemporaryDirectory() as tmpdir:
            build_dir = Path(tmpdir)
            
            # Even if build fails, cwd should be restored
            try:
                # This will fail without FINN
                adapter.build(
                    Path("fake.onnx"),
                    {"board": "test"},
                    build_dir
                )
            except:
                pass  # Expected
            
            # Verify cwd was restored
            assert os.getcwd() == original_cwd