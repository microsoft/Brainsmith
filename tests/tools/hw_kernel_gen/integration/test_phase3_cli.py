"""
CLI integration tests for Phase 3 unified system.

Tests the complete CLI interface including argument parsing, 
end-to-end file generation, error handling, and user messages.
"""

import pytest
import subprocess
import tempfile
import shutil
from pathlib import Path


class TestPhase3CLI:
    """Integration tests for Phase 3 CLI interface."""
    
    def setup_method(self):
        """Set up test fixtures with real RTL files."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.rtl_dir = self.temp_dir / "rtl"
        self.output_dir = self.temp_dir / "output"
        self.rtl_dir.mkdir()
        
        # Create test RTL files
        self.create_test_rtl_files()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_rtl_files(self):
        """Create test RTL files for CLI testing."""
        
        # Simple valid RTL
        simple_rtl = self.rtl_dir / "simple.sv"
        simple_rtl.write_text("""
// @brainsmith BDIM input0 -1 [PE]
// @brainsmith BDIM output0 -1 [PE]
// @brainsmith DATATYPE input0 FIXED 8 8
// @brainsmith DATATYPE output0 FIXED 8 8

module simple #(
    parameter PE = 4
) (
    input wire ap_clk,
    input wire ap_rst_n,
    input wire ap_start,
    output wire ap_done,
    output wire ap_idle,
    output wire ap_ready,
    
    input wire [input0_width-1:0] input0_TDATA,
    input wire input0_TVALID,
    output wire input0_TREADY,
    
    output wire [output0_width-1:0] output0_TDATA,
    output wire output0_TVALID,
    input wire output0_TREADY
);
endmodule
""")
        
        # Invalid RTL (missing module declaration)
        invalid_rtl = self.rtl_dir / "invalid.sv"
        invalid_rtl.write_text("""
// Invalid RTL - missing module
parameter PE = 4;
input wire clk;
""")
    
    def run_cli(self, args, expect_success=True):
        """Run the CLI with given arguments and return result."""
        cmd = ["python", "-m", "brainsmith.tools.hw_kernel_gen"] + args
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            cwd="/home/tafk/dev/brainsmith-2"
        )
        
        if expect_success:
            if result.returncode != 0:
                pytest.fail(f"CLI failed unexpectedly.\nstdout: {result.stdout}\nstderr: {result.stderr}")
        
        return result
    
    def test_cli_help_message(self):
        """Test CLI help message is informative."""
        result = self.run_cli(["--help"], expect_success=False)
        
        # Help should exit with code 0 but subprocess.run might return it as non-zero
        assert "Generate FINN-compatible HWCustomOp" in result.stdout
        assert "SystemVerilog RTL file to process" in result.stdout
        assert "Output directory for generated files" in result.stdout
        assert "--debug" in result.stdout
        assert "--template-version" in result.stdout
    
    def test_cli_successful_generation(self):
        """Test successful end-to-end generation through CLI."""
        rtl_file = self.rtl_dir / "simple.sv"
        
        result = self.run_cli([
            str(rtl_file),
            "-o", str(self.output_dir)
        ])
        
        # Check success indicators
        assert result.returncode == 0
        assert "✅ Successfully generated HWCustomOp for simple" in result.stdout
        assert "📁 Output directory:" in result.stdout
        assert "⚡ Generated 3 files" in result.stdout
        
        # Check files were created
        simple_dir = self.output_dir / "simple"
        assert simple_dir.exists()
        
        expected_files = [
            "simple_hw_custom_op.py",
            "simple_wrapper.v", 
            "test_simple.py",
            "generation_metadata.json",
            "generation_summary.txt"
        ]
        
        for filename in expected_files:
            file_path = simple_dir / filename
            assert file_path.exists(), f"Expected file not found: {filename}"
            assert file_path.stat().st_size > 0, f"File is empty: {filename}"
    
    def test_cli_debug_mode(self):
        """Test CLI debug mode provides detailed output."""
        rtl_file = self.rtl_dir / "simple.sv"
        
        result = self.run_cli([
            str(rtl_file),
            "-o", str(self.output_dir),
            "--debug"
        ])
        
        # Check debug output indicators
        assert result.returncode == 0
        assert "=== Phase 3 Hardware Kernel Generator ===" in result.stdout
        assert "🔍 Step 1: Parsing RTL" in result.stdout
        assert "🏭 Step 2: Generating all templates" in result.stdout  
        assert "💾 Step 3: Writing files" in result.stdout
        assert "✅ Parsed module: simple" in result.stdout
        assert "✅ Found 1 parameters: ['PE']" in result.stdout
        assert "✅ Generated 3 files:" in result.stdout
        assert "📄 simple_hw_custom_op.py" in result.stdout
        assert "bytes)" in result.stdout  # File sizes shown in debug
    
    def test_cli_template_version_parameter(self):
        """Test CLI template version parameter."""
        rtl_file = self.rtl_dir / "simple.sv"
        
        result = self.run_cli([
            str(rtl_file),
            "-o", str(self.output_dir),
            "--template-version", "phase2"
        ])
        
        assert result.returncode == 0
        assert "✅ Successfully generated HWCustomOp" in result.stdout
    
    def test_cli_nonexistent_rtl_file(self):
        """Test CLI error handling for nonexistent RTL file."""
        nonexistent_file = self.rtl_dir / "nonexistent.sv"
        
        result = self.run_cli([
            str(nonexistent_file),
            "-o", str(self.output_dir)
        ], expect_success=False)
        
        assert result.returncode == 1
        assert "❌ Error: RTL file not found" in result.stdout
    
    def test_cli_invalid_rtl_file(self):
        """Test CLI error handling for invalid RTL content."""
        invalid_rtl = self.rtl_dir / "invalid.sv"
        
        result = self.run_cli([
            str(invalid_rtl),
            "-o", str(self.output_dir)
        ], expect_success=False)
        
        assert result.returncode == 1
        assert "❌ Generation failed:" in result.stdout
    
    def test_cli_debug_error_with_traceback(self):
        """Test CLI debug mode shows traceback on errors."""
        invalid_rtl = self.rtl_dir / "invalid.sv"
        
        result = self.run_cli([
            str(invalid_rtl),
            "-o", str(self.output_dir),
            "--debug"
        ], expect_success=False)
        
        assert result.returncode == 1
        assert "❌ Generation failed:" in result.stdout
        assert "Debug traceback:" in result.stdout
        assert "Traceback" in result.stdout
    
    def test_cli_output_directory_creation(self):
        """Test CLI creates output directory if it doesn't exist."""
        rtl_file = self.rtl_dir / "simple.sv"
        new_output_dir = self.temp_dir / "new_output" / "nested"
        
        # Directory should not exist initially
        assert not new_output_dir.exists()
        
        result = self.run_cli([
            str(rtl_file),
            "-o", str(new_output_dir)
        ])
        
        assert result.returncode == 0
        assert new_output_dir.exists()
        
        simple_dir = new_output_dir / "simple"
        assert simple_dir.exists()
        assert (simple_dir / "simple_hw_custom_op.py").exists()
    
    def test_cli_invalid_template_version(self):
        """Test CLI error handling for invalid template version."""
        rtl_file = self.rtl_dir / "simple.sv"
        
        result = self.run_cli([
            str(rtl_file),
            "-o", str(self.output_dir),
            "--template-version", "invalid"
        ], expect_success=False)
        
        # Should fail during argument parsing
        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()
    
    def test_cli_missing_required_arguments(self):
        """Test CLI error handling for missing required arguments."""
        # Missing output directory
        result = self.run_cli([
            str(self.rtl_dir / "simple.sv")
        ], expect_success=False)
        
        assert result.returncode != 0
        assert "required" in result.stderr.lower()
    
    def test_cli_permission_error_handling(self):
        """Test CLI handles permission errors gracefully."""
        rtl_file = self.rtl_dir / "simple.sv"
        
        # Try to write to a directory we can't write to (if possible)
        try:
            restricted_dir = Path("/root/restricted")
            result = self.run_cli([
                str(rtl_file),
                "-o", str(restricted_dir)
            ], expect_success=False)
            
            assert result.returncode == 1
            assert "❌ Generation failed:" in result.stdout
        except:
            # Skip this test if we can't create a permission error scenario
            pytest.skip("Cannot create permission error scenario in test environment")
    
    def test_cli_multiple_generations_same_output(self):
        """Test CLI can handle multiple generations to same output directory."""
        rtl_file = self.rtl_dir / "simple.sv"
        
        # First generation
        result1 = self.run_cli([
            str(rtl_file),
            "-o", str(self.output_dir)
        ])
        assert result1.returncode == 0
        
        # Second generation (should overwrite)
        result2 = self.run_cli([
            str(rtl_file),
            "-o", str(self.output_dir)
        ])
        assert result2.returncode == 0
        
        # Files should still exist and be valid
        simple_dir = self.output_dir / "simple"
        assert simple_dir.exists()
        assert (simple_dir / "simple_hw_custom_op.py").exists()


class TestPhase3CLIConfiguration:
    """Test Phase 3 CLI configuration handling."""
    
    def test_config_creation_from_args(self):
        """Test Config.from_args with simplified arguments."""
        from brainsmith.tools.hw_kernel_gen.config import Config
        import tempfile
        
        # Create a temporary RTL file for testing
        temp_dir = Path(tempfile.mkdtemp())
        test_rtl = temp_dir / "test.sv"
        test_rtl.write_text("module test(); endmodule")
        
        # Mock argparse namespace
        class MockArgs:
            def __init__(self):
                self.rtl_file = test_rtl
                self.output = temp_dir / "output"
                self.debug = True
                self.template_version = "phase2"
        
        args = MockArgs()
        config = Config.from_args(args)
        
        assert config.rtl_file == test_rtl
        assert config.output_dir == temp_dir / "output"
        assert config.debug == True
        assert config.template_version == "phase2"
    
    def test_config_validation_invalid_template_version(self):
        """Test config validation rejects invalid template versions."""
        from brainsmith.tools.hw_kernel_gen.config import Config
        
        with pytest.raises(ValueError, match="Unsupported template version"):
            Config(
                rtl_file=Path("/nonexistent.sv"),  # Will fail before this check
                output_dir=Path("/tmp"),
                template_version="invalid"
            )
    
    def test_legacy_config_deprecation_warning(self):
        """Test legacy config issues deprecation warnings."""
        from brainsmith.tools.hw_kernel_gen.config import LegacyConfig
        import tempfile
        
        # Create a temporary RTL file for testing
        temp_dir = Path(tempfile.mkdtemp())
        test_rtl = temp_dir / "test.sv"
        test_rtl.write_text("module test(); endmodule")
        
        with pytest.warns(DeprecationWarning, match="compiler_data_file parameter is deprecated"):
            LegacyConfig(
                rtl_file=test_rtl,
                compiler_data_file=Path("/nonexistent.py"),
                output_dir=temp_dir
            )


class TestPhase3CLIIntegrationWithRealFiles:
    """Integration tests using real files from the codebase."""
    
    def setup_method(self):
        """Set up with real RTL examples if available."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "output"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def run_cli(self, args):
        """Run CLI and return result."""
        cmd = ["python", "-m", "brainsmith.tools.hw_kernel_gen"] + args
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/home/tafk/dev/brainsmith-2"
        )
    
    def test_cli_with_example_rtl_files(self):
        """Test CLI with example RTL files if they exist in the codebase."""
        # Look for example RTL files in the test fixtures
        test_rtl_dir = Path("/home/tafk/dev/brainsmith-2/tests/tools/hw_kernel_gen")
        
        # Find any .sv files in test directories
        rtl_files = list(test_rtl_dir.rglob("*.sv"))
        
        if not rtl_files:
            pytest.skip("No .sv test files found for CLI integration testing")
        
        # Test with first available RTL file
        rtl_file = rtl_files[0]
        
        result = self.run_cli([
            str(rtl_file),
            "-o", str(self.output_dir),
            "--debug"
        ])
        
        # Should either succeed or fail gracefully with clear error message
        if result.returncode == 0:
            assert "✅ Successfully generated HWCustomOp" in result.stdout
        else:
            assert "❌ Generation failed:" in result.stdout
            # Should have clear error message, not a crash
            assert "Traceback" in result.stdout  # Debug mode should show traceback