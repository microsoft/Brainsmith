"""
CLI Tests - Command Line Interface

Tests the BrainSmith CLI with minimal mocking approach.
Validates CLI commands, argument parsing, and user feedback.
"""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, Mock
import tempfile
from pathlib import Path

# Import CLI components
try:
    from brainsmith.core.cli import brainsmith, forge, validate, run
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False


@pytest.mark.core
class TestCLIBasicCommands:
    """Test basic CLI command functionality."""
    
    def setup_method(self):
        """Set up CLI test runner."""
        self.runner = CliRunner()
    
    def test_cli_help_command(self):
        """Test CLI help command."""
        if not CLI_AVAILABLE:
            pytest.skip("CLI not available")
        
        result = self.runner.invoke(brainsmith, ['--help'])
        
        assert result.exit_code == 0
        assert 'BrainSmith' in result.output
        assert 'forge' in result.output or 'forge-cmd' in result.output
        assert 'validate' in result.output
    
    def test_cli_version_command(self):
        """Test CLI version command."""
        if not CLI_AVAILABLE:
            pytest.skip("CLI not available")
        
        result = self.runner.invoke(brainsmith, ['--version'])
        
        assert result.exit_code == 0
        assert '0.5.0' in result.output or 'version' in result.output.lower()
    
    def test_forge_help(self):
        """Test forge command help."""
        if not CLI_AVAILABLE:
            pytest.skip("CLI not available")
        
        result = self.runner.invoke(brainsmith, ['forge', '--help'])
        
        assert result.exit_code == 0
        assert 'Generate FPGA accelerator' in result.output
        assert 'model_path' in result.output.lower()
        assert 'blueprint_path' in result.output.lower()
    
    def test_validate_cmd_help(self):
        """Test validate command help."""
        if not CLI_AVAILABLE:
            pytest.skip("CLI not available")
        
        result = self.runner.invoke(brainsmith, ['validate', '--help'])
        
        assert result.exit_code == 0
        assert 'Validate blueprint' in result.output
        assert 'blueprint_path' in result.output.lower()


@pytest.mark.core
class TestForgeCommand:
    """Test forge CLI command functionality."""
    
    def setup_method(self):
        """Set up CLI test runner."""
        self.runner = CliRunner()
    
    def test_forge_missing_args(self):
        """Test forge command with missing arguments."""
        if not CLI_AVAILABLE:
            pytest.skip("CLI not available")
        
        # Test with no arguments
        result = self.runner.invoke(brainsmith, ['forge'])
        assert result.exit_code != 0
        assert 'Missing argument' in result.output or 'Usage:' in result.output
        
        # Test with only model path
        result = self.runner.invoke(brainsmith, ['forge', 'model.onnx'])
        assert result.exit_code != 0
        assert 'Missing argument' in result.output or 'Usage:' in result.output
    
    def test_forge_nonexistent_files(self):
        """Test forge command with non-existent files."""
        if not CLI_AVAILABLE:
            pytest.skip("CLI not available")
        
        result = self.runner.invoke(brainsmith, [
            'forge', 
            'nonexistent_model.onnx', 
            'nonexistent_blueprint.yaml'
        ])
        
        # Should fail due to file not found
        assert result.exit_code != 0
        assert 'does not exist' in result.output or 'No such file' in result.output
    
    @patch('brainsmith.core.cli.forge')
    def test_forge_successful_execution(self, mock_forge, sample_model_path, sample_blueprint_path):
        """Test successful forge command execution."""
        if not CLI_AVAILABLE:
            pytest.skip("CLI not available")
        
        # Mock successful forge result
        mock_forge.return_value = {
            'success': True,
            'dataflow_graph': {'onnx_model': Mock()},
            'metrics': {'performance': {'throughput_ops_sec': 500.0}}
        }
        
        result = self.runner.invoke(brainsmith, [
            'forge',
            sample_model_path,
            sample_blueprint_path
        ])
        
        assert result.exit_code == 0
        assert '🔨 Forging accelerator' in result.output
        assert '✅ Forge completed successfully' in result.output
        assert mock_forge.called
        
        # Verify forge was called with correct arguments
        call_args = mock_forge.call_args
        assert call_args[0][0] == sample_model_path
        assert call_args[0][1] == sample_blueprint_path
    
    @patch('brainsmith.core.cli.forge')
    def test_forge_with_output_directory(self, mock_forge, sample_model_path, sample_blueprint_path, temp_test_dir):
        """Test forge command with output directory."""
        if not CLI_AVAILABLE:
            pytest.skip("CLI not available")
        
        mock_forge.return_value = {'success': True}
        output_dir = str(Path(temp_test_dir) / "test_output")
        
        result = self.runner.invoke(brainsmith, [
            'forge',
            sample_model_path,
            sample_blueprint_path,
            '--output', output_dir
        ])
        
        assert result.exit_code == 0
        assert '📁 Results saved to:' in result.output
        
        # Verify output_dir was passed to forge
        call_kwargs = mock_forge.call_args[1]
        assert call_kwargs['output_dir'] == output_dir
    
    @patch('brainsmith.core.cli.forge')
    def test_forge_error_handling(self, mock_forge, sample_model_path, sample_blueprint_path):
        """Test forge command error handling."""
        if not CLI_AVAILABLE:
            pytest.skip("CLI not available")
        
        # Mock forge to raise exception
        mock_forge.side_effect = Exception("Test error message")
        
        result = self.runner.invoke(brainsmith, [
            'forge',
            sample_model_path,
            sample_blueprint_path
        ])
        
        assert result.exit_code == 1
        assert '❌ Forge failed:' in result.output
        assert 'Test error message' in result.output
    
    @patch('brainsmith.core.cli.forge')
    def test_forge_warning_status(self, mock_forge, sample_model_path, sample_blueprint_path):
        """Test forge command with warning status."""
        if not CLI_AVAILABLE:
            pytest.skip("CLI not available")
        
        # Mock forge with warnings
        mock_forge.return_value = {
            'success': False,  # Completed but with warnings
            'warnings': ['Test warning']
        }
        
        result = self.runner.invoke(brainsmith, [
            'forge',
            sample_model_path,
            sample_blueprint_path
        ])
        
        assert result.exit_code == 0  # Should still succeed
        assert 'Status: Completed with warnings' in result.output


@pytest.mark.core
class TestValidateCommand:
    """Test validate CLI command functionality."""
    
    def setup_method(self):
        """Set up CLI test runner."""
        self.runner = CliRunner()
    
    def test_validate_cmd_missing_args(self):
        """Test validate command with missing arguments."""
        if not CLI_AVAILABLE:
            pytest.skip("CLI not available")
        
        result = self.runner.invoke(brainsmith, ['validate'])
        assert result.exit_code != 0
        assert 'Missing argument' in result.output or 'Usage:' in result.output
    
    def test_validate_cmd_nonexistent_file(self):
        """Test validate command with non-existent file."""
        if not CLI_AVAILABLE:
            pytest.skip("CLI not available")
        
        result = self.runner.invoke(brainsmith, ['validate', 'nonexistent_blueprint.yaml'])
        assert result.exit_code != 0
        assert 'does not exist' in result.output or 'No such file' in result.output
    
    @patch('brainsmith.core.cli.validate_blueprint')
    def test_validate_cmd_valid_blueprint(self, mock_validate, sample_blueprint_path):
        """Test validate command with valid blueprint."""
        if not CLI_AVAILABLE:
            pytest.skip("CLI not available")
        
        # Mock successful validation
        mock_validate.return_value = (True, [])
        
        result = self.runner.invoke(brainsmith, ['validate', sample_blueprint_path])
        
        assert result.exit_code == 0
        assert '🔍 Validating blueprint:' in result.output
        assert '✅ Blueprint is valid' in result.output
        
        mock_validate.assert_called_once_with(sample_blueprint_path)
    
    @patch('brainsmith.core.cli.validate_blueprint')
    def test_validate_cmd_invalid_blueprint(self, mock_validate, sample_blueprint_path):
        """Test validate command with invalid blueprint."""
        if not CLI_AVAILABLE:
            pytest.skip("CLI not available")
        
        # Mock validation failure
        mock_validate.return_value = (False, ['Error 1', 'Error 2'])
        
        result = self.runner.invoke(brainsmith, ['validate', sample_blueprint_path])
        
        assert result.exit_code == 1
        assert '❌ Blueprint validation failed:' in result.output
        assert '• Error 1' in result.output
        assert '• Error 2' in result.output
    
    @patch('brainsmith.core.cli.validate_blueprint')
    def test_validate_cmd_exception_handling(self, mock_validate, sample_blueprint_path):
        """Test validate command exception handling."""
        if not CLI_AVAILABLE:
            pytest.skip("CLI not available")
        
        # Mock validation to raise exception
        mock_validate.side_effect = Exception("Validation error")
        
        result = self.runner.invoke(brainsmith, ['validate', sample_blueprint_path])
        
        assert result.exit_code == 1
        assert '❌ Validation failed:' in result.output
        assert 'Validation error' in result.output


@pytest.mark.core
class TestRunCommand:
    """Test run command (alias for forge)."""
    
    def setup_method(self):
        """Set up CLI test runner."""
        self.runner = CliRunner()
    
    @patch('brainsmith.core.cli.forge')
    def test_run_cmd_alias(self, mock_forge, sample_model_path, sample_blueprint_path):
        """Test that run command works as alias for forge."""
        if not CLI_AVAILABLE:
            pytest.skip("CLI not available")
        
        mock_forge.return_value = {'success': True}
        
        result = self.runner.invoke(brainsmith, [
            'run',
            sample_model_path,
            sample_blueprint_path
        ])
        
        assert result.exit_code == 0
        assert '🔨 Forging accelerator' in result.output
        assert mock_forge.called


@pytest.mark.core
class TestCLIIntegration:
    """Integration tests for CLI components."""
    
    def setup_method(self):
        """Set up CLI test runner."""
        self.runner = CliRunner()
    
    def test_cli_command_structure(self):
        """Test overall CLI command structure."""
        if not CLI_AVAILABLE:
            pytest.skip("CLI not available")
        
        # Test main command
        result = self.runner.invoke(brainsmith, ['--help'])
        assert result.exit_code == 0
        
        # Should have subcommands
        assert 'Commands:' in result.output or 'forge' in result.output
    
    @patch('brainsmith.core.cli.forge')
    def test_cli_output_formatting(self, mock_forge, sample_model_path, sample_blueprint_path):
        """Test CLI output formatting and user experience."""
        if not CLI_AVAILABLE:
            pytest.skip("CLI not available")
        
        mock_forge.return_value = {
            'success': True,
            'metrics': {'performance': {'throughput_ops_sec': 500.0}}
        }
        
        result = self.runner.invoke(brainsmith, [
            'forge',
            sample_model_path,
            sample_blueprint_path
        ])
        
        # Check for proper emoji and formatting
        assert '🔨' in result.output  # Forge emoji
        assert '✅' in result.output  # Success emoji
        assert 'Model:' in result.output
        assert 'Blueprint:' in result.output
        assert sample_model_path in result.output
        assert sample_blueprint_path in result.output
    
    def test_cli_error_messages(self):
        """Test CLI error message formatting."""
        if not CLI_AVAILABLE:
            pytest.skip("CLI not available")
        
        # Test with clearly invalid input
        result = self.runner.invoke(brainsmith, ['forge', 'bad_file', 'bad_blueprint'])
        
        # Should have clear error message
        assert result.exit_code != 0
        assert 'does not exist' in result.output or 'No such file' in result.output


@pytest.mark.integration
def test_cli_end_to_end_workflow(sample_model_path, sample_blueprint_path):
    """Test complete CLI workflow from validation to forge."""
    if not CLI_AVAILABLE:
        pytest.skip("CLI not available")
    
    runner = CliRunner()
    
    with patch('brainsmith.core.cli.validate_blueprint') as mock_validate, \
         patch('brainsmith.core.cli.forge') as mock_forge:
        
        # Mock successful validation
        mock_validate.return_value = (True, [])
        mock_forge.return_value = {'success': True}
        
        # First validate
        validate_result = runner.invoke(brainsmith, ['validate', sample_blueprint_path])
        assert validate_result.exit_code == 0
        
        # Then forge
        forge_result = runner.invoke(brainsmith, ['forge', sample_model_path, sample_blueprint_path])
        assert forge_result.exit_code == 0
        
        # Both should have been called
        mock_validate.assert_called_once()
        mock_forge.assert_called_once()


# Helper functions for CLI testing
def invoke_cli_safely(runner, command_args):
    """Helper to safely invoke CLI commands for testing."""
    try:
        result = runner.invoke(brainsmith, command_args)
        return result
    except Exception as e:
        # Return mock result for test isolation
        class MockResult:
            def __init__(self, exception):
                self.exit_code = 1
                self.output = f"CLI Error: {str(exception)}"
                self.exception = exception
        
        return MockResult(e)


def assert_cli_success(result, expected_text=None):
    """Helper to assert CLI command success."""
    assert result.exit_code == 0, f"CLI command failed: {result.output}"
    if expected_text:
        assert expected_text in result.output


def assert_cli_failure(result, expected_error=None):
    """Helper to assert CLI command failure."""
    assert result.exit_code != 0, f"CLI command should have failed: {result.output}"
    if expected_error:
        assert expected_error in result.output