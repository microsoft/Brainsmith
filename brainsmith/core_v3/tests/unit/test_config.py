"""Unit tests for the global configuration system."""

import os
import pytest
import tempfile
from pathlib import Path
import yaml

from brainsmith.core_v3.config import BrainsmithConfig, load_config, get_config, reset_config


class TestBrainsmithConfig:
    """Test the BrainsmithConfig data structure."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = BrainsmithConfig()
        assert config.max_combinations == 100_000
        assert config.timeout_minutes == 60
    
    def test_custom_values(self):
        """Test creating config with custom values."""
        config = BrainsmithConfig(
            max_combinations=500_000,
            timeout_minutes=120
        )
        assert config.max_combinations == 500_000
        assert config.timeout_minutes == 120


class TestConfigLoading:
    """Test configuration loading from various sources."""
    
    def setup_method(self):
        """Reset config before each test."""
        reset_config()
        # Store original env vars
        self.orig_max_combinations = os.environ.get("BRAINSMITH_MAX_COMBINATIONS")
        self.orig_timeout = os.environ.get("BRAINSMITH_TIMEOUT_MINUTES")
        # Clear env vars
        os.environ.pop("BRAINSMITH_MAX_COMBINATIONS", None)
        os.environ.pop("BRAINSMITH_TIMEOUT_MINUTES", None)
    
    def teardown_method(self):
        """Restore original state after each test."""
        reset_config()
        # Restore original env vars
        if self.orig_max_combinations is not None:
            os.environ["BRAINSMITH_MAX_COMBINATIONS"] = self.orig_max_combinations
        else:
            os.environ.pop("BRAINSMITH_MAX_COMBINATIONS", None)
            
        if self.orig_timeout is not None:
            os.environ["BRAINSMITH_TIMEOUT_MINUTES"] = self.orig_timeout
        else:
            os.environ.pop("BRAINSMITH_TIMEOUT_MINUTES", None)
    
    def test_load_defaults(self):
        """Test loading default configuration."""
        config = load_config()
        assert config.max_combinations == 100_000
        assert config.timeout_minutes == 60
    
    def test_load_from_environment(self):
        """Test loading configuration from environment variables."""
        os.environ["BRAINSMITH_MAX_COMBINATIONS"] = "200000"
        os.environ["BRAINSMITH_TIMEOUT_MINUTES"] = "90"
        
        config = load_config()
        assert config.max_combinations == 200_000
        assert config.timeout_minutes == 90
    
    def test_invalid_environment_values(self):
        """Test handling of invalid environment variable values."""
        os.environ["BRAINSMITH_MAX_COMBINATIONS"] = "not_a_number"
        os.environ["BRAINSMITH_TIMEOUT_MINUTES"] = "invalid"
        
        # Should fall back to defaults
        config = load_config()
        assert config.max_combinations == 100_000
        assert config.timeout_minutes == 60
    
    def test_load_from_user_config(self, monkeypatch):
        """Test loading configuration from user config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake home directory
            fake_home = Path(tmpdir)
            config_dir = fake_home / ".brainsmith"
            config_dir.mkdir()
            
            # Create user config
            user_config = {
                "max_combinations": 300000,
                "timeout_minutes": 180
            }
            config_file = config_dir / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(user_config, f)
            
            # Monkey patch Path.home() to return our fake home
            monkeypatch.setattr(Path, "home", lambda: fake_home)
            
            config = load_config()
            assert config.max_combinations == 300_000
            assert config.timeout_minutes == 180
    
    def test_config_priority(self, monkeypatch):
        """Test configuration priority: env > user > defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake home directory
            fake_home = Path(tmpdir)
            config_dir = fake_home / ".brainsmith"
            config_dir.mkdir()
            
            # Create user config
            user_config = {
                "max_combinations": 300000,
                "timeout_minutes": 180
            }
            config_file = config_dir / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(user_config, f)
            
            # Monkey patch Path.home()
            monkeypatch.setattr(Path, "home", lambda: fake_home)
            
            # Set environment variables (should override user config)
            os.environ["BRAINSMITH_MAX_COMBINATIONS"] = "400000"
            # Don't set timeout - should use user config
            
            config = load_config()
            assert config.max_combinations == 400_000  # From env
            assert config.timeout_minutes == 180  # From user config
    
    def test_get_config_singleton(self):
        """Test that get_config returns a singleton."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
    
    def test_reset_config(self):
        """Test config reset functionality."""
        # Get initial config
        config1 = get_config()
        
        # Change environment
        os.environ["BRAINSMITH_MAX_COMBINATIONS"] = "500000"
        
        # Should still return cached config
        config2 = get_config()
        assert config2 is config1
        assert config2.max_combinations == 100_000  # Still default
        
        # Reset and get new config
        reset_config()
        config3 = get_config()
        assert config3 is not config1
        assert config3.max_combinations == 500_000  # New value from env