"""
Plugin Configuration System

BREAKING CHANGE: Runtime configuration replaces decoration-time configuration.
"""

import logging
from typing import Dict, Any, Optional, Union, List, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import yaml

logger = logging.getLogger(__name__)


class ConfigParam(ABC):
    """Base class for configuration parameters"""
    
    def __init__(self, default: Any = None, required: bool = False, 
                 description: str = ""):
        self.default = default
        self.required = required
        self.description = description
    
    @abstractmethod
    def validate(self, value: Any) -> tuple[bool, str]:
        """Validate a value. Returns (is_valid, error_message)"""
        pass
    
    def get_default(self) -> Any:
        """Get the default value"""
        return self.default


class StringParam(ConfigParam):
    """String configuration parameter"""
    
    def __init__(self, default: str = "", min_length: int = 0, 
                 max_length: int = None, pattern: str = None, **kwargs):
        super().__init__(default, **kwargs)
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern
    
    def validate(self, value: Any) -> tuple[bool, str]:
        if not isinstance(value, str):
            return False, f"Expected string, got {type(value).__name__}"
        
        if len(value) < self.min_length:
            return False, f"String too short (min {self.min_length})"
        
        if self.max_length and len(value) > self.max_length:
            return False, f"String too long (max {self.max_length})"
        
        if self.pattern:
            import re
            if not re.match(self.pattern, value):
                return False, f"String does not match pattern {self.pattern}"
        
        return True, ""


class IntParam(ConfigParam):
    """Integer configuration parameter"""
    
    def __init__(self, default: int = 0, min_value: int = None, 
                 max_value: int = None, **kwargs):
        super().__init__(default, **kwargs)
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, value: Any) -> tuple[bool, str]:
        if not isinstance(value, int):
            return False, f"Expected int, got {type(value).__name__}"
        
        if self.min_value is not None and value < self.min_value:
            return False, f"Value too small (min {self.min_value})"
        
        if self.max_value is not None and value > self.max_value:
            return False, f"Value too large (max {self.max_value})"
        
        return True, ""


class FloatParam(ConfigParam):
    """Float configuration parameter"""
    
    def __init__(self, default: float = 0.0, min_value: float = None, 
                 max_value: float = None, **kwargs):
        super().__init__(default, **kwargs)
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, value: Any) -> tuple[bool, str]:
        if not isinstance(value, (int, float)):
            return False, f"Expected number, got {type(value).__name__}"
        
        value = float(value)
        
        if self.min_value is not None and value < self.min_value:
            return False, f"Value too small (min {self.min_value})"
        
        if self.max_value is not None and value > self.max_value:
            return False, f"Value too large (max {self.max_value})"
        
        return True, ""


class BoolParam(ConfigParam):
    """Boolean configuration parameter"""
    
    def __init__(self, default: bool = False, **kwargs):
        super().__init__(default, **kwargs)
    
    def validate(self, value: Any) -> tuple[bool, str]:
        if not isinstance(value, bool):
            return False, f"Expected bool, got {type(value).__name__}"
        return True, ""


class EnumParam(ConfigParam):
    """Enumeration configuration parameter"""
    
    def __init__(self, choices: List[Any], default: Any = None, **kwargs):
        if default is None and choices:
            default = choices[0]
        super().__init__(default, **kwargs)
        self.choices = choices
    
    def validate(self, value: Any) -> tuple[bool, str]:
        if value not in self.choices:
            return False, f"Value must be one of {self.choices}"
        return True, ""


class ListParam(ConfigParam):
    """List configuration parameter"""
    
    def __init__(self, item_type: ConfigParam, default: List = None, 
                 min_items: int = 0, max_items: int = None, **kwargs):
        if default is None:
            default = []
        super().__init__(default, **kwargs)
        self.item_type = item_type
        self.min_items = min_items
        self.max_items = max_items
    
    def validate(self, value: Any) -> tuple[bool, str]:
        if not isinstance(value, list):
            return False, f"Expected list, got {type(value).__name__}"
        
        if len(value) < self.min_items:
            return False, f"Too few items (min {self.min_items})"
        
        if self.max_items and len(value) > self.max_items:
            return False, f"Too many items (max {self.max_items})"
        
        # Validate each item
        for i, item in enumerate(value):
            valid, error = self.item_type.validate(item)
            if not valid:
                return False, f"Item {i}: {error}"
        
        return True, ""


class DictParam(ConfigParam):
    """Dictionary configuration parameter"""
    
    def __init__(self, default: Dict = None, optional: bool = True, **kwargs):
        if default is None:
            default = {}
        super().__init__(default, **kwargs)
        self.optional = optional
    
    def validate(self, value: Any) -> tuple[bool, str]:
        if not isinstance(value, dict):
            return False, f"Expected dict, got {type(value).__name__}"
        return True, ""


@dataclass
class ConfigSchema:
    """Schema for plugin configuration"""
    parameters: Dict[str, ConfigParam]
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    
    def validate(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate a configuration against this schema"""
        errors = []
        
        # Check required parameters
        for param_name, param in self.parameters.items():
            if param.required and param_name not in config:
                errors.append(f"Required parameter '{param_name}' missing")
        
        # Validate provided parameters
        for param_name, value in config.items():
            if param_name not in self.parameters:
                errors.append(f"Unknown parameter '{param_name}'")
                continue
            
            param = self.parameters[param_name]
            valid, error = param.validate(value)
            if not valid:
                errors.append(f"Parameter '{param_name}': {error}")
        
        return len(errors) == 0, errors
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values"""
        defaults = {}
        for param_name, param in self.parameters.items():
            defaults[param_name] = param.get_default()
        return defaults
    
    def merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration with defaults"""
        result = self.get_defaults()
        result.update(config)
        return result


class PluginConfig:
    """
    Runtime configuration for plugins.
    
    BREAKING CHANGE: Replaces decoration-time configuration.
    """
    
    def __init__(self, config_data: Dict[str, Any] = None):
        self._config = config_data or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self._config[key] = value
    
    def update(self, config: Dict[str, Any]):
        """Update configuration with new values"""
        self._config.update(config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return self._config.copy()
    
    @classmethod
    def from_file(cls, file_path: str) -> 'PluginConfig':
        """Load configuration from file (JSON or YAML)"""
        with open(file_path, 'r') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        return cls(config_data)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PluginConfig':
        """Create configuration from dictionary"""
        return cls(config_dict)
    
    def save_to_file(self, file_path: str):
        """Save configuration to file"""
        with open(file_path, 'w') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                yaml.safe_dump(self._config, f, default_flow_style=False)
            else:
                json.dump(self._config, f, indent=2)


class ConfigurablePlugin:
    """
    Mixin for plugins that support runtime configuration.
    
    BREAKING CHANGE: New configuration approach.
    """
    
    def __init__(self, config: PluginConfig = None):
        self.config = config or PluginConfig()
        self._config_schema = self.get_config_schema()
        
        # Validate configuration if schema is provided
        if self._config_schema:
            valid, errors = self._config_schema.validate(self.config.to_dict())
            if not valid:
                logger.warning(f"Plugin {self.__class__.__name__} configuration validation failed: {errors}")
                # Merge with defaults to fix missing values
                merged_config = self._config_schema.merge_with_defaults(self.config.to_dict())
                self.config = PluginConfig(merged_config)
    
    def get_config_schema(self) -> Optional[ConfigSchema]:
        """
        Override this method to provide configuration schema.
        
        Returns:
            ConfigSchema defining the configuration parameters
        """
        return None
    
    def reconfigure(self, new_config: Dict[str, Any]):
        """
        Reconfigure the plugin with new settings.
        
        Args:
            new_config: New configuration values
        """
        if self._config_schema:
            valid, errors = self._config_schema.validate(new_config)
            if not valid:
                raise ValueError(f"Invalid configuration: {errors}")
        
        self.config.update(new_config)
        self.on_config_changed()
    
    def on_config_changed(self):
        """Called when configuration changes. Override to handle config updates."""
        pass