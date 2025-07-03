"""
Custom exceptions for the DSE V3 Design Space Constructor.

These exceptions provide clear error messages and help with debugging
blueprint parsing and validation issues.
"""


class BrainsmithError(Exception):
    """Base exception for all Brainsmith errors."""
    pass


class BlueprintParseError(BrainsmithError):
    """
    Raised when there's an error parsing a blueprint file.
    
    This includes YAML syntax errors, missing required fields,
    invalid data types, or unsupported blueprint versions.
    """
    def __init__(self, message: str, line: int = None, column: int = None):
        self.line = line
        self.column = column
        if line and column:
            message = f"Error at line {line}, column {column}: {message}"
        elif line:
            message = f"Error at line {line}: {message}"
        super().__init__(message)


class ValidationError(BrainsmithError):
    """
    Raised when design space validation fails.
    
    This includes invalid configurations, constraint violations,
    or incompatible settings.
    """
    def __init__(self, message: str, errors: list = None, warnings: list = None):
        self.errors = errors or []
        self.warnings = warnings or []
        
        if errors:
            message += "\n\nErrors:\n" + "\n".join(f"  - {e}" for e in errors)
        if warnings:
            message += "\n\nWarnings:\n" + "\n".join(f"  - {w}" for w in warnings)
        
        super().__init__(message)


class ConfigurationError(BrainsmithError):
    """
    Raised when there's an error in the configuration.
    
    This includes invalid parameter values, missing required
    configuration, or environment setup issues.
    """
    pass


class PluginNotFoundError(BlueprintParseError):
    """
    Raised when a referenced plugin doesn't exist in the registry.
    
    Provides helpful suggestions when possible.
    """
    
    def __init__(self, plugin_type: str, plugin_name: str, available: list = None):
        """
        Initialize error with plugin information.
        
        Args:
            plugin_type: Type of plugin (transform, kernel, backend)
            plugin_name: Name of the missing plugin
            available: List of available plugin names for suggestions
        """
        message = f"{plugin_type.capitalize()} '{plugin_name}' not found"
        
        # Always show available options, even if empty
        if available:
            # Show first 5 available options
            options = available[:5]
            if len(available) > 5:
                options.append("...")
            message += f". Available: {options}"
        else:
            message += ". Available: []"
        
        super().__init__(message)