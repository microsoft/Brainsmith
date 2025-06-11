"""
Registry Exception Classes

Defines the exception hierarchy for registry operations across all Brainsmith components.
"""


class RegistryError(Exception):
    """Base exception for all registry-related errors."""
    pass


class ComponentNotFoundError(RegistryError):
    """Raised when a requested component cannot be found in the registry."""
    
    def __init__(self, component_name: str, registry_type: str):
        self.component_name = component_name
        self.registry_type = registry_type
        super().__init__(f"Component '{component_name}' not found in {registry_type}")


class ValidationError(RegistryError):
    """Raised when component validation fails."""
    
    def __init__(self, component_name: str, errors: list):
        self.component_name = component_name
        self.errors = errors
        error_msg = f"Validation failed for component '{component_name}': {'; '.join(errors)}"
        super().__init__(error_msg)


class RegistryConfigurationError(RegistryError):
    """Raised when registry configuration is invalid."""
    pass


class ComponentLoadError(RegistryError):
    """Raised when a component cannot be loaded properly."""
    
    def __init__(self, component_name: str, original_error: Exception):
        self.component_name = component_name
        self.original_error = original_error
        super().__init__(f"Failed to load component '{component_name}': {original_error}")