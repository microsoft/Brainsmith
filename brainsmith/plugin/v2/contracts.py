"""
Plugin Contract System

Defines mandatory interfaces that all plugins must implement.
This is a BREAKING CHANGE - all existing plugins must be updated.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of plugin validation"""
    
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    @classmethod
    def success(cls, warnings: List[str] = None) -> 'ValidationResult':
        return cls(True, warnings=warnings)
    
    @classmethod
    def failure(cls, errors: List[str], warnings: List[str] = None) -> 'ValidationResult':
        return cls(False, errors, warnings)
    
    def combine(self, other: 'ValidationResult') -> 'ValidationResult':
        """Combine two validation results"""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings
        )


class PluginDependency:
    """Represents a plugin dependency"""
    
    def __init__(self, name: str, version_constraint: str = None, 
                 plugin_type: str = None, optional: bool = False):
        self.name = name
        self.version_constraint = version_constraint
        self.plugin_type = plugin_type
        self.optional = optional
    
    def __repr__(self):
        parts = [self.name]
        if self.version_constraint:
            parts.append(f"version={self.version_constraint}")
        if self.plugin_type:
            parts.append(f"type={self.plugin_type}")
        if self.optional:
            parts.append("optional=True")
        return f"PluginDependency({', '.join(parts)})"


class SystemDependency:
    """Represents a system-level dependency (Python packages, etc.)"""
    
    def __init__(self, name: str, version_constraint: str = None, 
                 import_name: str = None, optional: bool = False):
        self.name = name
        self.version_constraint = version_constraint
        self.import_name = import_name or name
        self.optional = optional
    
    def __repr__(self):
        return f"SystemDependency({self.name}, version={self.version_constraint})"


class PluginContract(ABC):
    """
    Base contract that all plugins must implement.
    
    This is a BREAKING CHANGE - existing plugins without these methods will fail.
    """
    
    @abstractmethod
    def validate_environment(self) -> ValidationResult:
        """
        Check if plugin can run in current environment.
        
        Returns:
            ValidationResult indicating if environment is suitable
        """
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[Union[PluginDependency, SystemDependency]]:
        """
        Return list of required dependencies.
        
        Returns:
            List of dependencies this plugin requires
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return plugin metadata.
        
        Returns:
            Dictionary of plugin metadata
        """
        pass
    
    def get_version(self) -> str:
        """Get plugin version"""
        return getattr(self, '__version__', '0.0.0')
    
    def get_name(self) -> str:
        """Get plugin name"""
        return getattr(self, '__name__', self.__class__.__name__)


class TransformContract(PluginContract):
    """
    Contract for transformation plugins.
    
    BREAKING CHANGE: All transforms must implement these methods.
    """
    
    @abstractmethod
    def apply(self, model) -> 'TransformResult':
        """
        Apply transformation to model.
        
        Args:
            model: Model to transform (typically ModelWrapper)
            
        Returns:
            TransformResult with transformed model and metadata
        """
        pass
    
    @abstractmethod
    def can_apply(self, model) -> bool:
        """
        Check if transform is applicable to model.
        
        Args:
            model: Model to check
            
        Returns:
            True if transform can be applied, False otherwise
        """
        pass
    
    def get_stage(self) -> str:
        """Get the compilation stage this transform belongs to"""
        return getattr(self, '_stage', 'unknown')
    
    def get_target_kernel(self) -> Optional[str]:
        """Get target kernel name if this is a kernel inference transform"""
        return getattr(self, '_target_kernel', None)
    
    def is_kernel_inference(self) -> bool:
        """Check if this is a kernel inference transform"""
        return self.get_target_kernel() is not None


class KernelContract(PluginContract):
    """
    Contract for kernel plugins.
    
    BREAKING CHANGE: All kernels must implement these methods.
    """
    
    @abstractmethod
    def get_nodeattr_types(self) -> Dict[str, tuple]:
        """
        Get node attribute types.
        
        Returns:
            Dictionary mapping attribute names to (type, required, default) tuples
        """
        pass
    
    @abstractmethod
    def execute_node(self, context, graph):
        """
        Execute the kernel operation.
        
        Args:
            context: Execution context
            graph: Graph being executed
        """
        pass
    
    def get_op_type(self) -> str:
        """Get ONNX operation type"""
        return getattr(self, '_op_type', self.get_name())
    
    def get_domain(self) -> str:
        """Get ONNX domain"""
        return getattr(self, '_domain', f"brainsmith.kernels.{self.get_name().lower()}")


class BackendContract(PluginContract):
    """
    Contract for backend plugins.
    
    BREAKING CHANGE: All backends must implement these methods.
    """
    
    @abstractmethod
    def get_backend_type(self) -> str:
        """
        Get backend type.
        
        Returns:
            Backend type ("hls" or "rtl")
        """
        pass
    
    @abstractmethod
    def get_target_kernel(self) -> str:
        """
        Get the kernel this backend implements.
        
        Returns:
            Name of the kernel this backend provides implementation for
        """
        pass
    
    @abstractmethod
    def generate_code(self, context) -> 'CodeGenerationResult':
        """
        Generate hardware code.
        
        Args:
            context: Code generation context
            
        Returns:
            CodeGenerationResult with generated code and metadata
        """
        pass
    
    def can_implement(self, kernel_spec) -> bool:
        """
        Check if this backend can implement the given kernel specification.
        
        Args:
            kernel_spec: Kernel specification to check
            
        Returns:
            True if backend can implement the kernel, False otherwise
        """
        return True  # Default implementation


class TransformResult:
    """Result of applying a transformation"""
    
    def __init__(self, model, graph_modified: bool = True, 
                 metadata: Dict[str, Any] = None, warnings: List[str] = None):
        self.model = model
        self.graph_modified = graph_modified
        self.metadata = metadata or {}
        self.warnings = warnings or []
    
    @classmethod
    def success(cls, model, graph_modified: bool = True, **metadata):
        return cls(model, graph_modified, metadata)
    
    @classmethod
    def no_change(cls, model, reason: str = None):
        metadata = {"reason": reason} if reason else {}
        return cls(model, False, metadata)


class CodeGenerationResult:
    """Result of code generation"""
    
    def __init__(self, code: str, language: str, metadata: Dict[str, Any] = None):
        self.code = code
        self.language = language
        self.metadata = metadata or {}
    
    @classmethod
    def hls(cls, code: str, **metadata):
        return cls(code, "hls", metadata)
    
    @classmethod
    def rtl(cls, code: str, **metadata):
        return cls(code, "rtl", metadata)


# Utility functions for contract validation
def validate_contract_compliance(plugin_class: type, expected_contract: type) -> ValidationResult:
    """
    Validate that a plugin class implements the expected contract.
    
    Args:
        plugin_class: Plugin class to validate
        expected_contract: Expected contract interface
        
    Returns:
        ValidationResult indicating compliance
    """
    errors = []
    warnings = []
    
    # Check inheritance
    if not issubclass(plugin_class, expected_contract):
        errors.append(f"Plugin {plugin_class.__name__} does not inherit from {expected_contract.__name__}")
        return ValidationResult.failure(errors)
    
    # Check abstract methods are implemented
    try:
        # This will raise TypeError if abstract methods aren't implemented
        instance = plugin_class.__new__(plugin_class)
        abstract_methods = getattr(plugin_class, '__abstractmethods__', set())
        if abstract_methods:
            errors.append(f"Plugin {plugin_class.__name__} has unimplemented abstract methods: {abstract_methods}")
    except TypeError as e:
        errors.append(f"Plugin {plugin_class.__name__} cannot be instantiated: {e}")
    
    if errors:
        return ValidationResult.failure(errors, warnings)
    else:
        return ValidationResult.success(warnings)