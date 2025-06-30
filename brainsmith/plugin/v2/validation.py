"""
Comprehensive Plugin Validation Framework

BREAKING CHANGE: Mandatory validation for all plugins.
No plugin can be activated without passing all validation checks.
"""

import logging
import importlib
import inspect
import sys
from typing import List, Dict, Any, Type, Optional, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .contracts import (
    PluginContract, TransformContract, KernelContract, BackendContract,
    ValidationResult, PluginDependency, SystemDependency
)

logger = logging.getLogger(__name__)


@dataclass
class SecurityIssue:
    """Represents a security concern with a plugin"""
    severity: str  # "low", "medium", "high", "critical"
    category: str  # "import", "code", "dependency", "permission"
    description: str
    recommendation: str


@dataclass
class PerformanceIssue:
    """Represents a performance concern with a plugin"""
    severity: str  # "low", "medium", "high"
    category: str  # "memory", "cpu", "io", "algorithm"
    description: str
    estimated_impact: str


class ValidationRule(ABC):
    """Abstract base for validation rules"""
    
    @abstractmethod
    def validate(self, plugin_spec: 'PluginSpec') -> ValidationResult:
        """Validate a plugin specification"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this validation rule"""
        pass


class NamingValidationRule(ValidationRule):
    """Validates plugin naming conventions"""
    
    @property
    def name(self) -> str:
        return "naming_validation"
    
    def validate(self, plugin_spec: 'PluginSpec') -> ValidationResult:
        errors = []
        warnings = []
        
        # Check for empty or whitespace names
        if not plugin_spec.name or not plugin_spec.name.strip():
            errors.append("Plugin name cannot be empty or whitespace")
            return ValidationResult(False, errors, warnings)
        
        # Check for malicious characters
        malicious_chars = [':', ';', '/', '\\', '<', '>', '"', "'", '`']
        if any(char in plugin_spec.name for char in malicious_chars):
            errors.append(f"Plugin name '{plugin_spec.name}' contains potentially malicious characters")
        
        # Check for reserved names
        reserved_names = ['__all__', '__init__', 'None', 'True', 'False', 'sys', 'os']
        if plugin_spec.name in reserved_names:
            errors.append(f"Plugin name '{plugin_spec.name}' is reserved")
        
        # Check naming conventions
        if not plugin_spec.name[0].isupper():
            warnings.append(f"Plugin name '{plugin_spec.name}' should start with uppercase letter")
        
        if '_' in plugin_spec.name:
            warnings.append(f"Plugin name '{plugin_spec.name}' contains underscores - consider CamelCase")
        
        return ValidationResult(len(errors) == 0, errors, warnings)


class ContractValidationRule(ValidationRule):
    """Validates plugin contract compliance"""
    
    @property
    def name(self) -> str:
        return "contract_validation"
    
    def validate(self, plugin_spec: 'PluginSpec') -> ValidationResult:
        errors = []
        warnings = []
        
        # Get expected contract based on plugin type
        contract_map = {
            'transform': TransformContract,
            'kernel': KernelContract,
            'backend': BackendContract
        }
        
        expected_contract = contract_map.get(plugin_spec.type)
        if not expected_contract:
            errors.append(f"Unknown plugin type: {plugin_spec.type}")
            return ValidationResult(False, errors, warnings)
        
        # Check inheritance
        if not issubclass(plugin_spec.plugin_class, expected_contract):
            errors.append(f"Plugin must inherit from {expected_contract.__name__}")
            return ValidationResult(False, errors, warnings)
        
        # Check abstract methods
        abstract_methods = getattr(plugin_spec.plugin_class, '__abstractmethods__', set())
        if abstract_methods:
            errors.append(f"Plugin has unimplemented abstract methods: {abstract_methods}")
        
        # Try instantiation
        try:
            instance = plugin_spec.plugin_class()
            
            # Test contract methods
            try:
                env_result = instance.validate_environment()
                if not isinstance(env_result, ValidationResult):
                    warnings.append("validate_environment() should return ValidationResult")
            except Exception as e:
                errors.append(f"validate_environment() failed: {e}")
            
            try:
                deps = instance.get_dependencies()
                if not isinstance(deps, list):
                    warnings.append("get_dependencies() should return a list")
            except Exception as e:
                errors.append(f"get_dependencies() failed: {e}")
            
            try:
                metadata = instance.get_metadata()
                if not isinstance(metadata, dict):
                    warnings.append("get_metadata() should return a dictionary")
            except Exception as e:
                errors.append(f"get_metadata() failed: {e}")
                
        except Exception as e:
            errors.append(f"Plugin instantiation failed: {e}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)


class DependencyValidationRule(ValidationRule):
    """Validates plugin dependencies"""
    
    @property
    def name(self) -> str:
        return "dependency_validation"
    
    def validate(self, plugin_spec: 'PluginSpec') -> ValidationResult:
        errors = []
        warnings = []
        
        for dependency in plugin_spec.dependencies:
            if isinstance(dependency, SystemDependency):
                result = self._validate_system_dependency(dependency)
            elif isinstance(dependency, PluginDependency):
                result = self._validate_plugin_dependency(dependency)
            else:
                errors.append(f"Unknown dependency type: {type(dependency)}")
                continue
            
            errors.extend(result.errors)
            warnings.extend(result.warnings)
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def _validate_system_dependency(self, dep: SystemDependency) -> ValidationResult:
        """Validate a system dependency"""
        errors = []
        warnings = []
        
        # Try to import the dependency
        try:
            module = importlib.import_module(dep.import_name)
            
            # Check version if specified
            if dep.version_constraint and hasattr(module, '__version__'):
                # Simple version check - could be enhanced with proper semver parsing
                if module.__version__ != dep.version_constraint.replace('>=', '').replace('==', ''):
                    warnings.append(f"Dependency {dep.name} version {module.__version__} "
                                  f"may not satisfy constraint {dep.version_constraint}")
            
        except ImportError as e:
            if dep.optional:
                warnings.append(f"Optional dependency {dep.name} not available: {e}")
            else:
                errors.append(f"Required dependency {dep.name} not available: {e}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def _validate_plugin_dependency(self, dep: PluginDependency) -> ValidationResult:
        """Validate a plugin dependency"""
        # Note: This would need access to the registry to check if plugin exists
        # For now, just validate the dependency specification
        errors = []
        warnings = []
        
        if not dep.name:
            errors.append("Plugin dependency must have a name")
        
        if dep.version_constraint:
            # Validate version constraint format
            if not any(op in dep.version_constraint for op in ['>=', '<=', '==', '>', '<']):
                warnings.append(f"Version constraint '{dep.version_constraint}' should use operators like >=, ==, etc.")
        
        return ValidationResult(len(errors) == 0, errors, warnings)


class SecurityValidationRule(ValidationRule):
    """Validates plugin security"""
    
    @property
    def name(self) -> str:
        return "security_validation"
    
    def validate(self, plugin_spec: 'PluginSpec') -> ValidationResult:
        errors = []
        warnings = []
        
        # Get plugin source code for analysis
        try:
            source = inspect.getsource(plugin_spec.plugin_class)
            security_issues = self._analyze_code_security(source)
            
            for issue in security_issues:
                if issue.severity == "critical":
                    errors.append(f"SECURITY: {issue.description}")
                elif issue.severity == "high":
                    warnings.append(f"SECURITY: {issue.description}")
                else:
                    warnings.append(f"Security note: {issue.description}")
                    
        except Exception as e:
            warnings.append(f"Could not analyze plugin source code for security: {e}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def _analyze_code_security(self, source_code: str) -> List[SecurityIssue]:
        """Analyze source code for security issues"""
        issues = []
        
        # Check for dangerous imports
        dangerous_imports = ['os', 'sys', 'subprocess', 'eval', 'exec', '__import__']
        for dangerous in dangerous_imports:
            if f"import {dangerous}" in source_code or f"from {dangerous}" in source_code:
                issues.append(SecurityIssue(
                    severity="medium",
                    category="import",
                    description=f"Plugin imports potentially dangerous module: {dangerous}",
                    recommendation=f"Review usage of {dangerous} module for security implications"
                ))
        
        # Check for eval/exec usage
        if "eval(" in source_code or "exec(" in source_code:
            issues.append(SecurityIssue(
                severity="high",
                category="code",
                description="Plugin uses eval() or exec() which can execute arbitrary code",
                recommendation="Remove eval/exec usage or use safer alternatives"
            ))
        
        # Check for file system access
        file_ops = ['open(', 'file(', 'os.path', 'pathlib']
        for file_op in file_ops:
            if file_op in source_code:
                issues.append(SecurityIssue(
                    severity="low",
                    category="io",
                    description=f"Plugin performs file system operations: {file_op}",
                    recommendation="Ensure file operations are safe and necessary"
                ))
        
        return issues


class PerformanceValidationRule(ValidationRule):
    """Validates plugin performance characteristics"""
    
    @property
    def name(self) -> str:
        return "performance_validation"
    
    def validate(self, plugin_spec: 'PluginSpec') -> ValidationResult:
        errors = []
        warnings = []
        
        try:
            source = inspect.getsource(plugin_spec.plugin_class)
            performance_issues = self._analyze_performance(source)
            
            for issue in performance_issues:
                if issue.severity == "high":
                    warnings.append(f"PERFORMANCE: {issue.description}")
                else:
                    warnings.append(f"Performance note: {issue.description}")
                    
        except Exception as e:
            warnings.append(f"Could not analyze plugin performance: {e}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def _analyze_performance(self, source_code: str) -> List[PerformanceIssue]:
        """Analyze source code for performance issues"""
        issues = []
        
        # Check for potential memory issues
        if "while True:" in source_code:
            issues.append(PerformanceIssue(
                severity="medium",
                category="cpu",
                description="Plugin contains infinite loop (while True)",
                estimated_impact="May cause high CPU usage or hanging"
            ))
        
        # Check for large data structures
        if "range(1000000)" in source_code or "* 1000000" in source_code:
            issues.append(PerformanceIssue(
                severity="medium",
                category="memory",
                description="Plugin may create large data structures",
                estimated_impact="High memory usage"
            ))
        
        return issues


class TypeValidationRule(ValidationRule):
    """Validates plugin type-specific requirements"""
    
    @property
    def name(self) -> str:
        return "type_validation"
    
    def validate(self, plugin_spec: 'PluginSpec') -> ValidationResult:
        errors = []
        warnings = []
        
        if plugin_spec.type == "transform":
            result = self._validate_transform(plugin_spec)
        elif plugin_spec.type == "kernel":
            result = self._validate_kernel(plugin_spec)
        elif plugin_spec.type == "backend":
            result = self._validate_backend(plugin_spec)
        else:
            errors.append(f"Unknown plugin type: {plugin_spec.type}")
            return ValidationResult(False, errors, warnings)
        
        errors.extend(result.errors)
        warnings.extend(result.warnings)
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def _validate_transform(self, plugin_spec: 'PluginSpec') -> ValidationResult:
        """Validate transform-specific requirements"""
        errors = []
        warnings = []
        
        # Check stage/target_kernel mutual exclusion
        has_stage = plugin_spec.stage is not None
        has_target_kernel = plugin_spec.target_kernel is not None
        
        if has_stage and has_target_kernel:
            errors.append("Transform cannot specify both 'stage' and 'target_kernel'")
        elif not has_stage and not has_target_kernel:
            errors.append("Transform must specify either 'stage' or 'target_kernel'")
        
        # Validate stage
        if has_stage:
            valid_stages = ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt"]
            if plugin_spec.stage not in valid_stages:
                warnings.append(f"Non-standard stage '{plugin_spec.stage}'. "
                              f"Standard stages: {valid_stages}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def _validate_kernel(self, plugin_spec: 'PluginSpec') -> ValidationResult:
        """Validate kernel-specific requirements"""
        errors = []
        warnings = []
        
        # Check for op_type and domain in custom metadata
        if 'op_type' not in plugin_spec.custom_metadata:
            warnings.append("Kernel should specify 'op_type'")
        
        if 'domain' not in plugin_spec.custom_metadata:
            warnings.append("Kernel should specify 'domain'")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def _validate_backend(self, plugin_spec: 'PluginSpec') -> ValidationResult:
        """Validate backend-specific requirements"""
        errors = []
        warnings = []
        
        # Check required fields
        if not plugin_spec.target_kernel:
            errors.append("Backend must specify 'target_kernel'")
        
        if not plugin_spec.backend_type:
            errors.append("Backend must specify 'backend_type'")
        elif plugin_spec.backend_type not in ["hls", "rtl"]:
            errors.append(f"Invalid backend_type '{plugin_spec.backend_type}'. Must be 'hls' or 'rtl'")
        
        return ValidationResult(len(errors) == 0, errors, warnings)


class PluginValidator:
    """
    Comprehensive plugin validator.
    
    BREAKING CHANGE: All plugins must pass validation before activation.
    """
    
    def __init__(self):
        self.rules: List[ValidationRule] = [
            NamingValidationRule(),
            ContractValidationRule(),
            DependencyValidationRule(),
            SecurityValidationRule(),
            PerformanceValidationRule(),
            TypeValidationRule(),
        ]
        
        self.rule_map = {rule.name: rule for rule in self.rules}
    
    def validate(self, plugin_spec: 'PluginSpec', 
                rule_names: Optional[List[str]] = None) -> ValidationResult:
        """
        Validate a plugin specification.
        
        Args:
            plugin_spec: Plugin to validate
            rule_names: Specific rules to run (None = all rules)
            
        Returns:
            Combined validation result
        """
        if rule_names is None:
            rules_to_run = self.rules
        else:
            rules_to_run = [self.rule_map[name] for name in rule_names 
                           if name in self.rule_map]
        
        all_errors = []
        all_warnings = []
        
        for rule in rules_to_run:
            try:
                result = rule.validate(plugin_spec)
                all_errors.extend(result.errors)
                all_warnings.extend(result.warnings)
            except Exception as e:
                all_errors.append(f"Validation rule '{rule.name}' failed: {e}")
        
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings
        )
    
    def validate_runtime_environment(self, plugin_spec: 'PluginSpec') -> ValidationResult:
        """
        Validate plugin in current runtime environment.
        
        This creates an instance and tests it works.
        """
        try:
            instance = plugin_spec.plugin_class()
            
            # Test environment validation
            env_result = instance.validate_environment()
            if not env_result.is_valid:
                return ValidationResult(False, 
                                      [f"Plugin environment validation failed: {env_result.errors}"],
                                      env_result.warnings)
            
            # Test basic functionality
            if plugin_spec.type == "transform" and hasattr(instance, 'can_apply'):
                # Create a dummy model for testing
                try:
                    # This would need a real model for proper testing
                    pass
                except Exception as e:
                    return ValidationResult(False, [f"Transform functionality test failed: {e}"])
            
            return ValidationResult(True, warnings=env_result.warnings)
            
        except Exception as e:
            return ValidationResult(False, [f"Runtime validation failed: {e}"])
    
    def get_available_rules(self) -> List[str]:
        """Get list of available validation rule names"""
        return list(self.rule_map.keys())