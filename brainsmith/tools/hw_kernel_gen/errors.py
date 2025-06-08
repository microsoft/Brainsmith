"""
Standardized error handling for Brainsmith Hardware Kernel Generator.

This module provides a consistent error handling framework with:
- Hierarchical exception structure
- Rich error context
- Actionable error messages
- Structured logging integration
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    ERROR = "error" 
    WARNING = "warning"
    INFO = "info"

class BrainsmithError(Exception):
    """
    Base exception for all Brainsmith Hardware Kernel Generator errors.
    
    Provides rich error context and consistent error handling patterns.
    """
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, 
                 severity: ErrorSeverity = ErrorSeverity.ERROR,
                 suggestions: Optional[list] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.severity = severity
        self.suggestions = suggestions or []
        self.timestamp = datetime.now().isoformat()
        
        # Log error automatically
        self._log_error()
    
    def _log_error(self):
        """Log error with appropriate level."""
        log_message = f"{self.message}"
        if self.context:
            log_message += f" Context: {self.context}"
        
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif self.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif self.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'context': self.context,
            'severity': self.severity.value,
            'suggestions': self.suggestions,
            'timestamp': self.timestamp
        }

class RTLParsingError(BrainsmithError):
    """Errors during RTL file parsing."""
    
    def __init__(self, message: str, file_path: str = None, line_number: int = None, **kwargs):
        context = kwargs.get('context', {})
        if file_path:
            context['file_path'] = file_path
        if line_number:
            context['line_number'] = line_number
        
        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions = [
                "Check SystemVerilog syntax",
                "Ensure ANSI-style port declarations",
                "Verify required interfaces (ap_clk, ap_rst_n)"
            ]
        
        super().__init__(message, context=context, suggestions=suggestions, **kwargs)

class InterfaceDetectionError(BrainsmithError):
    """Errors during interface detection and validation."""
    
    def __init__(self, message: str, interface_name: str = None, **kwargs):
        context = kwargs.get('context', {})
        if interface_name:
            context['interface_name'] = interface_name
        
        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions = [
                "Check AXI interface signal naming (s_axis_*, m_axis_*)",
                "Ensure required signals present (tdata, tvalid, tready)",
                "Verify global control signals (ap_clk, ap_rst_n)"
            ]
        
        super().__init__(message, context=context, suggestions=suggestions, **kwargs)

class PragmaProcessingError(BrainsmithError):
    """Errors during pragma processing."""
    
    def __init__(self, message: str, pragma_text: str = None, pragma_type: str = None, **kwargs):
        context = kwargs.get('context', {})
        if pragma_text:
            context['pragma_text'] = pragma_text
        if pragma_type:
            context['pragma_type'] = pragma_type
        
        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions = [
                "Check pragma syntax: // @brainsmith <TYPE> <args>",
                "Verify interface names match RTL ports",
                "Ensure parameter references are valid"
            ]
        
        super().__init__(message, context=context, suggestions=suggestions, **kwargs)

class CodeGenerationError(BrainsmithError):
    """Errors during code generation."""
    
    def __init__(self, message: str, generator_type: str = None, template_name: str = None, suggestion: str = None, **kwargs):
        context = kwargs.get('context', {})
        if generator_type:
            context['generator_type'] = generator_type
        if template_name:
            context['template_name'] = template_name
        
        suggestions = kwargs.get('suggestions', [])
        if suggestion and suggestion not in suggestions:
            suggestions.append(suggestion)
        if not suggestions:
            suggestions = [
                "Check template syntax and context variables",
                "Verify all required data is available",
                "Check file permissions for output directory"
            ]
        
        super().__init__(message, context=context, suggestions=suggestions, **kwargs)

class ValidationError(BrainsmithError):
    """Errors during validation."""
    
    def __init__(self, message: str, validation_type: str = None, suggestion: str = None, **kwargs):
        context = kwargs.get('context', {})
        if validation_type:
            context['validation_type'] = validation_type
        
        suggestions = kwargs.get('suggestions', [])
        if suggestion and suggestion not in suggestions:
            suggestions.append(suggestion)
        if not suggestions:
            suggestions = [
                "Check data types and formats",
                "Verify all required fields are present",
                "Ensure values are within acceptable ranges"
            ]
        
        super().__init__(message, context=context, suggestions=suggestions, **kwargs)

class ConfigurationError(BrainsmithError):
    """Errors in configuration setup and validation."""
    
    def __init__(self, message: str, config_section: str = None, suggestion: str = None, **kwargs):
        context = kwargs.get('context', {})
        if config_section:
            context['config_section'] = config_section
        
        suggestions = kwargs.get('suggestions', [])
        if suggestion and suggestion not in suggestions:
            suggestions.append(suggestion)
        if not suggestions:
            suggestions = [
                "Check configuration file syntax",
                "Verify all required fields are present",
                "Ensure file paths exist and are accessible"
            ]
        
        super().__init__(message, context=context, suggestions=suggestions, **kwargs)
        self.config_section = config_section

def handle_error_with_recovery(error: Exception, recovery_strategies: list = None) -> Any:
    """
    Handle errors with optional recovery strategies.
    
    Args:
        error: The exception that occurred
        recovery_strategies: List of functions to attempt for recovery
        
    Returns:
        Result from successful recovery strategy, or raises original error
    """
    if not recovery_strategies:
        raise error
    
    for strategy in recovery_strategies:
        try:
            result = strategy(error)
            logger.warning(f"Recovered from error using {strategy.__name__}: {error}")
            return result
        except Exception as recovery_error:
            logger.debug(f"Recovery strategy {strategy.__name__} failed: {recovery_error}")
            continue
    
    # All recovery strategies failed
    raise error

# Legacy compatibility
HardwareKernelGeneratorError = BrainsmithError
ParserError = RTLParsingError