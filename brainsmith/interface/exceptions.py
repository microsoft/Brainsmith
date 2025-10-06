# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Minimal exceptions for Brainsmith CLI.

This module provides only the exceptions that are actually used in the codebase.
Most error handling uses error_exit() directly.
"""

from typing import Optional, List


class BrainsmithError(Exception):
    """Base exception for CLI errors.
    
    Attributes:
        message: The main error message
        details: Optional list of details or suggestions
        exit_code: Exit code to use when this error causes program termination
    """
    
    def __init__(self, message: str, details: Optional[List[str]] = None, exit_code: int = 1):
        self.message = message
        self.details = details or []
        self.exit_code = exit_code
        super().__init__(message)


class ConfigurationError(BrainsmithError):
    """Configuration-specific errors."""
    
    def __init__(self, message: str, details: Optional[List[str]] = None):
        super().__init__(message, details, exit_code=2)


class ValidationError(BrainsmithError):
    """Input validation errors."""
    
    def __init__(self, message: str, details: Optional[List[str]] = None):
        super().__init__(message, details, exit_code=3)