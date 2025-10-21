# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CLI-specific exception hierarchy.

Provides a structured set of exceptions for CLI error handling,
allowing for consistent error reporting and handling patterns.
"""

from .constants import (
    EX_USAGE,
    EX_CONFIG,
    EX_DATAERR,
)


class CLIError(Exception):
    """Base exception for all CLI-related errors.

    Attributes:
        message: Main error message
        details: Optional list of additional detail lines
        exit_code: Suggested exit code for this error type (class attribute)
    """

    exit_code: int = EX_USAGE

    def __init__(self, message: str, details: list[str] | None = None):
        """Initialize CLI error.

        Args:
            message: Main error message
            details: Optional list of detail lines for user guidance
        """
        self.message = message
        self.details = details or []
        super().__init__(message)

    def format_for_console(self) -> str:
        """Format error message for console output.

        Returns:
            Formatted error message with details
        """
        lines = [f"[red]Error:[/red] {self.message}"]
        if self.details:
            lines.append("")
            for detail in self.details:
                lines.append(f"  • {detail}")
        return "\n".join(lines)


class ConfigurationError(CLIError):
    """Configuration-related errors.

    Raised when there are issues loading or validating configuration files.
    """

    exit_code = EX_CONFIG


class SetupError(CLIError):
    """Dependency setup and installation errors.

    Raised when setup operations (installing dependencies, building tools) fail.
    """

    exit_code = EX_USAGE


class ValidationError(CLIError):
    """Input validation errors.

    Raised when user input or arguments fail validation.
    """

    exit_code = EX_DATAERR


class CommandError(CLIError):
    """Command execution errors.

    Raised when a CLI command fails during execution.
    """

    exit_code = EX_USAGE
