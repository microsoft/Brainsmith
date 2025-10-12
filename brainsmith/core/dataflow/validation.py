############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Simple validation error type for dataflow constraints"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ValidationError:
    """Simple validation error with context and suggestions.

    Replaces string-based error returns with structured type.
    """
    message: str
    location: str  # e.g. "input.stream[1]", "output.block[0]"
    suggestions: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        msg = f"{self.location}: {self.message}"
        if self.suggestions:
            suggestions_str = ", ".join(self.suggestions)
            msg += f"\n  Suggestions: {suggestions_str}"
        return msg


__all__ = ['ValidationError']
