# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Brainsmith Transforms

Plugin-based transforms organized by compilation stage.
"""

# Import all transforms by category to trigger plugin registration
from . import cleanup
from . import kernel_opt
from . import post_proc

# Pattern-based transforms (Week 11)
from .apply_patterns import ApplyPatterns

__all__ = ["ApplyPatterns"]