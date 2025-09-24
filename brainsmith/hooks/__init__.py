# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Import hooks for Brainsmith.

This module contains import hooks that modify Python's import behavior
to ensure proper environment configuration for external dependencies.

NOTE: Import hooks are a temporary measure and should be avoided in
production PyPI packages. These hooks exist to improve the developer
experience until upstream dependencies adopt better configuration
management practices.
"""