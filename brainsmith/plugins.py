# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Core component registration.

Brainsmith components self-register in their respective modules:
- brainsmith/kernels/<kernel>/__init__.py (kernels & backends)
- brainsmith/steps/*.py (steps)

FINN components are loaded via manifest-based lazy loading in loader.py.
"""
