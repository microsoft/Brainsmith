# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Design module for Brainsmith.

This module handles design space definition, blueprint parsing,
and DSE tree construction.
"""

from .space import DesignSpace
from .parser import parse_blueprint
from .builder import DSETreeBuilder

__all__ = [
    'parse_blueprint',
    'DesignSpace',
    'DSETreeBuilder'
]