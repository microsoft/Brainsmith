############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Pragma definitions for the RTL parser.

This package contains all pragma implementations organized by type:
- base.py: Base classes and exceptions
- module.py: Module-level pragmas (TOP_MODULE)
- interface.py: Interface-related pragmas (DATATYPE, DATATYPE_PARAM, WEIGHT)
- parameter.py: Parameter-related pragmas (ALIAS, DERIVED_PARAMETER)
- dimension.py: Dimension pragmas (BDIM, SDIM)
"""

# Re-export all pragma classes for backward compatibility
from .base import Pragma, InterfacePragma, PragmaError
from .module import TopModulePragma
from .interface import DatatypePragma, DatatypeParamPragma, WeightPragma
from .parameter import AliasPragma, DerivedParameterPragma, AxiLiteParamPragma
from .dimension import BDimPragma, SDimPragma

__all__ = [
    # Base classes
    'Pragma',
    'InterfacePragma',
    'PragmaError',
    # Module pragmas
    'TopModulePragma',
    # Interface pragmas
    'DatatypePragma',
    'DatatypeParamPragma',
    'WeightPragma',
    # Parameter pragmas
    'AliasPragma',
    'DerivedParameterPragma',
    'AxiLiteParamPragma',
    # Dimension pragmas
    'BDimPragma',
    'SDimPragma',
]