############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Pragma definitions for the RTL parser.

This package contains all pragma implementations organized by type:
- base.py: Base classes and exceptions
- source.py: Source-related pragmas (TOP_MODULE, INCLUDE_RTL)
- interface.py: Interface-related pragmas (DATATYPE, DATATYPE_PARAM, WEIGHT)
- parameter.py: Parameter-related pragmas (ALIAS, DERIVED_PARAMETER)
- dimension.py: Dimension pragmas (BDIM, SDIM)
"""

# Re-export all pragma classes for backward compatibility
from .base import InterfacePragma, Pragma, PragmaError
from .dimension import BDimPragma, SDimPragma
from .interface import DatatypeConstraintPragma, DatatypePragma, WeightPragma
from .parameter import AliasPragma, AxiLiteParamPragma, DerivedParameterPragma
from .relationship import RelationshipPragma
from .source import IncludeRTLPragma, TopModulePragma

__all__ = [
    # Base classes
    "Pragma",
    "InterfacePragma",
    "PragmaError",
    # Source pragmas
    "TopModulePragma",
    "IncludeRTLPragma",
    # Interface pragmas
    "DatatypeConstraintPragma",
    "DatatypePragma",
    "WeightPragma",
    # Parameter pragmas
    "AliasPragma",
    "DerivedParameterPragma",
    "AxiLiteParamPragma",
    # Dimension pragmas
    "BDimPragma",
    "SDimPragma",
    # Relationship pragmas
    "RelationshipPragma",
]
