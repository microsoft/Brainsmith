############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unified Kernel Modeling Framework for FPGA AI Accelerators"""

from .core.types import Shape, RaggedShape, InterfaceDirection, DataType
from .core.interface import Interface
from .core.pragma import Pragma, TiePragma, ConstrPragma, parse_pragma
from .core.kernel import Kernel
from .core.graph import DataflowGraph, DataflowEdge

__version__ = "0.1.0"

__all__ = [
    # Types
    "Shape",
    "RaggedShape", 
    "InterfaceDirection",
    "DataType",
    # Core classes
    "Interface",
    "Kernel",
    "DataflowGraph",
    "DataflowEdge",
    # Pragmas
    "Pragma",
    "TiePragma",
    "ConstrPragma",
    "parse_pragma",
]