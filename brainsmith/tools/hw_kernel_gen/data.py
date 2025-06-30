############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

# /home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/data.py
"""
Data structures shared across Hardware Kernel Generator components.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class HWKernelPy:
    """
    Placeholder for structured Python data related to the HW Kernel.

    This data is expected to be provided alongside the RTL source and
    will contain compiler-specific information like ONNX pattern matching
    details, cost functions, etc. This structure will be expanded as
    the Python data parsing component is developed.
    """
    # Example fields - will be expanded later
    onnx_pattern: Optional[Any] = None # Placeholder for ONNX model/graph object
    cost_functions: Dict[str, str] = field(default_factory=dict) # Placeholder for cost function definitions
    metadata: Dict[str, Any] = field(default_factory=dict) # For any other relevant Python-defined info
