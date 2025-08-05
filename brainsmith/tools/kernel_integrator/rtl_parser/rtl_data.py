############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
RTL Parser Data Structures - Compatibility Shim.

This module now re-exports types from the new unified type system.
It maintains backward compatibility while the codebase is migrated.

DEPRECATED: Import directly from brainsmith.tools.kernel_integrator.types instead.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "rtl_data module is deprecated. Import from brainsmith.tools.kernel_integrator.types instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new type modules
from brainsmith.core.dataflow.types import InterfaceType
from brainsmith.tools.kernel_integrator.types.core import PortDirection
from brainsmith.tools.kernel_integrator.types.rtl import (
    Port,
    Parameter, 
    PortGroup,
    PragmaType,
    ProtocolValidationResult,
)

# Module exports (same as before for compatibility)
__all__ = [
    # Enums
    "PortDirection",
    "PragmaType",
    
    # RTL Data Structures  
    "Parameter",
    "Port", 
    "PortGroup",
    "ProtocolValidationResult",
    
    # Also export InterfaceType for convenience
    "InterfaceType",
]