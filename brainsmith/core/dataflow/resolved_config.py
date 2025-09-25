############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Resolved configuration structures for two-phase model creation.

These structures represent the intermediate state between schemas and models,
containing all nodeattr-resolved values but not yet bound to tensor information.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple


@dataclass(frozen=True)
class ResolvedInterfaceConfig:
    """Resolved configuration for a single interface.
    
    Contains all information that can be determined from schema + nodeattrs
    without needing access to the ModelWrapper.
    """
    
    name: str
    position: int
    
    # Resolved from schema + nodeattrs
    block_params: List[Union[int, str]]  # Resolved template with nodeattr values
    stream_params: Optional[List[Union[int, str]]] = None  # For inputs only
    datatype_attr: str = ""  # Nodeattr name for datatype
    
    # Schema properties carried forward
    is_weight: bool = False
    optional: bool = False


@dataclass(frozen=True)
class ResolvedKernelConfig:
    """Complete resolved configuration from schema + nodeattrs.
    
    This represents all the information we can determine about a kernel
    without access to the ONNX graph (ModelWrapper). It's the output
    of Phase 1 in the two-phase model creation process.
    """
    
    kernel_name: str
    inputs: List[ResolvedInterfaceConfig]
    outputs: List[ResolvedInterfaceConfig]
    parameters: Dict[str, Any]  # Extracted kernel parameters (PE, CHANNELS, etc.)
    
    # Performance parameters
    clock_freq_mhz: float = 100.0
    
    def get_input(self, name: str) -> Optional[ResolvedInterfaceConfig]:
        """Get input config by name."""
        for inp in self.inputs:
            if inp.name == name:
                return inp
        return None
    
    def get_output(self, name: str) -> Optional[ResolvedInterfaceConfig]:
        """Get output config by name."""
        for out in self.outputs:
            if out.name == name:
                return out
        return None