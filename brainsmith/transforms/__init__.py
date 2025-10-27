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

# Import individual transforms
from . import normalize_dataflow_layouts
from . import specialize_kernels

# Register SpecializeKernels as "SpecializeLayers" to shadow FINN's version
from brainsmith.core.plugins import get_registry
from brainsmith.transforms.specialize_kernels import SpecializeKernels
get_registry().register('transform', 'SpecializeLayers', SpecializeKernels)

__all__ = []