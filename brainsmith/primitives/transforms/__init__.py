# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Brainsmith Transforms

Graph transformation utilities for ONNX model optimization and compilation.

Provides QONNX Transformation classes for various graph rewrites including
normalization expansion, hardware-specific optimizations, and metadata extraction.
"""

# Explicit imports for public API
from .expand_norms import ExpandNorms
from .extract_shell_integration_metadata import ExtractShellIntegrationMetadata
from .infer_kernel import InferKernel
from .infer_kernels import InferKernels
from .insert_duplicate_streams import InsertDuplicateStreams
from .insert_infrastructure_kernels import InsertInfrastructureKernels
from .normalize_dataflow_layouts import NormalizeDataflowLayouts
from .parallelization import (
    ApplyParallelizationConfig,
    SetParallelization,
    get_node_interface,
    get_parallelization,
    set_parallelization,
)
from .refresh_design_points import RefreshKernelDesignPoints
from .set_pumped_compute import SetPumpedCompute
from .specialize_kernels import SpecializeKernels
from .temp_shuffle_fixer import TempShuffleFixer

# Backward compatibility alias
InferKernelList = InferKernels

__all__ = [
    "ExpandNorms",
    "SetPumpedCompute",
    "TempShuffleFixer",
    "ExtractShellIntegrationMetadata",
    "InferKernel",
    "InferKernels",
    "InferKernelList",  # Backward compatibility alias
    "SpecializeKernels",
    "NormalizeDataflowLayouts",
    "InsertDuplicateStreams",
    "InsertInfrastructureKernels",
    "RefreshKernelDesignPoints",
    "ApplyParallelizationConfig",
    "SetParallelization",
    "get_node_interface",
    "set_parallelization",
    "get_parallelization",
]
