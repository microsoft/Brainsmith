# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build dataflow graph step for hardware mapping.

This step orchestrates the complete dataflow graph construction through two phases:
1. Infrastructure kernels: Inserted via topology analysis (InsertInfrastructureKernels)
2. Computational kernels: Inferred via pattern matching (InferKernels)

The step automatically splits kernel_selections into these two categories based
on the is_infrastructure metadata flag, then dispatches to the appropriate transform.
"""
import logging
from typing import Any

from brainsmith.registry import get_kernel, get_component_metadata, step
from brainsmith.primitives.transforms import InferKernels, InsertInfrastructureKernels
from qonnx.transformation.general import GiveUniqueNodeNames

logger = logging.getLogger(__name__)


@step(name='build_dataflow_graph')
def build_dataflow_graph(model: Any, cfg: Any) -> Any:
    """Build complete dataflow graph from kernel selections (two-phase workflow).

    Extracts kernel classes from cfg.kernel_selections and splits them into:
    1. Infrastructure kernels (is_infrastructure=True) → InsertInfrastructureKernels
    2. Computational kernels (is_infrastructure=False) → InferKernels

    This two-phase approach ensures infrastructure nodes (DuplicateStreams, FIFO, etc.)
    are inserted first via topology analysis, then computational nodes are pattern-matched.

    Args:
        model: ONNX model to transform
        cfg: Build configuration with kernel_selections attribute

    Returns:
        Transformed model with complete dataflow graph (infrastructure + computational kernels)
    """
    kernel_selections = getattr(cfg, 'kernel_selections', None)
    if not kernel_selections:
        logger.debug("No kernel selections configured, skipping inference")
        return model

    logger.info(f"Processing {len(kernel_selections)} kernel(s)...")

    # Split kernel classes into infrastructure and computational
    infrastructure_kernels = []
    computational_kernels = []

    for kernel_name, _ in kernel_selections:
        try:
            kernel_class = get_kernel(kernel_name)
            metadata = get_component_metadata(kernel_name, 'kernel')

            if metadata.is_infrastructure:
                infrastructure_kernels.append(kernel_class)
                logger.info(f"  {kernel_name} (infrastructure)")
            else:
                computational_kernels.append(kernel_class)
                logger.info(f"  {kernel_name} (computational)")
        except KeyError:
            logger.warning(f"  Kernel not found in registry: {kernel_name}")

    # Phase 1: Insert infrastructure kernels via topology analysis
    if infrastructure_kernels:
        logger.info(f"Inserting {len(infrastructure_kernels)} infrastructure kernel(s)...")
        model = model.transform(InsertInfrastructureKernels(infrastructure_kernels))

    # Phase 2: Infer computational kernels via pattern matching
    if computational_kernels:
        logger.info(f"Inferring {len(computational_kernels)} computational kernel(s)...")
        model = model.transform(InferKernels(computational_kernels))

    # Ensure all nodes have unique names after graph construction
    # Some legacy FINN transforms (e.g., InferElementwiseBinaryOperation) create
    # nodes without names, which causes issues in downstream steps like partitioning
    model = model.transform(GiveUniqueNodeNames())
    logger.debug("Assigned unique names to all nodes after dataflow graph construction")

    return model
