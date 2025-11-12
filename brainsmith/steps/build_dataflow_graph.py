# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build dataflow graph steps for hardware mapping.

This module provides three step variants for dataflow graph construction:

1. build_dataflow_graph: Combined step (backward compatible) - runs both phases
2. insert_infrastructure_kernels: Phase 1 only - topology-based infrastructure insertion
3. infer_computational_kernels: Phase 2 only - pattern-based computational inference

The split steps enable finer control over the build pipeline, while the combined
step remains available for simpler blueprints.
"""
import logging
from typing import Any

from qonnx.transformation.general import GiveUniqueNodeNames

from brainsmith.primitives.transforms import InferKernels, InsertInfrastructureKernels
from brainsmith.registry import get_component_metadata, get_kernel, step

logger = logging.getLogger(__name__)


@step(name="build_dataflow_graph")
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
    kernel_selections = getattr(cfg, "kernel_selections", None)
    if not kernel_selections:
        logger.debug("No kernel selections configured, skipping inference")
        return model

    logger.debug(f"Processing {len(kernel_selections)} kernel(s)...")

    # Split kernel classes into infrastructure and computational
    infrastructure_kernels = []
    computational_kernels = []

    for kernel_name, _ in kernel_selections:
        try:
            kernel_class = get_kernel(kernel_name)
            metadata = get_component_metadata(kernel_name, "kernel")

            if metadata.is_infrastructure:
                infrastructure_kernels.append(kernel_class)
                logger.debug(f"  {kernel_name} (infrastructure)")
            else:
                computational_kernels.append(kernel_class)
                logger.debug(f"  {kernel_name} (computational)")
        except KeyError:
            logger.error(f"  Kernel not found in registry: {kernel_name}")

    # Phase 1: Insert infrastructure kernels via topology analysis
    if infrastructure_kernels:
        logger.debug(f"Inserting {len(infrastructure_kernels)} infrastructure kernel(s)...")
        model = model.transform(InsertInfrastructureKernels(infrastructure_kernels))

    # Phase 2: Infer computational kernels via pattern matching
    if computational_kernels:
        logger.debug(f"Inferring {len(computational_kernels)} computational kernel(s)...")
        model = model.transform(InferKernels(computational_kernels))

    # Ensure all nodes have unique names after graph construction
    # Some legacy FINN transforms (e.g., InferElementwiseBinaryOperation) create
    # nodes without names, which causes issues in downstream steps like partitioning
    model = model.transform(GiveUniqueNodeNames())
    logger.debug("Assigned unique names to all nodes after dataflow graph construction")

    return model


@step(name='insert_infrastructure_kernels')
def insert_infrastructure_kernels_step(model: Any, cfg: Any) -> Any:
    """Insert infrastructure kernels via topology analysis (Phase 1 of dataflow graph build).

    Infrastructure kernels are inserted based on graph topology and connectivity patterns,
    rather than pattern matching. Examples include:
    - DuplicateStreams (for fan-out)
    - FIFOs (for buffering)
    - AddStreams (for fan-in)

    This step extracts infrastructure kernels from cfg.kernel_selections (those with
    is_infrastructure=True metadata) and applies InsertInfrastructureKernels transform.

    Use this step when you want finer control over the build pipeline, running
    infrastructure insertion separately from computational kernel inference.

    Args:
        model: ONNX model to transform
        cfg: Build configuration with kernel_selections attribute

    Returns:
        Transformed model with infrastructure kernels inserted

    Blueprint usage:
        steps:
          - insert_infrastructure_kernels  # Phase 1: topology-based insertion
          - infer_computational_kernels    # Phase 2: pattern-based inference

    See also:
        - build_dataflow_graph: Combined step that runs both phases
        - infer_computational_kernels: Phase 2 only
    """
    kernel_selections = getattr(cfg, 'kernel_selections', None)
    if not kernel_selections:
        logger.debug("No kernel selections configured, skipping infrastructure insertion")
        return model

    logger.debug(f"Processing {len(kernel_selections)} kernel selection(s)...")

    # Extract only infrastructure kernels
    infrastructure_kernels = []

    for kernel_name, _ in kernel_selections:
        try:
            kernel_class = get_kernel(kernel_name)
            metadata = get_component_metadata(kernel_name, 'kernel')

            if metadata.is_infrastructure:
                infrastructure_kernels.append(kernel_class)
                logger.debug(f"  {kernel_name} (infrastructure)")
        except KeyError:
            logger.error(f"  Kernel not found in registry: {kernel_name}")

    # Insert infrastructure kernels via topology analysis
    if infrastructure_kernels:
        logger.debug(f"Inserting {len(infrastructure_kernels)} infrastructure kernel(s)...")
        model = model.transform(InsertInfrastructureKernels(infrastructure_kernels))
    else:
        logger.debug("No infrastructure kernels selected, skipping insertion")

    return model


@step(name='infer_computational_kernels')
def infer_computational_kernels_step(model: Any, cfg: Any) -> Any:
    """Infer computational kernels via pattern matching (Phase 2 of dataflow graph build).

    Computational kernels are inferred by matching ONNX node patterns against kernel
    transform patterns. Examples include:
    - MatMul → MVAU
    - LayerNorm → LayerNorm_hls
    - Transpose → Shuffle
    - Add/Mul → ElementwiseBinaryOp

    This step extracts computational kernels from cfg.kernel_selections (those with
    is_infrastructure=False metadata) and applies InferKernels transform.

    Use this step when you want finer control over the build pipeline, running
    computational inference separately from infrastructure insertion.

    Args:
        model: ONNX model to transform
        cfg: Build configuration with kernel_selections attribute

    Returns:
        Transformed model with computational kernels inferred and unique node names

    Blueprint usage:
        steps:
          - insert_infrastructure_kernels  # Phase 1: topology-based insertion
          - infer_computational_kernels    # Phase 2: pattern-based inference

    Implementation notes:
        - Applies GiveUniqueNodeNames after inference to fix legacy FINN transforms
        - Some FINN transforms (e.g., InferElementwiseBinaryOperation) create nodes
          without names, which causes issues in downstream partitioning

    See also:
        - build_dataflow_graph: Combined step that runs both phases
        - insert_infrastructure_kernels: Phase 1 only
    """
    kernel_selections = getattr(cfg, 'kernel_selections', None)
    if not kernel_selections:
        logger.debug("No kernel selections configured, skipping kernel inference")
        return model

    logger.debug(f"Processing {len(kernel_selections)} kernel selection(s)...")

    # Extract only computational kernels
    computational_kernels = []

    for kernel_name, _ in kernel_selections:
        try:
            kernel_class = get_kernel(kernel_name)
            metadata = get_component_metadata(kernel_name, 'kernel')

            if not metadata.is_infrastructure:
                computational_kernels.append(kernel_class)
                logger.debug(f"  {kernel_name} (computational)")
        except KeyError:
            logger.error(f"  Kernel not found in registry: {kernel_name}")

    # Infer computational kernels via pattern matching
    if computational_kernels:
        logger.debug(f"Inferring {len(computational_kernels)} computational kernel(s)...")
        model = model.transform(InferKernels(computational_kernels))
    else:
        logger.debug("No computational kernels selected, skipping inference")

    # Ensure all nodes have unique names after graph construction
    # Some legacy FINN transforms (e.g., InferElementwiseBinaryOperation) create
    # nodes without names, which causes issues in downstream steps like partitioning
    model = model.transform(GiveUniqueNodeNames())
    logger.debug("Assigned unique names to all nodes after computational kernel inference")

    return model
