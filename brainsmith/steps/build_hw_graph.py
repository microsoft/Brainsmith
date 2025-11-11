# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build hardware graph step combining partitioning and specialization.

This step combines two critical phases of the dataflow compilation pipeline:
1. Dataflow partitioning: Separates hardware-accelerated nodes into isolated subgraphs
2. Backend specialization: Converts generic kernel nodes to HLS/RTL implementations

The combined step simplifies the blueprint configuration and ensures proper
sequencing of these tightly-coupled transformations.
"""

import logging
import os
from typing import Any

from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.util.basic import getHWCustomOp
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import (
    ApplyConfig,
    GiveUniqueNodeNames,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.config import extract_model_config_to_json

from brainsmith.primitives.transforms.specialize_kernels import SpecializeKernels
from brainsmith.registry import step

logger = logging.getLogger(__name__)


@step(name='build_hw_graph')
def build_hw_graph(model: Any, cfg: Any) -> Any:
    """Build complete hardware dataflow graph via partitioning + specialization.

    This step combines create_dataflow_partition and specialize_layers into a
    unified transformation that:

    1. **Partitioning Phase**: Separates consecutive groups of HWCustomOp nodes
       into StreamingDataflowPartition nodes, which point to separate ONNX files.
       Only dataflow accelerator synthesis can be performed on these HW subgraphs.

    2. **Specialization Phase**: Converts generic hardware kernel nodes to
       specialized backend implementations (HLS or RTL) based on kernel_selections
       config and constraint checking.

    The step handles both Brainsmith KernelOp nodes and legacy FINN HWCustomOp nodes,
    ensuring compatibility with mixed graphs.

    Args:
        model: ModelWrapper containing the ONNX model with hardware kernel nodes
        cfg: Build configuration with:
            - output_dir: Output directory for intermediate models and configs
            - kernel_selections: Backend priority lists for specialization
            - specialize_layers_config_file: Optional user config for manual overrides

    Returns:
        ModelWrapper containing the specialized dataflow partition model

    Blueprint usage:
        steps:
          - build_dataflow_graph      # Infer kernels first
          - build_hw_graph            # Combined partitioning + specialization
          - apply_folding_config      # Then apply parallelization

    Implementation notes:
        - Creates template_specialize_layers_config.json for user reference
        - Supports single StreamingDataflowPartition only (FINN limitation)
        - Returns the dataflow partition model, not the parent model
        - Saves parent model to intermediate_models/dataflow_parent.onnx if enabled
    """
    logger.debug("Building hardware dataflow graph (partitioning + specialization)...")

    # ========================================================================
    # Phase 1: Create Dataflow Partition
    # ========================================================================

    logger.debug("Phase 1: Creating dataflow partition...")

    partition_dir = os.path.join(
        cfg.output_dir,
        "intermediate_models",
        "supported_op_partitions"
    )

    # Use FINN's CreateDataflowPartition to separate HW nodes
    parent_model = model.transform(CreateDataflowPartition(partition_model_dir=partition_dir))

    # Extract the dataflow partition model
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")

    if len(sdp_nodes) == 0:
        logger.error("No StreamingDataflowPartition nodes found after partitioning")
        logger.error("")
        logger.error("This typically means one or more nodes failed to be converted to hardware:")
        logger.error("  1. Kernel inference failed - ONNX nodes were not matched to any kernel")
        logger.error("     → Check that kernels are listed in blueprint design_space.kernels")
        logger.error("     → Verify nodes are supported by the selected kernels")
        logger.error("  2. Backend specialization failed - kernels lack viable backend implementations")
        logger.error("     → Check that backends are configured in kernel_selections")
        logger.error("     → Verify RTL backend constraints are satisfied (see SpecializeKernels)")
        logger.error("")
        logger.error("Debug steps:")
        logger.error("  - Inspect intermediate_models/ to see which nodes remain")
        logger.error("  - Check logs for kernel inference warnings")
        logger.error("  - Verify all ONNX ops have corresponding kernel transforms")
        raise RuntimeError(
            "No hardware dataflow partition created. "
            "One or more nodes failed kernel inference or backend specialization. "
            "See logs above for details."
        )

    if len(sdp_nodes) > 1:
        logger.warning(
            f"Found {len(sdp_nodes)} StreamingDataflowPartition nodes. "
            "Only single partition is officially supported by FINN."
        )

    # Get the dataflow partition model file
    sdp_node = sdp_nodes[0]
    sdp_node_inst = getHWCustomOp(sdp_node, parent_model)
    dataflow_model_filename = sdp_node_inst.get_nodeattr("model")

    logger.debug(f"Dataflow partition extracted: {dataflow_model_filename}")

    # Save parent model if requested
    if cfg.save_intermediate_models:
        parent_model_path = os.path.join(
            cfg.output_dir,
            "intermediate_models",
            "dataflow_parent.onnx"
        )
        parent_model.save(parent_model_path)
        logger.debug(f"Saved parent model: {parent_model_path}")

    # Load the dataflow partition for specialization
    model = ModelWrapper(dataflow_model_filename)

    # Create template config for user reference
    template_config_path = os.path.join(
        cfg.output_dir,
        "template_specialize_layers_config.json"
    )
    extract_model_config_to_json(
        model,
        template_config_path,
        ["preferred_impl_style"]
    )
    logger.debug(f"Created template config: {template_config_path}")

    # ========================================================================
    # Phase 2: Specialize Layers
    # ========================================================================

    logger.debug("Phase 2: Specializing hardware layers...")

    # Apply user config if provided (manual overrides)
    if cfg.specialize_layers_config_file is not None:
        logger.debug(f"Applying user config: {cfg.specialize_layers_config_file}")
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(ApplyConfig(cfg.specialize_layers_config_file))

    # Run registry-based backend specialization
    logger.debug("Running registry-based backend specialization...")
    model = model.transform(SpecializeKernels(cfg))

    # Clean up and infer properties
    logger.debug("Running cleanup transformations...")
    for transform in [
        GiveUniqueNodeNames(),
        InferShapes(),
        InferDataTypes()
    ]:
        model = model.transform(transform)

    return model
