# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Layout Normalization Build Step

Provides a preprocessing step that normalizes all tensor layouts to NHWC
(channel-last) format for dataflow acceleration. This eliminates the need
for per-kernel layout checking and ensures consistent channel-last layout
throughout the dataflow region.
"""

import logging
from typing import Any

from brainsmith.registry import step
from brainsmith.transforms.normalize_dataflow_layouts import NormalizeDataflowLayouts

logger = logging.getLogger(__name__)


@step(name="normalize_dataflow_layouts")
def normalize_dataflow_layouts_step(model: Any, cfg: Any) -> Any:
    """
    Normalize all tensor layouts to NHWC (channel-last).

    This preprocessing step converts all NCHW (channel-first) tensors in the graph
    to NHWC (channel-last) layout by inserting Transpose nodes. This ensures that
    all subsequent dataflow kernel operations can assume channel-last layout without
    individual layout checks.

    The transformation preserves the original layout contract for graph outputs by
    inserting reverse Transposes where needed.

    Args:
        model: ONNX model wrapper
        cfg: Build configuration (unused by this step)

    Returns:
        model: Transformed model with normalized layouts

    Usage in blueprint:
        steps:
          - "normalize_dataflow_layouts"  # Add before kernel inference
          - "infer_kernels"
          - ...
    """
    logger.info("Normalizing dataflow layouts to NHWC (channel-last)")

    # Apply the transformation (transforms are primitives, use direct import)
    model = model.transform(NormalizeDataflowLayouts())

    logger.info("Layout normalization complete")

    return model
