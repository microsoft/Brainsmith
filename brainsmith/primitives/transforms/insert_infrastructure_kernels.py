############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################

"""InsertInfrastructureKernels: Dynamic infrastructure kernel insertion.

This transform maps infrastructure kernel classes to their insertion transforms
and applies them to the model. Unlike computational kernels (which are pattern-matched
from ONNX nodes), infrastructure kernels are inserted based on topology analysis.

Example usage:
    from brainsmith.primitives.transforms import InsertInfrastructureKernels
    from brainsmith.kernels.duplicate_streams import DuplicateStreams

    # Insert specific infrastructure kernels
    model = model.transform(InsertInfrastructureKernels([DuplicateStreams]))

    # Or from registry lookup
    from brainsmith.registry import get_kernel
    kernel_class = get_kernel("DuplicateStreams")
    model = model.transform(InsertInfrastructureKernels([kernel_class]))
"""

import logging
from typing import List, Type, Dict

from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper

logger = logging.getLogger(__name__)


# Static mapping of infrastructure kernels to their insertion transforms
# This will be imported and used to resolve transforms
INFRASTRUCTURE_TRANSFORM_MAP: Dict[str, Type[Transformation]] = {}


def _register_infrastructure_transform(kernel_name: str, transform_cls: Type[Transformation]):
    """Register a kernel → transform mapping.

    Args:
        kernel_name: Name of the infrastructure kernel (e.g., "DuplicateStreams")
        transform_cls: Transform class that inserts this kernel
    """
    INFRASTRUCTURE_TRANSFORM_MAP[kernel_name] = transform_cls


class InsertInfrastructureKernels(Transformation):
    """Insert infrastructure kernels based on topology analysis.

    This transform takes a list of infrastructure kernel classes and runs their
    corresponding insertion transforms. The kernel → transform mapping is maintained
    in INFRASTRUCTURE_TRANSFORM_MAP.

    Infrastructure kernels are inserted based on graph topology analysis (e.g.,
    detecting fanout for DuplicateStreams), not pattern matching like computational
    kernels.

    Args:
        kernel_classes: List of infrastructure kernel classes to insert.
                       Each must have a registered insertion transform.

    Example:
        from brainsmith.kernels.duplicate_streams import DuplicateStreams

        # Insert DuplicateStreams nodes where needed
        model = model.transform(InsertInfrastructureKernels([DuplicateStreams]))

    Notes:
        - Only specified kernels are inserted (no auto-discovery)
        - Insertion transforms are run in the order provided
        - Each insertion transform decides WHERE to insert based on topology
        - If no matching transform found, warning logged and kernel skipped
    """

    def __init__(self, kernel_classes: List[Type]):
        """Initialize with list of infrastructure kernel classes.

        Args:
            kernel_classes: List of infrastructure kernel classes to insert
        """
        super().__init__()
        self.kernel_classes = kernel_classes or []

    def apply(self, model: ModelWrapper):
        """Apply infrastructure insertion transforms.

        For each kernel class:
        1. Look up kernel name
        2. Find corresponding insertion transform
        3. Run the transform
        4. Aggregate graph_modified flags

        Args:
            model: QONNX ModelWrapper to transform

        Returns:
            Tuple of (transformed_model, graph_modified_flag)
        """
        graph_modified = False

        if not self.kernel_classes:
            logger.debug("No infrastructure kernels to insert")
            return (model, False)

        logger.info(f"Inserting {len(self.kernel_classes)} infrastructure kernel(s)")

        for kernel_cls in self.kernel_classes:
            # Get kernel name for lookup
            kernel_name = getattr(kernel_cls, '__name__', str(kernel_cls))

            # Look up insertion transform
            transform_cls = INFRASTRUCTURE_TRANSFORM_MAP.get(kernel_name)

            if transform_cls is None:
                logger.warning(
                    f"No insertion transform registered for infrastructure kernel: {kernel_name}. "
                    f"Register with _register_infrastructure_transform() or add to INFRASTRUCTURE_TRANSFORM_MAP."
                )
                continue

            # Run insertion transform
            logger.debug(f"Running {transform_cls.__name__} for {kernel_name}")
            transform = transform_cls()
            model, modified = transform.apply(model)
            graph_modified = graph_modified or modified

            if modified:
                logger.info(f"  ✓ {kernel_name} inserted")
            else:
                logger.debug(f"  - {kernel_name} not needed (no insertion)")

        return (model, graph_modified)


# Register built-in infrastructure transforms
def _register_builtin_transforms():
    """Register Brainsmith's built-in infrastructure transforms."""
    try:
        from .insert_duplicate_streams import InsertDuplicateStreams
        _register_infrastructure_transform("DuplicateStreams", InsertDuplicateStreams)
        logger.debug("Registered DuplicateStreams → InsertDuplicateStreams")
    except ImportError as e:
        logger.warning(f"Could not register DuplicateStreams transform: {e}")


# Auto-register on module import
_register_builtin_transforms()
