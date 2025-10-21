# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Utility functions for working with transforms.
"""

from typing import Any, List


def apply_transforms(model: Any, transforms: List[Any]) -> Any:
    """Apply instantiated transforms in sequence.

    Args:
        model: The model to transform (typically QONNX ModelWrapper)
        transforms: List of instantiated transform objects

    Returns:
        The transformed model

    Example:
        from qonnx.transformation.fold_constants import FoldConstants
        from qonnx.transformation.general import RemoveIdentityOps

        model = apply_transforms(model, [
            FoldConstants(),
            RemoveIdentityOps()
        ])
    """
    for transform in transforms:
        model = model.transform(transform)
    return model
