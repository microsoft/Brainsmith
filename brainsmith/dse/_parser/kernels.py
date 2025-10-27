# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Blueprint kernel parsing.

Handles parsing of kernel specifications and backend resolution.
"""

from typing import List, Tuple, Type

from brainsmith.registry import list_backends_for_kernel, get_backend


def parse_kernels(kernels_data: List[str]) -> List[Tuple[str, List[Type]]]:
    """Parse kernels section.

    Each kernel in the blueprint automatically uses ALL registered backends.
    Backend selection/filtering is a planned future feature.

    Args:
        kernels_data: List of kernel names (e.g., ['LayerNorm', 'Crop'])

    Returns:
        List of (kernel_name, backend_classes) tuples where backend_classes
        contains ALL registered backends for that kernel

    Raises:
        ValueError: If kernel spec is not a string (dict format not supported)
        ValueError: If no backends found for kernel

    Example:
        >>> parse_kernels(['LayerNorm', 'Crop'])
        [('LayerNorm', [<class 'LayerNorm_hls'>, <class 'LayerNorm_rtl'>]),
         ('Crop', [<class 'Crop_hls'>])]

    Future Enhancement:
        Support backend filtering:
        - {'LayerNorm': ['hls']} → only HLS backends
        - {'LayerNorm': ['LayerNorm_hls_optimized']} → specific backend
    """
    kernel_backends = []

    for spec in kernels_data:
        # Only accept kernel name strings
        if not isinstance(spec, str):
            raise ValueError(
                f"Kernel spec must be a string, got {type(spec).__name__}: {spec}\n"
                f"Backend selection is not yet supported. Use kernel name only.\n"
                f"Example: kernels: ['LayerNorm', 'Crop']"
            )

        kernel_name = spec

        # Get ALL registered backends for this kernel
        backend_names = list_backends_for_kernel(kernel_name)

        if not backend_names:
            # TODO: Future enhancement - allow kernels with no backends
            # for abstract kernel definitions
            continue

        # Resolve backend names to classes
        try:
            backend_classes = [get_backend(name) for name in backend_names]
        except KeyError as e:
            raise ValueError(
                f"Backend not found for kernel '{kernel_name}': {e}"
            ) from e

        kernel_backends.append((kernel_name, backend_classes))

    return kernel_backends
