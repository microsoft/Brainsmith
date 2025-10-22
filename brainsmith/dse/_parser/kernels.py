# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Blueprint kernel parsing.

Handles parsing of kernel specifications and backend resolution.
"""

from typing import Any, List, Optional, Tuple, Type

from brainsmith.loader import list_backends_for_kernel, get_backend


def _extract_kernel_spec(spec) -> Tuple[str, Optional[List[str]]]:
    """Extract kernel name and optional backend names from spec.

    Args:
        spec: Kernel specification (string or dict)

    Returns:
        Tuple of (kernel_name, backend_names)

    Raises:
        ValueError: If kernel spec is invalid
    """
    if isinstance(spec, str):
        return spec, None
    elif isinstance(spec, dict) and len(spec) == 1:
        kernel_name, backend_specs = next(iter(spec.items()))
        backend_names = backend_specs if isinstance(backend_specs, list) else [backend_specs]
        return kernel_name, backend_names
    else:
        raise ValueError(f"Invalid kernel spec: {spec}")


def parse_kernels(kernels_data: List[Any]) -> List[Tuple[str, List[Type]]]:
    """Parse kernels section.

    Supports three specification formats:
    - String: 'LayerNorm' → all backends for kernel
    - Dict with language: {'LayerNorm': 'hls'} → HLS backends only
    - Dict with backend name: {'LayerNorm': 'LayerNorm_HLS'} → specific backend

    Args:
        kernels_data: Raw kernels data from blueprint YAML

    Returns:
        List of (kernel_name, backend_classes) tuples

    Raises:
        ValueError: If backend is not found in registry
    """
    kernel_backends = []

    for spec in kernels_data:
        kernel_name, backend_specs = _extract_kernel_spec(spec)

        if not backend_specs:
            # No backends specified - get all for this kernel
            backend_names = list_backends_for_kernel(kernel_name)
        else:
            # Specific backends/languages specified
            backend_names = []
            for spec in backend_specs:
                if spec in ('hls', 'rtl'):
                    # Language filter - use loader API
                    backend_names.extend(
                        list_backends_for_kernel(kernel_name, language=spec)
                    )
                else:
                    # Specific backend name - use directly
                    backend_names.append(spec)

        if not backend_names:
            continue

        # Get backend classes (loader handles name resolution)
        try:
            backend_classes = [get_backend(name) for name in backend_names]
        except KeyError as e:
            raise ValueError(
                f"Backend not found for kernel '{kernel_name}': {e}"
            ) from e

        kernel_backends.append((kernel_name, backend_classes))

    return kernel_backends

