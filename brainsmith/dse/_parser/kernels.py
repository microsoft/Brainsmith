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

    Args:
        kernels_data: Raw kernels data from blueprint YAML

    Returns:
        List of (kernel_name, backend_classes) tuples

    Raises:
        ValueError: If backend is not found in registry
    """
    kernel_backends = []

    for spec in kernels_data:
        kernel_name, backend_names = _extract_kernel_spec(spec)

        # If no backends specified, get all available
        if not backend_names:
            backend_names = list_backends_for_kernel(kernel_name)

        # Skip if no backends available
        if not backend_names:
            continue

        # Resolve backend classes
        backend_classes = []
        for backend_spec in backend_names:
            # Backend spec could be just language ('hls') or full name ('LayerNormHLS')
            # Try to extract language from the name
            language = backend_spec.lower()
            if language not in ['hls', 'rtl']:
                # Full backend name like 'LayerNormHLS' - extract language
                if backend_spec.endswith('HLS') or backend_spec.endswith('_hls'):
                    language = 'hls'
                elif backend_spec.endswith('RTL') or backend_spec.endswith('_rtl'):
                    language = 'rtl'
                else:
                    raise ValueError(f"Cannot determine language from backend name '{backend_spec}'")

            try:
                # Get all backends matching this kernel + language
                matching_backends = list_backends_for_kernel(kernel_name, language=language)
                if not matching_backends:
                    raise ValueError(f"No {language} backends found for kernel '{kernel_name}'")

                # Get the backend class (take first match if multiple)
                backend_class = get_backend(matching_backends[0])
                backend_classes.append(backend_class)
            except KeyError as e:
                raise ValueError(f"Backend not found for kernel '{kernel_name}', language '{language}': {e}") from e

        kernel_backends.append((kernel_name, backend_classes))

    return kernel_backends

