"""Generate and manage plugin manifests for fast discovery.

This module provides functionality to generate plugin manifests by scanning
installed packages (FINN, QONNX, etc.) and extracting component metadata
using runtime introspection. This allows for fast plugin discovery during
CLI invocations without importing all components.

The manifest generation imports package modules, introspects classes and
functions to discover kernels/backends/steps, then caches the metadata in
YAML files with version hashing for automatic invalidation on upgrades.

Manifest generation is slow (~6 seconds for FINN) but only runs once:
- During installation (post-install hook)
- When package version changes (automatic)
- Manual rebuild: `smith cache manifests rebuild`

Once generated, manifests enable fast CLI startup (~100ms) via lazy loading.
"""

import ast
import hashlib
import importlib.metadata
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# Manifest cache directory
def get_cache_dir() -> Path:
    """Get the manifest cache directory.

    Manifests are stored at {project_dir}/plugins/ to colocate with
    user plugin code. This allows multiple projects to have different
    dependency versions without conflicts.

    Returns:
        Path to {project_dir}/plugins/
    """
    from brainsmith.settings import get_config

    cache_dir = get_config().project_dir / 'plugins'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_manifest_path(package_name: str) -> Path:
    """Get the manifest file path for a package.

    Args:
        package_name: Name of the package (e.g., 'finn', 'qonnx')

    Returns:
        Path to the manifest YAML file
    """
    return get_cache_dir() / f"{package_name}.yaml"


def get_package_version(package_name: str) -> str:
    """Get the installed version of a package.

    Args:
        package_name: Name of the package

    Returns:
        Version string (e.g., '0.1.0')

    Raises:
        importlib.metadata.PackageNotFoundError: If package not installed
    """
    return importlib.metadata.version(package_name)


def compute_version_hash(package_name: str) -> str:
    """Compute a hash of the package version for informational purposes.

    Stored in manifests to track which package version was used during generation.
    Not used for automatic cache invalidation.

    Args:
        package_name: Name of the package

    Returns:
        SHA256 hash of the version string (first 16 chars)
    """
    try:
        version = get_package_version(package_name)
        return hashlib.sha256(version.encode()).hexdigest()[:16]
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def is_manifest_valid(package_name: str) -> bool:
    """Check if a manifest exists.

    Note: Version hash is stored in manifests for information only.
    Use eager_plugin_discovery=true or manual regeneration to update manifests.

    Args:
        package_name: Name of the package

    Returns:
        True if manifest file exists
    """
    manifest_path = get_manifest_path(package_name)
    return manifest_path.exists()


# ===== Introspection Helper Functions =====

def _find_infer_transform(kernel_name: str) -> Optional[str]:
    """Find InferTransform for a kernel.

    Maps kernel names to their corresponding InferTransform classes.
    This mapping is based on FINN's convert_to_hw_layers.py module.

    Args:
        kernel_name: Name of the kernel class

    Returns:
        Full path to InferTransform class, or None if no transform exists
    """
    # Mapping from kernel name to InferTransform
    # Based on finn/transformation/fpgadataflow/convert_to_hw_layers.py
    infer_map = {
        'MVAU': 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferQuantizedMatrixVectorActivation',
        'VVAU': 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferVectorVectorActivation',
        'Thresholding': 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferThresholdingLayer',
        'ConvolutionInputGenerator': 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferConvInpGen',
        'GlobalAccPool': 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferGlobalAccPoolLayer',
        'StreamingEltwise': 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferStreamingEltwise',
        'ChannelwiseOp': 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferChannelwiseLinearLayer',
        'Pool': 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferPool',
        'Lookup': 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferLookupLayer',
        'LabelSelect': 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferLabelSelectLayer',
        'AddStreams': 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferAddStreamsLayer',
        'DuplicateStreams': 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferDuplicateStreamsLayer',
        'StreamingConcat': 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferConcatLayer',
        'StreamingSplit': 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferSplitLayer',
        'UpsampleNearestNeighbour': 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferUpsample',
        'ElementwiseBinaryOperation': 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferElementwiseBinaryOperation',
    }
    return infer_map.get(kernel_name)


def _infer_target_kernel(backend_class: type, backend_name: str, language: str) -> str:
    """Infer target kernel from backend parent classes.

    Requires backend to inherit from kernel class (e.g., class MVAU_hls(MVAU, HLSBackend)).
    Fails hard if parent class doesn't reveal kernel.

    Args:
        backend_class: The backend class object
        backend_name: Name of the backend class
        language: 'hls' or 'rtl'

    Returns:
        Full kernel name with source prefix (e.g., 'finn:MVAU')

    Raises:
        ValueError: If target kernel cannot be inferred from parent classes
    """
    # Strategy: Check parent classes for kernel
    for base in backend_class.__bases__:
        base_module = getattr(base, '__module__', '')
        base_name = getattr(base, '__name__', '')

        if base_module.startswith('finn.custom_op.fpgadataflow') and \
           not base_name.endswith('Backend') and \
           base_name not in ('HWCustomOp', 'CustomOp'):
            return f'finn:{base_name}'

    # No fallback - fail hard with clear error message
    raise ValueError(
        f"Cannot infer target kernel for backend '{backend_name}'. "
        f"Backend must inherit from kernel class (e.g., class {backend_name}(KernelClass, {language.upper()}Backend)). "
        f"Parent classes: {[b.__name__ for b in backend_class.__bases__]}"
    )


def _scan_backend_dir(backend_dir: Path, base_class: type, language: str) -> List[Dict[str, Any]]:
    """Scan a backend directory for backend classes.

    Args:
        backend_dir: Path to backend directory (hls or rtl)
        base_class: Base class to filter by (HLSBackend or RTLBackend)
        language: 'hls' or 'rtl'

    Returns:
        List of backend metadata dicts
    """
    import importlib
    import inspect

    backends = []
    exclude = {'__init__.py'}

    if not backend_dir.exists():
        logger.debug(f"Backend directory does not exist: {backend_dir}")
        return backends

    for py_file in backend_dir.glob('*.py'):
        if py_file.name in exclude:
            continue

        module_name = f'finn.custom_op.fpgadataflow.{language}.{py_file.stem}'
        try:
            module = importlib.import_module(module_name)

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (obj.__module__ == module.__name__ and
                    issubclass(obj, base_class) and
                    obj is not base_class):

                    # Infer target kernel from class inheritance or name
                    target_kernel = _infer_target_kernel(obj, name, language)

                    backends.append({
                        'name': name,
                        'class_path': f'{module.__name__}.{name}',
                        'infer_path': None,
                        'params': {
                            'target_kernel': target_kernel,
                            'language': language
                        }
                    })
                    logger.debug(f"Found backend: {name} -> {target_kernel}")

        except Exception as e:
            logger.warning(f"Failed to scan {module_name}: {e}")

    return backends


# ===== Main Scanning Functions =====

def _scan_finn_kernels() -> List[Dict[str, Any]]:
    """Scan FINN for HWCustomOp subclasses (kernels)."""
    import importlib
    import inspect
    from pathlib import Path

    try:
        from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
    except ImportError:
        logger.warning("FINN not available, skipping kernel scan")
        return []

    kernels = []

    try:
        finn_root = Path(importlib.import_module('finn').__file__).parent
        kernel_dir = finn_root / 'custom_op' / 'fpgadataflow'

        # Exclude base classes and non-kernel files
        exclude = {'__init__.py', 'hwcustomop.py', 'hlsbackend.py', 'rtlbackend.py', 'templates.py',
                   'streamingdataflowpartition.py'}

        for py_file in kernel_dir.glob('*.py'):
            if py_file.name in exclude:
                continue

            module_name = f'finn.custom_op.fpgadataflow.{py_file.stem}'
            try:
                module = importlib.import_module(module_name)

                # Find HWCustomOp subclasses
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (obj.__module__ == module.__name__ and
                        issubclass(obj, HWCustomOp) and
                        obj is not HWCustomOp):

                        # Find associated InferTransform
                        infer_path = _find_infer_transform(name)

                        kernels.append({
                            'name': name,
                            'class_path': f'{module.__name__}.{name}',
                            'infer_path': infer_path,
                            'params': {}
                        })
                        logger.debug(f"Found kernel: {name}")

            except Exception as e:
                logger.warning(f"Failed to scan {module_name}: {e}")

        logger.info(f"Discovered {len(kernels)} FINN kernels")

    except Exception as e:
        logger.error(f"Failed to scan FINN kernels: {e}")

    return kernels


def _scan_finn_backends() -> List[Dict[str, Any]]:
    """Scan FINN for HLS/RTL backend classes."""
    import importlib
    from pathlib import Path

    try:
        from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
        from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
    except ImportError:
        logger.warning("FINN not available, skipping backend scan")
        return []

    backends = []

    try:
        finn_root = Path(importlib.import_module('finn').__file__).parent

        # Scan HLS backends
        hls_dir = finn_root / 'custom_op' / 'fpgadataflow' / 'hls'
        backends.extend(_scan_backend_dir(hls_dir, HLSBackend, 'hls'))

        # Scan RTL backends
        rtl_dir = finn_root / 'custom_op' / 'fpgadataflow' / 'rtl'
        backends.extend(_scan_backend_dir(rtl_dir, RTLBackend, 'rtl'))

        logger.info(f"Discovered {len(backends)} FINN backends")

    except Exception as e:
        logger.error(f"Failed to scan FINN backends: {e}")

    return backends


def _scan_finn_steps() -> List[Dict[str, Any]]:
    """Scan FINN builder for step functions."""
    import importlib
    import inspect

    steps = []
    module_name = 'finn.builder.build_dataflow_steps'

    try:
        module = importlib.import_module(module_name)

        # Find all functions starting with 'step_'
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith('step_') and obj.__module__ == module.__name__:
                # Remove 'step_' prefix for cleaner name
                step_name = name[5:]  # 'step_qonnx_to_finn' -> 'qonnx_to_finn'

                steps.append({
                    'name': step_name,
                    'callable_path': f'{module.__name__}.{name}',
                    'params': {}
                })
                logger.debug(f"Found step: {step_name}")

        logger.info(f"Discovered {len(steps)} FINN steps")

    except Exception as e:
        logger.warning(f"Failed to scan FINN steps: {e}")

    return steps


def scan_kernels(package_name: str) -> List[Dict[str, Any]]:
    """Scan a package for kernel components using runtime introspection.

    Imports package modules and discovers kernel classes via inspection.
    No hard-coded lists - all discovery is dynamic.

    Args:
        package_name: Name of the package to scan

    Returns:
        List of kernel metadata dicts with keys:
        - name: Component name
        - class_path: Full import path to class
        - infer_path: Full import path to infer transform
        - params: Additional parameters
    """
    if package_name == 'finn':
        return _scan_finn_kernels()
    return []


def scan_backends(package_name: str) -> List[Dict[str, Any]]:
    """Scan a package for backend components using runtime introspection.

    Imports package modules and discovers backend classes via inspection.
    No hard-coded lists - all discovery is dynamic.

    Args:
        package_name: Name of the package to scan

    Returns:
        List of backend metadata dicts
    """
    if package_name == 'finn':
        return _scan_finn_backends()
    return []


def scan_steps(package_name: str) -> List[Dict[str, Any]]:
    """Scan a package for step/transform components using runtime introspection.

    Imports package modules and discovers step functions via inspection.
    No hard-coded lists - all discovery is dynamic.

    Args:
        package_name: Name of the package to scan

    Returns:
        List of step metadata dicts with keys:
        - name: Component name
        - callable_path: Full import path to callable
        - params: Additional parameters
    """
    if package_name == 'finn':
        return _scan_finn_steps()
    return []


def generate_manifest(package_name: str, force: bool = False) -> Optional[Dict[str, Any]]:
    """Generate a plugin manifest for a package.

    This scans the package for plugin components and generates a manifest
    containing metadata for fast discovery without imports.

    Args:
        package_name: Name of the package to generate manifest for
        force: If True, regenerate even if valid manifest exists

    Returns:
        The generated manifest dict, or None if package not installed
    """
    # Check if package is installed
    try:
        version = get_package_version(package_name)
    except importlib.metadata.PackageNotFoundError:
        logger.warning(f"Package {package_name} not installed, skipping manifest generation")
        return None

    # Check if valid manifest already exists
    if not force and is_manifest_valid(package_name):
        logger.info(f"Valid manifest already exists for {package_name}")
        return load_manifest(package_name)

    logger.info(f"Generating manifest for {package_name} v{version}...")

    # Scan for components
    kernels = scan_kernels(package_name)
    backends = scan_backends(package_name)
    steps = scan_steps(package_name)

    # Build manifest
    manifest = {
        'package': package_name,
        'version': version,
        'version_hash': compute_version_hash(package_name),
        'kernels': kernels,
        'backends': backends,
        'steps': steps,
        'generated_at': None  # Will be set by save_manifest
    }

    logger.info(f"Found {len(kernels)} kernels, {len(backends)} backends, {len(steps)} steps")

    return manifest


def save_manifest(manifest: Dict[str, Any]) -> None:
    """Save a manifest to disk.

    Args:
        manifest: The manifest dict to save
    """
    from datetime import datetime

    package_name = manifest['package']
    manifest['generated_at'] = datetime.now().isoformat()

    manifest_path = get_manifest_path(package_name)
    with open(manifest_path, 'w') as f:
        yaml.safe_dump(manifest, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved manifest to {manifest_path}")


def load_manifest(package_name: str) -> Optional[Dict[str, Any]]:
    """Load a manifest from disk.

    Args:
        package_name: Name of the package

    Returns:
        The manifest dict, or None if not found
    """
    manifest_path = get_manifest_path(package_name)
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading manifest for {package_name}: {e}")
        return None


def generate_all_manifests(force: bool = False) -> Dict[str, bool]:
    """Generate manifest for FINN package.

    Args:
        force: If True, regenerate even if valid manifests exist

    Returns:
        Dict mapping package name to success status
    """
    packages = ['finn']
    results = {}

    for package_name in packages:
        try:
            manifest = generate_manifest(package_name, force=force)
            if manifest is not None:
                save_manifest(manifest)
                results[package_name] = True
            else:
                results[package_name] = False
        except Exception as e:
            logger.error(f"Error generating manifest for {package_name}: {e}")
            results[package_name] = False

    return results


def clear_manifests() -> int:
    """Clear all manifest cache files.

    Returns:
        Number of manifests deleted
    """
    cache_dir = get_cache_dir()
    count = 0

    for manifest_file in cache_dir.glob("*.yaml"):
        manifest_file.unlink()
        count += 1

    logger.info(f"Deleted {count} manifest files")
    return count
