"""
Generic package loader for kernel packages.
Loads any kernel package from kernel.yaml manifest without custom loader code.
"""

import yaml
from pathlib import Path
from typing import List

from .types import KernelPackage


def load_kernel_package(package_dir: str) -> KernelPackage:
    """
    Generic loader that creates KernelPackage from any kernel.yaml.
    No custom loader code required per kernel!
    
    Args:
        package_dir: Directory path containing kernel.yaml
        
    Returns:
        KernelPackage object loaded from manifest
        
    Raises:
        FileNotFoundError: If kernel.yaml not found
        ValueError: If kernel.yaml is invalid
    """
    package_path = Path(__file__).parent / package_dir
    manifest_path = package_path / "kernel.yaml"
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"No kernel.yaml found in {package_path}")
    
    try:
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {manifest_path}: {e}")
    
    if not isinstance(manifest, dict):
        raise ValueError(f"kernel.yaml must contain a dictionary, got {type(manifest)}")
    
    # Create KernelPackage directly from manifest
    return KernelPackage(
        name=manifest.get("name", package_dir),
        operator_type=manifest.get("operator_type", "Unknown"),
        backend=manifest.get("backend", "Unknown"),
        version=manifest.get("version", "1.0.0"),
        author=manifest.get("author", ""),
        license=manifest.get("license", ""),
        description=manifest.get("description", ""),
        parameters=manifest.get("parameters", {}),
        files=manifest.get("files", {}),
        performance=manifest.get("performance", {}),
        validation=manifest.get("validation", {}),
        repository=manifest.get("repository", {}),
        package_path=str(package_path)
    )