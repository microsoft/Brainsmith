# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""FINN-specific adapter to isolate workarounds.

All FINN-specific hacks, workarounds, and necessary evils are isolated here.
This allows the main executor to remain clean while documenting why these
workarounds exist.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import importlib.util


class FINNAdapter:
    """Adapter for FINN build system.
    
    Isolates all FINN-specific workarounds:
    - Working directory changes (os.chdir)
    - Model discovery in intermediate_models
    - Dynamic import handling
    - Configuration format conversion
    """
    
    def __init__(self):
        """Initialize FINN adapter."""
        # Check FINN availability with detailed error reporting
        self._check_finn_dependencies()
    
    def _check_finn_dependencies(self) -> None:
        """Check all FINN dependencies are available.
        
        Arete: Fail fast with clear error messages.
        """
        missing = []
        
        # Check core FINN modules
        required_modules = [
            ("finn", "finn-base"),
            ("finn.builder", "finn-base"),
            ("finn.builder.build_dataflow", "finn-base"),
            ("finn.builder.build_dataflow_config", "finn-base"),
        ]
        
        for module, package in required_modules:
            if importlib.util.find_spec(module) is None:
                missing.append((module, package))
        
        if missing:
            error_msg = "Missing FINN dependencies:\n"
            for module, package in missing:
                error_msg += f"  - {module} (from {package})\n"
            error_msg += "\nInstall with: pip install git+https://github.com/Xilinx/finn.git"
            raise RuntimeError(error_msg)
    
    def build(
        self,
        input_model: Path,
        config_dict: Dict[str, Any],
        output_dir: Path
    ) -> Optional[Path]:
        """Execute FINN build with all necessary workarounds.
        
        Args:
            input_model: Path to input ONNX model
            config_dict: FINN configuration as dictionary
            output_dir: Directory for build outputs
            
        Returns:
            Path to output model if successful, None otherwise
            
        Raises:
            RuntimeError: If build fails
        """
        # Import FINN lazily to avoid circular dependencies
        from finn.builder.build_dataflow import build_dataflow_cfg
        from finn.builder.build_dataflow_config import DataflowBuildConfig
        
        # FINN requires working directory change
        old_cwd = os.getcwd()
        
        try:
            os.chdir(output_dir)
            
            # Convert dict to DataflowBuildConfig
            print(f"Creating DataflowBuildConfig with: {config_dict}")
            config = DataflowBuildConfig(**config_dict)
            
            # Execute build
            print(f"Executing FINN build with model: {input_model}")
            print(f"Config steps: {config.steps}")
            exit_code = build_dataflow_cfg(str(input_model), config)
            
            if exit_code != 0:
                raise RuntimeError(f"FINN build failed with exit code {exit_code}")
            
            # FINN doesn't return output path, so we discover it
            return self._discover_output_model(output_dir)
            
        finally:
            # Always restore working directory
            os.chdir(old_cwd)
    
    def _discover_output_model(self, build_dir: Path) -> Optional[Path]:
        """Discover output model from FINN build directory.
        
        TECHNICAL DEBT: FINN doesn't return output paths from build_dataflow_cfg.
        This is a workaround that guesses the output by finding the most recently
        modified ONNX file in intermediate_models/.
        
        TODO: Submit PR to FINN to return output paths properly.
        
        Args:
            build_dir: Directory where build was executed
            
        Returns:
            Path to discovered model or None if not found
            
        Raises:
            RuntimeError: If no models found or discovery is ambiguous
        """
        intermediate_dir = build_dir / "intermediate_models"
        if not intermediate_dir.exists():
            raise RuntimeError(
                f"FINN build directory missing intermediate_models: {build_dir}"
            )
        
        # Find all ONNX files, sorted by modification time
        onnx_files = sorted(
            intermediate_dir.glob("*.onnx"),
            key=lambda p: p.stat().st_mtime
        )
        
        if not onnx_files:
            raise RuntimeError(
                f"No ONNX models found in {intermediate_dir}. "
                "FINN build may have failed silently."
            )
        
        # Validate the model exists and is readable
        output_model = onnx_files[-1]
        if not output_model.exists() or output_model.stat().st_size == 0:
            raise RuntimeError(
                f"Output model is invalid or empty: {output_model}"
            )
        
        # Log what we're doing for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(
            f"TECHNICAL DEBT: Guessing FINN output as {output_model.name} "
            f"(most recent of {len(onnx_files)} models)"
        )
        
        return output_model
    
    def prepare_model(self, source: Path, destination: Path) -> None:
        """Copy model to build directory.
        
        NECESSARY EVIL: FINN modifies input models in-place,
        so we must copy them to avoid corrupting originals.
        
        Args:
            source: Source model path
            destination: Destination model path
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)