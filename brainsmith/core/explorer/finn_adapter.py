# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""FINN-specific adapter to isolate workarounds.

All FINN-specific hacks, workarounds, and necessary evils are isolated here.
This allows the main executor to remain clean while documenting why these
workarounds exist.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import importlib.util

logger = logging.getLogger(__name__)


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
        """Check all FINN dependencies are available."""
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
        """Execute FINN build with proper path handling.
        
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
        
        # Convert to absolute paths before chdir
        abs_input = input_model.absolute()
        abs_output = output_dir.absolute()
        
        logger.info(f"FINN build: input={abs_input}, output={abs_output}")
        
        # FINN requires working directory change
        old_cwd = os.getcwd()
        
        try:
            os.chdir(abs_output)
            
            # Update config to use current directory
            config_dict = config_dict.copy()
            config_dict["output_dir"] = "."
            
            # Remove parameters that are not for DataflowBuildConfig
            finn_config = config_dict.copy()
            finn_config.pop("output_products", None)  # This is handled by generate_outputs
            
            # Convert dict to DataflowBuildConfig
            logger.debug(f"Creating DataflowBuildConfig with: {finn_config}")
            config = DataflowBuildConfig(**finn_config)
            
            # Execute build
            logger.info(f"Executing FINN build with {len(config.steps)} steps")
            exit_code = build_dataflow_cfg(str(abs_input), config)
            
            logger.info(f"FINN exit code: {exit_code}")
            
            if exit_code != 0:
                raise RuntimeError(f"FINN build failed with exit code {exit_code}")
            
            # Discovery now uses absolute path
            output_model = self._discover_output_model(abs_output)
            self._verify_output_model(output_model)
            
            logger.info(f"Found output: {output_model}")
            return output_model
            
        finally:
            # Always restore working directory
            os.chdir(old_cwd)
    
    def _discover_output_model(self, build_dir: Path) -> Optional[Path]:
        """Find the actual output model from FINN build.
        
        Args:
            build_dir: Directory where build was executed
            
        Returns:
            Path to discovered model
            
        Raises:
            RuntimeError: If no models found
        """
        intermediate_dir = build_dir / "intermediate_models"
        if not intermediate_dir.exists():
            raise RuntimeError(f"No intermediate_models directory found in {build_dir}")
        
        # Get all ONNX files
        onnx_files = list(intermediate_dir.glob("*.onnx"))
        if not onnx_files:
            raise RuntimeError(f"No ONNX files found in {intermediate_dir}")
        
        logger.debug(f"ONNX files in {intermediate_dir}: {[f.name for f in onnx_files]}")
        
        # Return the last (most recent) file
        # This is still imperfect but matches FINN's behavior
        onnx_files.sort(key=lambda p: p.stat().st_mtime)
        return onnx_files[-1]
    
    def _verify_output_model(self, model_path: Path) -> None:
        """Verify the output model exists and is valid ONNX.
        
        Args:
            model_path: Path to model to verify
            
        Raises:
            RuntimeError: If model is invalid
        """
        if not model_path.exists():
            raise RuntimeError(f"Output model does not exist: {model_path}")
        
        # Verify it's a valid ONNX file
        try:
            import onnx
            onnx.load(str(model_path))
        except Exception as e:
            raise RuntimeError(f"Invalid ONNX model at {model_path}: {e}")
    
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