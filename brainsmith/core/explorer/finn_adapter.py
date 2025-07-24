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
        # Check FINN availability once
        self._finn_available = self._check_finn_available()
        if not self._finn_available:
            raise RuntimeError(
                "FINN not installed. Please install finn-base: "
                "pip install git+https://github.com/Xilinx/finn.git"
            )
    
    def _check_finn_available(self) -> bool:
        """Check if FINN is available."""
        return importlib.util.find_spec("finn") is not None
    
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
            config = DataflowBuildConfig(**config_dict)
            
            # Execute build
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
        
        FINN puts the final model in intermediate_models directory
        with unpredictable naming. We find the last generated model.
        
        Args:
            build_dir: Directory where build was executed
            
        Returns:
            Path to discovered model or None if not found
        """
        intermediate_dir = build_dir / "intermediate_models"
        if not intermediate_dir.exists():
            return None
        
        # Find all ONNX files, sorted by modification time
        onnx_files = sorted(
            intermediate_dir.glob("*.onnx"),
            key=lambda p: p.stat().st_mtime
        )
        
        if not onnx_files:
            return None
        
        # Return the most recently modified model
        return onnx_files[-1]
    
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