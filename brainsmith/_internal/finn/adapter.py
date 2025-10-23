# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""FINN-specific adapter to isolate workarounds.

All FINN-specific hacks, workarounds, and necessary evils are isolated here.
This allows the main executor to remain clean while documenting why these
workarounds exist.
"""

import importlib.util
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class FINNAdapter:
    """Adapter for FINN build system.

    WARNING: This adapter is NOT thread-safe. FINN requires changing the
    working directory during builds (os.chdir), which affects the entire
    process. Do not run multiple builds concurrently in the same process.

    Isolates all FINN-specific workarounds:
    - Working directory changes (os.chdir) - FINN requirement, NOT thread-safe
    - Model discovery in intermediate_models directory
    - Dynamic import handling
    - Configuration format conversion
    - Environment variable management

    For concurrent builds, use separate processes instead of threads.
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
            error_msg += "\nPlease check the environment setup instructions in the README file."
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
        # Ensure FINN environment variables are set before importing FINN
        # TODO (blocked on FINN upstream): Move away from environment variables
        # FINN currently requires FINN_ROOT, FINN_BUILD_DIR, etc. to be set via
        # os.environ. Ideally these would be passed directly to FINN components.
        # This requires changes to finn-base library.
        from brainsmith.settings import load_config
        config = load_config()
        config.export_to_environment()

        # Import FINN lazily to avoid circular dependencies
        from finn.builder.build_dataflow import build_dataflow_cfg
        from finn.builder.build_dataflow_config import DataflowBuildConfig

        abs_input = input_model.absolute()
        abs_output = output_dir.absolute()

        logger.info("FINN build: input=%s, output=%s", abs_input, abs_output)

        # TAFK TODO: Figure this out
        # FINN requires working directory change - NOT THREAD-SAFE
        # This affects the entire process. Do not run concurrent builds in same process.
        old_cwd = os.getcwd()

        try:
            os.chdir(abs_output)

            config_dict = config_dict.copy()
            config_dict["output_dir"] = "."

            finn_config = config_dict.copy()
            finn_config.pop("output_products", None)

            # TODO (requires FINN API changes): Get output model path directly from FINN
            # Currently FINN's build_dataflow_cfg() doesn't return the final model path,
            # so we must:
            # 1. Mandate save_intermediate_models=True
            # 2. Discover the last model in intermediate_models/ directory
            # This coupling should be improved in future FINN versions.
            finn_config["save_intermediate_models"] = True

            # CRITICAL: Set True to prevent FINN from redirecting stdout/stderr
            # which conflicts with Rich console logging, causing hangs
            finn_config["no_stdout_redirect"] = True
            finn_config["verbose"] = True

            # Convert dict to DataflowBuildConfig
            logger.debug("Creating DataflowBuildConfig with: %s", finn_config)
            config = DataflowBuildConfig(**finn_config)

            # FINN output goes directly to console (controlled by no_stdout_redirect flag)
            steps_count = len(config.steps) if config.steps else 0
            logger.info("Executing FINN build with %d steps", steps_count)
            exit_code = build_dataflow_cfg(str(abs_input), config)

            logger.info("FINN exit code: %d", exit_code)

            if exit_code != 0:
                raise RuntimeError(f"FINN build failed with exit code {exit_code}")

            # Raises if not found
            output_model = self._discover_output_model(abs_output)
            self._verify_output_model(output_model)

            logger.info("Found output: %s", output_model)
            return output_model

        finally:
            os.chdir(old_cwd)

    def _discover_output_model(self, build_dir: Path) -> Path:
        """Find the actual output model from FINN build.

        Since we mandate save_intermediate_models=True, FINN will save
        one ONNX file per transform step in the intermediate_models directory.
        We return the most recent file as the final output.

        Args:
            build_dir: Directory where build was executed

        Returns:
            Path to discovered model (guaranteed to exist)

        Raises:
            RuntimeError: If no models found or intermediate_models missing
        """
        intermediate_dir = build_dir / "intermediate_models"
        if not intermediate_dir.exists():
            raise RuntimeError(f"No intermediate_models directory found in {build_dir}")

        # Get all ONNX files from intermediate_models
        onnx_files = list(intermediate_dir.glob("*.onnx"))
        if not onnx_files:
            raise RuntimeError(f"No ONNX files found in {intermediate_dir}")

        logger.debug("ONNX files in %s: %s", intermediate_dir, [f.name for f in onnx_files])

        # Return the last (most recent) file - this is the output of the last transform
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
            onnx.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Invalid ONNX model at {model_path}: {e}")

    def prepare_model(self, source: Path, destination: Path) -> None:
        """
        Copy model to build directory.

        Args:
            source: Source model path
            destination: Destination model path
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
