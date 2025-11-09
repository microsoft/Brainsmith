# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""FINN-specific adapter to isolate workarounds."""

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

    For concurrent builds, use separate processes instead of threads.

    IMPORTANT: Expects environment to be sourced before running:
        source .brainsmith/env.sh
    or:
        direnv allow

    FINN environment variables (FINN_ROOT, VIVADO_PATH, etc.) must be set
    externally before Python starts to ensure subprocesses inherit them.
    """

    def __init__(self):
        # Warn if environment not sourced (soft validation - let FINN errors surface naturally)
        from brainsmith.settings.validation import warn_if_environment_not_sourced
        warn_if_environment_not_sourced()

        self._check_finn_dependencies()

    def _check_finn_dependencies(self) -> None:
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
        config_dict: dict[str, Any],
        output_dir: Path
    ) -> Path | None:
        """Execute FINN build with proper path handling.

        Returns:
            Path to output model if successful, None otherwise

        Raises:
            RuntimeError: If build fails
        """
        # Import FINN lazily to avoid circular dependencies
        # Note: FINN environment (FINN_ROOT, etc.) configured in __init__()
        from finn.builder.build_dataflow import build_dataflow_cfg
        from finn.builder.build_dataflow_config import DataflowBuildConfig

        abs_input = input_model.absolute()
        abs_output = output_dir.absolute()

        logger.info("FINN build: input=%s, output=%s", abs_input, abs_output)

        # FINN requires os.chdir() for relative paths in build outputs
        # This is NOT THREAD-SAFE and affects the entire process.
        # Do not run concurrent builds in the same process.
        old_cwd = os.getcwd()

        try:
            os.chdir(abs_output)

            config_dict = config_dict.copy()
            config_dict["output_dir"] = "."

            finn_config = config_dict.copy()
            finn_config.pop("output_products", None)

            # WORKAROUND: FINN doesn't return output path, so we discover it from
            # intermediate_models/ directory (requires save_intermediate_models=True)
            finn_config["save_intermediate_models"] = True

            # CRITICAL: Set True to prevent FINN from redirecting stdout/stderr
            # which conflicts with Rich console logging, causing hangs
            finn_config["no_stdout_redirect"] = True
            finn_config["verbose"] = True

            # Disable pdb debugger for automated builds (pytest captures stdin/stdout)
            finn_config["enable_build_pdb_debug"] = False

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
        """Most recently modified .onnx file in build_dir"""
        intermediate_dir = build_dir / "intermediate_models"
        onnx_files = list(intermediate_dir.glob("*.onnx"))

        if not onnx_files:
            raise RuntimeError(
                f"No ONNX output found in {intermediate_dir}. "
                "FINN build may have failed silently."
            )

        logger.debug("ONNX files in %s: %s", intermediate_dir, [f.name for f in onnx_files])

        # Most recent file is the final output
        return max(onnx_files, key=lambda p: p.stat().st_mtime)

    def _verify_output_model(self, model_path: Path) -> None:
        if not model_path.exists():
            raise RuntimeError(f"Output model does not exist: {model_path}")

        # Verify it's a valid ONNX file
        try:
            import onnx
            onnx.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Invalid ONNX model at {model_path}: {e}")

    def prepare_model(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
