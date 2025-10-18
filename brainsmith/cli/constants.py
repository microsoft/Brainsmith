# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path

# ============================================================================
# Configuration Files
# ============================================================================

USER_CONFIG_DIR = Path.home() / ".brainsmith"
USER_CONFIG_FILE = "settings.yaml"
PROJECT_CONFIG_FILE = "brainsmith_settings.yaml"
PROJECT_CONFIG_FILE_ALT = ".brainsmith.yaml"

# ============================================================================
# Environment Variables
# ============================================================================

# Core settings
ENV_QUIET = "BSMITH_QUIET"

# Paths
ENV_BUILD_DIR = "BSMITH_BUILD_DIR"
ENV_DEPS_DIR = "BSMITH_DEPS_DIR"

# Toolchain settings
ENV_PLUGINS_STRICT = "BSMITH_PLUGINS_STRICT"
ENV_NETRON_PORT = "BSMITH_NETRON_PORT"
ENV_DEFAULT_WORKERS = "BSMITH_DEFAULT_WORKERS"

# Xilinx tools
ENV_XILINX_PATH = "BSMITH_XILINX_PATH"
ENV_XILINX_VERSION = "BSMITH_XILINX_VERSION"
ENV_VIVADO_PATH = "BSMITH_VIVADO_PATH"
ENV_VITIS_PATH = "BSMITH_VITIS_PATH"
ENV_VITIS_HLS_PATH = "BSMITH_VITIS_HLS_PATH"

# FINN configuration
ENV_FINN_ROOT = "BSMITH_FINN__FINN_ROOT"
ENV_FINN_BUILD_DIR = "BSMITH_FINN__FINN_BUILD_DIR"
ENV_FINN_DEPS_DIR = "BSMITH_FINN__FINN_DEPS_DIR"

# ============================================================================
# CLI Names and Subcommands
# ============================================================================

CLI_NAME_BRAINSMITH = "brainsmith"
CLI_NAME_SMITH = "smith"

# Subcommands (used for command routing logic)
SUBCOMMAND_DFC = "dfc"
SUBCOMMAND_KERNEL = "kernel"

# ============================================================================
# Exit Codes
# ============================================================================

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_CONFIG_ERROR = 2
EXIT_VALIDATION_ERROR = 3
EXIT_INTERRUPTED = 130  # Standard SIGINT exit code
