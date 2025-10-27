# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""User-facing messages and strings for CLI output.

Centralizes all UI text to separate presentation from business logic.
"""

# ============================================================================
# Package Metadata
# ============================================================================

PACKAGE_NAME = "brainsmith"

# ============================================================================
# Configuration Messages
# ============================================================================

CONFIG_EDIT_HINT = "\nEdit the file to customize settings for your environment."
CONFIG_CREATED_PROJECT = "These settings apply to this project only."
CONFIG_CREATED_USER = "These settings apply to all Brainsmith projects."

# ============================================================================
# Tool Status Messages
# ============================================================================

XILINX_NOT_CONFIGURED = "Not configured"
XILINX_NOT_FOUND = "Not found"

# ============================================================================
# Error Detail Messages
# ============================================================================

# Config command
CONFIG_OVERWRITE_HINT = "Run with --force flag to overwrite existing configuration"

# Kernel command
KERNEL_VALIDATION_HINTS = [
    "Verify the RTL file contains valid SystemVerilog with @brainsmith pragmas",
    "Check that all pragma syntax is correct",
    "Ensure any referenced RTL files in pragmas exist",
    "Confirm you have write permissions to the output directory"
]

KERNEL_TOOL_NOT_FOUND_HINTS = [
    "Ensure brainsmith is fully installed: pip install -e .",
    "Verify the kernel_integrator module exists in brainsmith/tools/"
]

# DFC command
DFC_ERROR_HINT = "Run with --logs=debug for detailed traceback"
