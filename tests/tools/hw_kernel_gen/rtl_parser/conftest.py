############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

import pytest
import logging
from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def parser():
    """Provides a configured RTLParser instance for tests in this directory."""
    logger.info("Setting up RTLParser fixture (module scope)")
    # Initialize RTLParser using its original __init__
    # It creates its own InterfaceBuilder internally
    # Assuming default grammar path finding works and debug=False is okay
    try:
        parser_instance = RTLParser(debug=False)
        logger.info("RTLParser fixture created successfully.")
    except Exception as e:
        logger.error(f"Failed to create RTLParser fixture: {e}", exc_info=True)
        pytest.fail(f"Failed to create RTLParser fixture: {e}")

    return parser_instance
