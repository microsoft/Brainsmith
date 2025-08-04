
import logging
import os

# Set default logging handler to avoid \"No handler found\" warnings.
# Libraries should NOT add other handlers or call basicConfig.
# The application using the library is responsible for configuring logging.
logger = logging.getLogger(__name__) # Get logger for the 'brainsmith' package
if not logger.hasHandlers():
    logger.addHandler(logging.NullHandler())

# Optional: You could set a default level for the library logger here,
# but it's often better to let the application control this entirely.
# logger.setLevel(logging.WARNING) # Example: Default to WARNING

# You can also expose key classes/functions here for easier import
# e.g., from .core import SomeClass
