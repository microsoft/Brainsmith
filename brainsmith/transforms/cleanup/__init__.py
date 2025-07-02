"""
graph_cleanup transforms
"""

# Import all transforms to trigger auto-registration
from . import remove_identity

__all__ = ["remove_identity"]
