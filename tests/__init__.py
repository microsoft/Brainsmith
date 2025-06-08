"""
Brainsmith Test Suite

Comprehensive testing for the extensible Brainsmith architecture
using existing components with hierarchical exit points.
"""

import sys
import os
from pathlib import Path

# Add brainsmith to Python path for testing
test_dir = Path(__file__).parent
project_root = test_dir.parent
sys.path.insert(0, str(project_root))

__version__ = "0.4.0"
__description__ = "Test suite for Brainsmith extensible architecture"