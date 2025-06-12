"""
Pytest configuration and shared fixtures for BrainSmith test suite.
"""

import pytest
import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def project_root_path():
    """Provide the project root path for tests."""
    return project_root

@pytest.fixture(scope="session") 
def brainsmith_libraries():
    """Import and provide access to brainsmith libraries."""
    import brainsmith.libraries.kernels as kernels
    import brainsmith.libraries.transforms as transforms
    import brainsmith.libraries.analysis as analysis
    import brainsmith.libraries.blueprints as blueprints
    import brainsmith.libraries.automation as automation
    
    return {
        'kernels': kernels,
        'transforms': transforms,
        'analysis': analysis,
        'blueprints': blueprints,
        'automation': automation
    }