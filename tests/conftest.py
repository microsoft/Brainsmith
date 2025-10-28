"""Global pytest configuration and fixtures.

This conftest.py is the root configuration for the DSE integration test suite.
"""

import pytest
import shutil
from pathlib import Path
from brainsmith.settings import reset_config, get_config
from brainsmith.registry import reset_registry

# Initialize configuration early (before FINN/QONNX imports)
# This exports FINN_DEPS_DIR and other env vars to prevent warnings
_config = get_config()

# Import test components - these register @step, @kernel, @backend decorators
# Available for tests that need globally-registered test components
import tests.fixtures.components.kernels
import tests.fixtures.components.backends
import tests.fixtures.components.steps

# Phase 4: Fixture imports for global availability
from tests.fixtures.models import *
from tests.fixtures.design_spaces import *
from tests.fixtures.blueprints import *

# Import kernel test helpers for easy access in all kernel tests
# Use these for unit testing kernels (schema, inference, transformation)
# For parity testing (comparing implementations), use tests/parity/ParityTestBase
from tests.fixtures.kernel_test_helpers import (
    OnnxModelBuilder,
    make_binary_op_model,
    make_parametric_op_model,
    make_unary_op_model,
    make_multithreshold_model,
    make_funclayernorm_model,
    make_vvau_model,
    make_broadcast_model,
    make_duplicate_streams_model,
)


@pytest.fixture(scope="session")
def test_workspace(tmp_path_factory):
    """Session-scoped workspace for FINN builds.

    All integration tests use this shared workspace for output directories.
    Avoids creating temporary directories per test.
    """
    return tmp_path_factory.mktemp("dse_workspace")


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Isolated environment with real project structure and test plugins.

    Creates:
    - tmp_path/test_project/.brainsmith/
    - tmp_path/test_project/plugins/
    - Real config.yaml
    - BSMITH_PROJECT_DIR env var

    No mocking. Real settings system. Real component discovery.

    Test components (kernels, steps, backends) are globally available via
    decorator registration in conftest.py imports, not via file discovery.
    """
    # Create project structure
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    brainsmith_dir = project_dir / ".brainsmith"
    brainsmith_dir.mkdir()

    plugins_dir = project_dir / "plugins"
    plugins_dir.mkdir()

    # Write real config file
    config_file = brainsmith_dir / "config.yaml"
    config_file.write_text("""
cache_components: true
component_sources: {}
""")

    # Set environment (monkeypatch auto-cleans on teardown)
    monkeypatch.setenv('BSMITH_PROJECT_DIR', str(project_dir))

    # Clear registry and config state using public API
    reset_registry()
    reset_config()

    yield project_dir

    # Cleanup using public API
    reset_registry()
    reset_config()


@pytest.fixture
def empty_env(tmp_path, monkeypatch):
    """Minimal environment with no component sources.

    Creates:
    - tmp_path/test_project/.brainsmith/
    - Real config.yaml (no component sources)
    - BSMITH_PROJECT_DIR env var

    No mocking. For tests that only need core brainsmith components.

    Use this fixture when you don't need test plugins, just core components.
    """
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    brainsmith_dir = project_dir / ".brainsmith"
    brainsmith_dir.mkdir()

    # Empty config - no component sources
    config_file = brainsmith_dir / "config.yaml"
    config_file.write_text("""
cache_components: false
component_sources: {}
""")

    # Set environment
    monkeypatch.setenv('BSMITH_PROJECT_DIR', str(project_dir))

    # Clear state using public API
    reset_registry()
    reset_config()

    yield project_dir

    # Cleanup using public API
    reset_registry()
    reset_config()


@pytest.fixture(scope="session")
def setup_parity_imports():
    """Setup imports for parity test modules.

    Adds tests/parity/ to sys.path to enable clean imports of parity test
    helpers in kernel parity tests.

    Usage in kernel parity tests:
        def test_something(setup_parity_imports):
            from base_parity_test import ParityTestBase
            ...

    This eliminates brittle sys.path manipulation in individual test files.
    """
    import sys
    from pathlib import Path

    repo_root = Path(__file__).parent.parent
    tests_parity_dir = repo_root / "tests" / "parity"

    if str(tests_parity_dir) not in sys.path:
        sys.path.insert(0, str(tests_parity_dir))

    yield

    # Cleanup: remove from sys.path
    if str(tests_parity_dir) in sys.path:
        sys.path.remove(str(tests_parity_dir))


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on location.

    Tests are automatically marked based on their directory:
    - integration/fast/ -> @pytest.mark.fast
    - integration/finn/ -> @pytest.mark.finn_build
    - integration/rtl/ -> @pytest.mark.rtlsim, @pytest.mark.slow
    - integration/hardware/ -> @pytest.mark.bitfile, @pytest.mark.hardware
    """
    for item in items:
        # Auto-mark based on directory
        if "integration/fast" in str(item.fspath):
            item.add_marker(pytest.mark.fast)
        elif "integration/finn" in str(item.fspath):
            item.add_marker(pytest.mark.finn_build)
        elif "integration/rtl" in str(item.fspath):
            item.add_marker(pytest.mark.rtlsim)
            item.add_marker(pytest.mark.slow)
        elif "integration/hardware" in str(item.fspath):
            item.add_marker(pytest.mark.bitfile)
            item.add_marker(pytest.mark.hardware)
