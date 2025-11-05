"""Global pytest configuration and fixtures.

This conftest.py is the root configuration for the DSE integration test suite.
"""

import pytest
import shutil
from pathlib import Path
from brainsmith.settings import reset_config
from brainsmith.settings.validation import ensure_environment_sourced
from brainsmith.registry import reset_registry

# Validate environment is sourced before any tests run
# This ensures FINN_ROOT, VIVADO_PATH, etc. are available for tests
ensure_environment_sourced()

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


def pytest_configure(config):
    """Configure pytest with custom settings.

    Validates that brainsmith environment is sourced before running tests.
    Environment must be activated via:
        source .brainsmith/env.sh
    or:
        direnv allow
    """
    # Validate environment is sourced before running tests
    from brainsmith.settings.validation import ensure_environment_sourced
    ensure_environment_sourced()

    # Markers already defined in pytest.ini
    pass


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

    NOTE: Old parity frameworks (ParityTestBase) have been replaced.
    Use new frameworks from tests/frameworks/ instead.

    Adds tests/parity/ to sys.path for remaining parity utilities:
    - assertions.py (parity-specific assertions)
    - test_fixtures.py (make_execution_context)

    New framework usage:
        from tests.frameworks.single_kernel_test import SingleKernelTest
        from tests.frameworks.dual_kernel_test import DualKernelTest

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
    """Auto-mark tests and handle parameterization sweeps.

    1. Auto-marks tests based on location:
       - integration/fast/ -> @pytest.mark.fast
       - integration/finn/ -> @pytest.mark.finn_build
       - integration/rtl/ -> @pytest.mark.rtlsim, @pytest.mark.slow
       - integration/hardware/ -> @pytest.mark.bitfile, @pytest.mark.hardware

    2. Parameterizes tests for classes with sweep methods:
       - Classes with get_dtype_sweep() get N copies per dtype config
       - Classes with get_shape_sweep() get N copies per shape config
    """
    # First pass: Auto-mark based on directory
    for item in items:
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

    # Second pass: Handle sweep parameterization
    # Build map of classes that need parameterization
    sweep_classes = {}  # cls -> (sweep_type, configs)

    for item in items:
        if item.cls is None:
            continue

        # Check if already processed this class
        if item.cls in sweep_classes:
            continue

        # Check for dtype sweep
        if hasattr(item.cls, "get_dtype_sweep"):
            instance = item.cls()
            sweep_configs = instance.get_dtype_sweep()
            if sweep_configs:
                sweep_classes[item.cls] = ("dtype", sweep_configs)
                continue

        # Check for shape sweep
        if hasattr(item.cls, "get_shape_sweep"):
            instance = item.cls()
            sweep_configs = instance.get_shape_sweep()
            if sweep_configs:
                sweep_classes[item.cls] = ("shape", sweep_configs)

    # If no sweep classes found, done
    if not sweep_classes:
        return

    # Third pass: Duplicate test items for sweep classes
    new_items = []

    for item in items:
        if item.cls not in sweep_classes:
            # Keep non-sweep tests as-is
            new_items.append(item)
            continue

        # This test belongs to a sweep class - create N copies
        sweep_type, configs = sweep_classes[item.cls]

        for idx, sweep_config in enumerate(configs):
            # Generate ID for this config
            if sweep_type == "dtype":
                id_parts = []
                for key, dtype in sweep_config.items():
                    dtype_name = str(dtype).split("[")[-1].rstrip("]").strip("'\"")
                    id_parts.append(f"{key}={dtype_name}")
                config_id = "_".join(id_parts)
            else:  # shape
                id_parts = []
                for key, value in sweep_config.items():
                    if isinstance(value, tuple):
                        shape_str = "x".join(str(d) for d in value)
                        id_parts.append(f"{key}={shape_str}")
                    else:
                        id_parts.append(f"{key}={value}")
                config_id = "_".join(id_parts)

            # Create new test item using pytest.Function
            # This is the proper way to create parameterized test items
            new_item = pytest.Function.from_parent(
                parent=item.parent,
                name=f"{item.name}[{config_id}]",
                callobj=item.obj,
            )

            # Store sweep config on the item for pytest_runtest_setup
            new_item._sweep_config = sweep_config
            new_item._sweep_type = sweep_type

            new_items.append(new_item)

    # Update items list
    items[:] = new_items


def pytest_runtest_setup(item):
    """Apply sweep configuration before each test runs.

    This hook runs before each test method executes and applies
    dtype/shape configuration to the test instance if present.
    """
    # Check if this test has sweep configuration (set by pytest_collection_modifyitems)
    if hasattr(item, "_sweep_config") and hasattr(item, "_sweep_type"):
        config = item._sweep_config
        sweep_type = item._sweep_type

        # Get the test instance (only available for class-based tests)
        if hasattr(item, "instance") and item.instance is not None:
            # Apply appropriate configuration
            if sweep_type == "dtype":
                if hasattr(item.instance, "configure_for_dtype_config"):
                    item.instance.configure_for_dtype_config(config)
            elif sweep_type == "shape":
                if hasattr(item.instance, "configure_for_shape_config"):
                    item.instance.configure_for_shape_config(config)
