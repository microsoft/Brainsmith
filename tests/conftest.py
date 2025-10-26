"""Global pytest configuration and fixtures.

This conftest.py is the root configuration for the DSE integration test suite.
Phase 3: Skeleton created
Phase 4: Fixture imports will be added when modules are populated
"""

import pytest

# TODO (Phase 4): Import test plugins for decorator registration
# CRITICAL: These imports trigger decorator side effects for plugin registration
# import tests.fixtures.plugins.kernels
# import tests.fixtures.plugins.steps
# import tests.fixtures.plugins.transforms

# TODO (Phase 4): Import fixtures to make available globally
# from tests.fixtures.models import *
# from tests.fixtures.design_spaces import *
# from tests.fixtures.blueprints import *


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Markers already defined in pytest.ini
    pass


@pytest.fixture(scope="session")
def test_workspace(tmp_path_factory):
    """Session-scoped workspace for FINN builds.

    All integration tests use this shared workspace for output directories.
    Avoids creating temporary directories per test.
    """
    return tmp_path_factory.mktemp("dse_workspace")


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on location.

    Tests are automatically marked based on their directory:
    - integration/fast/ -> @pytest.mark.fast
    - integration/finn/ -> @pytest.mark.finn_build
    - integration/rtl/ -> @pytest.mark.rtl_sim, @pytest.mark.slow
    - integration/hardware/ -> @pytest.mark.bitfile, @pytest.mark.hardware
    """
    for item in items:
        # Auto-mark based on directory
        if "integration/fast" in str(item.fspath):
            item.add_marker(pytest.mark.fast)
        elif "integration/finn" in str(item.fspath):
            item.add_marker(pytest.mark.finn_build)
        elif "integration/rtl" in str(item.fspath):
            item.add_marker(pytest.mark.rtl_sim)
            item.add_marker(pytest.mark.slow)
        elif "integration/hardware" in str(item.fspath):
            item.add_marker(pytest.mark.bitfile)
            item.add_marker(pytest.mark.hardware)
