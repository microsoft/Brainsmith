"""
Pytest fixtures for Phase 1 unit tests - Using REAL QONNX/FINN plugins only.
"""

import pytest
from pathlib import Path
import yaml

from brainsmith.core.plugins import get_registry


@pytest.fixture
def model_path(tmp_path):
    """Create a test ONNX model file."""
    model_file = tmp_path / "test_model.onnx"
    model_file.write_bytes(b"fake onnx model content")
    return str(model_file)


@pytest.fixture
def blueprint_factory(tmp_path):
    """Factory for creating blueprint YAML files."""
    def _create_blueprint(content):
        if isinstance(content, dict):
            content = yaml.dump(content)
        bp_file = tmp_path / "test_blueprint.yaml"
        bp_file.write_text(content)
        return str(bp_file)
    return _create_blueprint


@pytest.fixture
def minimal_blueprint_dict():
    """Minimal valid blueprint using real QONNX/FINN plugins."""
    return {
        "version": "3.0",
        "hw_compiler": {
            "kernels": ["LayerNorm"],  # Real QONNX kernel
            "transforms": ["FoldConstants"],  # Real QONNX transform
            "build_steps": ["ConvertToHW"],
        },
        "search": {"strategy": "exhaustive"},
        "global": {
            "output_stage": "rtl",
            "working_directory": "./builds"
        }
    }


@pytest.fixture
def parser_with_registry():
    """Create a parser with real QONNX/FINN registry."""
    from brainsmith.core.phase1.parser import BlueprintParser
    # Parser gets real registry via get_registry() in __init__
    return BlueprintParser()


@pytest.fixture
def validator_with_registry():
    """Create a validator with real QONNX/FINN registry."""
    from brainsmith.core.phase1.validator import DesignSpaceValidator
    # Validator gets real registry via get_registry() in __init__
    return DesignSpaceValidator()