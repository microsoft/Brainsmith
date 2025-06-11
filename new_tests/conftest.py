"""
BrainSmith Phase 1 Test Suite - pytest Configuration

Clean, focused test configuration for the new three-layer architecture.
Minimal mocking approach with real component testing.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_CONFIG = {
    'timeout': 30,
    'random_seed': 42,
    'temp_dir_prefix': 'brainsmith_new_test_',
}

# Set test environment
os.environ['BRAINSMITH_TEST_MODE'] = '1'
os.environ['BRAINSMITH_LOG_LEVEL'] = 'WARNING'


@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return TEST_CONFIG


@pytest.fixture(scope="session")
def temp_test_dir():
    """Create temporary directory for test artifacts."""
    temp_dir = tempfile.mkdtemp(prefix=TEST_CONFIG['temp_dir_prefix'])
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_model_path(temp_test_dir):
    """Create a sample ONNX model file for testing."""
    model_path = Path(temp_test_dir) / "sample_model.onnx"
    
    # Create minimal ONNX-like content for testing
    model_content = """<?xml version="1.0" encoding="UTF-8"?>
<!-- Sample ONNX model for testing -->
<ModelProto>
  <ir_version>8</ir_version>
  <producer_name>brainsmith_test</producer_name>
  <graph>
    <name>test_model</name>
    <input>
      <name>input</name>
      <type>
        <tensor_type>
          <elem_type>1</elem_type>
          <shape>
            <dim><dim_value>1</dim_value></dim>
            <dim><dim_value>3</dim_value></dim>
            <dim><dim_value>224</dim_value></dim>
            <dim><dim_value>224</dim_value></dim>
          </shape>
        </tensor_type>
      </type>
    </input>
    <output>
      <name>output</name>
      <type>
        <tensor_type>
          <elem_type>1</elem_type>
          <shape>
            <dim><dim_value>1</dim_value></dim>
            <dim><dim_value>1000</dim_value></dim>
          </shape>
        </tensor_type>
      </type>
    </output>
  </graph>
</ModelProto>"""
    
    with open(model_path, 'w') as f:
        f.write(model_content)
    
    return str(model_path)


@pytest.fixture
def sample_blueprint_path(temp_test_dir):
    """Create a sample blueprint YAML file for testing."""
    blueprint_path = Path(temp_test_dir) / "sample_blueprint.yaml"
    
    blueprint_content = """
name: "test_blueprint"
description: "Sample blueprint for testing"
version: "1.0.0"

build_steps:
  - "qonnx_to_finn"
  - "streamline"
  - "to_hw"
  - "optimize"
  - "synthesize"

parameters:
  pe_conv:
    type: "integer"
    range_min: 1
    range_max: 16
    default: 4
  simd_conv:
    type: "integer" 
    range_min: 1
    range_max: 8
    default: 2
  precision:
    type: "categorical"
    values: ["INT8", "INT16", "FP16"]
    default: "INT8"
  clock_freq:
    type: "float"
    range_min: 50.0
    range_max: 200.0
    default: 100.0

constraints:
  max_luts: 0.8
  max_dsps: 0.7
  max_brams: 0.6
  max_power: 20.0

objectives:
  throughput:
    direction: "maximize"
    weight: 1.0
  latency:
    direction: "minimize"
    weight: 0.8
  power:
    direction: "minimize"
    weight: 0.6

kernels:
  available: []

transforms:
  pipeline: []

hw_optimization:
  strategies: []

finn_interface:
  legacy_config:
    fpga_part: "xcvu9p-flga2104-2-i"
"""
    
    with open(blueprint_path, 'w') as f:
        f.write(blueprint_content)
    
    return str(blueprint_path)


@pytest.fixture
def invalid_blueprint_path(temp_test_dir):
    """Create an invalid blueprint for testing error scenarios."""
    blueprint_path = Path(temp_test_dir) / "invalid_blueprint.yaml"
    
    # Missing required fields
    invalid_content = """
name: "invalid_blueprint"
# Missing description, parameters, etc.
invalid_field: "this should not be here"
"""
    
    with open(blueprint_path, 'w') as f:
        f.write(invalid_content)
    
    return str(blueprint_path)


@pytest.fixture
def expected_forge_result():
    """Expected structure for forge() function results."""
    return {
        'dataflow_graph': {
            'onnx_model': None,  # Would contain actual ONNX model
            'metadata': {
                'kernel_mapping': {},
                'resource_estimates': {},
                'performance_estimates': {}
            }
        },
        'dataflow_core': None,  # Would contain generated core
        'dse_results': {
            'best_configuration': {},
            'pareto_frontier': [],
            'exploration_history': [],
            'convergence_metrics': {}
        },
        'metrics': {
            'performance': {
                'throughput_ops_sec': float,
                'latency_ms': float,
                'frequency_mhz': float
            },
            'resources': {
                'lut_utilization': float,
                'dsp_utilization': float,
                'bram_utilization': float,
                'power_consumption_w': float
            }
        },
        'analysis': {
            'design_space_coverage': float,
            'optimization_quality': float,
            'recommendations': list,
            'warnings': list
        }
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "core: Core layer tests")
    config.addinivalue_line("markers", "infrastructure: Infrastructure layer tests")
    config.addinivalue_line("markers", "compatibility: Backward compatibility tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers."""
    for item in items:
        # Add markers based on test location
        if "test_core" in str(item.fspath):
            item.add_marker(pytest.mark.core)
        elif "test_infrastructure" in str(item.fspath):
            item.add_marker(pytest.mark.infrastructure)
        elif "test_compatibility" in str(item.fspath):
            item.add_marker(pytest.mark.compatibility)
        elif "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically set up test environment for each test."""
    # Ensure deterministic behavior
    import random
    random.seed(TEST_CONFIG['random_seed'])
    
    yield
    
    # Cleanup after each test
    # Remove any temporary environment variables if needed
    pass


# Helper function for tests
def assert_result_structure(result: Dict[str, Any], expected_structure: Dict[str, Any]):
    """Helper to validate result structure matches expected format."""
    def _check_structure(actual, expected, path=""):
        for key, expected_type in expected.items():
            current_path = f"{path}.{key}" if path else key
            
            assert key in actual, f"Missing key: {current_path}"
            
            if expected_type == dict:
                assert isinstance(actual[key], dict), f"Expected dict at {current_path}"
            elif expected_type == list:
                assert isinstance(actual[key], list), f"Expected list at {current_path}"
            elif expected_type == float:
                assert isinstance(actual[key], (int, float)), f"Expected numeric at {current_path}"
            elif isinstance(expected_type, dict):
                _check_structure(actual[key], expected_type, current_path)
    
    _check_structure(result, expected_structure)