"""
BrainSmith Comprehensive Test Suite - Shared Test Fixtures

This module provides shared test fixtures, utilities, and configuration
for the comprehensive test suite.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import numpy if available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Test configuration
TEST_CONFIG = {
    'timeout': 300,
    'random_seed': 42,
    'temp_dir_prefix': 'brainsmith_test_',
    'benchmark_models': ['resnet18', 'bert_tiny', 'mobilenet'],
    'test_platforms': ['zynq', 'ultrascale', 'alveo'],
    'finn_versions': ['0.8.0', '0.9.0', '1.0.0']
}

# Set deterministic behavior if numpy available
if NUMPY_AVAILABLE:
    np.random.seed(TEST_CONFIG['random_seed'])


@pytest.fixture(scope="session")
def test_config():
    """Global test configuration fixture."""
    return TEST_CONFIG


@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for test artifacts."""
    temp_dir = tempfile.mkdtemp(prefix=TEST_CONFIG['temp_dir_prefix'])
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def synthetic_model():
    """Generate a synthetic neural network model for testing."""
    model_config = {
        'name': 'test_model',
        'layers': [
            {'type': 'conv2d', 'filters': 32, 'kernel_size': 3, 'input_shape': (224, 224, 3)},
            {'type': 'relu'},
            {'type': 'maxpool2d', 'pool_size': 2},
            {'type': 'conv2d', 'filters': 64, 'kernel_size': 3},
            {'type': 'relu'},
            {'type': 'maxpool2d', 'pool_size': 2},
            {'type': 'flatten'},
            {'type': 'dense', 'units': 128},
            {'type': 'relu'},
            {'type': 'dense', 'units': 10, 'activation': 'softmax'}
        ],
        'parameters': {
            'total_params': 1_234_567,
            'trainable_params': 1_234_567,
            'non_trainable_params': 0
        }
    }
    return model_config


@pytest.fixture
def design_space_config():
    """Standard design space configuration for testing."""
    return {
        'parameters': {
            'batch_size': {'type': 'integer', 'range': [1, 64], 'default': 32},
            'pe_conv': {'type': 'integer', 'range': [1, 32], 'default': 4},
            'simd_conv': {'type': 'integer', 'range': [1, 16], 'default': 2},
            'pe_dense': {'type': 'integer', 'range': [1, 64], 'default': 8},
            'simd_dense': {'type': 'integer', 'range': [1, 32], 'default': 4},
            'clock_freq': {'type': 'float', 'range': [50.0, 200.0], 'default': 100.0},
            'precision': {'type': 'categorical', 'values': ['INT8', 'INT16', 'FP16'], 'default': 'INT8'}
        },
        'constraints': {
            'max_luts': 0.8,
            'max_dsps': 0.7,
            'max_brams': 0.6,
            'max_power': 20.0
        },
        'objectives': [
            {'name': 'throughput', 'direction': 'maximize', 'weight': 1.0},
            {'name': 'latency', 'direction': 'minimize', 'weight': 0.8},
            {'name': 'power', 'direction': 'minimize', 'weight': 0.6}
        ]
    }


@pytest.fixture
def mock_finn_environment():
    """Mock FINN environment for testing without actual FINN installation."""
    finn_config = {
        'version': '1.0.0',
        'installation_path': '/mock/finn',
        'available_kernels': [
            'ConvolutionInputGenerator',
            'MatrixVectorActivation',
            'StreamingFCLayer',
            'VectorVectorActivation',
            'Thresholding'
        ],
        'supported_platforms': ['zynq', 'ultrascale', 'alveo'],
        'build_timeout': 3600
    }
    return finn_config


@pytest.fixture
def performance_benchmarks():
    """Standard performance benchmark data for validation."""
    return {
        'throughput_targets': {
            'small_model': 100,  # ops/sec
            'medium_model': 500,
            'large_model': 1000
        },
        'latency_targets': {
            'small_model': 50,   # ms
            'medium_model': 20,
            'large_model': 10
        },
        'resource_limits': {
            'luts': 200000,
            'dsps': 4000,
            'brams': 1000
        },
        'memory_limits': {
            'peak_mb': 2048,
            'sustained_mb': 1024,
            'leak_threshold_mb': 100
        }
    }


@pytest.fixture
def optimization_results():
    """Sample optimization results for analysis testing."""
    results = []
    
    if NUMPY_AVAILABLE:
        for i in range(50):
            result = {
                'iteration': i,
                'parameters': {
                    'pe_conv': np.random.randint(1, 17),
                    'simd_conv': np.random.randint(1, 9),
                    'pe_dense': np.random.randint(1, 33),
                    'simd_dense': np.random.randint(1, 17),
                    'clock_freq': np.random.uniform(50, 200)
                },
                'metrics': {
                    'throughput': np.random.uniform(100, 1000),
                    'latency': np.random.uniform(5, 50),
                    'power': np.random.uniform(5, 25),
                    'lut_utilization': np.random.uniform(0.3, 0.9),
                    'dsp_utilization': np.random.uniform(0.2, 0.8),
                    'bram_utilization': np.random.uniform(0.1, 0.7)
                },
                'build_success': np.random.choice([True, False], p=[0.9, 0.1]),
                'build_time': np.random.uniform(30, 300)
            }
            results.append(result)
    else:
        # Fallback without numpy
        import random
        random.seed(TEST_CONFIG['random_seed'])
        
        for i in range(50):
            result = {
                'iteration': i,
                'parameters': {
                    'pe_conv': random.randint(1, 16),
                    'simd_conv': random.randint(1, 8),
                    'pe_dense': random.randint(1, 32),
                    'simd_dense': random.randint(1, 16),
                    'clock_freq': random.uniform(50, 200)
                },
                'metrics': {
                    'throughput': random.uniform(100, 1000),
                    'latency': random.uniform(5, 50),
                    'power': random.uniform(5, 25),
                    'lut_utilization': random.uniform(0.3, 0.9),
                    'dsp_utilization': random.uniform(0.2, 0.8),
                    'bram_utilization': random.uniform(0.1, 0.7)
                },
                'build_success': random.choice([True, False]),
                'build_time': random.uniform(30, 300)
            }
            results.append(result)
    
    return results


@pytest.fixture
def brainsmith_api():
    """Mock BrainSmith API for high-level testing."""
    class MockBrainSmith:
        def __init__(self):
            self.version = "0.4.0"
            self.available_blueprints = ['bert_extensible', 'resnet', 'mobilenet']
            self.available_strategies = ['random', 'bayesian', 'genetic', 'adaptive']
        
        def optimize_model(self, model_path, blueprint_name, **kwargs):
            """Mock model optimization."""
            return {
                'success': True,
                'best_result': {
                    'parameters': {'pe': 8, 'simd': 4, 'clock_freq': 150.0},
                    'metrics': {'throughput': 750, 'latency': 15, 'power': 12.5}
                },
                'iterations': kwargs.get('max_evaluations', 50),
                'total_time': 1800
            }
        
        def explore_design_space(self, model_path, blueprint_name, **kwargs):
            """Mock design space exploration."""
            n_results = kwargs.get('max_evaluations', 50)
            return {
                'success': True,
                'results': [
                    {
                        'parameters': {'pe': i % 16 + 1, 'simd': i % 8 + 1},
                        'metrics': {'throughput': 500 + i * 10, 'latency': 30 - i * 0.2}
                    }
                    for i in range(n_results)
                ],
                'pareto_frontier': list(range(min(10, n_results))),
                'total_time': n_results * 30
            }
        
        def build_model(self, model_path, blueprint_name, parameters):
            """Mock single model build."""
            return {
                'success': True,
                'metrics': {
                    'throughput': parameters.get('pe', 1) * parameters.get('simd', 1) * 50,
                    'latency': 100 / (parameters.get('pe', 1) * parameters.get('simd', 1)),
                    'power': parameters.get('clock_freq', 100) * 0.1
                },
                'build_time': 120,
                'artifacts': ['bitstream.bit', 'driver.so', 'report.json']
            }
        
        def list_blueprints(self):
            return self.available_blueprints
        
        def list_strategies(self):
            return self.available_strategies
    
    return MockBrainSmith()


class TestDataManager:
    """Utility class for managing test data and artifacts."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def create_test_model(self, name: str, complexity: str = 'small') -> Path:
        """Create a test model file."""
        model_configs = {
            'small': {'layers': 5, 'params': 100_000},
            'medium': {'layers': 20, 'params': 1_000_000},
            'large': {'layers': 100, 'params': 10_000_000}
        }
        
        config = model_configs.get(complexity, model_configs['small'])
        model_path = self.base_dir / f"{name}_{complexity}.json"
        
        model_data = {
            'name': name,
            'complexity': complexity,
            'layer_count': config['layers'],
            'parameter_count': config['params'],
            'input_shape': [1, 3, 224, 224],
            'output_shape': [1, 1000]
        }
        
        import json
        with open(model_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        return model_path
    
    def create_test_config(self, name: str, config_data: Dict[str, Any]) -> Path:
        """Create a test configuration file."""
        config_path = self.base_dir / f"{name}_config.yaml"
        
        try:
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
        except ImportError:
            # Fallback to JSON if yaml not available
            config_path = self.base_dir / f"{name}_config.json"
            import json
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        
        return config_path
    
    def cleanup(self):
        """Clean up test data."""
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir, ignore_errors=True)


@pytest.fixture
def test_data_manager(temp_test_dir):
    """Test data manager fixture."""
    manager = TestDataManager(temp_test_dir)
    yield manager
    manager.cleanup()


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line("markers", "smoke: Quick smoke tests")
    config.addinivalue_line("markers", "core: Core functional tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "stress: Stress tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "ux: User experience tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers."""
    for item in items:
        # Add markers based on test file path
        if "test_stress" in str(item.fspath):
            item.add_marker(pytest.mark.stress)
        elif "test_performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        elif "ux" in str(item.fspath):
            item.add_marker(pytest.mark.ux)
        
        # Add slow marker for tests that might take time
        if any(keyword in item.name.lower() for keyword in ['stress', 'large', 'benchmark']):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically set up test environment for each test."""
    # Set environment variables for testing
    os.environ['BRAINSMITH_TEST_MODE'] = '1'
    os.environ['BRAINSMITH_LOG_LEVEL'] = 'WARNING'
    
    # Ensure deterministic behavior
    if NUMPY_AVAILABLE:
        np.random.seed(TEST_CONFIG['random_seed'])
    
    yield
    
    # Cleanup after test
    if 'BRAINSMITH_TEST_MODE' in os.environ:
        del os.environ['BRAINSMITH_TEST_MODE']
    if 'BRAINSMITH_LOG_LEVEL' in os.environ:
        del os.environ['BRAINSMITH_LOG_LEVEL']