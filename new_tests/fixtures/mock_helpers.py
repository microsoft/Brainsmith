"""
Mock Helpers for BrainSmith Phase 1 Tests

Minimal mocking utilities focused only on external dependencies.
Tests real components wherever possible.
"""

from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, Optional
from pathlib import Path


class MockFINNInterface:
    """Mock FINN interface for testing without FINN dependency."""
    
    def __init__(self):
        self.build_count = 0
        self.last_config = None
    
    def build_accelerator(self, model_path: str, blueprint_config: Dict[str, Any], 
                         output_dir: str = "./output") -> Dict[str, Any]:
        """Mock FINN accelerator build."""
        self.build_count += 1
        self.last_config = blueprint_config
        
        return {
            'ip_files': [f"{output_dir}/test_ip.v", f"{output_dir}/test_wrapper.v"],
            'synthesis_results': {
                'status': 'success',
                'lut_utilization': 0.45,
                'dsp_utilization': 0.32,
                'bram_utilization': 0.28,
                'frequency_mhz': 150.0
            },
            'driver_code': {
                'cpp_files': [f"{output_dir}/driver.cpp"],
                'header_files': [f"{output_dir}/driver.h"]
            },
            'bitstream': f"{output_dir}/design.bit",
            'reports': [f"{output_dir}/synthesis_report.txt"]
        }


class MockDSEResults:
    """Mock DSE results for testing."""
    
    def __init__(self, num_results: int = 10):
        self.results = []
        self.best_result = None
        self.pareto_points = []
        self.convergence = {}
        
        # Generate mock results
        for i in range(num_results):
            result = {
                'parameters': {
                    'pe_conv': 2 + (i % 8),
                    'simd_conv': 1 + (i % 4),
                    'clock_freq': 100 + (i * 10)
                },
                'metrics': {
                    'throughput': 100 + (i * 20),
                    'latency': 50 - (i * 2),
                    'lut_util': 0.3 + (i * 0.05),
                    'dsp_util': 0.2 + (i * 0.04),
                    'power': 10 + (i * 0.5)
                },
                'build_success': True
            }
            self.results.append(result)
        
        # Set best result (last one with highest throughput)
        self.best_result = self.results[-1] if self.results else {}
        self.best_result['dataflow_graph'] = MockONNXModel()


class MockONNXModel:
    """Mock ONNX model for testing."""
    
    def __init__(self, name: str = "test_model"):
        self.graph = Mock()
        self.graph.name = name
        self.graph.node = []
        self.ir_version = 8
        
    def __str__(self):
        return f"MockONNXModel({self.graph.name})"


class MockBlueprint:
    """Mock blueprint configuration."""
    
    def __init__(self, name: str = "test_blueprint"):
        self.name = name
        self.data = {
            'name': name,
            'description': 'Test blueprint',
            'parameters': {
                'pe_conv': {'type': 'integer', 'range_min': 1, 'range_max': 16, 'default': 4},
                'simd_conv': {'type': 'integer', 'range_min': 1, 'range_max': 8, 'default': 2}
            },
            'constraints': {'max_luts': 0.8, 'max_dsps': 0.7},
            'objectives': {
                'throughput': {'direction': 'maximize', 'weight': 1.0},
                'latency': {'direction': 'minimize', 'weight': 0.8}
            }
        }
    
    def get(self, key: str, default=None):
        return self.data.get(key, default)
    
    def __getitem__(self, key):
        return self.data[key]


def mock_forge_successful_result():
    """Create a mock successful forge result."""
    return {
        'dataflow_graph': {
            'onnx_model': MockONNXModel(),
            'metadata': {
                'kernel_mapping': {'conv2d': 'custom_op_conv2d'},
                'resource_estimates': {'luts': 45000, 'dsps': 320, 'brams': 180},
                'performance_estimates': {'throughput': 500.0, 'latency': 20.0}
            }
        },
        'dataflow_core': {
            'ip_files': ['test_ip.v'],
            'synthesis_results': {'status': 'success'},
            'driver_code': {'cpp_files': ['driver.cpp']},
            'bitstream': 'design.bit'
        },
        'dse_results': {
            'best_configuration': {'pe_conv': 8, 'simd_conv': 4},
            'pareto_frontier': [{'pe_conv': 4, 'simd_conv': 2}, {'pe_conv': 8, 'simd_conv': 4}],
            'exploration_history': [{'iteration': 0}, {'iteration': 1}],
            'convergence_metrics': {'best_score': 0.85}
        },
        'metrics': {
            'performance': {
                'throughput_ops_sec': 500.0,
                'latency_ms': 20.0,
                'frequency_mhz': 150.0
            },
            'resources': {
                'lut_utilization': 0.45,
                'dsp_utilization': 0.32,
                'bram_utilization': 0.28,
                'power_consumption_w': 12.5
            }
        },
        'analysis': {
            'design_space_coverage': 0.75,
            'optimization_quality': 0.85,
            'recommendations': ['Consider higher parallelism', 'Optimize memory usage'],
            'warnings': []
        },
        'analysis_data': {},
        'analysis_hooks': {
            'register_analyzer': Mock(),
            'get_raw_data': Mock(return_value=[]),
            'available_adapters': ['pandas', 'scipy', 'sklearn']
        }
    }


def mock_forge_fallback_result():
    """Create a mock fallback forge result (when components unavailable)."""
    return {
        'dataflow_graph': {
            'onnx_model': None,
            'metadata': {
                'kernel_mapping': {},
                'resource_estimates': {},
                'performance_estimates': {}
            }
        },
        'dataflow_core': {
            'ip_files': [],
            'synthesis_results': {'status': 'fallback_mode'},
            'driver_code': {},
            'bitstream': None,
            'fallback': True
        },
        'dse_results': {
            'best_configuration': {},
            'pareto_frontier': [],
            'exploration_history': [],
            'convergence_metrics': {}
        },
        'metrics': {
            'performance': {'throughput_ops_sec': 0.0, 'latency_ms': 0.0, 'frequency_mhz': 0.0},
            'resources': {'lut_utilization': 0.0, 'dsp_utilization': 0.0, 'bram_utilization': 0.0, 'power_consumption_w': 0.0}
        },
        'analysis': {
            'design_space_coverage': 0.0,
            'optimization_quality': 0.0,
            'recommendations': [],
            'warnings': ['Using fallback implementation - limited functionality']
        },
        'analysis_data': {},
        'analysis_hooks': {
            'register_analyzer': Mock(),
            'get_raw_data': Mock(return_value=[]),
            'available_adapters': []
        }
    }


# Context managers for mocking external dependencies
def mock_finn_unavailable():
    """Context manager to mock FINN as unavailable."""
    return patch('brainsmith.finn.build_accelerator', side_effect=ImportError("FINN not available"))


def mock_dse_unavailable():
    """Context manager to mock DSE system as unavailable."""
    return patch('brainsmith.dse.interface.DSEInterface', side_effect=ImportError("DSE not available"))


def mock_blueprint_system_unavailable():
    """Context manager to mock blueprint system as unavailable."""
    return patch('brainsmith.blueprints.functions.load_blueprint_yaml', side_effect=ImportError("Blueprint system not available"))


# Helper function for file operations
def create_temp_file(temp_dir: Path, filename: str, content: str) -> str:
    """Helper to create temporary files for testing."""
    file_path = temp_dir / filename
    with open(file_path, 'w') as f:
        f.write(content)
    return str(file_path)


# Assertion helpers
def assert_forge_result_structure(result: Dict[str, Any]):
    """Assert that forge result has the correct structure."""
    required_keys = ['dataflow_graph', 'dataflow_core', 'dse_results', 'metrics', 'analysis']
    
    for key in required_keys:
        assert key in result, f"Missing required key: {key}"
    
    # Check dataflow_graph structure
    dg = result['dataflow_graph']
    assert 'onnx_model' in dg
    assert 'metadata' in dg
    assert isinstance(dg['metadata'], dict)
    
    # Check metrics structure
    metrics = result['metrics']
    assert 'performance' in metrics
    assert 'resources' in metrics
    
    # Check analysis structure
    analysis = result['analysis']
    assert 'design_space_coverage' in analysis
    assert 'optimization_quality' in analysis
    assert 'recommendations' in analysis
    assert 'warnings' in analysis


def assert_metrics_valid(metrics: Dict[str, Any]):
    """Assert that metrics have valid values."""
    perf = metrics.get('performance', {})
    resources = metrics.get('resources', {})
    
    # Performance metrics should be non-negative
    for key, value in perf.items():
        if value is not None:
            assert value >= 0, f"Performance metric {key} should be non-negative"
    
    # Resource utilization should be between 0 and 1 (if specified as fraction)
    for key in ['lut_utilization', 'dsp_utilization', 'bram_utilization']:
        value = resources.get(key)
        if value is not None and key.endswith('_utilization'):
            assert 0 <= value <= 1, f"Resource utilization {key} should be between 0 and 1"