"""
Mock blueprints for backend testing.
"""


def get_legacy_blueprint():
    """Get a mock legacy workflow blueprint."""
    return {
        'name': 'test_legacy_blueprint',
        'description': 'Test blueprint for legacy FINN workflow',
        'finn_config': {
            'build_steps': [
                'step_qonnx_to_finn',
                'step_streamline',
                'step_convert_to_hw',
                'step_specialize_layers',
                'step_create_dataflow_partition',
                'step_target_fps_parallelization',
                'step_apply_folding_config',
                'step_minimize_bit_width',
                'step_hw_codegen',
                'step_hw_ipgen',
                'step_measure_rtlsim_performance',
                'step_deployment_package'
            ],
            'output_dir': 'output',
            'synth_clk_period_ns': 5.0,
            'fpga_part': 'xczu7ev-ffvc1156-2-e',
            'board': 'ZCU104'
        },
        'objectives': [
            {
                'name': 'throughput',
                'direction': 'maximize',
                'weight': 1.0
            }
        ],
        'constraints': [
            {
                'name': 'latency',
                'operator': '<=',
                'value': 10.0
            }
        ]
    }


def get_six_entrypoint_blueprint():
    """Get a mock 6-entrypoint workflow blueprint."""
    return {
        'name': 'test_6ep_blueprint',
        'description': 'Test blueprint for 6-entrypoint workflow',
        'nodes': {
            'canonical_ops': {
                'available': ['MatMul', 'LayerNorm', 'Softmax'],
                'exploration': {
                    'required': ['MatMul'],
                    'optional': ['LayerNorm', 'Softmax']
                }
            },
            'hw_kernels': {
                'available': [
                    {'MatMul': ['RTL_Dense_v1', 'RTL_Dense_v2', 'HLS_Dense']},
                    {'LayerNorm': ['RTL_LayerNorm', 'HLS_LayerNorm']},
                    {'Softmax': ['HLS_Softmax']}
                ],
                'exploration': {
                    'required': [],
                    'optional': ['RTL_Dense_v1', 'RTL_Dense_v2', 'HLS_Dense']
                }
            }
        },
        'transforms': {
            'model_topology': {
                'available': ['cleanup', 'streamline', 'constant_folding'],
                'exploration': {
                    'required': ['cleanup'],
                    'optional': ['streamline', 'constant_folding']
                }
            },
            'dataflow_partitioning': {
                'available': ['automatic', 'manual'],
                'exploration': {
                    'required': ['automatic']
                }
            },
            'hw_kernel_transforms': {
                'available': ['buffer_insertion', 'pipeline_balancing'],
                'exploration': {
                    'optional': ['buffer_insertion', 'pipeline_balancing']
                }
            },
            'board_deployment': {
                'available': ['ZCU104', 'U250'],
                'exploration': {
                    'required': ['ZCU104']
                }
            }
        },
        'objectives': [
            {
                'name': 'throughput',
                'direction': 'maximize',
                'weight': 0.7
            },
            {
                'name': 'resource_efficiency',
                'direction': 'maximize',
                'weight': 0.3
            }
        ],
        'constraints': [
            {
                'name': 'latency',
                'operator': '<=',
                'value': 10.0
            },
            {
                'name': 'lut_utilization',
                'operator': '<',
                'value': 0.8
            }
        ],
        'device': 'xczu7ev-ffvc1156-2-e'
    }


def get_minimal_legacy_blueprint():
    """Get a minimal legacy blueprint for testing."""
    return {
        'finn_config': {
            'build_steps': ['step_test']
        }
    }


def get_minimal_six_entrypoint_blueprint():
    """Get a minimal 6-entrypoint blueprint for testing."""
    return {
        'nodes': {
            'canonical_ops': ['MatMul']
        },
        'transforms': {
            'model_topology': ['cleanup']
        }
    }


def get_invalid_blueprint():
    """Get an invalid blueprint that should fail detection."""
    return {
        'name': 'invalid',
        'some_other_field': 'value'
    }