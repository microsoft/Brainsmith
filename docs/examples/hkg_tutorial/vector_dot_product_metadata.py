"""
Metadata file for Vector Dot Product Hardware Kernel
 
This file provides the Hardware Kernel Generator with essential information
about the RTL module for automatic FINN component generation.
"""

# Operation identification
operation_name = "VectorDotProduct"
operation_type = "dot_product"
description = "High-performance vector dot product accelerator for neural network inference"

# Hardware characteristics
target_device = "ultrascale_plus"
target_frequency = 250  # MHz
parallelism_factor = 8

# Data type configuration
input_precision = 8     # INT8 inputs
output_precision = 32   # INT32 result
accumulator_width = 32

# Dimensional configuration
vector_size = 768       # BERT hidden dimension
chunk_size = 96        # Processing granularity (768/8)
stream_width = 8       # Parallel elements per cycle

# Interface configuration
interfaces = {
    "s_axis_a": {
        "type": "INPUT",
        "protocol": "AXI_STREAM", 
        "role": "primary_input",
        "data_width": 64,  # 8 elements * 8 bits
        "qDim": 768,
        "tDim": 96,
        "stream_dims": 8,
        "dtype": "INT8"
    },
    "s_axis_b": {
        "type": "INPUT", 
        "protocol": "AXI_STREAM",
        "role": "secondary_input", 
        "data_width": 64,
        "qDim": 768,
        "tDim": 96,
        "stream_dims": 8,
        "dtype": "INT8"
    },
    "m_axis_result": {
        "type": "OUTPUT",
        "protocol": "AXI_STREAM",
        "role": "result_output",
        "data_width": 32,
        "qDim": 1,
        "tDim": 1, 
        "stream_dims": 1,
        "dtype": "INT32"
    },
    "config": {
        "type": "CONFIG",
        "protocol": "AXI_LITE",
        "role": "configuration",
        "address_width": 32,
        "data_width": 32
    },
    "control": {
        "type": "CONTROL",
        "protocol": "SIMPLE",
        "role": "status_control"
    }
}

# Performance characteristics
performance_metrics = {
    "latency_cycles": 96,           # Cycles to process one vector pair
    "throughput_ops_per_cycle": 1,  # Dot products per cycle
    "initiation_interval": 96,     # Cycles between starting new operations
    "pipeline_depth": 3            # Internal pipeline stages
}

# Resource estimation hints
resource_hints = {
    "lut_estimate": 2500,          # Conservative LUT estimate
    "dsp_estimate": 8,             # One DSP per parallel multiplier
    "bram_estimate": 0,            # No internal memory
    "uram_estimate": 0,
    "ff_estimate": 1200           # Flip-flop estimate
}

# Optimization configuration
optimization_config = {
    "mode": "balanced",            # balanced, latency, throughput, area
    "resource_sharing": True,      # Enable resource sharing optimizations
    "pipeline_balancing": True,    # Enable automatic pipeline balancing
    "clock_gating": False          # Disable for timing closure
}

# Validation configuration
validation_config = {
    "enable_assertions": True,     # Include SystemVerilog assertions
    "performance_monitoring": True, # Include performance counters
    "debug_interfaces": False,     # Exclude debug for production
    "formal_verification": True    # Enable formal property generation
}

# FINN integration specifics
finn_integration = {
    "model_precision": "INT8",
    "weight_precision": "INT8", 
    "activation_precision": "INT8",
    "output_precision": "INT32",
    "folding_factor": 8,           # Maps to stream_dims
    "simd_factor": 8,              # Parallel operations
    "pe_count": 1,                 # Single processing element
    "memory_mode": "external"      # No internal weight storage
}

# Test generation configuration
test_config = {
    "generate_unit_tests": True,
    "generate_integration_tests": True,
    "generate_performance_tests": True,
    "test_vectors": {
        "num_random_tests": 100,
        "edge_cases": True,
        "performance_benchmarks": True
    },
    "simulation_cycles": 10000
}

# Documentation configuration
documentation_config = {
    "generate_readme": True,
    "include_performance_analysis": True,
    "include_resource_analysis": True,
    "include_integration_guide": True,
    "include_timing_constraints": True
}

# Custom configuration for this specific operation
custom_config = {
    "attention_mechanism_support": True,  # Optimized for attention calculations
    "bert_compatibility": True,           # BERT model integration
    "quantization_aware": True,           # Supports quantized models
    "streaming_interface": True           # Continuous streaming operation
}

# Compiler data for FINN integration
finn_compiler_data = {
    "node_type": "VectorDotProduct",
    "input_shapes": [[1, 768], [1, 768]],  # Two input vectors
    "output_shapes": [[1, 1]],             # Single scalar result
    "attributes": {
        "vector_size": vector_size,
        "parallelism": parallelism_factor,
        "precision": input_precision
    }
}

# Export all configuration for HKG
__all__ = [
    'operation_name', 'operation_type', 'description',
    'target_device', 'target_frequency', 'parallelism_factor',
    'input_precision', 'output_precision', 'accumulator_width',
    'vector_size', 'chunk_size', 'stream_width',
    'interfaces', 'performance_metrics', 'resource_hints',
    'optimization_config', 'validation_config', 'finn_integration',
    'test_config', 'documentation_config', 'custom_config',
    'finn_compiler_data'
]