#!/usr/bin/env python3
"""Test performance improvements in nodeattr system."""

import time
from typing import List
import onnx.helper as helper

from brainsmith.core.dataflow import (
    KernelSchema,
    InputSchema,
    OutputSchema,
    SchemaCompiler
)
from brainsmith.core.finn.auto_hw_custom_op import AutoHWCustomOp


# Create test kernel schema
test_schema = KernelSchema(
    name="TestConv",
    inputs=[
        InputSchema(
            name="X",
            block_tiling=[":", ":", "PE", "SIMD"],
            stream_tiling=[":", ":", ":", "SIMD"],
        ),
        InputSchema(
            name="W",
            block_tiling=[":", ":", "PE", "SIMD", "PE_OUT"],
            is_weight=True
        )
    ],
    outputs=[
        OutputSchema(
            name="Y",
            block_tiling=[":", ":", ":", "PE_OUT"]
        )
    ]
)


class TestOp(AutoHWCustomOp):
    """Test custom op for benchmarking."""
    kernel_schema = test_schema
    
    def get_nodeattr_types(self):
        return {
            "PE": ("i", True, 8),
            "SIMD": ("i", True, 4),
            "PE_OUT": ("i", True, 16),
            "K": ("i", False, 3),
            "S": ("i", False, 1),
            "CHANNELS": ("i", False, 32),
            "clock_freq_mhz": ("f", False, 200.0),
            "input0Datatype": ("s", False, "INT8"),
            "input1Datatype": ("s", False, "INT8"),
            "output0Datatype": ("s", False, "INT8"),
        }
    
    # Implement required abstract methods (minimal implementation)
    def make_shape_compatible_op(self, model): pass
    def infer_node_datatype(self, model): pass
    def execute_node(self, context, graph): pass
    def verify_node(self): pass
    def get_exp_cycles(self): return 1000
    def get_folded_input_shape(self, ind=0): return [1, 32, 32, 8, 4]
    def get_folded_output_shape(self, ind=0): return [1, 32, 32, 16]
    def get_instream_width(self, ind=0): return 32
    def get_outstream_width(self, ind=0): return 128
    def get_number_output_values(self): return 32*32*64
    

def benchmark_set_nodeattr(op: AutoHWCustomOp, iterations: int = 10000) -> float:
    """Benchmark set_nodeattr performance."""
    attrs = ["PE", "SIMD", "PE_OUT", "K", "S", "CHANNELS", "clock_freq_mhz"]
    values = [8, 4, 16, 3, 1, 32, 200.0]
    
    start = time.time()
    for i in range(iterations):
        attr_idx = i % len(attrs)
        # Set to same value (no actual change) to measure overhead
        op.set_nodeattr(attrs[attr_idx], values[attr_idx])
    end = time.time()
    
    return (end - start) / iterations * 1_000_000  # microseconds per call


def benchmark_changing_values(op: AutoHWCustomOp, iterations: int = 1000) -> float:
    """Benchmark with actual value changes."""
    start = time.time()
    for i in range(iterations):
        # Actually change PE value
        op.set_nodeattr("PE", 8 + (i % 4))
    end = time.time()
    
    return (end - start) / iterations * 1_000_000  # microseconds per call


def test_compiled_schema():
    """Test compiled schema functionality."""
    compiled = SchemaCompiler.compile(test_schema)
    
    print("Compiled Schema Analysis:")
    print(f"  Total parameters: {len(compiled.all_parameters)}")
    print(f"  Parameters: {sorted(compiled.all_parameters)}")
    print()
    
    # Test dependency mapping
    print("Dependency Analysis:")
    test_attrs = ["PE", "SIMD", "clock_freq_mhz", "input0Datatype"]
    for attr in test_attrs:
        affected = compiled.get_affected_caches(attr)
        print(f"  {attr} affects: {affected}")
    print()
    
    # Test O(1) lookups
    print("Lookup Performance Test:")
    attrs_to_test = ["PE", "SIMD", "not_a_param", "clock_freq_mhz"]
    
    start = time.time()
    for _ in range(100000):
        for attr in attrs_to_test:
            _ = compiled.is_model_affecting(attr)
    end = time.time()
    
    print(f"  100k lookups x 4 attrs: {(end - start) * 1000:.2f} ms")
    print(f"  Per lookup: {(end - start) / 400000 * 1_000_000:.2f} μs")


def main():
    """Run performance tests."""
    print("=== Nodeattr Performance Test ===\n")
    
    # Test compiled schema
    test_compiled_schema()
    print()
    
    # Create test node
    node = helper.make_node(
        "TestOp",
        inputs=["x", "w"],
        outputs=["y"],
        PE=8,
        SIMD=4,
        PE_OUT=16
    )
    
    # Create test operator
    op = TestOp(node)
    
    # Warm up
    for _ in range(100):
        op.set_nodeattr("PE", 8)
    
    # Benchmark no-change scenario
    print("Benchmark Results:")
    no_change_time = benchmark_set_nodeattr(op, 10000)
    print(f"  No-change set_nodeattr: {no_change_time:.2f} μs/call")
    
    # Benchmark with changes
    change_time = benchmark_changing_values(op, 1000)
    print(f"  Value-changing set_nodeattr: {change_time:.2f} μs/call")
    
    # Test targeted invalidation
    print("\nTargeted Invalidation Test:")
    
    # Set up initial state
    op.refresh_nodeattr_config()
    
    # Test clock_freq_mhz (should only affect kernel_model)
    start = time.time()
    op.set_nodeattr("clock_freq_mhz", 250.0)
    end = time.time()
    print(f"  clock_freq_mhz change: {(end - start) * 1_000_000:.2f} μs")
    
    # Test PE (should affect all caches)
    start = time.time()
    op.set_nodeattr("PE", 16)
    end = time.time()
    print(f"  PE change: {(end - start) * 1_000_000:.2f} μs")
    
    print("\n✓ Performance tests completed")


if __name__ == "__main__":
    main()