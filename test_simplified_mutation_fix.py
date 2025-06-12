#!/usr/bin/env python3
"""
Test script to verify the simplified interface mutation fix.

This tests that the new atomic parallelism update approach works correctly
and doesn't suffer from the state corruption issues of the original implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from brainsmith.dataflow.core.dataflow_interface import DataflowInterface, DataflowDataType
from brainsmith.dataflow.core.dataflow_model import DataflowModel
from brainsmith.dataflow.core.interface_types import InterfaceType

def test_atomic_parallelism_updates():
    """Test that parallelism updates are atomic and don't corrupt state."""
    print("Testing atomic parallelism updates...")
    
    # Create test interfaces
    dtype = DataflowDataType(base_type="UINT", bitwidth=8, signed=False, finn_type="UINT8")
    
    input_interface = DataflowInterface(
        name="in0",
        interface_type=InterfaceType.INPUT,
        tensor_dims=[32],
        block_dims=[8],
        stream_dims=[1],
        dtype=dtype
    )
    
    weight_interface = DataflowInterface(
        name="weights",
        interface_type=InterfaceType.WEIGHT,
        tensor_dims=[16],
        block_dims=[4],
        stream_dims=[1],
        dtype=dtype
    )
    
    output_interface = DataflowInterface(
        name="out0",
        interface_type=InterfaceType.OUTPUT,
        tensor_dims=[32],
        block_dims=[8],
        stream_dims=[1],
        dtype=dtype
    )
    
    # Create model
    interfaces = [input_interface, weight_interface, output_interface]
    model = DataflowModel(interfaces, {})
    
    # Store original stream_dims
    original_input_stream = input_interface.stream_dims.copy()
    original_weight_stream = weight_interface.stream_dims.copy()
    original_output_stream = output_interface.stream_dims.copy()
    
    print(f"Original stream dims - Input: {original_input_stream}, Weight: {original_weight_stream}, Output: {original_output_stream}")
    
    # Test 1: Apply parallelism and verify atomic update
    intervals1 = model.apply_parallelism({"in0": 4}, {"weights": 2})
    
    print(f"After first apply_parallelism - Input: {input_interface.stream_dims}, Weight: {weight_interface.stream_dims}, Output: {output_interface.stream_dims}")
    print(f"First calculation results - cII: {intervals1.cII}, eII: {intervals1.eII}, L: {intervals1.L}")
    
    # Verify stream dims were updated
    assert input_interface.stream_dims[0] == 4, f"Expected input stream_dims[0] = 4, got {input_interface.stream_dims[0]}"
    assert weight_interface.stream_dims[0] == 8, f"Expected weight stream_dims[0] = 8 (2*4*1), got {weight_interface.stream_dims[0]}"
    
    # Test 2: Apply different parallelism and verify state is completely replaced
    intervals2 = model.apply_parallelism({"in0": 2}, {"weights": 1})
    
    print(f"After second apply_parallelism - Input: {input_interface.stream_dims}, Weight: {weight_interface.stream_dims}, Output: {output_interface.stream_dims}")
    print(f"Second calculation results - cII: {intervals2.cII}, eII: {intervals2.eII}, L: {intervals2.L}")
    
    # Verify stream dims were updated to new values
    assert input_interface.stream_dims[0] == 2, f"Expected input stream_dims[0] = 2, got {input_interface.stream_dims[0]}"
    assert weight_interface.stream_dims[0] == 2, f"Expected weight stream_dims[0] = 2 (1*2*1), got {weight_interface.stream_dims[0]}"
    
    # Test 3: Verify calculations are deterministic
    intervals3 = model.apply_parallelism({"in0": 4}, {"weights": 2})
    
    print(f"After third apply_parallelism - Input: {input_interface.stream_dims}, Weight: {weight_interface.stream_dims}, Output: {output_interface.stream_dims}")
    print(f"Third calculation results - cII: {intervals3.cII}, eII: {intervals3.eII}, L: {intervals3.L}")
    
    # Should match first calculation
    assert intervals3.cII == intervals1.cII, f"Non-deterministic cII: {intervals3.cII} != {intervals1.cII}"
    assert intervals3.eII == intervals1.eII, f"Non-deterministic eII: {intervals3.eII} != {intervals1.eII}"
    assert intervals3.L == intervals1.L, f"Non-deterministic L: {intervals3.L} != {intervals1.L}"
    
    # Test 4: Test cached results
    cached = model.get_current_intervals()
    assert cached is not None, "Cached intervals should not be None"
    assert cached.L == intervals3.L, f"Cached L doesn't match: {cached.L} != {intervals3.L}"
    
    # Test 5: Test reset functionality
    model.reset_parallelism()
    
    print(f"After reset - Input: {input_interface.stream_dims}, Weight: {weight_interface.stream_dims}, Output: {output_interface.stream_dims}")
    
    # Should be back to all 1s
    assert all(dim == 1 for dim in input_interface.stream_dims), f"Input stream_dims not reset: {input_interface.stream_dims}"
    assert all(dim == 1 for dim in weight_interface.stream_dims), f"Weight stream_dims not reset: {weight_interface.stream_dims}"
    assert all(dim == 1 for dim in output_interface.stream_dims), f"Output stream_dims not reset: {output_interface.stream_dims}"
    
    # Cached results should be cleared
    cached_after_reset = model.get_current_intervals()
    assert cached_after_reset is None, "Cached intervals should be None after reset"
    
    print("✅ All atomic parallelism tests passed!")

def test_sequential_calculation_safety():
    """Test that sequential calculations don't interfere with each other."""
    print("\nTesting sequential calculation safety...")
    
    # Create simple test interface
    dtype = DataflowDataType(base_type="UINT", bitwidth=8, signed=False, finn_type="UINT8")
    
    input_interface = DataflowInterface(
        name="in0",
        interface_type=InterfaceType.INPUT,
        tensor_dims=[64],
        block_dims=[16],
        stream_dims=[1],
        dtype=dtype
    )
    
    model = DataflowModel([input_interface], {})
    
    # Perform multiple sequential calculations
    results = []
    parallelisms = [1, 2, 4, 8, 4, 2, 1]  # Mix of values including repeats
    
    for i, par in enumerate(parallelisms):
        intervals = model.apply_parallelism({"in0": par}, {})
        results.append((par, intervals.cII["in0"], intervals.L))
        print(f"Calculation {i+1}: iPar={par}, cII={intervals.cII['in0']}, L={intervals.L}")
    
    # Verify identical parallelism produces identical results
    assert results[0][1] == results[6][1], f"Non-deterministic results for iPar=1: {results[0][1]} != {results[6][1]}"
    assert results[1][1] == results[5][1], f"Non-deterministic results for iPar=2: {results[1][1]} != {results[5][1]}"
    assert results[2][1] == results[4][1], f"Non-deterministic results for iPar=4: {results[2][1]} != {results[4][1]}"
    
    print("✅ Sequential calculation safety tests passed!")

def test_interface_reset_methods():
    """Test the new reset methods on DataflowInterface."""
    print("\nTesting interface reset methods...")
    
    dtype = DataflowDataType(base_type="UINT", bitwidth=8, signed=False, finn_type="UINT8")
    
    interface = DataflowInterface(
        name="test",
        interface_type=InterfaceType.INPUT,
        tensor_dims=[32],
        block_dims=[8],
        stream_dims=[4, 2],  # Multi-dimensional for testing
        dtype=dtype
    )
    
    # Test default_stream_dims property
    default_dims = interface.default_stream_dims
    assert default_dims == [1, 1], f"Expected [1, 1], got {default_dims}"
    
    # Modify stream_dims
    interface.stream_dims = [8, 4]
    assert interface.stream_dims == [8, 4], "Stream dims should be modified"
    
    # Test reset
    interface.reset_stream_dims()
    assert interface.stream_dims == [1, 1], f"Expected [1, 1] after reset, got {interface.stream_dims}"
    
    print("✅ Interface reset method tests passed!")

if __name__ == "__main__":
    print("Testing Simplified Interface Mutation Fix")
    print("=" * 50)
    
    try:
        test_atomic_parallelism_updates()
        test_sequential_calculation_safety()
        test_interface_reset_methods()
        
        print("\n🎉 All tests passed! The simplified interface mutation fix is working correctly.")
        print("\nKey benefits demonstrated:")
        print("- ✅ Atomic updates: All parallelism changes happen in one method")
        print("- ✅ State consistency: Stream dims and performance metrics always match")
        print("- ✅ Sequential safety: Each calculation is independent and deterministic")
        print("- ✅ No residual effects: Previous calculations don't affect new ones")
        print("- ✅ Simple API: Clear single-purpose methods")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)