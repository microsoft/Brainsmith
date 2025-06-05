#!/usr/bin/env python3
"""
Simple validation script for Phase 1 implementation

This script tests the core functionality of the dataflow framework
implementation to ensure Phase 1 objectives are met.
"""

import sys
import traceback

def test_imports():
    """Test that all core modules can be imported"""
    print("Testing imports...")
    
    try:
        from brainsmith.dataflow.core.validation import ValidationError, ValidationResult, ValidationSeverity
        print("  ✓ validation module")
    except Exception as e:
        print(f"  ✗ validation module: {e}")
        return False
    
    try:
        from brainsmith.dataflow.core.dataflow_interface import (
            DataflowInterface, DataflowInterfaceType, DataflowDataType, DataTypeConstraint
        )
        print("  ✓ dataflow_interface module")
    except Exception as e:
        print(f"  ✗ dataflow_interface module: {e}")
        return False
        
    try:
        from brainsmith.dataflow.core.dataflow_model import DataflowModel, InitiationIntervals
        print("  ✓ dataflow_model module")
    except Exception as e:
        print(f"  ✗ dataflow_model module: {e}")
        return False
        
    try:
        from brainsmith.dataflow.core.tensor_chunking import TensorChunking, TDimPragma
        print("  ✓ tensor_chunking module")
    except Exception as e:
        print(f"  ✗ tensor_chunking module: {e}")
        return False
    
    return True

def test_datatype_functionality():
    """Test DataflowDataType and DataTypeConstraint functionality"""
    print("\nTesting datatype functionality...")
    
    try:
        from brainsmith.dataflow.core.dataflow_interface import DataflowDataType, DataTypeConstraint
        
        # Test datatype creation
        dtype = DataflowDataType("INT", 8, True, "")
        assert dtype.finn_type == "INT8", f"Expected INT8, got {dtype.finn_type}"
        print("  ✓ DataflowDataType creation")
        
        # Test constraint creation
        constraint = DataTypeConstraint(
            base_types=["INT", "UINT"],
            min_bitwidth=1,
            max_bitwidth=32,
            signed_allowed=True,
            unsigned_allowed=True
        )
        print("  ✓ DataTypeConstraint creation")
        
        # Test constraint validation
        valid = constraint.is_valid_datatype(dtype)
        assert valid, "Expected datatype to be valid"
        print("  ✓ Constraint validation")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Datatype functionality: {e}")
        traceback.print_exc()
        return False

def test_interface_functionality():
    """Test DataflowInterface functionality"""
    print("\nTesting interface functionality...")
    
    try:
        from brainsmith.dataflow.core.dataflow_interface import (
            DataflowInterface, DataflowInterfaceType, DataflowDataType, DataTypeConstraint
        )
        
        # Create datatype and constraint
        dtype = DataflowDataType("INT", 8, True, "")
        constraint = DataTypeConstraint(
            base_types=["INT", "UINT"],
            min_bitwidth=1,
            max_bitwidth=32,
            signed_allowed=True,
            unsigned_allowed=True
        )
        
        # Create interface
        interface = DataflowInterface(
            name="test_input",
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[64],
            tDim=[16],
            sDim=[4],
            dtype=dtype,
            allowed_datatypes=[constraint]
        )
        print("  ✓ DataflowInterface creation")
        
        # Test validation
        result = interface.validate_constraints()
        assert result.success, f"Interface validation failed: {[e.message for e in result.errors]}"
        print("  ✓ Interface constraint validation")
        
        # Test stream width calculation
        width = interface.calculate_stream_width()
        expected_width = 32  # 4 elements * 8 bits = 32 bits
        assert width == expected_width, f"Expected width {expected_width}, got {width}"
        print("  ✓ Stream width calculation")
        
        # Test AXI signal generation
        signals = interface.get_axi_signals()
        assert "test_input_TDATA" in signals, "Missing TDATA signal"
        assert "test_input_TVALID" in signals, "Missing TVALID signal"
        assert "test_input_TREADY" in signals, "Missing TREADY signal"
        print("  ✓ AXI signal generation")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Interface functionality: {e}")
        traceback.print_exc()
        return False

def test_unified_model_functionality():
    """Test DataflowModel unified computational functionality"""
    print("\nTesting unified model functionality...")
    
    try:
        from brainsmith.dataflow.core.dataflow_interface import (
            DataflowInterface, DataflowInterfaceType, DataflowDataType
        )
        from brainsmith.dataflow.core.dataflow_model import DataflowModel
        
        # Create interfaces
        dtype = DataflowDataType("INT", 8, True, "")
        
        input_interface = DataflowInterface(
            name="input0",
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[64],
            tDim=[16],
            sDim=[4],
            dtype=dtype
        )
        
        weight_interface = DataflowInterface(
            name="weights",
            interface_type=DataflowInterfaceType.WEIGHT,
            qDim=[128],
            tDim=[32],
            sDim=[8],
            dtype=dtype
        )
        
        output_interface = DataflowInterface(
            name="output0",
            interface_type=DataflowInterfaceType.OUTPUT,
            qDim=[64],
            tDim=[16],
            sDim=[4],
            dtype=dtype
        )
        
        # Create model
        model = DataflowModel([input_interface, weight_interface, output_interface], {})
        print("  ✓ DataflowModel creation")
        
        # Test interface organization
        assert len(model.input_interfaces) == 1, f"Expected 1 input, got {len(model.input_interfaces)}"
        assert len(model.weight_interfaces) == 1, f"Expected 1 weight, got {len(model.weight_interfaces)}"
        assert len(model.output_interfaces) == 1, f"Expected 1 output, got {len(model.output_interfaces)}"
        print("  ✓ Interface organization")
        
        # Test unified calculation
        iPar = {"input0": 4}
        wPar = {"weights": 8}
        
        intervals = model.calculate_initiation_intervals(iPar, wPar)
        
        # Verify structure
        assert "input0" in intervals.cII, "Missing cII for input0"
        assert "input0" in intervals.eII, "Missing eII for input0"
        assert intervals.L > 0, f"Invalid latency: {intervals.L}"
        assert "bottleneck_input" in intervals.bottleneck_analysis, "Missing bottleneck analysis"
        
        print("  ✓ Unified initiation interval calculation")
        
        # Test parallelism bounds
        bounds = model.get_parallelism_bounds()
        assert "input0_iPar" in bounds, "Missing input parallelism bounds"
        assert "weights_wPar" in bounds, "Missing weight parallelism bounds"
        print("  ✓ Parallelism bounds generation")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Unified model functionality: {e}")
        traceback.print_exc()
        return False

def test_tensor_chunking_functionality():
    """Test TensorChunking functionality"""
    print("\nTesting tensor chunking functionality...")
    
    try:
        from brainsmith.dataflow.core.tensor_chunking import TensorChunking, TDimPragma
        
        # Test ONNX layout inference
        onnx_layout = "[N, C, H, W]"
        shape = [1, 64, 32, 32]
        
        qDim, tDim = TensorChunking.infer_dimensions(onnx_layout, shape)
        assert len(qDim) > 0, "qDim should not be empty"
        assert len(tDim) > 0, "tDim should not be empty"
        print("  ✓ ONNX layout inference")
        
        # Test TDIM pragma
        pragma = TDimPragma("test_interface", ["param1 * 2", "param2 + 1"])
        parameters = {"param1": 8, "param2": 15}
        
        evaluated = pragma.evaluate_expressions(parameters)
        expected = [16, 16]  # param1 * 2 = 8 * 2 = 16, param2 + 1 = 15 + 1 = 16
        assert evaluated == expected, f"Expected {expected}, got {evaluated}"
        print("  ✓ TDIM pragma evaluation")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Tensor chunking functionality: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all Phase 1 validation tests"""
    print("Phase 1 Implementation Validation")
    print("=" * 40)
    
    all_passed = True
    
    # Run all test functions
    test_functions = [
        test_imports,
        test_datatype_functionality,
        test_interface_functionality, 
        test_unified_model_functionality,
        test_tensor_chunking_functionality
    ]
    
    for test_func in test_functions:
        try:
            passed = test_func()
            all_passed = all_passed and passed
        except Exception as e:
            print(f"Test {test_func.__name__} failed with exception: {e}")
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ Phase 1 Implementation: ALL TESTS PASSED")
        print("\nPhase 1 Success Criteria Met:")
        print("  ✓ Core data structures implemented with constraint support")
        print("  ✓ Unified mathematical framework operational")
        print("  ✓ Datatype constraint system functional")
        print("  ✓ Enhanced tensor chunking with TDIM pragma support")
        print("  ✓ FINN optimization integration via parallelism bounds")
        print("\nReady to proceed to Phase 2: Integration")
    else:
        print("✗ Phase 1 Implementation: SOME TESTS FAILED")
        print("Please review and fix issues before proceeding to Phase 2")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
