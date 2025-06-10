"""
Basic usage example for Interface-Wise Dataflow Modeling Framework

This example demonstrates how to create interfaces, build a dataflow model,
and perform unified computational calculations with constraint support.
"""

from brainsmith.dataflow.core.dataflow_interface import (
    DataflowInterface,
    DataflowInterfaceType,
    DataflowDataType,
    DataTypeConstraint
)
from brainsmith.dataflow.core.dataflow_model import DataflowModel
from brainsmith.dataflow.core.block_chunking import TensorChunking

def main():
    """Demonstrate basic dataflow framework usage"""
    
    print("Interface-Wise Dataflow Modeling Framework - Basic Usage Example")
    print("=" * 70)
    
    # Step 1: Create datatype with constraint support
    print("\n1. Creating datatype with constraints...")
    
    # Create a datatype
    input_dtype = DataflowDataType(
        base_type="INT",
        bitwidth=8,
        signed=True,
        finn_type=""
    )
    print(f"   Created datatype: {input_dtype.finn_type}")
    
    # Create datatype constraints
    flexible_constraint = DataTypeConstraint(
        base_types=["INT", "UINT"],
        min_bitwidth=4,
        max_bitwidth=16,
        signed_allowed=True,
        unsigned_allowed=True
    )
    print(f"   Created constraint: {flexible_constraint.base_types}, {flexible_constraint.min_bitwidth}-{flexible_constraint.max_bitwidth} bits")
    
    # Step 2: Create dataflow interfaces
    print("\n2. Creating dataflow interfaces...")
    
    # Input interface
    input_interface = DataflowInterface(
        name="input0",
        interface_type=DataflowInterfaceType.INPUT,
        tensor_dims=[64],
        block_dims=[16],
        stream_dims=[4],
        dtype=input_dtype,
        allowed_datatypes=[flexible_constraint]
    )
    print(f"   Input interface: {input_interface}")
    
    # Weight interface
    weight_interface = DataflowInterface(
        name="weights",
        interface_type=DataflowInterfaceType.WEIGHT,
        tensor_dims=[128],
        block_dims=[32],
        stream_dims=[8],
        dtype=input_dtype,
        allowed_datatypes=[flexible_constraint]
    )
    print(f"   Weight interface: {weight_interface}")
    
    # Output interface
    output_interface = DataflowInterface(
        name="output0",
        interface_type=DataflowInterfaceType.OUTPUT,
        tensor_dims=[64],
        block_dims=[16],
        stream_dims=[4],
        dtype=input_dtype,
        allowed_datatypes=[flexible_constraint]
    )
    print(f"   Output interface: {output_interface}")
    
    # Step 3: Validate interfaces
    print("\n3. Validating interface constraints...")
    
    for interface in [input_interface, weight_interface, output_interface]:
        result = interface.validate_constraints()
        if result.success:
            print(f"   ✓ {interface.name}: Valid")
        else:
            print(f"   ✗ {interface.name}: {len(result.errors)} errors")
            for error in result.errors:
                print(f"     - {error.message}")
    
    # Step 4: Test datatype constraint validation
    print("\n4. Testing datatype constraint validation...")
    
    # Valid datatype
    valid_dtype = DataflowDataType("UINT", 8, False, "")
    if input_interface.validate_datatype(valid_dtype):
        print(f"   ✓ {valid_dtype.finn_type}: Allowed")
    else:
        print(f"   ✗ {valid_dtype.finn_type}: Not allowed")
    
    # Invalid datatype (too many bits)
    invalid_dtype = DataflowDataType("INT", 32, True, "")
    if input_interface.validate_datatype(invalid_dtype):
        print(f"   ✓ {invalid_dtype.finn_type}: Allowed")
    else:
        print(f"   ✗ {invalid_dtype.finn_type}: Not allowed (exceeds max bitwidth)")
    
    # Step 5: Create dataflow model
    print("\n5. Creating dataflow model...")
    
    interfaces = [input_interface, weight_interface, output_interface]
    parameters = {"kernel_size": 3, "stride": 1}
    
    model = DataflowModel(interfaces, parameters)
    print(f"   Model created with {len(model.interfaces)} interfaces")
    print(f"   - Input interfaces: {len(model.input_interfaces)}")
    print(f"   - Weight interfaces: {len(model.weight_interfaces)}")
    print(f"   - Output interfaces: {len(model.output_interfaces)}")
    
    # Step 6: Unified initiation interval calculations
    print("\n6. Performing unified initiation interval calculations...")
    
    # Define parallelism parameters
    iPar = {"input0": 4}
    wPar = {"weights": 8}
    
    # Calculate using unified method
    intervals = model.calculate_initiation_intervals(iPar, wPar)
    
    print(f"   Calculation Initiation Intervals (cII):")
    for interface_name, cii in intervals.cII.items():
        print(f"     {interface_name}: {cii}")
    
    print(f"   Execution Initiation Intervals (eII):")
    for interface_name, eii in intervals.eII.items():
        print(f"     {interface_name}: {eii}")
    
    print(f"   Overall Latency (L): {intervals.L}")
    
    print(f"   Bottleneck Analysis:")
    bottleneck = intervals.bottleneck_analysis
    print(f"     Bottleneck input: {bottleneck['bottleneck_input']}")
    print(f"     Bottleneck eII: {bottleneck['bottleneck_eII']}")
    print(f"     Total inputs: {bottleneck['total_inputs']}")
    print(f"     Total weights: {bottleneck['total_weights']}")
    
    # Step 7: Generate parallelism bounds for FINN optimization
    print("\n7. Generating parallelism bounds for FINN optimization...")
    
    bounds = model.get_parallelism_bounds()
    
    for param_name, bound in bounds.items():
        print(f"   {param_name}:")
        print(f"     Range: [{bound.min_value}, {bound.max_value}]")
        print(f"     Valid divisors: {bound.divisibility_constraints[:5]}{'...' if len(bound.divisibility_constraints) > 5 else ''}")
    
    # Step 8: Demonstrate tensor chunking
    print("\n8. Demonstrating tensor chunking...")
    
    # ONNX layout inference
    onnx_layout = "[N, C, H, W]"
    shape = [1, 64, 32, 32]  # Batch=1, Channels=64, Height=32, Width=32
    
    tensor_dims, block_dims = TensorChunking.infer_dimensions(onnx_layout, shape)
    print(f"   ONNX layout: {onnx_layout}")
    print(f"   Shape: {shape}")
    print(f"   Inferred tensor_dims: {tensor_dims}")
    print(f"   Inferred block_dims: {block_dims}")
    
    # Step 9: Calculate AXI signal specifications
    print("\n9. Generating AXI signal specifications...")
    
    input_signals = input_interface.get_axi_signals()
    print(f"   Input interface signals:")
    for signal_name, signal_info in input_signals.items():
        print(f"     {signal_name}: {signal_info['direction']} ({signal_info['width']} bits)")
    
    output_signals = output_interface.get_axi_signals()
    print(f"   Output interface signals:")
    for signal_name, signal_info in output_signals.items():
        print(f"     {signal_name}: {signal_info['direction']} ({signal_info['width']} bits)")
    
    # Step 10: Resource estimation
    print("\n10. Resource estimation...")
    
    print(f"   Memory footprints:")
    for interface in [input_interface, weight_interface]:
        footprint = interface.get_memory_footprint()
        print(f"     {interface.name}: {footprint} bits ({footprint // 8} bytes)")
    
    print(f"   Transfer cycles:")
    for interface in [input_interface, weight_interface, output_interface]:
        cycles = interface.get_transfer_cycles()
        print(f"     {interface.name}: {cycles} cycles")
    
    print(f"\n✓ Basic usage example completed successfully!")
    print("  The framework provides:")
    print("  - Unified computational model with single calculate_initiation_intervals() method")
    print("  - Datatype constraint system for RTL creator flexibility")
    print("  - FINN optimization integration via parallelism bounds")
    print("  - Comprehensive validation and error handling")

if __name__ == "__main__":
    main()
