#!/usr/bin/env python3
"""
Test script for direct factory implementation.
"""

from qonnx.core.datatype import DataType
from brainsmith.core.dataflow import (
    KernelSchema, InputSchema, OutputSchema,
    DatatypeConstraintGroup, RelationType,
    TensorContext, TensorInfo,
    build_kernel_model_direct,
    KernelBuilder
)
from brainsmith.core.dataflow.direct_factory import DirectKernelFactory


def test_direct_factory():
    """Test the direct factory architecture."""
    
    # 1. Create a simple kernel schema
    schema = KernelSchema(name="TestConv")
    
    schema.add_input(InputSchema(
        name="input",
        block_tiling=["BATCH", "CHANNELS", "HEIGHT", "WIDTH"],
        stream_tiling=["SIMD"],
        datatype_constraints=[
            DatatypeConstraintGroup("INT", 8, 16),
            DatatypeConstraintGroup("UINT", 8, 16)
        ]
    ))
    
    schema.add_input(InputSchema(
        name="weights",
        block_tiling=["PE", "CHANNELS", "K", "K"],
        stream_tiling=["SIMD"],
        is_weight=True
    ))
    
    schema.add_output(OutputSchema(
        name="output",
        block_tiling=["BATCH", "PE", ":", ":"],
        datatype_constraints=[
            DatatypeConstraintGroup("INT", 16, 32)
        ]
    ))
    
    # Add relationships
    schema.add_relationship(
        "input", "output", RelationType.EQUAL,
        source_dim=0, target_dim=0  # Batch dims must match
    )
    
    # 2. Create a mock tensor context
    tensor_context = TensorContext(
        inputs={
            "input": TensorInfo(
                shape=[1, 64, 28, 28],
                datatype=DataType["INT8"]
            ),
            "weights": TensorInfo(
                shape=[16, 64, 3, 3],
                datatype=DataType["INT8"]
            )
        },
        outputs={
            "output": TensorInfo(
                shape=[1, 16, 26, 26],
                datatype=DataType["INT32"]
            )
        }
    )
    
    # 3. Define nodeattrs
    nodeattrs = {
        "BATCH": 1,
        "CHANNELS": 64,
        "HEIGHT": 28,
        "WIDTH": 28,
        "PE": 16,
        "K": 3,
        "SIMD": 8,
        "outputDatatype": "INT16"  # Override output type
    }
    
    # 4. Test direct factory
    print("Testing DirectKernelFactory...")
    try:
        model = DirectKernelFactory.create_model(schema, tensor_context, nodeattrs)
        print("✓ Direct factory succeeded")
        print(f"  Input shape: {model.inputs[0].tensor_dims}")
        print(f"  Input block: {model.inputs[0].block_dims}")
        print(f"  Input stream: {model.inputs[0].stream_dims}")
        print(f"  Output datatype: {model.outputs[0].datatype}")
    except Exception as e:
        print(f"✗ Direct factory failed: {e}")
        return False
    
    # 5. Test builder with direct flow
    print("\nTesting KernelBuilder with direct flow...")
    try:
        model2 = (KernelBuilder()
            .from_schema(schema)
            .with_tensor_context(tensor_context)
            .with_nodeattrs(nodeattrs)
            .build())
        print("✓ Builder direct flow succeeded")
    except Exception as e:
        print(f"✗ Builder direct flow failed: {e}")
        return False
    
    # 6. Test convenience function
    print("\nTesting build_kernel_model_direct...")
    try:
        model3 = build_kernel_model_direct(schema, tensor_context, nodeattrs)
        print("✓ Convenience function succeeded")
    except Exception as e:
        print(f"✗ Convenience function failed: {e}")
        return False
    
    # 7. Verify all models are equivalent
    assert model.inputs[0].tensor_dims == model2.inputs[0].tensor_dims == model3.inputs[0].tensor_dims
    assert model.outputs[0].datatype == model2.outputs[0].datatype == model3.outputs[0].datatype
    print("\n✓ All models are equivalent")
    
    return True


if __name__ == "__main__":
    success = test_direct_factory()
    if success:
        print("\n✅ Direct factory tests passed!")
    else:
        print("\n❌ Direct factory tests failed!")
        exit(1)