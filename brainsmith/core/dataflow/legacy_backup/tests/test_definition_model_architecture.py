############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Tests for Definition/Model architecture split"""

import pytest
from brainsmith.core.dataflow.interface_definition import InterfaceDefinition
from brainsmith.core.dataflow.interface_model import InterfaceModel
from brainsmith.core.dataflow.kernel_definition import KernelDefinition
from brainsmith.core.dataflow.kernel_model import KernelModel
from brainsmith.core.dataflow.base import ParameterBinding
from brainsmith.core.dataflow.relationships import RelationType
from brainsmith.core.dataflow.types import InterfaceDirection, DataType


def test_interface_definition_validation():
    """Test interface definition constraint validation"""
    # Create valid interface definition
    intf_def = InterfaceDefinition(
        name="input",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("INT8"),
        alignment=64,
        min_dims=(16, 16),
        max_dims=(1024, 1024),
        granularity=(8, 8)
    )
    
    # Validation should pass
    errors = intf_def.validate()
    assert len(errors) == 0
    
    # Test constraint checking methods
    assert intf_def.has_constraint("alignment")
    assert intf_def.has_constraint("bounds")
    assert intf_def.has_constraint("granularity")
    assert not intf_def.has_constraint("dataflow")
    
    # Test constraint summary
    summary = intf_def.get_constraint_summary()
    assert summary["alignment"] == 64
    assert summary["min_dims"] == (16, 16)
    assert summary["max_dims"] == (1024, 1024)
    assert summary["granularity"] == (8, 8)


def test_interface_definition_model_validation():
    """Test interface definition validates model dimensions"""
    intf_def = InterfaceDefinition(
        name="test",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("INT8"),
        min_dims=(16, 16),
        max_dims=(512, 512),
        granularity=(8, 8)
    )
    
    # Valid dimensions
    errors = intf_def.validate_model_dimensions((64, 64))
    assert len(errors) == 0
    
    # Invalid dimensions (too small)
    errors = intf_def.validate_model_dimensions((8, 64))
    assert len(errors) > 0
    assert "< min=" in errors[0]
    
    # Invalid dimensions (too large)
    errors = intf_def.validate_model_dimensions((1024, 64))
    assert len(errors) > 0
    assert "> max=" in errors[0]
    
    # Invalid granularity
    errors = intf_def.validate_model_dimensions((65, 64))
    assert len(errors) > 0
    assert "not divisible by granularity" in errors[0]


def test_interface_model_creation():
    """Test creating interface model from definition"""
    intf_def = InterfaceDefinition(
        name="data",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("INT8"),
        alignment=64,
        min_dims=(16, 16),
        granularity=(8, 8)
    )
    
    # Create valid model
    model = intf_def.create_model(
        tensor_dims=(64, 64),
        block_dims=(8, 8), 
        stream_dims=(2, 2)
    )
    
    assert isinstance(model, InterfaceModel)
    assert model.definition == intf_def
    assert model.tensor_dims == (64, 64)
    assert model.block_dims == [(8, 8)]
    assert model.stream_dims == (2, 2)
    assert model.ipar == 4  # 2 * 2
    
    # Test invalid model creation
    with pytest.raises(ValueError, match="violate definition constraints"):
        intf_def.create_model(
            tensor_dims=(8, 64),  # Too small
            block_dims=(8, 8),
            stream_dims=(1, 1)
        )


def test_interface_model_performance():
    """Test interface model performance calculations"""
    # Create definition first
    intf_def = InterfaceDefinition(
        name="test",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("INT8")
    )
    
    # Create simple model for testing
    model = InterfaceModel(
        tensor_dims=(128, 128),
        block_dims=(16, 16),
        stream_dims=(4, 4),
        actual_utilization=0.8,
        definition=intf_def
    )
    
    # Test basic properties
    assert model.ipar == 16  # 4 * 4
    assert model.n_phases == 1
    assert not model.is_csdf
    
    # Test performance calculations
    metrics = model.calculate_performance_metrics()
    assert metrics["interface_parallelism"] == 16
    assert metrics["actual_utilization"] == 0.8
    assert metrics["total_tensor_size"] == 16384  # 128 * 128
    
    # Test bandwidth calculations
    assert model.bandwidth_bytes == 16.0  # 16 parallelism * 8 bits / 8 = 16 bytes per cycle
    effective_bw = model.effective_bandwidth(100.0)  # 100 MHz
    assert effective_bw == 0.8 * 16.0 * 100.0  # utilization * bytes_per_cycle * freq_mhz


def test_kernel_definition_creation():
    """Test creating kernel definition with relationships"""
    # Create interface definitions
    input_def = InterfaceDefinition(
        name="input",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("INT8"),
        min_dims=(16, 16)
    )
    
    weights_def = InterfaceDefinition(
        name="weights", 
        direction=InterfaceDirection.WEIGHT,
        dtype=DataType.from_string("INT8"),
        granularity=(16, None)  # Output channels multiple of 16
    )
    
    output_def = InterfaceDefinition(
        name="output",
        direction=InterfaceDirection.OUTPUT,
        dtype=DataType.from_string("INT16"),
        alignment=128
    )
    
    # Create kernel definition
    kernel_def = KernelDefinition(
        name="conv2d",
        hw_module="convolution_2d",
        interface_definitions=[input_def, weights_def, output_def],
        memory_architecture="HBM",
        requires_burst_alignment=True
    )
    
    # Add relationships
    kernel_def.add_relationship(
        "weights", "input", RelationType.EQUAL,
        source_dim=1, target_dim=1,
        description="Weight input channels = input channels"
    )
    
    kernel_def.add_constraint(
        "dsp_limit",
        "weights.stream[0] * weights.stream[1]", "<=", 16,
        "DSP resource limitation"
    )
    
    # Validate definition
    errors = kernel_def.validate()
    assert len(errors) == 0
    
    # Test properties
    assert len(kernel_def.interface_definitions) == 3
    assert len(kernel_def.relationships) == 1
    assert len(kernel_def.constraints) == 1
    assert kernel_def.memory_architecture == "HBM"
    assert kernel_def.requires_burst_alignment


def test_kernel_model_creation():
    """Test creating kernel model from definition"""
    # Create simplified kernel definition
    input_def = InterfaceDefinition(
        name="A",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("INT8")
    )
    
    output_def = InterfaceDefinition(
        name="B", 
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("INT8")
    )
    
    result_def = InterfaceDefinition(
        name="C",
        direction=InterfaceDirection.OUTPUT,
        dtype=DataType.from_string("INT32")
    )
    
    kernel_def = KernelDefinition(
        name="matmul",
        interface_definitions=[input_def, output_def, result_def]
    )
    
    # Add matrix multiplication relationship
    kernel_def.add_relationship("A", "B", RelationType.EQUAL, source_dim=1, target_dim=0)
    
    # Create interface models
    input_model = input_def.create_model(
        tensor_dims=(512, 256),
        block_dims=(64, 32),
        stream_dims=(8, 4)
    )
    
    output_model = output_def.create_model(
        tensor_dims=(256, 512),
        block_dims=(32, 64), 
        stream_dims=(4, 8)
    )
    
    result_model = result_def.create_model(
        tensor_dims=(512, 512),
        block_dims=(64, 64),
        stream_dims=(8, 8)
    )
    
    # Create parameter binding
    params = ParameterBinding({
        "total_operations": 512 * 256 * 512,
        "clock_freq": 100.0
    })
    
    # Create kernel model
    kernel_model = kernel_def.create_model(
        interface_models=[input_model, output_model, result_model],
        parameter_binding=params,
        clock_freq_mhz=100.0
    )
    
    assert isinstance(kernel_model, KernelModel)
    assert kernel_model.definition == kernel_def
    assert len(kernel_model.interface_models) == 3
    assert kernel_model.clock_freq_mhz == 100.0
    
    # Test performance calculations
    metrics = kernel_model.calculate_performance_metrics()
    assert metrics["name"] == "matmul"
    assert metrics["n_interfaces"] == 3
    assert "throughput_fps" in metrics
    assert "resource_estimates" in metrics


def test_kernel_model_performance_analysis():
    """Test comprehensive kernel model performance analysis"""
    # Create simple kernel model for testing
    input_model = InterfaceModel(
        tensor_dims=(1024,),
        block_dims=(64,), 
        stream_dims=(8,),
        actual_utilization=0.9
    )
    
    output_model = InterfaceModel(
        tensor_dims=(1024,),
        block_dims=(64,),
        stream_dims=(8,),
        actual_utilization=0.9
    )
    
    kernel_model = KernelModel(
        interface_models=[input_model, output_model],
        parameter_binding=ParameterBinding({"total_operations": 1024}),
        latency_cycles=(100, 90),
        calculation_ii=1,
        clock_freq_mhz=200.0,
        actual_efficiency=0.85
    )
    
    # Test basic calculations
    assert kernel_model.initiation_interval() == 1
    assert kernel_model.inference_latency() == 100  # Uses worst-case latency
    
    # Test throughput
    throughput = kernel_model.throughput_fps()
    assert throughput > 0
    
    # Test simulation
    simulation = kernel_model.simulate_execution(n_inferences=5)
    assert simulation["n_inferences"] == 5
    assert simulation["total_cycles"] > 0
    assert simulation["pipeline_utilization"] > 0


def test_definition_model_workflow():
    """Test complete workflow from definition to performance analysis"""
    # Step 1: Create reusable definitions
    conv_input_def = InterfaceDefinition(
        name="ifmap",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("UINT8"),
        alignment=128,
        min_dims=(1, 1, 7, 7)
    )
    
    conv_weights_def = InterfaceDefinition(
        name="weights",
        direction=InterfaceDirection.WEIGHT,
        dtype=DataType.from_string("INT8"),
        granularity=(16, None, None, None)
    )
    
    conv_output_def = InterfaceDefinition(
        name="ofmap",
        direction=InterfaceDirection.OUTPUT,
        dtype=DataType.from_string("INT16"),
        alignment=128
    )
    
    # Step 2: Create kernel definition
    conv_def = KernelDefinition(
        name="conv2d_3x3",
        hw_module="conv2d_v2",
        interface_definitions=[conv_input_def, conv_weights_def, conv_output_def],
        pipeline_style="streaming"
    )
    
    conv_def.add_relationship("weights", "ifmap", RelationType.EQUAL, source_dim=1, target_dim=1)
    conv_def.add_constraint("min_channels", "ifmap[1]", ">=", 8, "Minimum channels for efficiency")
    
    # Step 3: Create multiple model instances with different configurations
    configs = [
        {"ifmap": (1, 32, 224, 224), "weights": (64, 32, 3, 3), "ofmap": (1, 64, 222, 222)},
        {"ifmap": (1, 64, 112, 112), "weights": (128, 64, 3, 3), "ofmap": (1, 128, 110, 110)},
    ]
    
    models = []
    for i, config in enumerate(configs):
        # Create interface models
        ifmap_model = conv_input_def.create_model(
            tensor_dims=config["ifmap"],
            block_dims=(config["ifmap"][1], 14, 14),
            stream_dims=(4, 1, 1)
        )
        
        weights_model = conv_weights_def.create_model(
            tensor_dims=config["weights"],
            block_dims=(16, config["weights"][1], 3, 3),
            stream_dims=(4, 4, 1, 1)
        )
        
        ofmap_model = conv_output_def.create_model(
            tensor_dims=config["ofmap"],
            block_dims=(16, 12, 12),
            stream_dims=(4, 1, 1)
        )
        
        # Create kernel model
        kernel_model = conv_def.create_model(
            interface_models=[ifmap_model, weights_model, ofmap_model],
            parameter_binding=ParameterBinding({
                "total_operations": config["ifmap"][1] * config["weights"][0] * 
                                  config["ofmap"][2] * config["ofmap"][3] * 9
            }),
            clock_freq_mhz=150.0 + i * 50  # Different frequencies
        )
        
        models.append(kernel_model)
    
    # Step 4: Compare performance
    comparison = models[0].compare_with(models[1])
    assert "throughput_ratio" in comparison
    assert "latency_ratio" in comparison
    assert "resource_ratios" in comparison
    
    # Step 5: Analyze individual performance
    for i, model in enumerate(models):
        metrics = model.calculate_performance_metrics()
        print(f"\nModel {i+1} Performance:")
        print(f"  Throughput: {metrics['throughput_fps']:.1f} FPS")
        print(f"  Bandwidth: {metrics['total_bandwidth_mbps']:.1f} MB/s") 
        print(f"  Resources: LUT={metrics['resource_estimates']['LUT']:.0f}, DSP={metrics['resource_estimates']['DSP']:.0f}")


if __name__ == "__main__":
    test_interface_definition_validation()
    test_interface_definition_model_validation()
    test_interface_model_creation()
    test_interface_model_performance()
    test_kernel_definition_creation()
    test_kernel_model_creation()
    test_kernel_model_performance_analysis()
    test_definition_model_workflow()
    print("All Definition/Model architecture tests passed!")