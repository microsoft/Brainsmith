############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Showcase of Definition/Model architecture benefits"""

from brainsmith.core.dataflow.interface_definition import InterfaceDefinition
from brainsmith.core.dataflow.kernel_definition import KernelDefinition
from brainsmith.core.dataflow.base import ParameterBinding, DEFINITION_REGISTRY
from brainsmith.core.dataflow.relationships import RelationType
from brainsmith.core.dataflow.types import InterfaceDirection, DataType


def create_convolution_definition():
    """Create a reusable convolution kernel definition"""
    
    # Define interface templates with rich constraints
    ifmap_def = InterfaceDefinition(
        name="ifmap",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("UINT8"),
        alignment=128,  # DMA burst alignment
        min_dims=(1, 1, 7, 7),  # Minimum tile size
        granularity=(1, 8, 1, 1),  # Channels must be multiple of 8
        produces={"ofmap"},
        synchronized_with={"weights"}
    )
    
    weights_def = InterfaceDefinition(
        name="weights",
        direction=InterfaceDirection.WEIGHT,
        dtype=DataType.from_string("INT8"),
        granularity=(16, None, None, None),  # Output channels multiple of 16
        produces={"ofmap"},
        synchronized_with={"ifmap"}
    )
    
    ofmap_def = InterfaceDefinition(
        name="ofmap",
        direction=InterfaceDirection.OUTPUT,
        dtype=DataType.from_string("INT16"),
        alignment=128,
        consumes={"ifmap", "weights"}
    )
    
    # Create kernel definition with relationships
    conv_def = KernelDefinition(
        name="conv2d_streaming",
        hw_module="convolution_streaming_v3",
        interface_definitions=[ifmap_def, weights_def, ofmap_def],
        memory_architecture="distributed",
        pipeline_style="streaming",
        requires_burst_alignment=True
    )
    
    # Add mathematical relationships
    conv_def.add_relationship(
        "weights", "ifmap", RelationType.EQUAL,
        source_dim=1, target_dim=1,
        description="Weight input channels = ifmap channels"
    )
    
    conv_def.add_relationship(
        "ofmap", "weights", RelationType.EQUAL,
        source_dim=1, target_dim=0,
        description="Output channels = weight output channels"
    )
    
    # Add spatial constraints (for valid padding)
    conv_def.add_constraint(
        "output_height",
        "ofmap[2]", "==", "ifmap[2] - weights[2] + 1",
        "Output height with valid padding"
    )
    
    conv_def.add_constraint(
        "output_width",
        "ofmap[3]", "==", "ifmap[3] - weights[3] + 1",
        "Output width with valid padding"
    )
    
    # Add hardware constraints
    conv_def.add_constraint(
        "dsp_limit",
        "weights.stream[0] * weights.stream[1]", "<=", 16,
        "DSP slice limitation"
    )
    
    conv_def.add_constraint(
        "memory_bandwidth",
        "ifmap.bandwidth + weights.bandwidth + ofmap.bandwidth", "<=", 25600,
        "Memory bandwidth limit (25.6 GB/s)"
    )
    
    # Add parameter dependencies
    conv_def.add_dependency(
        "mac_operations",
        "ifmap[1] * weights[0] * ofmap[2] * ofmap[3] * weights[2] * weights[3]",
        "Total multiply-accumulate operations"
    )
    
    conv_def.add_dependency(
        "line_buffer_size",
        "ifmap[1] * weights[2] * ifmap[3] * 2",  # 2 bytes per element
        "Line buffer memory requirement"
    )
    
    return conv_def


def demonstrate_definition_reuse():
    """Demonstrate how one definition can create multiple optimized models"""
    
    print("=== Definition/Model Architecture Showcase ===\n")
    
    # Create and register the convolution definition
    conv_def = create_convolution_definition()
    DEFINITION_REGISTRY.register_kernel_definition("conv2d_streaming", conv_def)
    
    print(f"Created kernel definition: {conv_def.name}")
    print(f"  - {len(conv_def.interface_definitions)} interfaces")
    print(f"  - {len(conv_def.relationships)} relationships")
    print(f"  - {len(conv_def.constraints)} constraints")
    print(f"  - {len(conv_def.dependencies)} dependencies")
    
    # Define different model configurations
    configurations = [
        {
            "name": "Mobile (3x3 conv)",
            "ifmap": (1, 32, 112, 112),
            "weights": (64, 32, 3, 3),
            "ofmap": (1, 64, 110, 110),
            "block_size": (8, 14, 14),
            "parallelism": (4, 2, 1, 1),
            "freq_mhz": 100.0,
            "efficiency": 0.8
        },
        {
            "name": "Server (5x5 conv)",
            "ifmap": (1, 64, 224, 224),
            "weights": (128, 64, 5, 5),
            "ofmap": (1, 128, 220, 220),
            "block_size": (16, 28, 28),
            "parallelism": (8, 4, 1, 1),
            "freq_mhz": 200.0,
            "efficiency": 0.9
        },
        {
            "name": "Edge (1x1 conv)",
            "ifmap": (1, 128, 56, 56),
            "weights": (256, 128, 1, 1),
            "ofmap": (1, 256, 56, 56),
            "block_size": (32, 14, 14),
            "parallelism": (16, 8, 1, 1),
            "freq_mhz": 150.0,
            "efficiency": 0.85
        }
    ]
    
    models = []
    
    for config in configurations:
        print(f"\n--- Creating model: {config['name']} ---")
        
        # Create interface models from definitions
        ifmap_model = conv_def.get_interface_definition("ifmap").create_model(
            tensor_dims=config["ifmap"],
            block_dims=(config["ifmap"][1], config["block_size"][1], config["block_size"][2]),
            stream_dims=config["parallelism"]
        )
        
        # Adjust parallelism to respect DSP constraint (weights.stream[0] * weights.stream[1] <= 16)
        weights_parallelism = config["parallelism"]
        if weights_parallelism[0] * weights_parallelism[1] > 16:
            # Scale down to fit constraint
            scale_factor = (16 / (weights_parallelism[0] * weights_parallelism[1])) ** 0.5
            weights_parallelism = (
                max(1, int(weights_parallelism[0] * scale_factor)),
                max(1, int(weights_parallelism[1] * scale_factor)),
                weights_parallelism[2],
                weights_parallelism[3]
            )
        
        weights_model = conv_def.get_interface_definition("weights").create_model(
            tensor_dims=config["weights"],
            block_dims=(config["block_size"][0], config["weights"][1], config["weights"][2], config["weights"][3]),
            stream_dims=weights_parallelism
        )
        
        ofmap_model = conv_def.get_interface_definition("ofmap").create_model(
            tensor_dims=config["ofmap"],
            block_dims=(config["block_size"][0], config["block_size"][1], config["block_size"][2]),
            stream_dims=config["parallelism"]
        )
        
        # Calculate operations
        mac_ops = (config["ifmap"][1] * config["weights"][0] * 
                  config["ofmap"][2] * config["ofmap"][3] * 
                  config["weights"][2] * config["weights"][3])
        
        # Create parameter binding
        params = ParameterBinding({
            "mac_operations": mac_ops,
            "line_buffer_size": config["ifmap"][1] * config["weights"][2] * config["ifmap"][3] * 2,
            "clock_freq": config["freq_mhz"]
        })
        
        # Create kernel model from definition
        kernel_model = conv_def.create_model(
            interface_models=[ifmap_model, weights_model, ofmap_model],
            parameter_binding=params,
            clock_freq_mhz=config["freq_mhz"],
            actual_efficiency=config["efficiency"],
            latency_cycles=(100, 85),  # Estimated latency
            calculation_ii=1
        )
        
        models.append((config["name"], kernel_model))
        
        # Analyze performance
        metrics = kernel_model.calculate_performance_metrics()
        
        print(f"  Tensor dims: {config['ifmap']} -> {config['ofmap']}")
        print(f"  Parallelism: {config['parallelism']}")
        print(f"  Clock freq: {config['freq_mhz']} MHz")
        print(f"  Throughput: {metrics['throughput_fps']:.1f} FPS")
        print(f"  GOPS: {metrics['throughput_gops']:.2f}")
        print(f"  Bandwidth: {metrics['total_bandwidth_mbps']:.1f} MB/s")
        print(f"  DSP usage: {metrics['resource_estimates']['DSP']:.0f}")
        print(f"  Power est: {metrics['power_estimates']['total_watts']:.2f} W")
    
    # Compare models
    print(f"\n--- Model Comparisons ---")
    for i, (name_a, model_a) in enumerate(models):
        for j, (name_b, model_b) in enumerate(models[i+1:], i+1):
            comparison = model_a.compare_with(model_b)
            print(f"\n{name_a} vs {name_b}:")
            print(f"  Throughput ratio: {comparison['throughput_ratio']:.2f}x")
            print(f"  DSP ratio: {comparison['resource_ratios']['DSP']:.2f}x")
            print(f"  Bandwidth ratio: {comparison['bandwidth_ratio']:.2f}x")
    
    # Demonstrate constraint validation
    print(f"\n--- Constraint Validation ---")
    try:
        # Try to create an invalid model (violates minimum dimensions)
        bad_ifmap = conv_def.get_interface_definition("ifmap").create_model(
            tensor_dims=(1, 4, 3, 3),  # Too small - violates min_dims constraint
            block_dims=(4, 3, 3),
            stream_dims=(1, 1, 1, 1)
        )
    except ValueError as e:
        print(f"✓ Caught constraint violation: {str(e)[:60]}...")
    
    try:
        # Try to create model with bad granularity
        bad_weights = conv_def.get_interface_definition("weights").create_model(
            tensor_dims=(15, 32, 3, 3),  # 15 not multiple of 16
            block_dims=(15, 32, 3, 3),
            stream_dims=(1, 1, 1, 1)
        )
    except ValueError as e:
        print(f"✓ Caught granularity violation: {str(e)[:60]}...")
    
    print(f"\n--- Registry Usage ---")
    print(f"Registered definitions: {DEFINITION_REGISTRY.list_kernel_definitions()}")
    
    # Retrieve and reuse definition
    retrieved_def = DEFINITION_REGISTRY.get_kernel_definition("conv2d_streaming")
    print(f"Retrieved definition: {retrieved_def.name if retrieved_def else 'None'}")
    
    return models


def demonstrate_performance_analysis():
    """Demonstrate advanced performance analysis capabilities"""
    
    print(f"\n=== Advanced Performance Analysis ===\n")
    
    # Create a simple model for detailed analysis
    conv_def = DEFINITION_REGISTRY.get_kernel_definition("conv2d_streaming")
    
    # Create a model instance
    ifmap_model = conv_def.get_interface_definition("ifmap").create_model(
        tensor_dims=(1, 64, 224, 224),
        block_dims=(16, 28, 28),
        stream_dims=(8, 1, 1),
        actual_utilization=0.85
    )
    
    weights_model = conv_def.get_interface_definition("weights").create_model(
        tensor_dims=(128, 64, 3, 3),
        block_dims=(16, 64, 3, 3),
        stream_dims=(4, 4, 1, 1),
        actual_utilization=0.9
    )
    
    ofmap_model = conv_def.get_interface_definition("ofmap").create_model(
        tensor_dims=(1, 128, 222, 222),
        block_dims=(16, 28, 28),
        stream_dims=(8, 1, 1),
        actual_utilization=0.8
    )
    
    kernel_model = conv_def.create_model(
        interface_models=[ifmap_model, weights_model, ofmap_model],
        parameter_binding=ParameterBinding({
            "mac_operations": 64 * 128 * 222 * 222 * 9,
            "line_buffer_size": 64 * 3 * 224 * 2
        }),
        clock_freq_mhz=175.0,
        actual_efficiency=0.88
    )
    
    print("Model Configuration:")
    print(f"  Input: {ifmap_model.tensor_dims}")
    print(f"  Weights: {weights_model.tensor_dims}")
    print(f"  Output: {ofmap_model.tensor_dims}")
    print(f"  Clock: {kernel_model.clock_freq_mhz} MHz")
    
    # Detailed interface analysis
    print(f"\nInterface Analysis:")
    for intf_model in kernel_model.interface_models:
        if intf_model.definition:
            metrics = intf_model.calculate_performance_metrics()
            print(f"  {intf_model.definition.name}:")
            print(f"    Parallelism: {metrics['interface_parallelism']}")
            print(f"    Utilization: {metrics['actual_utilization']:.1%}")
            print(f"    Bandwidth: {metrics['effective_bandwidth_100mhz_mbps']:.1f} MB/s @ 100MHz")
            print(f"    Tokens/inference: {metrics['tokens_per_inference']}")
    
    # Simulation
    print(f"\nExecution Simulation:")
    simulation = kernel_model.simulate_execution(n_inferences=10, detailed=False)
    print(f"  10 inferences: {simulation['total_cycles']} cycles")
    print(f"  Pipeline utilization: {simulation['pipeline_utilization']:.1%}")
    print(f"  Resource utilization:")
    for resource, value in simulation['resource_utilization'].items():
        print(f"    {resource}: {value:.0f}")
    
    # Frequency scaling analysis
    print(f"\nFrequency Scaling Analysis:")
    frequencies = [100, 150, 200, 250, 300]
    for freq in frequencies:
        kernel_model.update_clock_frequency(freq)
        throughput = kernel_model.throughput_fps()
        gops = kernel_model.throughput_gops()
        bandwidth = kernel_model.total_bandwidth_mbps()
        power = kernel_model.estimate_power()["total_watts"]
        
        print(f"  {freq:3d} MHz: {throughput:6.1f} FPS, {gops:5.1f} GOPS, {bandwidth:6.0f} MB/s, {power:4.1f}W")
    
    print(f"\n=== Architecture Benefits Demonstrated ===")
    print("✓ Single definition -> multiple optimized models")
    print("✓ Rich constraint validation and error reporting")
    print("✓ Performance-optimized model calculations")
    print("✓ Detailed simulation and analysis capabilities")
    print("✓ Registry for definition reuse across projects")
    print("✓ Clean separation of specification vs runtime")


if __name__ == "__main__":
    models = demonstrate_definition_reuse()
    demonstrate_performance_analysis()
    print(f"\nDefinition/Model architecture showcase completed successfully!")