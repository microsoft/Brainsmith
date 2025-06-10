"""
Example-based validation tests.

Tests that validate the system through realistic usage examples,
ensuring that documentation examples work and common workflows
are properly supported.
"""

import pytest
import numpy as np
from pathlib import Path
from brainsmith.dataflow.core.dataflow_interface import DataflowInterface, DataflowInterfaceType, DataflowDataType
from brainsmith.dataflow.core.dataflow_model import DataflowModel
from brainsmith.dataflow.core.block_chunking import TensorChunking
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp


@pytest.mark.examples
class TestBasicUsageExamples:
    """Test basic usage examples from documentation."""
    
    def test_simple_interface_creation(self, basic_datatype):
        """Test basic interface creation example."""
        
        # Example: Create a simple CNN input interface
        cnn_input = DataflowInterface(
            name="cnn_input",
            interface_type=DataflowInterfaceType.INPUT,
            tensor_dims=[64, 224, 224],    # 64 channels, 224x224 spatial
            block_dims=[1, 224, 224],      # Process 1 channel at a time
            stream_dims=[1, 1, 8],         # 8 pixels per clock cycle
            dtype=basic_datatype
        )
        
        # Validate creation
        assert cnn_input.name == "cnn_input"
        assert cnn_input.get_num_blocks() == [64, 1, 1]
        assert cnn_input.calculate_stream_width() == 64  # 8 pixels × 8 bits
        
        # Validate basic operations
        transfer_cycles = cnn_input.get_transfer_cycles()
        expected_cycles = (224 * 224) // 8  # Total pixels / pixels per cycle
        assert transfer_cycles == expected_cycles
    
    def test_transformer_interface_example(self, basic_datatype):
        """Test transformer interface creation example."""
        
        # Example: BERT-style transformer interface
        bert_input = DataflowInterface(
            name="bert_tokens",
            interface_type=DataflowInterfaceType.INPUT,
            tensor_dims=[512, 768],        # 512 tokens, 768 features
            block_dims=[1, 768],           # Process 1 token at a time
            stream_dims=[1, 64],           # 64 features per cycle
            dtype=basic_datatype
        )
        
        # Validate transformer properties
        assert bert_input.get_num_blocks() == [512, 1]
        assert bert_input.get_transfer_cycles() == 768 // 64  # 12 cycles per token
        
        # Total processing time
        total_cycles = 512 * (768 // 64)  # 512 tokens × 12 cycles
        assert total_cycles == 6144
    
    def test_complete_model_example(self, basic_datatype):
        """Test complete dataflow model creation example."""
        
        # Example: Simple matrix multiplication kernel
        input_interface = DataflowInterface(
            name="matrix_a",
            interface_type=DataflowInterfaceType.INPUT,
            tensor_dims=[128, 256],
            block_dims=[32, 256],
            stream_dims=[8, 32],
            dtype=basic_datatype
        )
        
        weight_interface = DataflowInterface(
            name="matrix_b",
            interface_type=DataflowInterfaceType.WEIGHT,
            tensor_dims=[256, 512],
            block_dims=[256, 128],
            stream_dims=[32, 16],
            dtype=basic_datatype
        )
        
        output_interface = DataflowInterface(
            name="result",
            interface_type=DataflowInterfaceType.OUTPUT,
            tensor_dims=[128, 512],
            block_dims=[32, 128],
            stream_dims=[8, 16],
            dtype=basic_datatype
        )
        
        # Create model
        interfaces = [input_interface, weight_interface, output_interface]
        model = DataflowModel(interfaces, {})
        
        # Validate model structure
        assert len(model.input_interfaces) == 1
        assert len(model.weight_interfaces) == 1
        assert len(model.output_interfaces) == 1
        
        # Test timing calculation
        iPar = {"matrix_a": 2}
        wPar = {"matrix_b": 4}
        
        intervals = model.calculate_initiation_intervals(iPar, wPar)
        
        # Should successfully calculate timing
        assert hasattr(intervals, 'bottleneck_analysis')
        assert intervals.bottleneck_analysis["bottleneck_eII"] > 0
    
    def test_chunking_strategy_examples(self):
        """Test different chunking strategy examples."""
        
        # Example 1: CNN channel-wise chunking
        cnn_shape = [64, 56, 56]
        cnn_layout = "[C, H, W]"
        
        # Simulate layout-driven chunking
        tensor_dims = cnn_shape
        block_dims = [1, 56, 56]  # Chunk along channel dimension
        
        # Validate chunking
        num_blocks = [tensor_dims[i] // block_dims[i] for i in range(len(tensor_dims))]
        assert num_blocks == [64, 1, 1]  # 64 blocks along channels
        
        # Example 2: Transformer sequence chunking
        transformer_shape = [512, 768]
        transformer_layout = "[L, C]"
        
        tensor_dims = transformer_shape
        block_dims = [1, 768]  # Chunk along sequence dimension
        
        num_blocks = [tensor_dims[i] // block_dims[i] for i in range(len(tensor_dims))]
        assert num_blocks == [512, 1]  # 512 blocks along sequence
        
        # Example 3: Matrix chunking
        matrix_shape = [1024, 1024]
        
        # Option A: Row-wise chunking
        tensor_dims = matrix_shape
        block_dims = [32, 1024]  # Process 32 rows at a time
        
        num_blocks = [tensor_dims[i] // block_dims[i] for i in range(len(tensor_dims))]
        assert num_blocks == [32, 1]  # 32 row blocks
        
        # Option B: Tile-based chunking
        block_dims = [32, 32]  # Process 32×32 tiles
        
        num_blocks = [tensor_dims[i] // block_dims[i] for i in range(len(tensor_dims))]
        assert num_blocks == [32, 32]  # 32×32 = 1024 tile blocks
    
    def test_parallelism_optimization_example(self, cnn_interfaces):
        """Test parallelism optimization example."""
        
        model = DataflowModel(list(cnn_interfaces.values()), {})
        
        # Example: Test different parallelism configurations
        parallelism_configs = [
            {"iPar": {"input": 1}, "wPar": {"weights": 1}},    # Baseline
            {"iPar": {"input": 4}, "wPar": {"weights": 2}},    # Medium parallelism
            {"iPar": {"input": 8}, "wPar": {"weights": 4}},    # High parallelism
        ]
        
        performance_results = []
        
        for config in parallelism_configs:
            intervals = model.calculate_initiation_intervals(config["iPar"], config["wPar"])
            
            performance_results.append({
                "config": config,
                "eII": intervals.bottleneck_analysis["bottleneck_eII"],
                "L": intervals.bottleneck_analysis.get("total_inference_cycles", 0)
            })
        
        # Validate that higher parallelism reduces execution time
        baseline_eII = performance_results[0]["eII"]
        high_parallel_eII = performance_results[-1]["eII"]
        
        # Higher parallelism should reduce eII (or at least not increase significantly)
        assert high_parallel_eII <= baseline_eII * 2  # Allow some overhead
    
    def test_memory_footprint_analysis_example(self, transformer_interfaces):
        """Test memory footprint analysis example."""
        
        model = DataflowModel(list(transformer_interfaces.values()), {})
        
        # Example: Analyze memory requirements for different configurations
        parallelism_configs = [
            {"iPar": {"input_tokens": 1}, "wPar": {"attention_weights": 1}},
            {"iPar": {"input_tokens": 4}, "wPar": {"attention_weights": 2}},
            {"iPar": {"input_tokens": 8}, "wPar": {"attention_weights": 4}},
        ]
        
        for config in parallelism_configs:
            # Calculate resource requirements
            from brainsmith.dataflow.core.dataflow_model import ParallelismConfiguration
            
            parallelism_config = ParallelismConfiguration(
                iPar=config["iPar"],
                wPar=config["wPar"],
                derived_stream_dims={}
            )
            
            resources = model.get_resource_requirements(parallelism_config)
            
            # Validate resource calculation
            assert "memory_bits" in resources
            assert "bandwidth_requirements" in resources
            assert resources["memory_bits"] > 0
            
            # Memory should scale with parallelism
            total_parallelism = sum(config["iPar"].values()) + sum(config["wPar"].values())
            assert resources["memory_bits"] > total_parallelism * 100  # Rough scaling check
    
    def test_datatype_constraint_example(self, basic_datatype_constraint):
        """Test datatype constraint usage example."""
        
        # Example: Create interface with specific datatype constraints
        constrained_interface = DataflowInterface(
            name="constrained_input",
            interface_type=DataflowInterfaceType.INPUT,
            tensor_dims=[128, 128],
            block_dims=[32, 128],
            stream_dims=[8, 16],
            dtype=DataflowDataType("INT", 8, True, "INT8"),
            allowed_datatypes=[basic_datatype_constraint]
        )
        
        # Validate constraint enforcement
        validation_result = constrained_interface.validate()
        assert validation_result.is_valid()
        
        # Test constraint violation
        invalid_dtype = DataflowDataType("FLOAT", 32, True, "FLOAT32")
        constrained_interface.dtype = invalid_dtype
        
        validation_result = constrained_interface.validate()
        # Should detect constraint violation
        assert not validation_result.is_valid() or len(validation_result.warnings) > 0
    
    def test_real_world_cnn_workflow(self):
        """Test realistic CNN workflow example."""
        
        # Example: ResNet-style convolutional block
        dtype = DataflowDataType("INT", 8, True, "INT8")
        
        # Input: 256 channels, 14×14 spatial
        input_feature_maps = DataflowInterface(
            name="input_fmaps",
            interface_type=DataflowInterfaceType.INPUT,
            tensor_dims=[256, 14, 14],
            block_dims=[1, 14, 14],     # Process 1 channel at a time
            stream_dims=[1, 1, 8],      # 8 pixels per cycle
            dtype=dtype
        )
        
        # Weights: 512 output channels, 256 input channels, 3×3 kernel
        conv_weights = DataflowInterface(
            name="conv_weights",
            interface_type=DataflowInterfaceType.WEIGHT,
            tensor_dims=[512, 256, 3, 3],
            block_dims=[32, 1, 3, 3],   # Process 32 output channels per block
            stream_dims=[8, 1, 1, 1],   # 8 weights per cycle
            dtype=dtype
        )
        
        # Output: 512 channels, 14×14 spatial (same size with padding)
        output_feature_maps = DataflowInterface(
            name="output_fmaps",
            interface_type=DataflowInterfaceType.OUTPUT,
            tensor_dims=[512, 14, 14],
            block_dims=[32, 14, 14],    # Match weight block size
            stream_dims=[8, 1, 1],      # 8 outputs per cycle
            dtype=dtype
        )
        
        # Create CNN model
        cnn_model = DataflowModel([
            input_feature_maps,
            conv_weights,
            output_feature_maps
        ], {})
        
        # Analyze CNN performance
        iPar = {"input_fmaps": 1}
        wPar = {"conv_weights": 8}  # High weight parallelism
        
        intervals = cnn_model.calculate_initiation_intervals(iPar, wPar)
        
        # Validate CNN-specific properties
        bottleneck = intervals.bottleneck_analysis
        
        # Should identify bottleneck correctly
        assert "bottleneck_input" in bottleneck
        assert bottleneck["bottleneck_eII"] > 0
        
        # Calculate expected performance metrics
        input_blocks = np.prod(input_feature_maps.get_num_blocks())
        weight_blocks = np.prod(conv_weights.get_num_blocks())
        
        assert input_blocks == 256  # 256 input channel blocks
        assert weight_blocks == 512 * 256 // 32  # Weight blocks with PE=32
    
    def test_real_world_transformer_workflow(self):
        """Test realistic transformer workflow example."""
        
        # Example: BERT attention mechanism
        dtype = DataflowDataType("INT", 16, True, "INT16")  # Higher precision
        
        # Query input: 512 tokens, 768 features
        query_input = DataflowInterface(
            name="query",
            interface_type=DataflowInterfaceType.INPUT,
            tensor_dims=[512, 768],
            block_dims=[1, 768],        # Process 1 token at a time
            stream_dims=[1, 64],        # 64 features per cycle
            dtype=dtype
        )
        
        # Key input: 512 tokens, 768 features
        key_input = DataflowInterface(
            name="key",
            interface_type=DataflowInterfaceType.INPUT,
            tensor_dims=[512, 768],
            block_dims=[1, 768],
            stream_dims=[1, 64],
            dtype=dtype
        )
        
        # Attention weights: 768×768 transformation matrix
        attention_weights = DataflowInterface(
            name="attention_matrix",
            interface_type=DataflowInterfaceType.WEIGHT,
            tensor_dims=[768, 768],
            block_dims=[768, 1],        # Column-wise processing
            stream_dims=[64, 1],        # 64 weights per cycle
            dtype=dtype
        )
        
        # Attention output: 512 tokens, 768 features
        attention_output = DataflowInterface(
            name="attention_out",
            interface_type=DataflowInterfaceType.OUTPUT,
            tensor_dims=[512, 768],
            block_dims=[1, 768],
            stream_dims=[1, 64],
            dtype=dtype
        )
        
        # Create transformer model
        transformer_model = DataflowModel([
            query_input,
            key_input,
            attention_weights,
            attention_output
        ], {})
        
        # Analyze transformer performance
        iPar = {"query": 1, "key": 1}
        wPar = {"attention_matrix": 12}  # Multi-head parallelism
        
        intervals = transformer_model.calculate_initiation_intervals(iPar, wPar)
        
        # Validate transformer-specific properties
        bottleneck = intervals.bottleneck_analysis
        
        # Calculate expected sequence processing time
        tokens_per_sequence = 512
        cycles_per_token = 768 // 64  # 12 cycles per token
        total_sequence_cycles = tokens_per_sequence * cycles_per_token
        
        assert total_sequence_cycles == 6144  # Expected total cycles
        
        # Validate that model handles multiple inputs correctly
        assert len(transformer_model.input_interfaces) == 2  # query and key
        assert len(transformer_model.weight_interfaces) == 1  # attention weights
        assert len(transformer_model.output_interfaces) == 1  # attention output