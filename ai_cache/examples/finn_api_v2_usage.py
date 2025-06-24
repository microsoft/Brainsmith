"""
FINN API V2 Usage Examples

This file demonstrates how to use the new FINN-Brainsmith API for various
compilation scenarios.
"""

from brainsmith.core.finn_v2 import (
    FINNCompiler,
    CompilationConfig,
    HighPerformanceStrategy,
    AreaOptimizedStrategy,
    BalancedStrategy,
    CompilationStrategy,
    KernelSelection,
    TransformSequence,
    CompilationStage
)


# =============================================================================
# Example 1: Simple Compilation
# =============================================================================

def example_simple_compilation():
    """Simplest way to compile a model."""
    
    # Create compiler
    compiler = FINNCompiler()
    
    # Compile with default settings
    result = compiler.compile("model.onnx")
    
    print(f"Compilation successful: {result['success']}")
    print(f"Output directory: {result['finn_result']['output_dir']}")


# =============================================================================
# Example 2: Strategy Selection
# =============================================================================

def example_strategy_selection():
    """Using different built-in strategies."""
    
    compiler = FINNCompiler()
    
    # High performance strategy
    result = compiler.compile(
        "model.onnx",
        strategy="performance"
    )
    print(f"Performance strategy FPS: {result['finn_result']['metrics']['estimated_fps']}")
    
    # Area optimized strategy
    result = compiler.compile(
        "model.onnx",
        strategy="area"
    )
    print(f"Area strategy resources: {result['finn_result']['metrics']['resource_usage']}")
    
    # Balanced strategy
    result = compiler.compile(
        "model.onnx",
        strategy="balanced"
    )
    print(f"Balanced strategy: {result['strategy_used']}")


# =============================================================================
# Example 3: Custom Configuration
# =============================================================================

def example_custom_configuration():
    """Compilation with custom configuration."""
    
    # Create configuration
    config = CompilationConfig(
        output_dir="./my_accelerator",
        target_device="U250",  # Alveo U250
        target_frequency_mhz=300.0,  # 300 MHz
        constraints={
            "target_fps": 5000,
            "max_power_w": 75,
            "max_lut_utilization": 0.85
        },
        debug_level=1,  # Enable debug output
        save_intermediate=True  # Save intermediate models
    )
    
    # Compile with custom config
    compiler = FINNCompiler()
    result = compiler.compile(
        "model.onnx",
        strategy="performance",
        config=config
    )
    
    print(f"Custom build completed: {result['success']}")


# =============================================================================
# Example 4: BERT-Specific Optimization
# =============================================================================

def example_bert_optimization():
    """BERT model optimization example."""
    
    from brainsmith.core.finn_v2.strategies import BERTOptimizedStrategy
    
    # BERT configuration
    config = CompilationConfig(
        output_dir="./bert_accelerator",
        target_device="U280",
        target_frequency_mhz=250.0,
        constraints={
            "target_fps": 3000,
            "max_seq_length": 512,
            "batch_size": 1,
            "folding_config": "configs/bert_folding.json"
        }
    )
    
    # Create BERT-specific strategy
    bert_strategy = BERTOptimizedStrategy(
        config=config,
        num_heads=12,
        hidden_size=768
    )
    
    # Compile
    compiler = FINNCompiler()
    result = compiler.compile(
        "bert_base.onnx",
        strategy=bert_strategy,
        config=config
    )
    
    print(f"BERT kernels: {result['kernels_selected']}")
    print(f"BERT transforms: {result['transforms_applied']}")


# =============================================================================
# Example 5: Custom Strategy Definition
# =============================================================================

def example_custom_strategy():
    """Creating a completely custom strategy."""
    
    class VisionTransformerStrategy(CompilationStrategy):
        """Custom strategy for Vision Transformers."""
        
        def select_kernels(self) -> KernelSelection:
            """ViT-specific kernel selection."""
            return KernelSelection().use_kernel(
                "MatMul", "MatMulRTL"
            ).use_kernel(
                "LayerNorm", "LayerNormRTL"
            ).use_kernel(
                "Softmax", "SoftmaxHLS"
            ).use_kernel(
                "PatchEmbed", "PatchEmbedCustom"  # Custom kernel
            )
        
        def build_transform_sequence(self) -> Dict[CompilationStage, TransformSequence]:
            """ViT-specific transforms."""
            sequences = {}
            
            # ViT-specific cleanup
            sequences[CompilationStage.GRAPH_CLEANUP] = TransformSequence().add(
                "RemoveDropout"
            ).add(
                "FoldConstants"
            )
            
            # ViT topology optimization
            sequences[CompilationStage.TOPOLOGY_OPTIMIZATION] = TransformSequence().add(
                "FuseMultiHeadAttention", {"num_heads": 12}
            ).add(
                "OptimizePatchEmbedding", {"patch_size": 16}
            ).add(
                "FuseLayerNormLinear"
            )
            
            # ViT kernel optimization
            sequences[CompilationStage.KERNEL_OPTIMIZATION] = TransformSequence().add(
                "OptimizeAttentionTiling", {
                    "tile_size": 64,
                    "num_heads": 12
                }
            )
            
            return sequences
        
        def get_finn_parameters(self) -> Dict[str, Any]:
            """ViT-specific FINN parameters."""
            return {
                "mvau_wwidth_max": 48,
                "minimize_bit_width": True,
                "mem_mode": "decoupled"
            }
    
    # Use custom strategy
    config = CompilationConfig(
        output_dir="./vit_build",
        target_device="U250",
        target_frequency_mhz=200.0,
        constraints={"target_fps": 100}  # Lower FPS for larger model
    )
    
    vit_strategy = VisionTransformerStrategy(config)
    
    compiler = FINNCompiler()
    result = compiler.compile(
        "vit_base.onnx",
        strategy=vit_strategy,
        config=config
    )
    
    print(f"ViT compilation: {result['success']}")


# =============================================================================
# Example 6: Querying Available Components
# =============================================================================

def example_query_components():
    """Query available kernels and transforms."""
    
    compiler = FINNCompiler()
    registry = compiler.registry
    
    # List all available kernels
    print("Available Kernels:")
    kernels = registry.list_kernels()
    for name, kernel_class in kernels:
        metadata = kernel_class._plugin_metadata
        print(f"  - {name}: {metadata.get('description', 'No description')}")
    
    # List transforms by stage
    print("\nTransforms by Stage:")
    for stage in CompilationStage:
        print(f"\n{stage.value}:")
        transforms = registry.list_transforms(stage=stage.value)
        for name, transform_class in transforms:
            metadata = transform_class._plugin_metadata
            print(f"  - {name}: {metadata.get('description', '')}")
    
    # Search for specific components
    print("\nSearch results for 'norm':")
    results = registry.search_plugins("norm")
    for result in results:
        print(f"  - {result['type']}: {result['name']}")


# =============================================================================
# Example 7: Debug Mode
# =============================================================================

def example_debug_mode():
    """Using debug mode to inspect compilation."""
    
    # Enable maximum debug output
    config = CompilationConfig(
        output_dir="./debug_build",
        target_device="Pynq-Z1",
        target_frequency_mhz=200.0,
        debug_level=2,  # Maximum debug
        save_intermediate=True
    )
    
    compiler = FINNCompiler()
    result = compiler.compile(
        "model.onnx",
        strategy="balanced",
        config=config
    )
    
    # In debug mode, we get the model and config without FINN execution
    if result.get('finn_result', {}).get('debug'):
        print("Debug mode - compilation config:")
        print(f"  FINN config: {result['finn_result']['finn_config']}")
        print(f"  Selected kernels: {result['kernels_selected']}")
        print(f"  Applied transforms: {result['transforms_applied']}")


# =============================================================================
# Example 8: Programmatic Transform Addition
# =============================================================================

def example_programmatic_transforms():
    """Building transform sequences programmatically."""
    
    from brainsmith.plugin import PluginRegistry
    
    # Get available transforms
    registry = PluginRegistry()
    
    # Build custom transform sequence
    my_cleanup = TransformSequence()
    my_cleanup.add("RemoveIdentityOps")
    my_cleanup.add("FoldConstants", {"aggressive": True})
    my_cleanup.add("EliminateDeadCode")
    
    my_optimization = TransformSequence()
    my_optimization.add("StreamlineActivations", {"level": 3})
    my_optimization.add("FuseOperations", {"patterns": ["conv_bn_relu", "linear_gelu"]})
    
    # Create custom strategy with these sequences
    class MyStrategy(CompilationStrategy):
        def select_kernels(self) -> KernelSelection:
            return KernelSelection().use_kernel("MatMul", "MatMulHLS")
        
        def build_transform_sequence(self) -> Dict[CompilationStage, TransformSequence]:
            return {
                CompilationStage.GRAPH_CLEANUP: my_cleanup,
                CompilationStage.TOPOLOGY_OPTIMIZATION: my_optimization
            }
        
        def get_finn_parameters(self) -> Dict[str, Any]:
            return {"minimize_bit_width": True}
    
    # Use it
    config = CompilationConfig(
        output_dir="./custom_transforms",
        target_device="Pynq-Z1",
        target_frequency_mhz=200.0
    )
    
    compiler = FINNCompiler()
    result = compiler.compile(
        "model.onnx",
        strategy=MyStrategy(config)
    )


# =============================================================================
# Example 9: Migration from Old API
# =============================================================================

def example_migration_from_old_api():
    """Shows how to migrate from the old 6-entrypoint system."""
    
    # Old 6-entrypoint configuration
    old_config = {
        'entrypoint_1': ['LayerNorm', 'Softmax', 'GELU'],
        'entrypoint_2': ['cleanup', 'streamlining'],
        'entrypoint_3': ['MatMul', 'LayerNorm'],
        'entrypoint_4': ['matmul_hls', 'layernorm_rtl'],
        'entrypoint_5': ['target_fps_parallelization'],
        'entrypoint_6': ['set_fifo_depths']
    }
    
    # New approach - much cleaner!
    config = CompilationConfig(
        output_dir="./migrated_build",
        target_device="Pynq-Z1",
        target_frequency_mhz=200.0,
        constraints={"target_fps": 1000}
    )
    
    # The complexity is hidden in the strategy
    compiler = FINNCompiler()
    result = compiler.compile(
        "model.onnx",
        strategy="balanced",  # Strategies encapsulate the complexity
        config=config
    )
    
    print("Migration complete - no more 6-entrypoint confusion!")


# =============================================================================
# Example 10: Batch Compilation
# =============================================================================

def example_batch_compilation():
    """Compile multiple models with different strategies."""
    
    models = [
        ("resnet18.onnx", "performance"),
        ("mobilenet.onnx", "area"),
        ("bert_tiny.onnx", "balanced")
    ]
    
    compiler = FINNCompiler()
    
    results = []
    for model_path, strategy in models:
        config = CompilationConfig(
            output_dir=f"./batch_{strategy}",
            target_device="Pynq-Z1",
            target_frequency_mhz=200.0
        )
        
        result = compiler.compile(model_path, strategy, config)
        results.append({
            "model": model_path,
            "strategy": strategy,
            "success": result['success'],
            "fps": result.get('finn_result', {}).get('metrics', {}).get('estimated_fps')
        })
    
    # Summary
    print("Batch Compilation Results:")
    for r in results:
        print(f"  {r['model']}: {r['strategy']} - FPS: {r['fps']}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Run examples
    print("FINN API V2 Usage Examples\n")
    
    # Note: These are example functions showing API usage
    # In real use, you would call the specific example you need
    
    print("Available examples:")
    print("1. example_simple_compilation() - Basic compilation")
    print("2. example_strategy_selection() - Using different strategies")
    print("3. example_custom_configuration() - Custom configuration")
    print("4. example_bert_optimization() - BERT-specific optimization")
    print("5. example_custom_strategy() - Creating custom strategies")
    print("6. example_query_components() - Querying available components")
    print("7. example_debug_mode() - Debug mode usage")
    print("8. example_programmatic_transforms() - Building transform sequences")
    print("9. example_migration_from_old_api() - Migration guide")
    print("10. example_batch_compilation() - Batch processing")
    
    print("\nTo run an example, call the function directly.")
    print("e.g., example_simple_compilation()")