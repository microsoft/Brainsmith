# FINN-Brainsmith API V2 Design

**Version**: 2.0  
**Date**: December 2024  
**Purpose**: Complete redesign of FINN integration with cleaner abstractions

## Executive Summary

This design replaces the flawed "6-entrypoint" system with a cleaner, more intuitive API that properly separates concerns between kernels, transforms, and compilation strategies.

## Core Concepts

### 1. Kernels and Kernel Backends

**Kernel**: A hardware-acceleratable operation (e.g., MatMul, Conv2D, LayerNorm)  
**Kernel Backend**: A specific implementation of a kernel (e.g., HLS, RTL, custom)

```python
@dataclass
class Kernel:
    name: str                    # e.g., "MatMul"
    operation_type: str          # e.g., "gemm"
    supported_backends: List[str] # e.g., ["hls", "rtl", "brainsmith_rtl"]
    constraints: Dict[str, Any]   # Backend-specific constraints

@dataclass 
class KernelBackend:
    kernel_name: str             # e.g., "MatMul"
    backend_type: str            # e.g., "hls"
    implementation_path: str     # Path to implementation
    parameters: Dict[str, Any]   # Backend-specific parameters
```

### 2. Transform Pipeline

**Transform**: A graph transformation operation applied at specific compilation stages

```python
@dataclass
class Transform:
    name: str                    # e.g., "FoldConstants"
    stage: CompilationStage      # When to apply
    category: TransformCategory  # Type of transform
    function: Callable           # Actual transformation function

class CompilationStage(Enum):
    GRAPH_CLEANUP = "graph_cleanup"
    TOPOLOGY_OPTIMIZATION = "topology_optimization"  
    KERNEL_MAPPING = "kernel_mapping"
    KERNEL_OPTIMIZATION = "kernel_optimization"
    GRAPH_OPTIMIZATION = "graph_optimization"

class TransformCategory(Enum):
    CLEANUP = "cleanup"          # Graph cleanup ops
    STREAMLINING = "streamlining" # Topology optimization
    LOWERING = "lowering"        # Convert to hardware ops
    SPECIALIZATION = "specialization" # Kernel-specific opts
    SYSTEM = "system"            # System-level opts
```

### 3. Compilation Strategy

**CompilationStrategy**: High-level specification of how to compile a model

```python
@dataclass
class CompilationStrategy:
    name: str                    # e.g., "bert_optimized"
    description: str             # Human-readable description
    kernels: List[KernelSpec]    # Kernels to use
    transforms: List[TransformSpec] # Transforms to apply
    parameters: Dict[str, Any]   # Strategy parameters

@dataclass
class KernelSpec:
    kernel: Kernel
    preferred_backend: Optional[str]
    backend_config: Dict[str, Any]

@dataclass
class TransformSpec:
    transform: Transform
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
```

## Architecture Overview

```
┌─────────────────────────────────────────┐
│         Plugin System                   │
│  (Community kernels & transforms)       │
├─────────────────┴───────────────────────┤
│  • Plugin Registry & Discovery          │
│  • BrainSmith Hub Integration           │
│  • Quality Certification                │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         CompilationStrategy             │
│  (High-level compilation specification) │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         FINNCompiler                    │
│  (Orchestrates compilation process)     │
├─────────────────┴───────────────────────┤
│  • KernelRegistry (built-in + plugins)  │
│  • TransformRegistry (built-in + plugins)│
│  • StageExecutor                        │
│  • PluginLoader & Sandbox               │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         Legacy FINN Adapter             │
│  (Converts to DataflowBuildConfig)      │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         FINN Build Process              │
│  (Actual FINN execution)                │
└─────────────────────────────────────────┘
```

## Implementation Design

### 1. FINNCompiler - Main Orchestrator

```python
class FINNCompiler:
    """Main compiler orchestrating the FINN build process."""
    
    def __init__(self):
        self.kernel_registry = KernelRegistry()
        self.transform_registry = TransformRegistry()
        self.stage_executor = StageExecutor()
    
    def compile(self, 
                model_path: str,
                strategy: CompilationStrategy,
                output_dir: str) -> CompilationResult:
        """
        Compile model using specified strategy.
        
        Args:
            model_path: Path to ONNX model
            strategy: Compilation strategy to use
            output_dir: Output directory for artifacts
            
        Returns:
            CompilationResult with metrics and artifacts
        """
        # Initialize compilation context
        context = CompilationContext(
            model_path=model_path,
            strategy=strategy,
            output_dir=output_dir,
            metadata={}
        )
        
        # Execute compilation stages
        for stage in CompilationStage:
            context = self.stage_executor.execute_stage(
                stage, context, strategy
            )
        
        # Extract results
        return self._extract_results(context)
```

### 2. KernelRegistry - Hardware Kernel Management with Plugin Support

```python
class KernelRegistry:
    """Registry for available kernels and backends with plugin support."""
    
    def __init__(self, plugin_loader: Optional[PluginLoader] = None):
        self._kernels: Dict[str, Kernel] = {}
        self._backends: Dict[Tuple[str, str], KernelBackend] = {}
        self._plugin_loader = plugin_loader or PluginLoader()
        self._initialize_builtin_kernels()
        self._discover_plugin_kernels()
    
    def register_kernel(self, kernel: Kernel, source: str = "builtin"):
        """Register a new kernel."""
        self._kernels[kernel.name] = kernel
        kernel.metadata["source"] = source  # Track if builtin or plugin
        logger.info(f"Registered {source} kernel: {kernel.name}")
    
    def register_backend(self, backend: KernelBackend):
        """Register a kernel backend implementation."""
        key = (backend.kernel_name, backend.backend_type)
        self._backends[key] = backend
    
    def get_kernel(self, name: str) -> Optional[Kernel]:
        """Get kernel by name (builtin or plugin)."""
        return self._kernels.get(name)
    
    def search_kernels(self, query: str, filters: Dict[str, Any] = None) -> List[Kernel]:
        """Search for kernels with advanced filtering."""
        results = []
        for kernel in self._kernels.values():
            if self._matches_search(kernel, query, filters):
                results.append(kernel)
        return results
    
    def _discover_plugin_kernels(self):
        """Auto-discover and register plugin kernels."""
        plugins = self._plugin_loader.discover_kernels()
        for plugin_kernel in plugins:
            self.register_kernel(plugin_kernel, source="plugin")
    
    def _initialize_builtin_kernels(self):
        """Initialize built-in FINN and BrainSmith kernels."""
        # FINN standard kernels
        self.register_kernel(Kernel(
            name="MatMul",
            operation_type="gemm",
            supported_backends=["hls", "rtl"],
            constraints={"min_size": 16}
        ))
        
        # BrainSmith custom kernels
        self.register_kernel(Kernel(
            name="LayerNorm",
            operation_type="layer_norm",
            supported_backends=["brainsmith_rtl", "hls"],
            constraints={"epsilon": 1e-5}
        ))
```

### 3. TransformRegistry - Transform Management

```python
class TransformRegistry:
    """Registry for available transforms."""
    
    def __init__(self):
        self._transforms: Dict[str, Transform] = {}
        self._initialize_builtin_transforms()
    
    def register_transform(self, transform: Transform):
        """Register a new transform."""
        self._transforms[transform.name] = transform
    
    def get_transforms_for_stage(self, 
                                stage: CompilationStage) -> List[Transform]:
        """Get all transforms for a specific stage."""
        return [t for t in self._transforms.values() 
                if t.stage == stage]
    
    def _initialize_builtin_transforms(self):
        """Initialize built-in transforms."""
        # Graph cleanup transforms
        self.register_transform(Transform(
            name="FoldConstants",
            stage=CompilationStage.GRAPH_CLEANUP,
            category=TransformCategory.CLEANUP,
            function=fold_constants_transform
        ))
        
        # Topology optimization transforms
        self.register_transform(Transform(
            name="Streamline",
            stage=CompilationStage.TOPOLOGY_OPTIMIZATION,
            category=TransformCategory.STREAMLINING,
            function=streamline_transform
        ))
```

### 4. Legacy FINN Adapter

```python
class LegacyFINNAdapter:
    """Adapts modern API to legacy FINN DataflowBuildConfig."""
    
    def convert_to_dataflow_config(self,
                                  context: CompilationContext) -> Any:
        """
        Convert compilation context to FINN DataflowBuildConfig.
        
        Args:
            context: Current compilation context
            
        Returns:
            FINN DataflowBuildConfig object
        """
        from finn.builder.build_dataflow_config import DataflowBuildConfig
        
        # Build step functions from context
        steps = self._build_step_functions(context)
        
        # Extract FINN parameters
        params = self._extract_finn_params(context)
        
        # Create DataflowBuildConfig
        return DataflowBuildConfig(
            steps=steps,
            output_dir=context.output_dir,
            **params
        )
    
    def _build_step_functions(self, 
                            context: CompilationContext) -> List[Callable]:
        """Build FINN step functions from compilation context."""
        steps = []
        
        # Convert applied transforms to step functions
        for stage_name, transforms in context.applied_transforms.items():
            for transform in transforms:
                step_func = self._wrap_transform_as_step(transform)
                steps.append(step_func)
        
        # Add standard FINN steps
        steps.extend(self._get_standard_finn_steps())
        
        return steps
```

## Usage Examples

### Example 1: BERT Compilation

```python
# Define BERT-optimized compilation strategy
bert_strategy = CompilationStrategy(
    name="bert_optimized",
    description="Optimized compilation for BERT models",
    kernels=[
        KernelSpec(
            kernel=kernel_registry.get_kernel("MatMul"),
            preferred_backend="hls",
            backend_config={"optimization": "latency"}
        ),
        KernelSpec(
            kernel=kernel_registry.get_kernel("LayerNorm"),
            preferred_backend="brainsmith_rtl",
            backend_config={"precision": "int8"}
        ),
        KernelSpec(
            kernel=kernel_registry.get_kernel("Softmax"),
            preferred_backend="hls",
            backend_config={}
        )
    ],
    transforms=[
        TransformSpec(
            transform=transform_registry.get_transform("ExpandNorms"),
            enabled=True,
            config={"mode": "aggressive"}
        ),
        TransformSpec(
            transform=transform_registry.get_transform("Streamline"),
            enabled=True,
            config={"level": 2}
        )
    ],
    parameters={
        "target_frequency_mhz": 200,
        "target_throughput_fps": 3000,
        "folding_config": "configs/bert_folding.json"
    }
)

# Execute compilation
compiler = FINNCompiler()
result = compiler.compile(
    model_path="bert_model.onnx",
    strategy=bert_strategy,
    output_dir="./bert_build"
)

print(f"Compilation successful: {result.success}")
print(f"Throughput: {result.metrics['throughput']} FPS")
print(f"Resource utilization: {result.metrics['resource_utilization']}%")
```

### Example 2: Custom Transform Registration

```python
# Define custom transform
def custom_attention_optimization(model, config):
    """Custom optimization for attention layers."""
    # Transform implementation
    return optimized_model

# Register transform
transform_registry.register_transform(Transform(
    name="OptimizeAttention",
    stage=CompilationStage.KERNEL_OPTIMIZATION,
    category=TransformCategory.SPECIALIZATION,
    function=custom_attention_optimization
))

# Use in strategy
strategy = CompilationStrategy(
    name="custom_attention",
    transforms=[
        TransformSpec(
            transform=transform_registry.get_transform("OptimizeAttention"),
            enabled=True,
            config={"heads": 12}
        )
    ],
    # ... other configuration
)
```

## Key Improvements Over Current System

1. **Clear Abstractions**: Kernels, transforms, and strategies are clearly separated
2. **Plugin System**: Seamless integration of community contributions
3. **Declarative Registration**: Simple decorators for creating components
4. **Auto-Discovery**: Plugins found automatically from multiple sources
5. **Type Safety**: Strong typing throughout with dataclasses
6. **Stage-Based Execution**: Clear compilation stages instead of mixed steps
7. **Registry Pattern**: Central management of components with search
8. **Quality Assurance**: Automated validation and certification
9. **Community Hub**: Central platform for sharing and discovery
10. **Legacy Compatibility**: Clean adapter pattern for FINN integration

## Migration Path

1. **Phase 1**: Implement core abstractions (Kernel, Transform, Strategy)
2. **Phase 2**: Build registries and populate with existing components
3. **Phase 3**: Implement FINNCompiler with stage-based execution
4. **Phase 4**: Create Legacy FINN Adapter
5. **Phase 5**: Migrate existing code to use new API
6. **Phase 6**: Deprecate old 6-entrypoint system

## Conclusion

This design provides a clean, extensible API that properly separates concerns and provides clear abstractions for hardware kernels, graph transforms, and compilation strategies. It maintains compatibility with legacy FINN while providing a modern interface for future development.