# FINN-Brainsmith API V2: Clean Design

**Date**: December 2024  
**Status**: Design Document  
**Purpose**: Complete redesign of FINN integration leveraging the new plugin system

## Executive Summary

This document presents a complete redesign of the FINN-Brainsmith integration, moving away from the flawed "6-entrypoint" concept to a clean architecture that:
- Leverages the new plugin system for kernels and transforms
- Provides clear compilation strategies instead of arbitrary entrypoints
- Enables dynamic workflow composition
- Maintains FINN compatibility through a clean adapter pattern

## 1. Problem Analysis

The current system's "6-entrypoint" concept is fundamentally flawed:
- It's an artificial abstraction that doesn't exist in FINN
- Mixes DSE components with build steps
- Creates unnecessary complexity through multiple translation layers
- Doesn't properly separate kernels (hardware) from transforms (graph operations)

## 2. Core Architecture

### 2.1 Key Abstractions

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from qonnx.core.modelwrapper import ModelWrapper

# Leverage the plugin system
from brainsmith.plugin import PluginRegistry

@dataclass
class CompilationConfig:
    """Configuration for the compilation process."""
    output_dir: str
    target_device: str  # e.g., "Pynq-Z1", "U250"
    target_frequency_mhz: float
    constraints: Dict[str, Any]  # Performance, area, power constraints
    debug_level: int = 0
    save_intermediate: bool = True

class CompilationStage(Enum):
    """Stages of the compilation pipeline."""
    GRAPH_CLEANUP = "graph_cleanup"
    TOPOLOGY_OPTIMIZATION = "topology_optimization"
    KERNEL_MAPPING = "kernel_mapping"
    KERNEL_OPTIMIZATION = "kernel_optimization"
    GRAPH_OPTIMIZATION = "graph_optimization"

@dataclass
class TransformSequence:
    """Ordered sequence of transforms to apply."""
    transforms: List[Tuple[str, Dict[str, Any]]]  # (transform_name, params)
    
    def add(self, transform_name: str, params: Optional[Dict] = None):
        """Add a transform to the sequence."""
        self.transforms.append((transform_name, params or {}))
        return self

@dataclass
class KernelSelection:
    """Specification of kernels and their backends."""
    kernels: Dict[str, str]  # kernel_name -> backend_name
    
    def use_kernel(self, kernel: str, backend: str):
        """Select a backend for a kernel."""
        self.kernels[kernel] = backend
        return self
```

### 2.2 Compilation Strategy

```python
from abc import ABC, abstractmethod

class CompilationStrategy(ABC):
    """Abstract base class for compilation strategies."""
    
    def __init__(self, config: CompilationConfig):
        self.config = config
        self.registry = PluginRegistry()
    
    @abstractmethod
    def select_kernels(self) -> KernelSelection:
        """Select kernels and backends for this strategy."""
        pass
    
    @abstractmethod
    def build_transform_sequence(self) -> Dict[CompilationStage, TransformSequence]:
        """Build transform sequences for each compilation stage."""
        pass
    
    @abstractmethod
    def get_finn_parameters(self) -> Dict[str, Any]:
        """Get FINN-specific parameters for this strategy."""
        pass
```

### 2.3 Concrete Strategies

```python
class HighPerformanceStrategy(CompilationStrategy):
    """Strategy optimized for maximum performance."""
    
    def select_kernels(self) -> KernelSelection:
        selection = KernelSelection({})
        
        # Prefer RTL backends for performance
        available_kernels = self.registry.list_kernels()
        for kernel_name, _ in available_kernels:
            # Check for RTL backend
            rtl_backend = f"{kernel_name}RTL"
            hls_backend = f"{kernel_name}HLS"
            
            if self.registry.get_backend(rtl_backend):
                selection.use_kernel(kernel_name, rtl_backend)
            elif self.registry.get_backend(hls_backend):
                selection.use_kernel(kernel_name, hls_backend)
        
        return selection
    
    def build_transform_sequence(self) -> Dict[CompilationStage, TransformSequence]:
        sequences = {}
        
        # Minimal cleanup for performance
        sequences[CompilationStage.GRAPH_CLEANUP] = TransformSequence([
            ("RemoveIdentityOps", {}),
            ("FoldConstants", {"aggressive": False})
        ])
        
        # Aggressive topology optimization
        sequences[CompilationStage.TOPOLOGY_OPTIMIZATION] = TransformSequence([
            ("ExpandNorms", {"mode": "full"}),
            ("StreamlineActivations", {"level": 3}),
            ("FuseOperations", {"aggressive": True})
        ])
        
        # Performance-oriented kernel optimization
        sequences[CompilationStage.KERNEL_OPTIMIZATION] = TransformSequence([
            ("MaximizeParallelism", {
                "target_fps": self.config.constraints.get("target_fps", 1000)
            }),
            ("UnrollLoops", {"factor": 4})
        ])
        
        return sequences
    
    def get_finn_parameters(self) -> Dict[str, Any]:
        return {
            "mvau_wwidth_max": 72,  # Wide MVAU for performance
            "standalone_thresholds": False,  # Fused for performance
            "minimize_bit_width": False,  # Keep full precision
            "auto_fifo_depths": True,
            "auto_fifo_strategy": "largefifo_rtlsim"
        }

class AreaOptimizedStrategy(CompilationStrategy):
    """Strategy optimized for minimum area."""
    
    def select_kernels(self) -> KernelSelection:
        selection = KernelSelection({})
        
        # Prefer HLS backends for resource sharing
        available_kernels = self.registry.list_kernels()
        for kernel_name, _ in available_kernels:
            hls_backend = f"{kernel_name}HLS"
            if self.registry.get_backend(hls_backend):
                selection.use_kernel(kernel_name, hls_backend)
        
        return selection
    
    def build_transform_sequence(self) -> Dict[CompilationStage, TransformSequence]:
        sequences = {}
        
        # Aggressive cleanup for area
        sequences[CompilationStage.GRAPH_CLEANUP] = TransformSequence([
            ("RemoveIdentityOps", {}),
            ("FoldConstants", {"aggressive": True}),
            ("EliminateDeadCode", {}),
            ("MergeConsecutiveOps", {})
        ])
        
        # Area-focused optimizations
        sequences[CompilationStage.KERNEL_OPTIMIZATION] = TransformSequence([
            ("MinimizeBitWidth", {"aggressive": True}),
            ("EnableResourceSharing", {"level": "maximum"}),
            ("ReduceParallelism", {"target_area": 0.8})
        ])
        
        return sequences
    
    def get_finn_parameters(self) -> Dict[str, Any]:
        return {
            "mvau_wwidth_max": 18,  # Narrow MVAU for area
            "standalone_thresholds": True,  # Allows sharing
            "minimize_bit_width": True,
            "auto_fifo_depths": False,  # Manual small FIFOs
            "large_fifo_mem_style": "distributed"
        }

class BalancedStrategy(CompilationStrategy):
    """Balanced strategy between performance and area."""
    
    def select_kernels(self) -> KernelSelection:
        selection = KernelSelection({})
        
        # Mix of RTL and HLS based on operation type
        compute_intensive = ["MatMul", "Conv2D"]
        control_intensive = ["LayerNorm", "Softmax", "Pooling"]
        
        for kernel_name, _ in self.registry.list_kernels():
            if kernel_name in compute_intensive:
                # Prefer RTL for compute
                rtl = f"{kernel_name}RTL"
                if self.registry.get_backend(rtl):
                    selection.use_kernel(kernel_name, rtl)
                else:
                    selection.use_kernel(kernel_name, f"{kernel_name}HLS")
            elif kernel_name in control_intensive:
                # Prefer HLS for control
                selection.use_kernel(kernel_name, f"{kernel_name}HLS")
        
        return selection
    
    def build_transform_sequence(self) -> Dict[CompilationStage, TransformSequence]:
        # Balanced set of optimizations
        sequences = {}
        
        sequences[CompilationStage.GRAPH_CLEANUP] = TransformSequence([
            ("RemoveIdentityOps", {}),
            ("FoldConstants", {})
        ])
        
        sequences[CompilationStage.TOPOLOGY_OPTIMIZATION] = TransformSequence([
            ("ExpandNorms", {}),
            ("StreamlineActivations", {"level": 2})
        ])
        
        if "target_fps" in self.config.constraints:
            sequences[CompilationStage.KERNEL_OPTIMIZATION] = TransformSequence([
                ("BalancedParallelization", {
                    "target_fps": self.config.constraints["target_fps"],
                    "area_budget": 0.8
                })
            ])
        
        return sequences
    
    def get_finn_parameters(self) -> Dict[str, Any]:
        return {
            "mvau_wwidth_max": 36,  # Default
            "minimize_bit_width": True,
            "auto_fifo_depths": True
        }
```

### 2.4 Workflow Engine

```python
class FINNCompiler:
    """Main compiler that orchestrates the FINN build process."""
    
    def __init__(self):
        self.registry = PluginRegistry()
        self.strategies = {
            "performance": HighPerformanceStrategy,
            "area": AreaOptimizedStrategy,
            "balanced": BalancedStrategy
        }
    
    def compile(self,
                model_path: str,
                strategy: str = "balanced",
                config: Optional[CompilationConfig] = None) -> Dict[str, Any]:
        """
        Compile a model using the specified strategy.
        
        Args:
            model_path: Path to ONNX model
            strategy: Strategy name or custom strategy instance
            config: Compilation configuration
            
        Returns:
            Compilation results including metrics and artifacts
        """
        # Default config if not provided
        if config is None:
            config = CompilationConfig(
                output_dir="./finn_output",
                target_device="Pynq-Z1",
                target_frequency_mhz=200.0,
                constraints={}
            )
        
        # Create strategy instance
        if isinstance(strategy, str):
            if strategy not in self.strategies:
                raise ValueError(f"Unknown strategy: {strategy}")
            strategy_instance = self.strategies[strategy](config)
        else:
            strategy_instance = strategy
        
        # Execute compilation workflow
        return self._execute_workflow(model_path, strategy_instance, config)
    
    def _execute_workflow(self,
                         model_path: str,
                         strategy: CompilationStrategy,
                         config: CompilationConfig) -> Dict[str, Any]:
        """Execute the compilation workflow."""
        
        # Load model
        model = ModelWrapper(model_path)
        
        # Select kernels
        kernel_selection = strategy.select_kernels()
        
        # Build transform sequences
        transform_sequences = strategy.build_transform_sequence()
        
        # Apply transforms stage by stage
        for stage in CompilationStage:
            if stage in transform_sequences:
                model = self._apply_transform_sequence(
                    model, 
                    transform_sequences[stage]
                )
        
        # Generate FINN configuration
        finn_config = self._build_finn_config(
            strategy, 
            kernel_selection, 
            config
        )
        
        # Execute FINN build
        if config.debug_level == 0:
            # Normal execution
            result = self._execute_finn_build(model, finn_config)
        else:
            # Debug mode - just return config
            result = {
                "model": model,
                "finn_config": finn_config,
                "debug": True
            }
        
        return {
            "success": True,
            "strategy_used": strategy.__class__.__name__,
            "kernels_selected": kernel_selection.kernels,
            "transforms_applied": self._get_applied_transforms(transform_sequences),
            "finn_result": result
        }
    
    def _apply_transform_sequence(self,
                                 model: ModelWrapper,
                                 sequence: TransformSequence) -> ModelWrapper:
        """Apply a sequence of transforms to the model."""
        for transform_name, params in sequence.transforms:
            transform_class = self.registry.get_transform(transform_name)
            if not transform_class:
                raise ValueError(f"Transform not found: {transform_name}")
            
            transform = transform_class(**params)
            model, _ = transform.apply(model)
        
        return model
    
    def _build_finn_config(self,
                          strategy: CompilationStrategy,
                          kernel_selection: KernelSelection,
                          config: CompilationConfig) -> Dict:
        """Build FINN configuration from strategy."""
        finn_params = strategy.get_finn_parameters()
        
        # Base configuration
        finn_config = {
            "output_dir": config.output_dir,
            "synth_clk_period_ns": 1000.0 / config.target_frequency_mhz,
            "board": config.target_device,
            "save_intermediate_models": config.save_intermediate,
            **finn_params
        }
        
        # Add constraints
        if "target_fps" in config.constraints:
            finn_config["target_fps"] = config.constraints["target_fps"]
        
        # Kernel backend mapping
        finn_config["kernel_backends"] = kernel_selection.kernels
        
        return finn_config
    
    def _execute_finn_build(self,
                           model: ModelWrapper,
                           finn_config: Dict) -> Dict:
        """Execute actual FINN build (or delegate to legacy adapter)."""
        # This would use the LegacyFINNAdapter to convert to DataflowBuildConfig
        # and execute the actual FINN build
        adapter = LegacyFINNAdapter()
        return adapter.execute_build(model, finn_config)
```

### 2.5 Legacy FINN Adapter

```python
class LegacyFINNAdapter:
    """Adapter to interface with legacy FINN DataflowBuildConfig."""
    
    def execute_build(self, model: ModelWrapper, config: Dict) -> Dict:
        """Execute FINN build using legacy interface."""
        from finn.builder.build_dataflow_config import (
            DataflowBuildConfig, 
            DataflowOutputType
        )
        from finn.builder.build_dataflow import build_dataflow_cfg
        
        # Convert to DataflowBuildConfig
        dataflow_config = self._create_dataflow_config(config)
        
        # Save model to temporary location
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            model.save(tmp.name)
            model_path = tmp.name
        
        try:
            # Execute FINN build
            result = build_dataflow_cfg(model_path, dataflow_config)
            
            # Extract metrics
            metrics = self._extract_metrics(dataflow_config.output_dir)
            
            return {
                "success": result == 0,
                "output_dir": dataflow_config.output_dir,
                "metrics": metrics
            }
        finally:
            # Cleanup
            import os
            if os.path.exists(model_path):
                os.remove(model_path)
    
    def _create_dataflow_config(self, config: Dict) -> Any:
        """Create DataflowBuildConfig from our config format."""
        from finn.builder.build_dataflow_config import (
            DataflowBuildConfig,
            DataflowOutputType
        )
        
        # Map our config to FINN config
        return DataflowBuildConfig(
            output_dir=config["output_dir"],
            synth_clk_period_ns=config["synth_clk_period_ns"],
            board=config.get("board"),
            fpga_part=config.get("fpga_part"),
            target_fps=config.get("target_fps"),
            save_intermediate_models=config.get("save_intermediate_models", True),
            minimize_bit_width=config.get("minimize_bit_width", True),
            mvau_wwidth_max=config.get("mvau_wwidth_max", 36),
            auto_fifo_depths=config.get("auto_fifo_depths", True),
            generate_outputs=[
                DataflowOutputType.STITCHED_IP,
                DataflowOutputType.ESTIMATE_REPORTS
            ]
        )
    
    def _extract_metrics(self, output_dir: str) -> Dict[str, Any]:
        """Extract metrics from FINN build output."""
        metrics = {}
        
        # Read estimate reports
        import json
        import os
        
        perf_file = os.path.join(output_dir, "report", "estimate_network_performance.json")
        if os.path.exists(perf_file):
            with open(perf_file, 'r') as f:
                perf = json.load(f)
                metrics["estimated_fps"] = perf.get("estimated_throughput_fps")
                metrics["latency_cycles"] = perf.get("critical_path_cycles")
        
        res_file = os.path.join(output_dir, "report", "estimate_layer_resources.json")
        if os.path.exists(res_file):
            with open(res_file, 'r') as f:
                res = json.load(f)
                if "total" in res:
                    metrics["resource_usage"] = res["total"]
        
        return metrics
```

## 3. Usage Examples

### 3.1 Basic Usage

```python
# Simple compilation with default strategy
compiler = FINNCompiler()
result = compiler.compile("model.onnx", strategy="balanced")

print(f"Compilation {'succeeded' if result['success'] else 'failed'}")
print(f"Kernels used: {result['kernels_selected']}")
print(f"Estimated FPS: {result['finn_result']['metrics'].get('estimated_fps')}")
```

### 3.2 Custom Configuration

```python
# Custom configuration for high-performance BERT
config = CompilationConfig(
    output_dir="./bert_accelerator",
    target_device="U250",
    target_frequency_mhz=300.0,
    constraints={
        "target_fps": 5000,
        "max_power_w": 75,
        "max_lut_utilization": 0.85
    }
)

compiler = FINNCompiler()
result = compiler.compile(
    "bert_model.onnx",
    strategy="performance",
    config=config
)
```

### 3.3 Custom Strategy

```python
class BERTOptimizedStrategy(CompilationStrategy):
    """Custom strategy optimized for BERT models."""
    
    def select_kernels(self) -> KernelSelection:
        return KernelSelection({
            "MatMul": "MatMulRTL",
            "LayerNorm": "LayerNormBrainSmith",  # Custom kernel
            "Softmax": "SoftmaxHLS",
            "GELU": "GELUHLS"
        })
    
    def build_transform_sequence(self) -> Dict[CompilationStage, TransformSequence]:
        sequences = {}
        
        # BERT-specific optimizations
        sequences[CompilationStage.TOPOLOGY_OPTIMIZATION] = TransformSequence([
            ("ExpandNorms", {}),
            ("FuseAttentionOps", {"num_heads": 12}),
            ("OptimizeGELU", {})
        ])
        
        sequences[CompilationStage.KERNEL_OPTIMIZATION] = TransformSequence([
            ("BERTFoldingOptimization", {
                "sequence_length": 512,
                "hidden_size": 768
            })
        ])
        
        return sequences
    
    def get_finn_parameters(self) -> Dict[str, Any]:
        return {
            "mvau_wwidth_max": 64,
            "folding_config_file": "configs/bert_folding.json"
        }

# Use custom strategy
bert_strategy = BERTOptimizedStrategy(config)
result = compiler.compile("bert.onnx", strategy=bert_strategy)
```

### 3.4 Querying Available Components

```python
compiler = FINNCompiler()

# List all available transforms by stage
for stage in CompilationStage:
    transforms = compiler.registry.list_transforms(stage=stage.value)
    print(f"\n{stage.name} transforms:")
    for name, transform_class in transforms:
        metadata = transform_class._plugin_metadata
        print(f"  - {name}: {metadata.get('description', 'No description')}")

# List all kernels and their backends
kernels = compiler.registry.list_kernels()
for kernel_name, _ in kernels:
    print(f"\nKernel: {kernel_name}")
    backends = compiler.registry.list_backends()
    for backend_name, _ in backends:
        if backend_name.startswith(kernel_name):
            print(f"  - Backend: {backend_name}")
```

## 4. Benefits of This Design

### 4.1 Clear Separation of Concerns
- **Kernels**: Hardware implementations (via plugin system)
- **Transforms**: Graph operations (via plugin system)
- **Strategies**: High-level compilation approaches
- **Compiler**: Orchestration and workflow management

### 4.2 Leverages Plugin System
- All kernels and transforms use the existing plugin infrastructure
- Automatic discovery and registration
- Easy extension by contributors

### 4.3 No Artificial Abstractions
- No "6-entrypoint" concept
- Direct mapping to FINN's actual architecture
- Clear compilation stages that match the process

### 4.4 Flexible and Extensible
- Easy to add new strategies
- Custom strategies are first-class citizens
- Dynamic transform composition

### 4.5 Clean Legacy Integration
- Adapter pattern isolates FINN dependencies
- Easy to update when FINN API changes
- Clear boundary between modern and legacy code

## 5. Implementation Plan

### Phase 1: Core Infrastructure
1. Implement base classes (CompilationConfig, Strategy, etc.)
2. Create standard strategies (Performance, Area, Balanced)
3. Build FINNCompiler orchestrator

### Phase 2: Legacy Integration
1. Implement LegacyFINNAdapter
2. Add metrics extraction
3. Test with existing FINN builds

### Phase 3: Migration
1. Port existing transforms to plugin system
2. Register existing kernels
3. Create migration guide

### Phase 4: Deprecation
1. Mark old 6-entrypoint system as deprecated
2. Update documentation
3. Remove old code in next major version

## 6. Conclusion

This design provides a clean, modern API that:
- Properly separates kernels, transforms, and strategies
- Leverages the plugin system effectively
- Eliminates conceptual confusion
- Provides clear compilation workflows
- Maintains backward compatibility

The result is a system that is easier to understand, extend, and maintain while providing more power and flexibility than the current implementation.