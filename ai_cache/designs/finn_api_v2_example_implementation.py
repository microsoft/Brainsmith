#!/usr/bin/env python3
"""
FINN-Brainsmith API V2 - Example Implementation

This file demonstrates the new API design with concrete implementations
of the core components. This serves as a reference for the actual implementation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Core Data Models
# =============================================================================

class CompilationStage(Enum):
    """Stages of the compilation process."""
    GRAPH_CLEANUP = "graph_cleanup"
    TOPOLOGY_OPTIMIZATION = "topology_optimization"
    KERNEL_MAPPING = "kernel_mapping"
    KERNEL_OPTIMIZATION = "kernel_optimization"
    GRAPH_OPTIMIZATION = "graph_optimization"


class TransformCategory(Enum):
    """Categories of transforms."""
    CLEANUP = "cleanup"
    STREAMLINING = "streamlining"
    LOWERING = "lowering"
    SPECIALIZATION = "specialization"
    SYSTEM = "system"


@dataclass
class Kernel:
    """Hardware-acceleratable operation."""
    name: str
    operation_type: str
    supported_backends: List[str]
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KernelBackend:
    """Specific implementation of a kernel."""
    kernel_name: str
    backend_type: str
    implementation_path: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transform:
    """Graph transformation operation."""
    name: str
    stage: CompilationStage
    category: TransformCategory
    function: Callable
    dependencies: List[str] = field(default_factory=list)


@dataclass
class KernelSpec:
    """Kernel specification in a compilation strategy."""
    kernel: Kernel
    preferred_backend: Optional[str] = None
    backend_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformSpec:
    """Transform specification in a compilation strategy."""
    transform: Transform
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompilationStrategy:
    """High-level compilation specification."""
    name: str
    description: str
    kernels: List[KernelSpec]
    transforms: List[TransformSpec]
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompilationContext:
    """Context maintained throughout compilation."""
    model_path: str
    strategy: CompilationStrategy
    output_dir: str
    model: Any = None  # ONNX ModelWrapper
    applied_transforms: Dict[str, List[Transform]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompilationResult:
    """Result of compilation."""
    success: bool
    output_dir: str
    metrics: Dict[str, float]
    artifacts: Dict[str, str]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Registry Implementations
# =============================================================================

class KernelRegistry:
    """Registry for kernels and backends."""
    
    def __init__(self):
        self._kernels: Dict[str, Kernel] = {}
        self._backends: Dict[Tuple[str, str], KernelBackend] = {}
        self._initialize_builtin()
    
    def register_kernel(self, kernel: Kernel):
        """Register a kernel."""
        self._kernels[kernel.name] = kernel
        logger.info(f"Registered kernel: {kernel.name}")
    
    def register_backend(self, backend: KernelBackend):
        """Register a kernel backend."""
        key = (backend.kernel_name, backend.backend_type)
        self._backends[key] = backend
        logger.info(f"Registered backend: {backend.kernel_name}/{backend.backend_type}")
    
    def get_kernel(self, name: str) -> Optional[Kernel]:
        """Get kernel by name."""
        return self._kernels.get(name)
    
    def get_backend(self, kernel_name: str, backend_type: str) -> Optional[KernelBackend]:
        """Get specific backend for kernel."""
        return self._backends.get((kernel_name, backend_type))
    
    def list_kernels(self) -> List[str]:
        """List all registered kernels."""
        return list(self._kernels.keys())
    
    def _initialize_builtin(self):
        """Initialize built-in kernels."""
        # FINN standard kernels
        self.register_kernel(Kernel(
            name="MatMul",
            operation_type="gemm",
            supported_backends=["hls", "rtl"],
            constraints={"min_batch": 1, "min_size": 16}
        ))
        
        self.register_kernel(Kernel(
            name="Conv2D",
            operation_type="conv",
            supported_backends=["hls", "rtl"],
            constraints={"min_kernel": 1, "max_kernel": 11}
        ))
        
        # BrainSmith custom kernels
        self.register_kernel(Kernel(
            name="LayerNorm",
            operation_type="layer_norm",
            supported_backends=["brainsmith_rtl", "hls"],
            constraints={"epsilon": 1e-5}
        ))
        
        self.register_kernel(Kernel(
            name="Softmax",
            operation_type="softmax",
            supported_backends=["hls", "brainsmith_hls"],
            constraints={"axis": -1}
        ))
        
        # Register example backends
        self.register_backend(KernelBackend(
            kernel_name="MatMul",
            backend_type="hls",
            implementation_path="finn.custom_op.matmul.MatMulHLS",
            parameters={"optimization": "latency"}
        ))
        
        self.register_backend(KernelBackend(
            kernel_name="LayerNorm",
            backend_type="brainsmith_rtl",
            implementation_path="brainsmith.kernels.rtl.LayerNormRTL",
            parameters={"precision": "int8"}
        ))


class TransformRegistry:
    """Registry for transforms."""
    
    def __init__(self):
        self._transforms: Dict[str, Transform] = {}
        self._initialize_builtin()
    
    def register_transform(self, transform: Transform):
        """Register a transform."""
        self._transforms[transform.name] = transform
        logger.info(f"Registered transform: {transform.name}")
    
    def get_transform(self, name: str) -> Optional[Transform]:
        """Get transform by name."""
        return self._transforms.get(name)
    
    def get_transforms_for_stage(self, stage: CompilationStage) -> List[Transform]:
        """Get all transforms for a specific stage."""
        return [t for t in self._transforms.values() if t.stage == stage]
    
    def list_transforms(self) -> List[str]:
        """List all registered transforms."""
        return list(self._transforms.keys())
    
    def _initialize_builtin(self):
        """Initialize built-in transforms."""
        # Graph cleanup transforms
        self.register_transform(Transform(
            name="FoldConstants",
            stage=CompilationStage.GRAPH_CLEANUP,
            category=TransformCategory.CLEANUP,
            function=self._fold_constants_transform
        ))
        
        self.register_transform(Transform(
            name="RemoveUnusedNodes",
            stage=CompilationStage.GRAPH_CLEANUP,
            category=TransformCategory.CLEANUP,
            function=self._remove_unused_transform
        ))
        
        # Topology optimization transforms
        self.register_transform(Transform(
            name="Streamline",
            stage=CompilationStage.TOPOLOGY_OPTIMIZATION,
            category=TransformCategory.STREAMLINING,
            function=self._streamline_transform,
            dependencies=["FoldConstants"]
        ))
        
        self.register_transform(Transform(
            name="ExpandNorms",
            stage=CompilationStage.TOPOLOGY_OPTIMIZATION,
            category=TransformCategory.STREAMLINING,
            function=self._expand_norms_transform
        ))
        
        # Hardware lowering transforms
        self.register_transform(Transform(
            name="InferHardware",
            stage=CompilationStage.KERNEL_MAPPING,
            category=TransformCategory.LOWERING,
            function=self._infer_hardware_transform,
            dependencies=["Streamline"]
        ))
        
        # Kernel optimization transforms
        self.register_transform(Transform(
            name="SetFolding",
            stage=CompilationStage.KERNEL_OPTIMIZATION,
            category=TransformCategory.SPECIALIZATION,
            function=self._set_folding_transform
        ))
        
        # Graph optimization transforms
        self.register_transform(Transform(
            name="SetFIFODepths",
            stage=CompilationStage.GRAPH_OPTIMIZATION,
            category=TransformCategory.SYSTEM,
            function=self._set_fifo_depths_transform
        ))
    
    # Example transform implementations (would be actual transforms in real code)
    def _fold_constants_transform(self, model, config):
        logger.debug("Applying FoldConstants transform")
        # In real implementation, would call QONNX FoldConstants
        return model
    
    def _remove_unused_transform(self, model, config):
        logger.debug("Applying RemoveUnusedNodes transform")
        return model
    
    def _streamline_transform(self, model, config):
        logger.debug("Applying Streamline transform")
        return model
    
    def _expand_norms_transform(self, model, config):
        logger.debug("Applying ExpandNorms transform")
        return model
    
    def _infer_hardware_transform(self, model, config):
        logger.debug("Applying InferHardware transform")
        return model
    
    def _set_folding_transform(self, model, config):
        logger.debug("Applying SetFolding transform")
        return model
    
    def _set_fifo_depths_transform(self, model, config):
        logger.debug("Applying SetFIFODepths transform")
        return model


# =============================================================================
# Stage Executor
# =============================================================================

class StageExecutor:
    """Executes compilation stages."""
    
    def __init__(self, kernel_registry: KernelRegistry, transform_registry: TransformRegistry):
        self.kernel_registry = kernel_registry
        self.transform_registry = transform_registry
    
    def execute_stage(self, stage: CompilationStage, 
                     context: CompilationContext) -> CompilationContext:
        """Execute a compilation stage."""
        logger.info(f"Executing stage: {stage.value}")
        
        # Get transforms for this stage from strategy
        stage_transforms = self._get_stage_transforms(stage, context.strategy)
        
        # Apply each transform
        for transform_spec in stage_transforms:
            if transform_spec.enabled:
                context = self._apply_transform(transform_spec, context)
        
        # Record applied transforms
        if stage.value not in context.applied_transforms:
            context.applied_transforms[stage.value] = []
        context.applied_transforms[stage.value].extend(
            [ts.transform for ts in stage_transforms if ts.enabled]
        )
        
        return context
    
    def _get_stage_transforms(self, stage: CompilationStage, 
                            strategy: CompilationStrategy) -> List[TransformSpec]:
        """Get transforms for a specific stage from strategy."""
        return [ts for ts in strategy.transforms 
                if ts.transform.stage == stage]
    
    def _apply_transform(self, transform_spec: TransformSpec, 
                        context: CompilationContext) -> CompilationContext:
        """Apply a single transform."""
        transform = transform_spec.transform
        config = transform_spec.config
        
        logger.debug(f"Applying transform: {transform.name}")
        
        # Check dependencies
        for dep in transform.dependencies:
            if not self._is_transform_applied(dep, context):
                logger.warning(f"Dependency {dep} not satisfied for {transform.name}")
        
        # Apply transform
        try:
            context.model = transform.function(context.model, config)
            context.metadata[f"transform_{transform.name}"] = "success"
        except Exception as e:
            logger.error(f"Transform {transform.name} failed: {e}")
            context.metadata[f"transform_{transform.name}"] = f"failed: {e}"
        
        return context
    
    def _is_transform_applied(self, transform_name: str, 
                            context: CompilationContext) -> bool:
        """Check if a transform has been applied."""
        for transforms in context.applied_transforms.values():
            if any(t.name == transform_name for t in transforms):
                return True
        return False


# =============================================================================
# Main Compiler
# =============================================================================

class FINNCompiler:
    """Main compiler orchestrating the build process."""
    
    def __init__(self):
        self.kernel_registry = KernelRegistry()
        self.transform_registry = TransformRegistry()
        self.stage_executor = StageExecutor(self.kernel_registry, self.transform_registry)
    
    def compile(self, model_path: str, strategy: CompilationStrategy, 
               output_dir: str) -> CompilationResult:
        """
        Compile model using specified strategy.
        
        Args:
            model_path: Path to ONNX model
            strategy: Compilation strategy
            output_dir: Output directory
            
        Returns:
            CompilationResult
        """
        logger.info(f"Starting compilation with strategy: {strategy.name}")
        
        # Initialize context
        context = CompilationContext(
            model_path=model_path,
            strategy=strategy,
            output_dir=output_dir,
            metadata={"strategy": strategy.name}
        )
        
        try:
            # Load model
            context = self._load_model(context)
            
            # Execute all compilation stages
            for stage in CompilationStage:
                context = self.stage_executor.execute_stage(stage, context)
            
            # Generate outputs
            artifacts = self._generate_outputs(context)
            
            # Extract metrics
            metrics = self._extract_metrics(context)
            
            return CompilationResult(
                success=True,
                output_dir=output_dir,
                metrics=metrics,
                artifacts=artifacts
            )
            
        except Exception as e:
            logger.error(f"Compilation failed: {e}")
            return CompilationResult(
                success=False,
                output_dir=output_dir,
                metrics={},
                artifacts={},
                errors=[str(e)]
            )
    
    def _load_model(self, context: CompilationContext) -> CompilationContext:
        """Load ONNX model."""
        logger.info(f"Loading model: {context.model_path}")
        # In real implementation, would use ModelWrapper
        context.model = f"MockModel({context.model_path})"
        return context
    
    def _generate_outputs(self, context: CompilationContext) -> Dict[str, str]:
        """Generate compilation outputs."""
        artifacts = {
            "stitched_ip": f"{context.output_dir}/stitched_ip",
            "estimates": f"{context.output_dir}/estimates.json",
            "folding_config": f"{context.output_dir}/folding.json"
        }
        logger.info(f"Generated artifacts: {list(artifacts.keys())}")
        return artifacts
    
    def _extract_metrics(self, context: CompilationContext) -> Dict[str, float]:
        """Extract performance metrics."""
        # In real implementation, would extract from FINN results
        metrics = {
            "throughput": 1000.0,  # FPS
            "latency": 5.0,        # ms
            "lut_utilization": 0.75,
            "dsp_utilization": 0.60,
            "bram_utilization": 0.80,
            "power_consumption": 10.5  # W
        }
        logger.info(f"Extracted metrics: throughput={metrics['throughput']} FPS")
        return metrics


# =============================================================================
# Legacy FINN Adapter
# =============================================================================

class LegacyFINNAdapter:
    """Adapts new API to legacy FINN DataflowBuildConfig."""
    
    def __init__(self, kernel_registry: KernelRegistry, transform_registry: TransformRegistry):
        self.kernel_registry = kernel_registry
        self.transform_registry = transform_registry
    
    def convert_to_dataflow_config(self, context: CompilationContext):
        """Convert compilation context to FINN DataflowBuildConfig."""
        # Build step functions from applied transforms
        steps = self._build_step_functions(context)
        
        # Extract FINN parameters
        params = self._extract_finn_params(context)
        
        # In real implementation, would create actual DataflowBuildConfig
        config = {
            "steps": steps,
            "output_dir": context.output_dir,
            **params
        }
        
        logger.info(f"Created DataflowBuildConfig with {len(steps)} steps")
        return config
    
    def _build_step_functions(self, context: CompilationContext) -> List[Callable]:
        """Build FINN step functions from context."""
        steps = []
        
        # Convert applied transforms to steps
        for stage_name, transforms in context.applied_transforms.items():
            for transform in transforms:
                # Wrap transform as FINN step
                step = self._wrap_transform_as_step(transform)
                steps.append(step)
        
        # Add standard FINN steps
        steps.extend([
            "step_create_dataflow_partition",
            "step_hw_codegen",
            "step_hw_ipgen",
            "step_create_stitched_ip"
        ])
        
        return steps
    
    def _wrap_transform_as_step(self, transform: Transform) -> Callable:
        """Wrap transform as FINN step function."""
        def step_function(model, cfg):
            return transform.function(model, {})
        step_function.__name__ = f"step_{transform.name.lower()}"
        return step_function
    
    def _extract_finn_params(self, context: CompilationContext) -> Dict[str, Any]:
        """Extract FINN parameters from context."""
        strategy_params = context.strategy.parameters
        
        params = {
            "synth_clk_period_ns": 1000.0 / strategy_params.get("target_frequency_mhz", 200),
            "target_fps": strategy_params.get("target_throughput_fps"),
            "folding_config_file": strategy_params.get("folding_config"),
            "auto_fifo_depths": True,
            "save_intermediate_models": True
        }
        
        return params


# =============================================================================
# Example Usage
# =============================================================================

def create_bert_strategy(kernel_registry: KernelRegistry, 
                        transform_registry: TransformRegistry) -> CompilationStrategy:
    """Create BERT-optimized compilation strategy."""
    return CompilationStrategy(
        name="bert_optimized",
        description="Optimized compilation strategy for BERT models",
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
                preferred_backend="brainsmith_hls",
                backend_config={}
            )
        ],
        transforms=[
            TransformSpec(
                transform=transform_registry.get_transform("FoldConstants"),
                enabled=True
            ),
            TransformSpec(
                transform=transform_registry.get_transform("ExpandNorms"),
                enabled=True,
                config={"mode": "aggressive"}
            ),
            TransformSpec(
                transform=transform_registry.get_transform("Streamline"),
                enabled=True,
                config={"level": 2}
            ),
            TransformSpec(
                transform=transform_registry.get_transform("InferHardware"),
                enabled=True
            ),
            TransformSpec(
                transform=transform_registry.get_transform("SetFolding"),
                enabled=True,
                config={"target_fps": 3000}
            ),
            TransformSpec(
                transform=transform_registry.get_transform("SetFIFODepths"),
                enabled=True,
                config={"mode": "auto"}
            )
        ],
        parameters={
            "target_frequency_mhz": 200,
            "target_throughput_fps": 3000,
            "folding_config": "configs/bert_folding.json"
        }
    )


def main():
    """Demonstrate the new API."""
    # Initialize compiler
    compiler = FINNCompiler()
    
    # Create BERT strategy
    bert_strategy = create_bert_strategy(
        compiler.kernel_registry,
        compiler.transform_registry
    )
    
    # Compile model
    result = compiler.compile(
        model_path="bert_model.onnx",
        strategy=bert_strategy,
        output_dir="./bert_build"
    )
    
    # Print results
    print(f"\nCompilation {'succeeded' if result.success else 'failed'}")
    if result.success:
        print(f"Output directory: {result.output_dir}")
        print("\nMetrics:")
        for metric, value in result.metrics.items():
            print(f"  {metric}: {value}")
        print("\nArtifacts:")
        for name, path in result.artifacts.items():
            print(f"  {name}: {path}")
    else:
        print(f"Errors: {result.errors}")
    
    # Demonstrate legacy adapter
    print("\n--- Legacy FINN Adapter Demo ---")
    adapter = LegacyFINNAdapter(
        compiler.kernel_registry,
        compiler.transform_registry
    )
    
    # Create mock context
    context = CompilationContext(
        model_path="bert_model.onnx",
        strategy=bert_strategy,
        output_dir="./bert_build",
        applied_transforms={
            "graph_cleanup": [
                compiler.transform_registry.get_transform("FoldConstants")
            ],
            "topology_optimization": [
                compiler.transform_registry.get_transform("Streamline")
            ]
        }
    )
    
    # Convert to legacy format
    legacy_config = adapter.convert_to_dataflow_config(context)
    print(f"Legacy config has {len(legacy_config['steps'])} steps")
    print(f"Output dir: {legacy_config['output_dir']}")
    print(f"Clock period: {legacy_config['synth_clk_period_ns']} ns")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()