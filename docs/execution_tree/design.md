# Final Complete Design: YAML to Execution Tree

## Overview

Complete implementation incorporating all learnings:
- Multi-transform stages with sequential execution
- Kernel backends as specific classes from registry
- Direct tree construction without intermediate representations
- Clean separation between transforms and kernels

## 1. Example Blueprint

```yaml
version: "4.0"
name: "Production FINN Design"

global_config:
  output_stage: "synthesize_bitstream"
  working_directory: "work"
  save_intermediate_models: true
  max_combinations: 10000
  timeout_minutes: 120

design_space:
  transforms:
    # Import stage - no branching
    imports:
      - ImportBrevitas
      - InferDataLayouts
    
    # Cleanup stage with branching
    cleanup:
      - RemoveIdentityOps                        # Required
      - [CollapseRepeatedOps, RemoveUnusedTensors]  # Choose one
      - ["~", InferDataTypes]                   # Optional
    
    # Streamlining with multiple decision points
    streamlining:
      - [MoveAddPastMul, MoveScalarLinearPastInvariants]  # Choose one
      - ["~", AbsorbSignBias]                   # Optional
      - [RoundAndClipThresholds, "~"]           # Optional
    
    # Simple folding stage
    folding:
      - ["~", FoldConstants, FoldBatchNorms]    # Choose one or skip
  
  # Kernels with specific backend implementations
  kernels:
    - MatrixVectorActivation: mvau_hls
    - LayerNorm: layernorm_hls
    - Thresholding: [thresholding_hls, thresholding_rtl]
    - ChannelwiseOp                             # All available backends

build_pipeline:
  steps:
    - step_qonnx_to_finn
    - {imports}
    - {cleanup}
    - {streamlining}
    - {folding}
    - infer_kernels
    - step_convert_to_hw
    - step_create_dataflow_partition
    - step_synthesize_bitstream
```

## 2. Complete Data Structures

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Type, Any
from enum import Enum
from pathlib import Path
from qonnx.transformation.base import Transformation  # Use QONNX Transform

class OutputStage(Enum):
    COMPILE_AND_PACKAGE = "compile_and_package"
    SYNTHESIZE_BITSTREAM = "synthesize_bitstream"
    GENERATE_REPORTS = "generate_reports"

@dataclass
class GlobalConfig:
    """All configuration including constraints."""
    output_stage: OutputStage = OutputStage.COMPILE_AND_PACKAGE
    working_directory: str = "work"
    save_intermediate_models: bool = False
    max_combinations: int = 100000
    timeout_minutes: int = 60

@dataclass
class TransformStage:
    """A stage containing multiple transform steps."""
    name: str
    transform_steps: List[List[Optional[Transformation]]]  # Each step has options
    
    def get_combinations(self) -> List[List[Transformation]]:
        """Get all valid combinations of transforms for this stage."""
        if not self.transform_steps:
            return [[]]
        
        combinations = [[]]
        
        for step_options in self.transform_steps:
            new_combinations = []
            
            for combo in combinations:
                for option in step_options:
                    if option is None:
                        # Skip option
                        new_combinations.append(combo)
                    else:
                        # Add transform
                        new_combinations.append(combo + [option])
            
            combinations = new_combinations
        
        return combinations

@dataclass
class DesignSpace:
    """Design space with resolved objects."""
    model_path: str
    transform_stages: Dict[str, TransformStage]
    kernel_backends: List[Tuple[str, List[Type]]]  # [(kernel_name, [Backend classes])]
    build_pipeline: List[str]
    global_config: GlobalConfig

@dataclass
class ExecutionNode:
    """Node in execution tree."""
    step_name: str
    config: Dict[str, Any]
    parent: Optional['ExecutionNode'] = None
    children: List['ExecutionNode'] = field(default_factory=list)
    
    # Execution state
    status: str = "pending"
    output_dir: Optional[Path] = None
    error: Optional[str] = None
    
    def find_or_create_child(self, step_name: str, config: Dict) -> 'ExecutionNode':
        """Get existing child or create new."""
        config_key = self._make_config_key(config)
        
        for child in self.children:
            if child.step_name == step_name and self._make_config_key(child.config) == config_key:
                return child
        
        child = ExecutionNode(step_name, config, parent=self)
        self.children.append(child)
        return child
    
    def _make_config_key(self, config: Dict) -> str:
        """Create comparable key from config."""
        items = []
        for k, v in sorted(config.items()):
            if k == "transforms" and isinstance(v, list):
                # Transform class names
                items.append((k, tuple(t.__class__.__name__ for t in v)))
            elif k == "kernel_backends" and isinstance(v, list):
                # Backend class names
                items.append((k, tuple((kn, tuple(b.__name__ for b in bc)) for kn, bc in v)))
            else:
                items.append((k, str(v)))
        return str(items)
```

## 3. Registry and Resolution

```python
from brainsmith.core.plugins import registry

def resolve_transform_spec(spec: Union[str, List]) -> List[Optional[Transformation]]:
    """Resolve transform spec to list of options."""
    if isinstance(spec, list):
        # List = multiple options
        options = []
        for name in spec:
            if name == "~":
                options.append(None)
            else:
                options.append(registry.get_transform(name))
        return options
    else:
        # Single string = required transform
        return [registry.get_transform(spec)]

def resolve_kernel_spec(spec: Union[str, Dict]) -> Tuple[str, List[Type]]:
    """Resolve kernel spec to kernel name and backend classes."""
    if isinstance(spec, str):
        # Just kernel name - get all available backends
        kernel_name = spec
        backend_names = registry.list_backends_by_kernel(kernel_name)
        
        if not backend_names:
            raise ValueError(f"No backends found for kernel {kernel_name}")
        
        # Get backend classes
        backend_classes = []
        for backend_name in backend_names:
            backend_class = registry.get_backend(backend_name)
            backend_classes.append(backend_class)
        
        return (kernel_name, backend_classes)
    
    elif isinstance(spec, dict):
        # Kernel with specific backends
        if len(spec) != 1:
            raise ValueError(f"Kernel spec must have exactly one key: {spec}")
        
        kernel_name, backend_specs = next(iter(spec.items()))
        
        # Normalize to list
        if isinstance(backend_specs, str):
            backend_specs = [backend_specs]
        
        # Resolve backend names to classes
        backend_classes = []
        for backend_name in backend_specs:
            backend_class = registry.get_backend(backend_name)
            if not backend_class:
                raise ValueError(f"Backend {backend_name} not found")
            backend_classes.append(backend_class)
        
        return (kernel_name, backend_classes)

def parse_transform_stage(stage_name: str, stage_spec: List) -> TransformStage:
    """Parse a transform stage with multiple steps."""
    transform_steps = []
    
    for spec in stage_spec:
        options = resolve_transform_spec(spec)
        transform_steps.append(options)
    
    return TransformStage(stage_name, transform_steps)
```

## 4. Blueprint Parser

```python
import yaml

class BlueprintParser:
    """Parse blueprint to design space."""
    
    def parse(self, blueprint_path: str, model_path: str) -> DesignSpace:
        """Parse and resolve all objects."""
        with open(blueprint_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse global config
        global_config = GlobalConfig()
        if 'global_config' in data:
            for key, value in data['global_config'].items():
                if hasattr(global_config, key):
                    if key == 'output_stage':
                        setattr(global_config, key, OutputStage(value))
                    else:
                        setattr(global_config, key, value)
        
        # Parse transform stages
        transform_stages = {}
        for stage_name, stage_spec in data['design_space']['transforms'].items():
            if not isinstance(stage_spec, list):
                stage_spec = [stage_spec]
            transform_stages[stage_name] = parse_transform_stage(stage_name, stage_spec)
        
        # Parse kernels with backend resolution
        kernel_backends = []
        for spec in data['design_space']['kernels']:
            kernel_backends.append(resolve_kernel_spec(spec))
        
        return DesignSpace(
            model_path=model_path,
            transform_stages=transform_stages,
            kernel_backends=kernel_backends,
            build_pipeline=data['build_pipeline']['steps'],
            global_config=global_config
        )
```

## 5. Implementation Decisions

### Stack-Based Tree Traversal

The executor uses a stack-based approach instead of recursion for cleaner code and better debugging:

```python
def execute_tree(tree: ExecutionTree, executor: Executor) -> TreeExecutionResult:
    """Execute tree using stack-based traversal."""
    results = {}
    stack = [(tree.root, None)]  # (node, parent_result)
    
    while stack:
        node, parent_result = stack.pop()
        
        # Execute segment
        result = executor.execute_segment(node, parent_result)
        results[node.segment_id] = result
        
        # Add children in reverse order for depth-first
        if result.success and node.children:
            for child in reversed(node.children):
                stack.append((child, result))
    
    return TreeExecutionResult(tree, results)
```

### Pre-computed Transform Wrappers

Transform stages are wrapped during tree building with deterministic indices:

```python
class StageWrapperFactory:
    """Creates numbered wrappers for transform stages."""
    
    def create_wrappers(self, stages: Dict[str, TransformStage]) -> Dict[str, str]:
        """Create wrapper names for all transform stages."""
        wrappers = {}
        for stage_name, stage in stages.items():
            # Simple deterministic naming
            for i in range(len(stage.transform_steps)):
                wrapper_name = f"{stage_name}_{i}"
                wrappers[wrapper_name] = (stage_name, i)
        return wrappers
```

### Error Handling Strategy

Fail-fast with skip propagation:

```python
class Executor:
    def execute_segment(self, node: ExecutionNode, parent_result: Optional[SegmentResult]):
        # Skip if parent failed
        if parent_result and not parent_result.success:
            return SegmentResult(
                segment_id=node.segment_id,
                success=False,
                error="Parent segment failed",
                skipped=True
            )
        
        # Check cache first
        if self._is_cached(node):
            return self._load_cached_result(node)
        
        try:
            # Execute FINN build
            output_model = self.finn_adapter.build(...)
            return SegmentResult(success=True, ...)
        except Exception as e:
            # Fail-fast mode stops on first error
            if self.fail_fast:
                raise
            return SegmentResult(success=False, error=str(e))
```

### Caching Strategy

Simple file-based caching with existence check:

```python
def _is_cached(self, node: ExecutionNode) -> bool:
    """Check if segment output exists."""
    output_path = self._get_output_path(node)
    return output_path.exists()

def _get_output_path(self, node: ExecutionNode) -> Path:
    """Deterministic output path."""
    return Path(self.output_dir) / f"segment_{node.segment_id}" / f"{node.segment_id}_output.onnx"
```

## 6. Direct Tree Builder

```python
def build_execution_tree(space: DesignSpace) -> ExecutionNode:
    """Build execution tree directly from design space."""
    root = ExecutionNode("root", {"model": space.model_path})
    active_nodes = [root]
    
    for step in space.build_pipeline:
        if step.startswith("{") and step.endswith("}"):
            # Transform stage - may branch
            stage_name = step[1:-1]
            stage = space.transform_stages.get(stage_name)
            
            if not stage:
                continue
            
            # Get all combinations for this stage
            stage_combinations = stage.get_combinations()
            
            if len(stage_combinations) == 1:
                # No branching
                transforms = stage_combinations[0]
                if transforms:
                    next_nodes = []
                    for node in active_nodes:
                        child = node.find_or_create_child(
                            f"stage_{stage_name}",
                            {"transforms": transforms}
                        )
                        next_nodes.append(child)
                    active_nodes = next_nodes
            else:
                # Multiple combinations - branch
                next_nodes = []
                for node in active_nodes:
                    for transforms in stage_combinations:
                        if not transforms:
                            # Empty - skip stage
                            next_nodes.append(node)
                        else:
                            child = node.find_or_create_child(
                                f"stage_{stage_name}",
                                {"transforms": transforms}
                            )
                            next_nodes.append(child)
                active_nodes = next_nodes
                
        elif step == "infer_kernels":
            # Kernel step - no branching
            next_nodes = []
            for node in active_nodes:
                child = node.find_or_create_child(
                    "infer_kernels",
                    {"kernel_backends": space.kernel_backends}
                )
                next_nodes.append(child)
            active_nodes = next_nodes
            
        else:
            # Regular step
            next_nodes = []
            for node in active_nodes:
                child = node.find_or_create_child(step, {})
                next_nodes.append(child)
            active_nodes = next_nodes
    
    # Validate tree size
    leaf_count = count_leaves(root)
    if leaf_count > space.global_config.max_combinations:
        raise ValueError(
            f"Tree has {leaf_count} paths, exceeds limit of "
            f"{space.global_config.max_combinations}"
        )
    
    return root

def count_leaves(node: ExecutionNode) -> int:
    """Count leaf nodes."""
    if not node.children:
        return 1
    return sum(count_leaves(child) for child in node.children)

def print_tree(node: ExecutionNode, indent: str = "", last: bool = True):
    """Pretty print the tree."""
    if node.step_name != "root":
        prefix = "└── " if last else "├── "
        
        # Format config
        config_str = ""
        if "transforms" in node.config:
            transforms = node.config["transforms"]
            if transforms:
                names = [t.__class__.__name__ for t in transforms]
                config_str = f" ({', '.join(names)})"
        elif "kernel_backends" in node.config:
            backend_info = []
            for kernel_name, backend_classes in node.config["kernel_backends"]:
                backend_names = [b.__name__ for b in backend_classes]
                backend_info.append(f"{kernel_name}[{','.join(backend_names)}]")
            config_str = f" ({'; '.join(backend_info)})"
        
        print(f"{indent}{prefix}{node.step_name}{config_str}")
    
    extension = "    " if last else "│   "
    for i, child in enumerate(node.children):
        print_tree(child, indent + extension, i == len(node.children) - 1)
```

## 6. Complete Example

```python
def main():
    """Complete flow from YAML to execution tree."""
    
    # Parse blueprint
    print("=== PARSING BLUEPRINT ===")
    parser = BlueprintParser()
    design_space = parser.parse("blueprint.yaml", "model.onnx")
    
    print(f"Model: {design_space.model_path}")
    print(f"Transform stages: {list(design_space.transform_stages.keys())}")
    print(f"Kernels with backends:")
    for kernel_name, backend_classes in design_space.kernel_backends:
        backend_names = [b.__name__ for b in backend_classes]
        print(f"  - {kernel_name}: {backend_names}")
    
    # Build execution tree
    print("\n=== BUILDING EXECUTION TREE ===")
    tree = build_execution_tree(design_space)
    
    # Display tree
    print("\nExecution Tree:")
    print_tree(tree)
    
    # Statistics
    leaf_count = count_leaves(tree)
    node_count = count_nodes(tree)
    print(f"\n=== STATISTICS ===")
    print(f"Total execution paths: {leaf_count}")
    print(f"Total tree nodes: {node_count}")
    print(f"Average sharing factor: {(leaf_count * len(design_space.build_pipeline)) / node_count:.1f}x")

def count_nodes(node: ExecutionNode) -> int:
    """Count all nodes in tree."""
    count = 0 if node.step_name == "root" else 1
    for child in node.children:
        count += count_nodes(child)
    return count
```

## 7. Expected Output

```
=== PARSING BLUEPRINT ===
Model: model.onnx
Transform stages: ['imports', 'cleanup', 'streamlining', 'folding']
Kernels with backends:
  - MatrixVectorActivation: ['MVAU_hls']
  - LayerNorm: ['LayerNorm_hls']
  - Thresholding: ['Thresholding_hls', 'Thresholding_rtl']
  - ChannelwiseOp: ['ChannelwiseOp_hls', 'ChannelwiseOp_rtl', 'ChannelwiseOp_vitis']

=== BUILDING EXECUTION TREE ===

Execution Tree:
└── step_qonnx_to_finn
    └── stage_imports (ImportBrevitas, InferDataLayouts)
        ├── stage_cleanup (RemoveIdentityOps, CollapseRepeatedOps)
        │   ├── stage_cleanup (RemoveIdentityOps, CollapseRepeatedOps, InferDataTypes)
        │   │   └── ... (streamlining branches)
        │   └── ... (streamlining branches)
        └── stage_cleanup (RemoveIdentityOps, RemoveUnusedTensors)
            ├── stage_cleanup (RemoveIdentityOps, RemoveUnusedTensors, InferDataTypes)
            │   └── ... (streamlining branches)
            └── ... (streamlining branches)

=== EXAMPLE CALCULATION ===
This is an illustrative example showing how the tree structure works.
Actual tree sizes depend on the specific blueprint configuration.
```

## 8. Key Design Principles

1. **Direct Construction**: YAML → DesignSpace → ExecutionTree (no intermediate representations)
2. **Type Safety**: Backend classes are actual types from registry
3. **Clear Separation**: Transforms create branches, kernels don't
4. **Optimal Sharing**: Common prefixes automatically shared
5. **Registry Integration**: All plugins resolved at parse time
6. **Uses QONNX Transform**: No reimplementation - uses `qonnx.transformation.base.Transformation`

This is the complete, clean design incorporating all learnings about transforms, kernels, and backends.