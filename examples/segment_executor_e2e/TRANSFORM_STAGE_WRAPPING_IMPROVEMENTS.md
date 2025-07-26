# Transform Stage Wrapping Improvements

## Understanding Transform Stages

Transform stages are **semantic units** that represent a logical group of transforms:
- `cleanup`: Identity removal, tensor cleanup, naming strategies
- `optional_step`: Constant folding, shape inference, parameter handling

These stages:
1. Can contain multiple transforms executed sequentially
2. May have branching options (mutually exclusive choices)
3. Get wrapped as single FINN build steps
4. Preserve semantic meaning for debugging/analysis

## Current Flow Analysis

```yaml
# Blueprint defines stages with choices
cleanup:
  - RemoveIdentityOps              # Always run
  - RemoveUnusedTensors           # Always run  
  - [GiveUniqueNodeNames, GiveRandomTensorNames]  # Choice point

# Tree builder creates branches
cleanup_opt0: [RemoveIdentityOps, RemoveUnusedTensors, GiveUniqueNodeNames]
cleanup_opt1: [RemoveIdentityOps, RemoveUnusedTensors, GiveRandomTensorNames]

# Each branch becomes a FINN step
"cleanup_RemoveIdentityOps_RemoveUnusedTensors"  # Wrapped function name
```

## Problems with Current Approach

1. **Late Wrapping**: Functions created during execution, not planning
2. **Name Length**: "cleanup_RemoveIdentityOps_RemoveUnusedTensors" can exceed limits
3. **Global Mutation**: Each execution mutates FINN's global registry
4. **No Stage Awareness**: Wrapped names lose semantic stage information

## Improved Design

### 1. Stage-Aware Wrapper Factory

```python
class StageWrapperFactory:
    """Create wrapped functions that preserve stage semantics"""
    
    def __init__(self, registry: PluginRegistry):
        self.registry = registry
        self._wrapper_cache = {}
        self._stage_registry = {}  # Track stage → wrapper mapping
    
    def create_stage_wrapper(self, stage_name: str, 
                           transforms: List[str],
                           branch_index: int) -> Tuple[str, callable]:
        """Create wrapper for a transform stage branch"""
        
        # Generate semantic name that preserves stage identity
        wrapper_name = self._generate_stage_name(stage_name, branch_index, transforms)
        
        # Check cache
        cache_key = (stage_name, tuple(transforms))
        if cache_key in self._wrapper_cache:
            return wrapper_name, self._wrapper_cache[cache_key]
        
        # Create wrapper that knows it's a stage
        wrapper = self._create_stage_wrapper(stage_name, transforms)
        wrapper.__name__ = wrapper_name
        
        self._wrapper_cache[cache_key] = wrapper
        self._stage_registry[wrapper_name] = {
            "stage": stage_name,
            "transforms": transforms,
            "branch_index": branch_index
        }
        
        return wrapper_name, wrapper
    
    def _generate_stage_name(self, stage: str, branch_index: int, transforms: List[str]) -> str:
        """Generate concise but informative stage step name"""
        # Keep stage identity clear with simple numeric index
        if not transforms:
            return f"{stage}_skip"
        
        # Use simple numeric index for branches
        return f"{stage}_{branch_index}"
    
    def _create_stage_wrapper(self, stage_name: str, transform_names: List[str]) -> callable:
        """Create wrapper that preserves stage context"""
        # Resolve transforms once
        transforms = [self.registry.get_plugin(name) for name in transform_names]
        
        def stage_wrapper(model, cfg):
            # Log with stage context
            if transforms:
                print(f"[{stage_name}] Executing {len(transforms)} transforms")
                for i, transform_cls in enumerate(transforms, 1):
                    print(f"  [{i}/{len(transforms)}] {transform_cls.__name__}")
                    model = model.transform(transform_cls())
            else:
                print(f"[{stage_name}] Skipping (no transforms selected)")
            return model
        
        # Attach metadata for debugging
        stage_wrapper._stage_info = {
            "stage": stage_name,
            "transforms": transform_names
        }
        
        return stage_wrapper
```

### 2. Pre-compute During Tree Building

Integrate wrapper creation into the tree building phase:

```python
def build_execution_tree_with_wrappers(space: DesignSpace, 
                                      wrapper_factory: StageWrapperFactory) -> ExecutionNode:
    """Build tree and pre-compute all stage wrappers"""
    
    # Track all wrapped functions
    wrapped_functions = {}
    
    # Original tree building logic with wrapper integration
    root = ExecutionNode(segment_steps=[], finn_config=_extract_finn_config(space.global_config))
    
    # When creating branches for transform stages
    def _create_branches_with_wrappers(segments, stage, step_name, combinations):
        new_segments = []
        
        for segment in segments:
            for i, transforms in enumerate(combinations):
                # Create wrapper during tree building with simple index
                wrapper_name, wrapper_fn = wrapper_factory.create_stage_wrapper(
                    step_name,
                    [t.__name__ for t in transforms],
                    i  # Simple numeric index
                )
                
                # Store for later FINN registration
                wrapped_functions[wrapper_name] = wrapper_fn
                
                # Create child with wrapper name
                child = segment.add_child(f"{step_name}_opt{i}", [{
                    "transforms": transforms,
                    "stage_name": step_name,
                    "finn_step_name": wrapper_name  # Store wrapper name
                }])
                new_segments.append(child)
        
        return new_segments
    
    # ... rest of tree building ...
    
    # Register all wrappers with FINN at once
    _register_all_wrappers(wrapped_functions)
    
    return root
```

### 3. Enhanced Segment Structure

Store both semantic and execution information:

```python
@dataclass
class StageStep:
    """Transform stage step with full metadata"""
    stage_name: str              # Semantic stage name
    transforms: List[str]        # Transform class names
    finn_step_name: str         # Wrapped function name for FINN
    branch_index: int           # Simple numeric branch index
```

### 4. Simplified Execution

During execution, use pre-computed wrapper names:

```python
def _make_finn_config(self, segment: ExecutionNode, output_dir: Path) -> Dict[str, Any]:
    """Create FINN configuration using pre-wrapped stages"""
    config = self.base_finn_config.copy()
    config["output_dir"] = str(output_dir)
    
    steps = []
    for step in segment.segment_steps:
        if "finn_step_name" in step:
            # Use pre-computed wrapper name
            steps.append(step["finn_step_name"])
        else:
            # Regular FINN step
            steps.append(step["name"])
    
    config["steps"] = steps
    return config
```

### 5. Stage-Aware Caching

Cache based on semantic stage, not just transforms:

```python
def _check_stage_cache(self, segment: ExecutionNode, output_dir: Path) -> Optional[Path]:
    """Check if this stage combination was already executed"""
    # Use stage-aware cache key
    stage_key = self._get_stage_cache_key(segment)
    cache_file = output_dir / f".stage_cache/{stage_key}.onnx"
    
    if cache_file.exists():
        print(f"✓ Cached: {segment.segment_id} (stage: {stage_key})")
        return cache_file
    
    return None
```

## Benefits

1. **Semantic Preservation**: Stage names and purposes remain clear
2. **Early Validation**: Wrapper creation failures caught during planning
3. **Better Names**: `cleanup_opt0` instead of `cleanup_RemoveIdentityOps_RemoveUnusedTensors`
4. **Stage-Aware Caching**: Can cache based on semantic stages
5. **Debugging**: Clear mapping from stages to execution steps
6. **Performance**: Wrappers created once, reused across runs

## Example Execution

```python
# During tree building
factory = StageWrapperFactory(registry)
tree = build_execution_tree_with_wrappers(design_space, factory)

# Tree node contains:
{
    "transforms": ["RemoveIdentityOps", "RemoveUnusedTensors", "GiveUniqueNodeNames"],
    "stage_name": "cleanup",
    "finn_step_name": "cleanup_0",  # Pre-computed wrapper name with simple index
    "branch_index": 0
}

# During execution
config["steps"] = [
    "step_qonnx_to_finn",
    "cleanup_0",              # Stage with branch 0
    "optional_step_1",        # Stage with branch 1 (skip FoldConstants)
    "step_create_dataflow_partition"
]

# Debugging shows stage context
[cleanup] Executing 3 transforms
  [1/3] RemoveIdentityOps
  [2/3] RemoveUnusedTensors
  [3/3] GiveUniqueNodeNames
```

## Migration Path

1. Add `StageWrapperFactory` to explorer/utils.py
2. Update tree builder to use factory during branch creation
3. Store `finn_step_name` in segment steps
4. Update executor to use pre-computed names
5. Add stage-aware caching logic

This design respects the semantic importance of transform stages while improving the technical implementation.

Arete!