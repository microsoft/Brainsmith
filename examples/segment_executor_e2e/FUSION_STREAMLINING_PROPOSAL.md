# Fusion & Elimination Streamlining Proposal

## Principle: Fuse redundant layers while preserving extensibility

## Analysis of Current Architecture

The current architecture has good bones:
- Clear separation of concerns
- Extensible plugin system
- Robust tree-based execution model
- Clean segment abstraction

The issue is redundant transformations between layers that can be fused.

## Fusion Opportunities

### 1. Fuse DesignSpace + ExecutionNode Creation

**Current Flow**:
```
BlueprintParser.parse() → DesignSpace → build_execution_tree() → ExecutionNode
```

**Proposed Fusion**:
```python
# In blueprint_parser.py
def parse(self, blueprint_path: Path) -> Tuple[DesignSpace, ExecutionNode]:
    """Parse blueprint and build tree in one pass"""
    design_space = self._build_design_space(data)
    execution_tree = self._build_tree_during_parse(design_space)
    return design_space, execution_tree
```

**Benefit**: Eliminate separate tree building pass, reduce memory allocation

### 2. Fuse Transform Resolution + Tree Building

**Current**: Resolve transforms to classes, then build tree with classes
**Proposed**: Build tree with transform names, resolve on execution

```python
# Store names in tree, resolve during segment execution
segment.transform_names = ["RemoveIdentityOps", "RemoveUnusedTensors"]
# Resolve only when needed
transforms = [registry.get(name) for name in segment.transform_names]
```

**Benefit**: Lazy resolution, simpler serialization

### 3. Fuse TreeExecutor + SegmentExecutor

**Current**: TreeExecutor orchestrates, SegmentExecutor executes
**Proposed**: Single executor with both capabilities

```python
class UnifiedExecutor:
    def execute_tree(self, tree: ExecutionNode, output_dir: Path) -> TreeExecutionResult:
        """Handles both tree traversal and segment execution"""
        # Stack-based traversal
        for segment in self._traverse(tree):
            self._execute_segment(segment, output_dir)
```

**Benefit**: Eliminate abstraction layer, reduce call stack depth

### 4. Fuse Artifact Sharing into Execution

**Current**: Separate share_artifacts_at_branch() function
**Proposed**: Inline into execution flow

```python
# In executor
if segment.parent_id:
    parent_output = self._get_parent_output(segment.parent_id)
    shutil.copy2(parent_output, segment_dir / "input.onnx")
```

**Benefit**: Clear data flow, no separate artifact management

## Elimination Opportunities

### 1. Eliminate ExecutionNodeCompat

Just update the tests - maintaining compatibility adds complexity.

### 2. Eliminate Redundant Result Types

**Current**: SegmentResult, TreeExecutionResult with computed stats
**Proposed**: Single result dict with lazily computed stats

```python
@property
def stats(self) -> Dict[str, int]:
    """Compute on demand"""
    return calculate_stats(self.segment_results)
```

### 3. Eliminate Complex Serialization

Use dataclass.asdict() instead of custom serialize_tree().

### 4. Eliminate TransformStage from execution_tree.py

Already defined conceptually in design_space.py.

## Streamlined Workflow

```mermaid
graph LR
    A[Blueprint YAML] --> B[Parser + Tree Builder]
    B --> C[Unified Executor]
    C --> D[FINN Builds]
    D --> E[Results]
```

## Implementation Plan

### Phase 1: Fuse Parser + Tree Builder
```python
# blueprint_parser.py
class BlueprintParser:
    def parse(self, blueprint_path: Path) -> Tuple[DesignSpace, ExecutionNode]:
        """Parse and build tree in one pass"""
        with open(blueprint_path) as f:
            data = yaml.safe_load(f)
        
        design_space = DesignSpace(...)
        
        # Build tree while we have the data
        tree = self._build_tree(design_space)
        
        return design_space, tree
```

### Phase 2: Fuse Executors
```python
# explorer/executor.py (merge tree_executor.py + segment_executor.py)
class Executor:
    def __init__(self, registry: PluginRegistry):
        self.registry = registry
        self._finn_imported = False
    
    def execute(self, tree: ExecutionNode, output_dir: Path, 
                fail_fast: bool = True) -> Dict[str, Any]:
        """Unified execution handling both tree and segments"""
        results = {}
        stack = [tree]
        
        while stack:
            node = stack.pop()
            segment = node.segment
            
            # Check cache
            if self._is_cached(segment, output_dir):
                results[segment.segment_id] = self._cached_result(segment)
                continue
            
            # Execute
            result = self._execute_segment(segment, output_dir)
            results[segment.segment_id] = result
            
            # Add children
            if result.success or not fail_fast:
                stack.extend(reversed(node.children))
        
        return {
            "segment_results": results,
            "total_time": sum(r.get("time", 0) for r in results.values())
        }
```

### Phase 3: Simplify Registry Usage
```python
# Keep the registry but simplify usage
# Store transform names in segments, resolve on demand
class Executor:
    def _get_transforms(self, names: List[str]) -> List[Type]:
        """Lazy transform resolution"""
        return [self.registry.get_plugin(name) for name in names]
```

### Phase 4: Inline Artifact Management
```python
# Remove share_artifacts_at_branch(), inline the logic
def _prepare_segment_input(self, segment: Segment, output_dir: Path):
    segment_dir = output_dir / segment.segment_id
    
    if segment.parent_segment_id:
        parent_output = self._find_parent_output(segment.parent_segment_id)
        shutil.copy2(parent_output, segment_dir / "input.onnx")
    else:
        shutil.copy2(segment.initial_model, segment_dir / "input.onnx")
```

## Benefits of This Approach

1. **Preserves Extensibility**: Plugin system remains intact
2. **Reduces Layers**: 3 execution layers → 1
3. **Clearer Data Flow**: Direct path from blueprint to results
4. **Less Memory**: Fewer intermediate data structures
5. **Simpler Testing**: Fewer abstractions to mock
6. **Maintains Features**: Caching, fail-fast, progress tracking all preserved

## What We Keep

- Plugin registry (extensibility)
- Tree structure (parallel exploration)
- Segment abstraction (clean execution units)
- FINN workarounds (necessary evils)
- Blueprint format (user interface)

## What We Fuse/Eliminate

- Separate tree building phase
- TreeExecutor/SegmentExecutor split
- Complex serialization
- Redundant result types
- Compatibility layers

This achieves ~25% code reduction while maintaining all functionality and extensibility.

Arete!