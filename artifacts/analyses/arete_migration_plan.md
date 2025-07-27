# Arete Migration Plan: @brainsmith/core

## Phase 1: Critical Fixes (Week 1)

### Day 1: Fix validation.py Import Error

**Problem**: `validation.py` imports from non-existent `design_space_v2`

**Steps**:
1. **Identify actual imports needed**
   ```bash
   grep -n "design_space_v2" brainsmith/core/validation.py
   # Line 10: from .design_space_v2 import BuildConfig, OutputStage
   ```

2. **Update import to use existing classes**
   ```python
   # Replace line 10:
   from .design_space import ForgeConfig, OutputStage
   ```

3. **Update BuildConfig references to ForgeConfig**
   ```python
   # Line 40: build_config: BuildConfig → forge_config: ForgeConfig
   # Line 51: build_config.output_stage → forge_config.output_stage
   ```

4. **Run tests to verify fix**
   ```bash
   ./smithy exec python -m pytest tests/test_validation.py
   ```

### Day 2-3: Resolve Circular Imports

**Problem**: forge.py imports explorer inside function to avoid circular dependency

**Steps**:

1. **Analyze the circular dependency**
   ```
   forge.py → explorer/__init__.py → explorer.py → executor.py → plugins → steps → core → forge.py
   ```

2. **Break the cycle - Option A: Move explore_execution_tree to forge.py**
   ```python
   # In forge.py, add:
   def _explore_tree(tree, model_path, output_dir, forge_config, design_space):
       """Internal tree exploration logic."""
       from .explorer.finn_adapter import FINNAdapter
       from .explorer.executor import Executor
       from .explorer.utils import serialize_tree, serialize_results
       
       # ... implementation from explorer.py
   ```

3. **Option B (Preferred): Create interface module**
   ```python
   # Create brainsmith/core/interfaces.py
   """Clean interfaces to avoid circular imports."""
   
   def run_exploration(tree, model_path, output_dir, forge_config, design_space):
       """Deferred import wrapper."""
       from .explorer import explore_execution_tree
       return explore_execution_tree(tree, model_path, output_dir, forge_config, design_space)
   ```

4. **Update forge.py**
   ```python
   from .interfaces import run_exploration
   # Remove the import inside function
   ```

5. **Test thoroughly**
   ```bash
   ./smithy exec python artifacts/tests/test_forge_e2e.py
   ```

### Day 4: Standardize Logging

**Problem**: Mix of logger and print() statements

**Steps**:

1. **Find all print() statements in core**
   ```bash
   grep -n "print(" brainsmith/core/*.py
   ```

2. **Replace print_tree_summary in forge.py**
   ```python
   # Current (lines 116-133):
   print("\nExecution Tree Structure:")
   print("=" * 60)
   
   # Replace with:
   logger.info("\nExecution Tree Structure:")
   logger.info("=" * 60)
   ```

3. **Create proper logging formatter for tree display**
   ```python
   # In forge.py
   def log_tree_summary(tree: ExecutionNode, max_depth: int = 3) -> None:
       """Log tree summary using logger."""
       import io
       from contextlib import redirect_stdout
       
       buffer = io.StringIO()
       with redirect_stdout(buffer):
           _print_tree_limited(tree, max_depth=max_depth)
       
       for line in buffer.getvalue().splitlines():
           logger.info(line)
   ```

4. **Configure logging in tests**
   ```python
   import logging
   logging.basicConfig(level=logging.INFO, format='%(message)s')
   ```

## Phase 2: Simplification (Week 2)

### Day 5-6: Simplify Config Extraction

**Current complexity**: 23 lines of manual parameter checking

**Steps**:

1. **Create clean config extractor**
   ```python
   # In blueprint_parser.py, replace _extract_config_and_mappings:
   def _extract_config_and_mappings(self, data: Dict[str, Any]) -> ForgeConfig:
       """Extract ForgeConfig from blueprint data."""
       # Merge all config sources
       all_config = {
           **data.get('global_config', {}),
           **{k: v for k, v in data.items() if k not in ['design_space', 'extends']}
       }
       
       # Extract ForgeConfig fields
       forge_fields = {}
       for field in ForgeConfig.__dataclass_fields__:
           if field in all_config:
               value = all_config.pop(field)
               if field == 'output_stage' and isinstance(value, str):
                   value = OutputStage(value)
               forge_fields[field] = value
       
       # Handle legacy mappings
       finn_params = data.get('finn_config', {})
       if 'platform' in all_config:
           finn_params['board'] = all_config.pop('platform')
       if 'target_clk' in all_config:
           finn_params['synth_clk_period_ns'] = self._parse_time_with_units(all_config.pop('target_clk'))
       
       return ForgeConfig(**forge_fields, finn_params=finn_params)
   ```

2. **Remove _parse_forge_config method** (lines 375-390)
   - No longer needed with simplified extraction

3. **Update tests to verify all config paths**
   ```python
   # Test nested config, top-level config, and legacy params
   ```

### Day 7: Delete Tree Printing Code

**Target**: Remove 65 lines of custom tree printing

**Steps**:

1. **Install lightweight tree library**
   ```bash
   # Add to requirements.txt
   anytree>=2.8.0
   ```

2. **Create simple tree renderer**
   ```python
   # In forge.py, replace entire print_tree_summary and _print_tree_limited:
   def log_tree_stats(tree: ExecutionNode) -> None:
       """Log tree statistics."""
       stats = get_tree_stats(tree)
       logger.info(f"Tree Statistics:")
       logger.info(f"  Total paths: {stats['total_paths']:,}")
       logger.info(f"  Total segments: {stats['total_segments']:,}")
       logger.info(f"  Efficiency: {stats['segment_efficiency']}%")
   ```

3. **If tree visualization needed, use anytree**
   ```python
   # In separate visualization module if needed
   from anytree import Node, RenderTree
   
   def visualize_tree(root: ExecutionNode):
       # Convert to anytree and render
       pass
   ```

4. **Delete lines 108-172 from forge.py**

### Day 8: Standardize Path Handling

**Target**: Use pathlib everywhere

**Steps**:

1. **Find all os.path usage**
   ```bash
   grep -n "os\.path" brainsmith/core/*.py
   ```

2. **Update forge.py**
   ```python
   # Line 43-46:
   if not Path(model_path).exists():
       raise FileNotFoundError(f"Model file not found: {model_path}")
   if not Path(blueprint_path).exists():
       raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
   
   # Line 50-52:
   build_dir = Path(os.environ.get("BSMITH_BUILD_DIR", "./build"))
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   output_dir = build_dir / f"forge_{timestamp}"
   
   # Line 64:
   design_space, tree = parser.parse(blueprint_path, str(Path(model_path).absolute()))
   ```

3. **Update blueprint_parser.py**
   ```python
   # Line 133-136:
   parent_path = Path(blueprint_path).parent / data['extends']
   ```

4. **Update validation and utils modules similarly**

5. **Run full test suite**
   ```bash
   ./smithy exec python -m pytest tests/
   ```

## Verification Checklist

### Phase 1 Complete When:
- [ ] validation.py imports successfully
- [ ] No circular import warnings
- [ ] No print() statements in core modules
- [ ] All tests pass

### Phase 2 Complete When:
- [ ] Config extraction ≤ 15 lines
- [ ] No custom tree printing code
- [ ] pathlib used consistently
- [ ] All tests pass

## Expected Outcomes

### Code Reduction
- Phase 1: -10 lines (mostly comments about circular imports)
- Phase 2: -150 lines (tree printing, config complexity)
- **Total**: 160 lines deleted (8% of codebase)

### Quality Improvements
- No runtime import errors
- No circular dependencies
- Consistent modern Python patterns
- Cleaner, more maintainable code

### Time Investment
- Phase 1: 4 days
- Phase 2: 4 days
- **Total**: 8 days

Arete!