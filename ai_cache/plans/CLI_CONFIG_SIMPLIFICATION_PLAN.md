# CLI & Config Simplification Plan

## Current State: Wildly Over-Engineered

### CLI Problems:
- **453 lines** with dual generation paths, multi-phase execution, complex fallbacks
- Trying to be "simple by default, powerful when needed" but failing at both
- Complex abstractions (GenerationResult, complexity levels) that add no value
- Duplicate code and unnecessary sophistication

### Config Problems:
- **461 lines** for what should be a 20-line dataclass
- 7 different "use case" configurations solving imaginary problems
- Performance profiling, resource estimation, usage analytics in a config class
- Export/import functionality that nobody asked for

## Target State: Radically Simplified

### New CLI Goals:
1. **Single generation path** - no "enhanced" vs "legacy"
2. **Simple argument parsing** - basic argparse with 4-5 arguments max
3. **Direct execution** - parse RTL → generate files → done
4. **~100 lines total** including error handling

### New Config Goals:
1. **Pure dataclass** with just the essential fields
2. **Basic validation** - file exists, create output dir
3. **~30 lines total** including validation
4. **No performance profiling, analytics, or optimization advice**

## Implementation Plan

### Phase 1: Create Simplified Config (15 minutes)
```python
@dataclass
class Config:
    """Simple configuration for RTL-to-template generation."""
    rtl_file: Path
    compiler_data_file: Path  
    output_dir: Path
    debug: bool = False
    
    def __post_init__(self):
        # Basic validation only
        if not self.rtl_file.exists():
            raise ValueError(f"RTL file not found: {self.rtl_file}")
        if not self.compiler_data_file.exists():
            raise ValueError(f"Compiler data not found: {self.compiler_data_file}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_args(cls, args):
        return cls(
            rtl_file=Path(args.rtl_file),
            compiler_data_file=Path(args.compiler_data),
            output_dir=Path(args.output),
            debug=args.debug
        )
```

### Phase 2: Create Simplified CLI (30 minutes)
```python
def main():
    parser = argparse.ArgumentParser(description="Generate FINN components from RTL")
    parser.add_argument('rtl_file', help='SystemVerilog RTL file')
    parser.add_argument('compiler_data', help='Python compiler data file')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Debug output')
    
    args = parser.parse_args()
    config = Config.from_args(args)
    
    # Single, simple generation path
    try:
        parsed_data = parse_rtl_file(config.rtl_file)
        
        generators = [
            HWCustomOpGenerator(),
            RTLBackendGenerator(),
            TestSuiteGenerator()
        ]
        
        generated_files = []
        for generator in generators:
            output_file = generator.generate(parsed_data, config.output_dir)
            generated_files.append(output_file)
            if config.debug:
                print(f"Generated: {output_file}")
        
        print(f"✅ Generated {len(generated_files)} files in {config.output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
```

## What We're Removing

### From CLI (saving ~350 lines):
- ❌ `GenerationResult` class with success tracking
- ❌ "Enhanced" vs "legacy" generation paths  
- ❌ Multi-phase execution with 6 phases
- ❌ Complexity level detection
- ❌ Advanced pragma flags
- ❌ Stop-after functionality
- ❌ Fallback logic and duplicate code paths
- ❌ Complex error reporting with warnings/errors lists
- ❌ Debug output scattered everywhere

### From Config (saving ~430 lines):
- ❌ `complexity_level` property and logic
- ❌ `performance_profile` and resource estimation
- ❌ Use case configurations (production, development, research, etc.)
- ❌ Optimization recommendations
- ❌ Configuration export/import
- ❌ Usage analytics
- ❌ Resource efficiency calculations
- ❌ Complex validation for edge cases
- ❌ Template directory validation (generators handle this)

## Benefits of Simplification

### 1. **Maintainability**
- **90% less code** to understand and maintain
- Single code path eliminates branching complexity
- Clear, linear execution flow

### 2. **Reliability** 
- Fewer code paths = fewer bugs
- Simple validation reduces configuration errors
- No complex fallback logic to fail

### 3. **Performance**
- Single generation path eliminates overhead
- No resource estimation or analytics computation
- Direct execution without abstraction layers

### 4. **Usability**
- 4 simple arguments instead of 10+ flags
- Predictable behavior without "sophistication levels"
- Clear error messages without categorization

### 5. **Testing**
- Simple functions easy to unit test
- No complex state management
- Reduced test matrix (no multiple execution paths)

## File Structure After Simplification

```
brainsmith/tools/hw_kernel_gen/
├── cli.py                 # ~100 lines (down from 453)
├── config.py              # ~30 lines (down from 461)  
├── rtl_parser/            # Unchanged
├── generators/            # Unchanged
└── templates/             # Unchanged
```

## Migration Strategy

### 1. **Backup Originals**
```bash
mv cli.py cli_complex.py
mv config.py config_complex.py
```

### 2. **Create Simplified Versions**
- New `cli.py` with single generation path
- New `config.py` as simple dataclass

### 3. **Update Imports**
- Remove references to removed classes/functions
- Update tests to use simplified interfaces

### 4. **Validate**
- Run existing tests with simplified CLI
- Ensure generated output quality unchanged

## Validation Plan

### Before/After Comparison:
1. **Generate same test RTL** with old and new CLI
2. **Compare output files** - should be identical
3. **Measure execution time** - should be faster
4. **Count lines of code** - should be 85% reduction

### Success Criteria:
- ✅ Same quality output files generated
- ✅ Faster execution (no dual paths/complexity)
- ✅ CLI usage is simpler (4 args instead of 10+)
- ✅ Code is 85% smaller and easier to understand
- ✅ All existing functionality preserved

## Timeline

- **Phase 1** (15 min): Create simplified config.py
- **Phase 2** (30 min): Create simplified cli.py  
- **Phase 3** (15 min): Update imports and test
- **Total**: 1 hour to eliminate 90% of unnecessary complexity

## The Core Problem

Both files suffer from **premature optimization** and **feature creep**:
- Solving problems that don't exist (configuration export, usage analytics)
- Complex abstractions for simple tasks (GenerationResult for file list)
- Multiple execution paths when one would suffice
- "Enterprise" features (use cases, performance profiling) in a simple tool

The simplified version will do exactly the same job with 90% less code and complexity.