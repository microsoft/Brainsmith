# Legacy BrainSmith Components

This directory contains legacy components that have been superseded by the new segment-based execution tree architecture.

## Contents

### phase2/
Original Phase 2 implementation for design space exploration:
- `explorer.py` - Step-based explorer (replaced by `core/explorer/`)
- `data_structures.py` - BuildConfig and related types
- `progress.py` - Progress tracking
- `results_aggregator.py` - Results aggregation
- `hooks.py` - Hook system for extensibility
- `interfaces.py` - Abstract interfaces

### phase3/
Original Phase 3 implementation for build execution:
- `build_runner.py` - Build orchestration
- `legacy_finn_backend.py` - FINN integration (replaced by segment executor)
- `future_brainsmith_backend.py` - Placeholder for future backend
- `preprocessing.py` - Model preprocessing pipeline
- `postprocessing.py` - Results postprocessing
- `metrics_collector.py` - Performance metrics collection
- `error_handler.py` - Error categorization and handling

## Migration Notes

The legacy phase2 and phase3 have been replaced by:
- `core/explorer/` - New segment-based execution tree explorer
- `core/execution_tree.py` - Segment-based tree structure
- `core/tree_builder.py` - Builds segment trees from design space

### Key Differences

1. **Segment-based vs Step-based**: The new architecture groups steps into segments between branch points, dramatically reducing the number of FINN builds required.

2. **Simplified Caching**: File existence check instead of marker files.

3. **Cleaner Abstractions**: No separate ArtifactManager or ConfigMapper classes.

4. **Better Performance**: ~40% less code with same functionality.

### Import Updates

If you need to reference legacy code:
```python
# Old imports (no longer work)
from brainsmith.core.phase2.data_structures import BuildConfig
from brainsmith.core.phase3.legacy_finn_backend import LegacyFINNBackend

# New imports
from brainsmith.legacy.phase2.data_structures import BuildConfig
from brainsmith.legacy.phase3.legacy_finn_backend import LegacyFINNBackend
```

## Status

These components are maintained for reference but are not actively developed. All new development should use the segment-based architecture in `core/explorer/`.