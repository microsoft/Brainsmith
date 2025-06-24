# Design Space Exploration (DSE) Module Analysis

## Overview

The DSE module in `brainsmith/core/dse/` implements a comprehensive design space exploration framework for Blueprint V2. It provides automated exploration of hardware accelerator design spaces for FPGA AI implementations.

## Architecture

### Module Structure

```
brainsmith/core/dse/
├── __init__.py                # Module exports and documentation
├── combination_generator.py    # Component combination generation
├── space_explorer.py          # Main exploration orchestration
├── strategy_executor.py       # Exploration strategy implementations
└── results_analyzer.py        # Results analysis and Pareto frontier
```

### Core Components

#### 1. ComponentCombination (`combination_generator.py`)

**Purpose**: Represents a specific combination of components across all 6 FINN entrypoints.

**Key Attributes**:
- **Node Components** (Entrypoints 1, 3, 4):
  - `canonical_ops`: List of canonical operations
  - `hw_kernels`: Dict mapping components to chosen implementation options
  
- **Transform Components** (Entrypoints 2, 5, 6):
  - `model_topology`: Model-level transformations
  - `hw_kernel_transforms`: Hardware kernel transformations
  - `hw_graph_transforms`: Hardware graph transformations

**Features**:
- Automatic unique ID generation based on component composition
- Validation state tracking
- Hashable for deduplication
- Serialization support

#### 2. CombinationGenerator (`combination_generator.py`)

**Purpose**: Generates valid component combinations from design space definitions while respecting exploration rules.

**Key Methods**:
- `generate_all_combinations()`: Exhaustive generation with optional limit
- `generate_sample_combinations()`: Strategic sampling (random, diverse, balanced)
- Handles exploration rules:
  - Required components
  - Optional components
  - Mutually exclusive groups
  - Component dependencies

**Process Flow**:
1. Generate node combinations (canonical ops + hw kernels)
2. Generate transform combinations (all three transform types)
3. Cross-product to create complete combinations
4. Validate against rules
5. Deduplicate

#### 3. DesignSpaceExplorer (`space_explorer.py`)

**Purpose**: Main orchestration class that coordinates the entire exploration process.

**Key Features**:
- **Evaluation Management**:
  - Sequential or parallel evaluation
  - Caching with persistent storage
  - Error handling and recovery
  
- **Progress Tracking**:
  - Real-time progress monitoring
  - Time estimation
  - Checkpoint saving
  
- **Early Termination**:
  - Performance plateau detection
  - Configurable patience and thresholds

**Configuration Options** (`ExplorationConfig`):
```python
max_evaluations: int = 100
strategy_name: Optional[str] = None
enable_caching: bool = True
cache_directory: Optional[str] = None
parallel_evaluations: int = 1
early_termination_patience: int = 20
early_termination_threshold: float = 0.01
progress_callback: Optional[Callable] = None
checkpoint_frequency: int = 10
```

**Integration with FINN**:
- Automatic FINN bridge instantiation when no custom evaluation function provided
- Passes Blueprint configuration to FINN for evaluation
- Handles metrics extraction and normalization

#### 4. Strategy Executor (`strategy_executor.py`)

**Purpose**: Implements different exploration strategies for traversing the design space.

**Available Strategies**:

1. **HierarchicalExplorationStrategy**:
   - Three-phase approach:
     - Phase 1 (40%): Kernel selection exploration
     - Phase 2 (40%): Transform selection with best kernels
     - Phase 3 (20%): Fine-tuning best combinations
   - Groups combinations by kernel signatures
   - Progressive refinement

2. **AdaptiveExplorationStrategy**:
   - Analyzes performance trends
   - Balances exploration vs exploitation (70/30 default)
   - Identifies promising regions
   - Adapts sampling based on results

3. **ParetoGuidedStrategy**:
   - Multi-objective optimization
   - Maintains Pareto frontier
   - Guides sampling toward frontier
   - Supports configurable objectives

**Strategy Context** (`ExplorationContext`):
- Tracks exploration state
- Maintains performance history
- Current phase tracking
- Budget management

#### 5. Results Analyzer (`results_analyzer.py`)

**Purpose**: Comprehensive analysis of exploration results.

**Analysis Capabilities**:

1. **Summary Statistics**:
   - Mean, std, min, max, median for all metrics
   - Success rates
   - Coverage calculations

2. **Performance Trends**:
   - Moving averages
   - Trend direction detection
   - Convergence rate
   - Plateau detection

3. **Component Impact Analysis**:
   - Component frequency in successful runs
   - Performance statistics per component
   - Impact scores with confidence weighting

4. **Correlation Analysis**:
   - Metric correlations
   - Strong correlation identification

5. **Outlier Detection**:
   - IQR-based outlier identification
   - Outlier classification (high/low)

6. **Pareto Frontier Analysis**:
   - Multi-objective frontier calculation
   - Frontier diversity metrics
   - Component distribution analysis

## Design Space Definition Flow

1. **Input**: Blueprint V2 design space definition
2. **Combination Generation**: 
   - Respects component exploration rules
   - Generates valid combinations only
3. **Strategy Selection**:
   - Uses primary strategy from blueprint
   - Falls back to defaults if needed
4. **Evaluation Loop**:
   - Batch evaluation with parallelization
   - Result caching
   - Progress tracking
5. **Results Analysis**:
   - Multi-dimensional analysis
   - Pareto frontier extraction
   - Recommendations generation

## Key Design Patterns

### 1. Separation of Concerns
- Clear boundaries between generation, exploration, and analysis
- Each component has a single responsibility

### 2. Strategy Pattern
- Abstract base class for exploration strategies
- Pluggable strategy implementations
- Runtime strategy selection

### 3. Builder Pattern
- ComponentCombination builds incrementally
- Validation after construction

### 4. Observer Pattern
- Progress callbacks for UI integration
- Event-driven updates

### 5. Cache-Aside Pattern
- Check cache before evaluation
- Update cache after evaluation
- Persistent cache storage

## Integration Points

### Blueprint V2 Integration
- Reads design space definitions
- Respects exploration rules
- Uses DSE strategy configurations

### FINN Integration
- Automatic bridge creation
- Passes blueprint config to FINN
- Extracts evaluation metrics

### Future Extensions
- Custom strategy plugins
- Additional analysis metrics
- Machine learning-guided exploration
- Distributed evaluation support

## Performance Considerations

1. **Combination Explosion**:
   - Limits on total combinations
   - Strategic sampling methods
   - Early termination

2. **Evaluation Cost**:
   - Caching to avoid re-evaluation
   - Parallel evaluation support
   - Checkpoint/resume capability

3. **Memory Usage**:
   - Streaming results processing
   - Selective history storage
   - Efficient data structures

## Usage Example

```python
from brainsmith.core.dse import DesignSpaceExplorer
from brainsmith.core.blueprint import load_blueprint

# Load blueprint with design space
blueprint = load_blueprint("bert_accelerator.yaml")

# Configure exploration
config = ExplorationConfig(
    max_evaluations=100,
    parallel_evaluations=4,
    enable_caching=True,
    cache_directory="./dse_cache"
)

# Create explorer
explorer = DesignSpaceExplorer(blueprint.design_space, config)

# Run exploration
results = explorer.explore_design_space("model.onnx")

# Access results
print(f"Best combination: {results.best_combination.combination_id}")
print(f"Best score: {results.best_score}")
print(f"Pareto frontier size: {len(results.pareto_frontier)}")
```

## Summary

The DSE module provides a sophisticated framework for exploring hardware accelerator design spaces. It balances comprehensive exploration with practical constraints through:

- Intelligent combination generation
- Multiple exploration strategies
- Efficient evaluation management
- Comprehensive results analysis
- Strong integration with the Blueprint V2 system

The modular design allows for easy extension and customization while maintaining clean interfaces between components.