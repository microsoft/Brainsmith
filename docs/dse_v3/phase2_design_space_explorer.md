# Phase 2: Design Space Explorer - Detailed Design

## Overview

The Design Space Explorer takes a DesignSpace object from Phase 1 and systematically explores all configurations, managing the exploration process and collecting results. The initial implementation focuses on exhaustive exploration with hooks for future intelligent search strategies.

## Core Responsibilities

1. **Combination Generation** - Generate all valid configurations from the design space
2. **Exploration Management** - Orchestrate the exploration process with progress tracking
3. **Build Submission** - Submit configurations to the Build Runner (Phase 3)
4. **Results Collection** - Aggregate results and provide analysis
5. **Hook System** - Enable extensibility for future enhancements

## Data Structures

### 1. Build Configuration

```python
@dataclass
class BuildConfig:
    """Configuration for a single build run."""
    id: str                              # Unique identifier (e.g., "config_001")
    design_space_id: str                 # Links to parent design space
    
    # Specific selections from the design space
    kernels: List[Tuple[str, List[str]]] # Selected kernel configurations
    transforms: List[str]                 # Selected transform sequence
    preprocessing: List[ProcessingStep]   # Selected preprocessing steps
    postprocessing: List[ProcessingStep]  # Selected postprocessing steps
    
    # Fixed configuration from design space
    build_steps: List[str]               # From hw_compiler_space
    config_flags: Dict[str, Any]         # From hw_compiler_space
    global_config: GlobalConfig          # From design space
    
    # Metadata
    timestamp: datetime
    combination_index: int               # Which combination number this is
    total_combinations: int              # Total in the design space
```

### 2. Build Result

```python
@dataclass
class BuildResult:
    """Result from a single build run."""
    config_id: str                       # Links back to BuildConfig
    status: BuildStatus                  # Success, failure, timeout, skipped
    
    # Metrics (if successful)
    metrics: Optional[BuildMetrics]      # From Phase 1 data structures
    
    # Timing information
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    
    # Artifacts and logs
    artifacts: Dict[str, str]            # artifact_name -> file_path
    logs: Dict[str, str]                 # log_type -> content
    error_message: Optional[str]         # If failed
    
class BuildStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"  # Due to constraints or early stopping
```

### 3. Exploration Results

```python
@dataclass
class ExplorationResults:
    """Aggregated results from design space exploration."""
    design_space_id: str
    start_time: datetime
    end_time: datetime
    
    # All build results
    evaluations: List[BuildResult]
    
    # Summary statistics
    total_combinations: int
    evaluated_count: int
    success_count: int
    failure_count: int
    skipped_count: int
    
    # Best configurations
    best_config: Optional[BuildConfig]      # Based on primary metric
    pareto_optimal: List[BuildConfig]       # Pareto frontier
    
    # Aggregated metrics
    metrics_summary: Dict[str, Dict[str, float]]  # metric -> {min, max, mean, std}
    
    def get_successful_results(self) -> List[BuildResult]:
        """Get only successful build results."""
        return [r for r in self.evaluations if r.status == BuildStatus.SUCCESS]
```

## Core Components

### 1. Explorer Engine

```python
class ExplorerEngine:
    """Main orchestrator for design space exploration."""
    
    def __init__(
        self,
        build_runner_factory: Callable[[], BuildRunnerInterface],
        hooks: Optional[List[ExplorationHook]] = None
    ):
        self.build_runner_factory = build_runner_factory
        self.hooks = hooks or []
        self.results_aggregator = ResultsAggregator()
        
    def explore(
        self, 
        design_space: DesignSpace,
        resume_from: Optional[str] = None
    ) -> ExplorationResults:
        """
        Explore the given design space.
        
        Args:
            design_space: The design space to explore
            resume_from: Optional checkpoint ID to resume from
            
        Returns:
            Aggregated exploration results
        """
        # Initialize exploration
        exploration_id = self._generate_exploration_id()
        self._fire_hook("on_exploration_start", design_space)
        
        # Generate combinations
        combination_generator = CombinationGenerator()
        all_configs = combination_generator.generate_all(design_space)
        
        # Filter based on resume point if provided
        configs_to_evaluate = self._filter_for_resume(all_configs, resume_from)
        
        # Execute exploration
        with self._create_progress_tracker(len(configs_to_evaluate)) as progress:
            for config in configs_to_evaluate:
                # Check early stopping
                if self._should_stop_early():
                    break
                
                # Fire pre-build hook
                self._fire_hook("on_combination_generated", config)
                
                # Submit build
                result = self._evaluate_config(config)
                
                # Update progress
                progress.update(result)
                
                # Fire post-build hook
                self._fire_hook("on_build_complete", result)
                
                # Add to results
                self.results_aggregator.add_result(result)
        
        # Finalize results
        results = self.results_aggregator.finalize()
        self._fire_hook("on_exploration_complete", results)
        
        return results
```

### 2. Combination Generator

```python
class CombinationGenerator:
    """Generate all valid combinations from a design space."""
    
    def generate_all(self, design_space: DesignSpace) -> List[BuildConfig]:
        """Generate all valid combinations from the design space."""
        configs = []
        
        # Get all combinations from each component
        kernel_combos = design_space.hw_compiler_space.get_kernel_combinations()
        transform_combos = design_space.hw_compiler_space.get_transform_combinations()
        preproc_combos = design_space.processing_space.get_preprocessing_combinations()
        postproc_combos = design_space.processing_space.get_postprocessing_combinations()
        
        # Generate cartesian product
        combo_index = 0
        total = len(kernel_combos) * len(transform_combos) * len(preproc_combos) * len(postproc_combos)
        
        for kernels in kernel_combos:
            for transforms in transform_combos:
                for preprocessing in preproc_combos:
                    for postprocessing in postproc_combos:
                        # Filter out empty/skipped elements
                        active_kernels = [k for k in kernels if k[0]]  # Non-empty names
                        active_transforms = [t for t in transforms if t]  # Non-empty strings
                        
                        config = BuildConfig(
                            id=f"config_{combo_index:05d}",
                            design_space_id=self._generate_design_space_id(design_space),
                            kernels=active_kernels,
                            transforms=active_transforms,
                            preprocessing=list(preprocessing),
                            postprocessing=list(postprocessing),
                            build_steps=design_space.hw_compiler_space.build_steps,
                            config_flags=design_space.hw_compiler_space.config_flags,
                            global_config=design_space.global_config,
                            timestamp=datetime.now(),
                            combination_index=combo_index,
                            total_combinations=total
                        )
                        
                        # Apply constraints
                        if self._satisfies_constraints(config, design_space.search_config.constraints):
                            configs.append(config)
                        
                        combo_index += 1
        
        return configs
    
    def _satisfies_constraints(
        self, 
        config: BuildConfig, 
        constraints: List[SearchConstraint]
    ) -> bool:
        """Check if a configuration satisfies all constraints."""
        # For now, constraints are checked after evaluation
        # In future, some constraints could be checked here
        return True
```

### 3. Results Aggregator

```python
class ResultsAggregator:
    """Aggregate and analyze exploration results."""
    
    def __init__(self):
        self.results: List[BuildResult] = []
        self.start_time = datetime.now()
        
    def add_result(self, result: BuildResult):
        """Add a build result to the aggregation."""
        self.results.append(result)
        
    def finalize(self) -> ExplorationResults:
        """Finalize and return aggregated results."""
        successful_results = [r for r in self.results if r.status == BuildStatus.SUCCESS]
        
        # Find best configuration
        best_config = self._find_best_config(successful_results)
        
        # Find Pareto optimal set
        pareto_optimal = self._find_pareto_optimal(successful_results)
        
        # Calculate metrics summary
        metrics_summary = self._calculate_metrics_summary(successful_results)
        
        return ExplorationResults(
            design_space_id=self.results[0].config_id.split('_')[0] if self.results else "",
            start_time=self.start_time,
            end_time=datetime.now(),
            evaluations=self.results,
            total_combinations=len(self.results),
            evaluated_count=len(self.results),
            success_count=len(successful_results),
            failure_count=len([r for r in self.results if r.status == BuildStatus.FAILED]),
            skipped_count=len([r for r in self.results if r.status == BuildStatus.SKIPPED]),
            best_config=best_config,
            pareto_optimal=pareto_optimal,
            metrics_summary=metrics_summary
        )
    
    def _find_best_config(self, results: List[BuildResult]) -> Optional[BuildConfig]:
        """Find the best configuration based on primary metric."""
        if not results:
            return None
            
        # For now, optimize for throughput
        best_result = max(results, key=lambda r: r.metrics.throughput if r.metrics else 0)
        return self._get_config_for_result(best_result)
    
    def _find_pareto_optimal(self, results: List[BuildResult]) -> List[BuildConfig]:
        """Find Pareto optimal configurations."""
        # Simple 2D Pareto frontier for throughput vs resource utilization
        pareto_set = []
        
        for r1 in results:
            if not r1.metrics:
                continue
                
            is_dominated = False
            for r2 in results:
                if r1 == r2 or not r2.metrics:
                    continue
                    
                # Check if r2 dominates r1 (better in all metrics)
                if (r2.metrics.throughput >= r1.metrics.throughput and
                    r2.metrics.lut_utilization <= r1.metrics.lut_utilization and
                    (r2.metrics.throughput > r1.metrics.throughput or
                     r2.metrics.lut_utilization < r1.metrics.lut_utilization)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_set.append(self._get_config_for_result(r1))
        
        return pareto_set
```

### 4. Hook System

```python
class ExplorationHook(ABC):
    """Base class for exploration hooks."""
    
    @abstractmethod
    def on_exploration_start(self, design_space: DesignSpace):
        """Called when exploration begins."""
        pass
    
    @abstractmethod
    def on_combination_generated(self, config: BuildConfig):
        """Called for each generated combination."""
        pass
    
    @abstractmethod
    def on_build_complete(self, result: BuildResult):
        """Called after each build completes."""
        pass
    
    @abstractmethod
    def on_exploration_complete(self, results: ExplorationResults):
        """Called when exploration finishes."""
        pass

# Example concrete hooks

class LoggingHook(ExplorationHook):
    """Log exploration progress."""
    
    def on_exploration_start(self, design_space: DesignSpace):
        logger.info(f"Starting exploration of {design_space.get_total_combinations()} combinations")
    
    def on_combination_generated(self, config: BuildConfig):
        logger.debug(f"Generated config {config.id}")
    
    def on_build_complete(self, result: BuildResult):
        logger.info(f"Build {result.config_id} completed with status: {result.status.value}")
    
    def on_exploration_complete(self, results: ExplorationResults):
        logger.info(f"Exploration complete: {results.success_count}/{results.evaluated_count} successful")

class CachingHook(ExplorationHook):
    """Cache build results for reuse."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def on_build_complete(self, result: BuildResult):
        # Save result to cache
        cache_file = self.cache_dir / f"{result.config_id}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
```

### 5. Progress Tracking

```python
@dataclass
class ProgressTracker:
    """Track exploration progress."""
    total_configs: int
    completed: int = 0
    successful: int = 0
    failed: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    def update(self, result: BuildResult):
        """Update progress with a new result."""
        self.completed += 1
        if result.status == BuildStatus.SUCCESS:
            self.successful += 1
        elif result.status == BuildStatus.FAILED:
            self.failed += 1
    
    def get_eta(self) -> Optional[datetime]:
        """Estimate time remaining."""
        if self.completed == 0:
            return None
            
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.completed / elapsed
        remaining = self.total_configs - self.completed
        eta_seconds = remaining / rate
        
        return datetime.now() + timedelta(seconds=eta_seconds)
    
    def get_summary(self) -> str:
        """Get progress summary string."""
        pct = (self.completed / self.total_configs) * 100
        eta = self.get_eta()
        eta_str = eta.strftime("%H:%M:%S") if eta else "Unknown"
        
        return (
            f"Progress: {self.completed}/{self.total_configs} ({pct:.1f}%) | "
            f"Success: {self.successful} | Failed: {self.failed} | "
            f"ETA: {eta_str}"
        )
```

## Integration with Phase 3

The Explorer will interface with the Build Runner through a simple interface:

```python
class BuildRunnerInterface(ABC):
    """Interface for Phase 3 Build Runner."""
    
    @abstractmethod
    def run(self, config: BuildConfig) -> BuildResult:
        """Execute a build with the given configuration."""
        pass
```

## Usage Example

```python
from brainsmith.core_v3 import forge
from brainsmith.core_v3.phase2 import ExplorerEngine, LoggingHook, CachingHook
from brainsmith.core_v3.phase3 import FINNBuildRunner

# Phase 1: Create design space
design_space = forge("model.onnx", "blueprint.yaml")

# Phase 2: Explore design space
explorer = ExplorerEngine(
    build_runner_factory=lambda: FINNBuildRunner(),
    hooks=[
        LoggingHook(),
        CachingHook("./cache"),
    ]
)

results = explorer.explore(design_space)

# Analyze results
print(f"Best configuration: {results.best_config.id}")
print(f"Success rate: {results.success_count}/{results.evaluated_count}")
print(f"Pareto optimal set size: {len(results.pareto_optimal)}")
```

## Future Extensions

The hook system enables future enhancements without modifying core code:

1. **Smart Sampling Hook** - Skip similar configurations
2. **Early Stopping Hook** - Stop when convergence detected
3. **ML Prediction Hook** - Predict performance without building
4. **Distributed Execution Hook** - Distribute builds across machines
5. **Checkpoint/Resume Hook** - Save and restore exploration state

## Key Design Decisions

1. **Exhaustive First** - Start with simple exhaustive exploration
2. **Hook-Based Extensibility** - All advanced features via hooks
3. **Clear Interfaces** - Clean separation from Phase 1 and Phase 3
4. **Progress Visibility** - Rich progress tracking and ETA
5. **Result Analysis** - Built-in Pareto optimization and metrics

## Implementation Plan

1. Implement core data structures
2. Create CombinationGenerator with constraint filtering
3. Implement ExplorerEngine with basic exploration loop
4. Add ResultsAggregator with analysis methods
5. Create example hooks (Logging, Caching)
6. Add progress tracking
7. Write comprehensive tests
8. Create mock BuildRunner for testing