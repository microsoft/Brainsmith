# Phase 3: Build Runner - Design Document

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Backend System](#backend-system)
6. [Pipeline System](#pipeline-system)
7. [Metrics and Error Handling](#metrics-and-error-handling)
8. [Integration Points](#integration-points)
9. [Design Rationale](#design-rationale)

## Overview

The Build Runner (Phase 3) executes individual build configurations received from Phase 2, managing the actual hardware compilation process through a BuildRunner orchestrator and multiple backend implementations. The BuildRunner orchestrator coordinates preprocessing, backend execution, and postprocessing to provide a unified interface for different compilation targets with consistent pipeline execution.

### Key Responsibilities
- Orchestrate the complete build lifecycle through BuildRunner
- Execute builds through appropriate backends with preprocessed models
- Apply shared preprocessing and postprocessing pipelines
- Collect and standardize metrics
- Handle errors with detailed categorization
- Support multiple output stages (RTL, STITCHED_IP, etc.)

## Architecture

```mermaid
graph TB
    subgraph "Phase 3: Build Runner"
        BR[BuildRunner]
        BRI[BuildRunnerInterface]
        LFB[LegacyFINNBackend]
        FFB[FutureBrainsmithBackend]
        PP[PreprocessingPipeline]
        PoP[PostprocessingPipeline]
        MC[MetricsCollector]
        EH[BuildErrorHandler]
        F[Factory]
        
        BR --> PP
        BR --> PoP
        BR --> BRI
        BRI --> LFB
        BRI --> FFB
        LFB --> MC
        F --> BR
        BR -.wraps.-> LFB
        BR -.wraps.-> FFB
    end
    
    subgraph "Phase 2"
        BC[BuildConfig]
        MP[Model Path]
    end
    
    subgraph "External"
        FINN[FINN Builder]
        FS[File System]
    end
    
    BC --> BR
    MP --> BR
    BR --> BResult[BuildResult]
    LFB --> FINN
    MC --> FS
    EH --> BResult
```

### Component Relationships

```mermaid
classDiagram
    class BuildRunner {
        -backend: BuildRunnerInterface
        -preprocessing_pipeline: PreprocessingPipeline
        -postprocessing_pipeline: PostprocessingPipeline
        +run(config: BuildConfig, model_path: str) BuildResult
        +get_backend_name() str
        +get_supported_output_stages() List[OutputStage]
    }
    
    class BuildRunnerInterface {
        <<interface>>
        +run(config: BuildConfig, model_path: str) BuildResult
        +get_backend_name() str
        +get_supported_output_stages() List[OutputStage]
    }
    
    class LegacyFINNBackend {
        -finn_build_dir: str
        -temp_cleanup: bool
        -preserve_intermediate: bool
        +run(config: BuildConfig, model_path: str) BuildResult
        -_create_dataflow_config(config) Dict
        -_execute_finn_build(model_path, finn_config) int
        -_collect_artifacts(output_dir) Dict
    }
    
    class FutureBrainsmithBackend {
        -mock_success_rate: float
        -mock_build_time_range: Tuple
        +run(config: BuildConfig, model_path: str) BuildResult
        -_prepare_finn_brainsmith_config(config) Dict
        -_execute_finn_brainsmith_build(model_path, config) bool
        -_generate_mock_metrics(config) BuildMetrics
    }
    
    class PreprocessingPipeline {
        +execute(config, model_path) str
        -_apply_qonnx_transform(step, model_path, output_dir) str
    }
    
    class PostprocessingPipeline {
        +analyze(config, result) void
        -_apply_qonnx_analysis(step, config, result, output_dir) void
    }
    
    class MetricsCollector {
        +collect_from_finn_output(output_dir) BuildMetrics
        -_extract_resource_estimates(output_dir) Dict
        -_extract_performance_data(output_dir) Dict
    }
    
    BuildRunner --> BuildRunnerInterface
    BuildRunner --> PreprocessingPipeline
    BuildRunner --> PostprocessingPipeline
    BuildRunnerInterface <|-- LegacyFINNBackend
    BuildRunnerInterface <|-- FutureBrainsmithBackend
    LegacyFINNBackend --> MetricsCollector
```

## Core Components

### 0. BuildRunner (NEW - Orchestrator)

The central orchestrator that manages the complete build lifecycle, introduced to separate preprocessing and postprocessing from backend implementations.

```python
class BuildRunner:
    """
    Orchestrator that manages the complete build lifecycle.
    
    Key Benefits:
    - Separates preprocessing/postprocessing from backend implementations
    - Provides consistent behavior across all backends
    - Simplifies backend implementations
    - Improves testability and maintainability
    """
    
    def __init__(self, backend: BuildRunnerInterface):
        self.backend = backend
        self.preprocessing_pipeline = PreprocessingPipeline()
        self.postprocessing_pipeline = PostprocessingPipeline()
    
    def run(self, config: BuildConfig, model_path: str) -> BuildResult:
        """Execute complete build lifecycle."""
        # 1. Preprocessing
        processed_model_path = self.preprocessing_pipeline.execute(config, model_path)
        
        # 2. Backend execution with preprocessed model
        result = self.backend.run(config, processed_model_path)
        
        # 3. Postprocessing (only if successful)
        if result.is_successful():
            self.postprocessing_pipeline.analyze(config, result)
            
        return result
```

### 1. BuildRunnerInterface

The abstract interface that all backends must implement.

```python
class BuildRunnerInterface(ABC):
    """
    Abstract interface for build execution backends.
    
    Responsibilities:
    - Define contract for build execution
    - Declare backend capabilities
    - Ensure consistent result format
    """
    
    @abstractmethod
    def run(self, config: BuildConfig, model_path: str) -> BuildResult:
        """Execute build with preprocessed model and return results."""
        pass
        
    @abstractmethod
    def get_backend_name(self) -> str:
        """Return human-readable backend name."""
        pass
        
    @abstractmethod
    def get_supported_output_stages(self) -> List[OutputStage]:
        """Return list of supported output stages."""
        pass
```

### 2. Data Structures

```mermaid
classDiagram
    class BuildStatus {
        <<enumeration>>
        SUCCESS
        FAILED
        TIMEOUT
        SKIPPED
    }
    
    class BuildMetrics {
        +throughput: Optional[float]
        +latency: Optional[float]
        +clock_frequency: Optional[float]
        +lut_utilization: Optional[float]
        +dsp_utilization: Optional[float]
        +bram_utilization: Optional[float]
        +uram_utilization: Optional[float]
        +total_power: Optional[float]
        +accuracy: Optional[float]
        +raw_metrics: Dict[str, Any]
    }
    
    class BuildResult {
        +config_id: str
        +status: BuildStatus
        +metrics: Optional[BuildMetrics]
        +start_time: datetime
        +end_time: Optional[datetime]
        +duration_seconds: float
        +artifacts: Dict[str, str]
        +logs: Dict[str, str]
        +error_message: Optional[str]
        +complete(status, error_message)
        +is_successful() bool
        +has_metrics() bool
    }
    
    BuildResult --> BuildStatus
    BuildResult --> BuildMetrics
```

## Data Flow

### Build Execution Lifecycle

```mermaid
sequenceDiagram
    participant Phase2
    participant Factory
    participant BuildRunner
    participant PreProc
    participant Backend
    participant FINN
    participant PostProc
    participant Metrics
    
    Phase2->>Factory: create_build_runner(type)
    Factory->>Factory: Create backend
    Factory->>BuildRunner: new BuildRunner(backend)
    Factory-->>Phase2: build_runner
    
    Phase2->>BuildRunner: run(BuildConfig, model_path)
    BuildRunner->>BuildRunner: Create output directory
    
    BuildRunner->>PreProc: execute(config, model_path)
    PreProc->>PreProc: Apply QONNX transforms (placeholder)
    PreProc-->>BuildRunner: processed_model_path
    
    BuildRunner->>Backend: run(config, processed_model_path)
    
    alt Legacy Backend
        Backend->>Backend: Create FINN config
        Backend->>FINN: build_dataflow_cfg()
        FINN-->>Backend: Build outputs
        Backend->>Metrics: collect_from_finn_output()
        Metrics-->>Backend: BuildMetrics
    else Future Backend
        Backend->>Backend: Prepare config
        Backend->>Backend: Simulate build
        Backend-->>Backend: Mock outputs & metrics
    end
    
    Backend-->>BuildRunner: BuildResult
    
    alt Build Successful
        BuildRunner->>PostProc: analyze(config, result)
        PostProc->>PostProc: Apply QONNX analysis (placeholder)
        PostProc-->>BuildRunner: Analysis artifacts
    end
    
    BuildRunner-->>Phase2: BuildResult
```

### Error Handling Flow

```mermaid
graph TD
    E[Execute Build] --> T{Try Block}
    T --> |Success| SM[Success Metrics]
    T --> |Exception| C[Catch Exception]
    
    C --> EH[ErrorHandler]
    EH --> Cat[Categorize Error]
    Cat --> |Timing| TE[Timing Error]
    Cat --> |Resource| RE[Resource Error]
    Cat --> |Synthesis| SE[Synthesis Error]
    Cat --> |Memory| ME[Memory Error]
    
    TE --> Tips[Get Troubleshooting Tips]
    RE --> Tips
    SE --> Tips
    ME --> Tips
    
    Tips --> ER[Generate Error Report]
    ER --> FR[Failed Result]
    
    SM --> SR[Success Result]
```

## Backend System

### Legacy FINN Backend

Integrates with existing FINN builder infrastructure using explicit build steps.

```mermaid
graph LR
    subgraph "Legacy Backend Flow"
        BC[BuildConfig] --> DFC[DataflowConfig]
        DFC --> FB[FINN Builder]
        FB --> Outputs[Build Outputs]
        
        subgraph "Config Mapping"
            OS[Output Stage] --> GO[Generate Outputs]
            BS[Build Steps] --> FS[FINN Steps]
            CF[Config Flags] --> FP[FINN Params]
        end
        
        subgraph "Output Mapping"
            RTL --> |inference_cost<br/>rtlsim_perf<br/>rtlsim_reports| RO[RTL Outputs]
            IP[STITCHED_IP] --> |+bitfile<br/>+pynq_driver<br/>+deployment| IPO[IP Outputs]
        end
    end
```

**Key Features:**
- Maps BuildConfig to FINN DataflowBuildConfig format
- Manages FINN_BUILD_DIR environment variable
- Collects artifacts from known FINN output locations
- Handles cleanup of temporary files

### Future FINN-Brainsmith Backend (Stub)

Robust stub implementation for future direct integration.

```mermaid
graph LR
    subgraph "Future Backend Flow"
        BC[BuildConfig] --> Marshal[Marshal Data]
        
        subgraph "Configuration Structure"
            K[Kernels with metadata] --> Config
            T[Transform stages] --> Config
            G[Global settings] --> Config
            M[Build metadata] --> Config
        end
        
        Config --> Save[Save JSON]
        Config --> Sim[Simulate Build]
        
        Sim --> Mock[Generate Mock Data]
        Mock --> |Correlated with<br/>complexity| Metrics
        Mock --> Artifacts
    end
```

**Key Features:**
- Marshals kernels and transforms for future API
- Generates realistic mock metrics based on complexity
- Saves configuration for interface development
- Configurable success rate for testing

## Pipeline System

### Preprocessing Pipeline

Shared preprocessing steps applied before build execution. The pipeline provides simple handling logic that expects QONNX transforms and uses placeholder implementations.

```mermaid
stateDiagram-v2
    [*] --> CheckSteps: Check enabled steps
    CheckSteps --> [*]: No steps enabled
    CheckSteps --> ApplyTransform: Apply QONNX transform
    
    ApplyTransform --> LogTransform: Log transform name & params
    LogTransform --> CopyModel: Copy model (placeholder)
    CopyModel --> NextStep: Continue to next step
    NextStep --> ApplyTransform: More steps?
    NextStep --> [*]: All steps complete
```

### Postprocessing Pipeline

Analysis steps applied after build completion. The pipeline provides simple handling logic that expects QONNX analysis transforms and uses placeholder implementations.

```mermaid
stateDiagram-v2
    [*] --> CheckSteps: Check enabled steps
    CheckSteps --> [*]: No steps enabled
    CheckSteps --> ApplyAnalysis: Apply QONNX analysis
    
    ApplyAnalysis --> LogAnalysis: Log analysis name & params
    LogAnalysis --> CreateArtifact: Create analysis artifact (placeholder)
    CreateArtifact --> NextStep: Continue to next step
    NextStep --> ApplyAnalysis: More steps?
    NextStep --> [*]: All steps complete
```

## Metrics and Error Handling

### Metrics Collection

```mermaid
graph TB
    subgraph "FINN Output Files"
        ELR[estimate_layer_resources_hls.json]
        RSP[rtlsim_performance.json]
        TPS[time_per_step.json]
        VRP[vivado_reports/]
    end
    
    subgraph "Metrics Extraction"
        MC[MetricsCollector]
        RE[Resource Extraction]
        PE[Performance Extraction]
        TE[Timing Extraction]
    end
    
    subgraph "BuildMetrics"
        PM[Performance Metrics<br/>throughput, latency, clock]
        RM[Resource Metrics<br/>LUT, DSP, BRAM, URAM]
        QM[Quality Metrics<br/>accuracy, power]
        Raw[raw_metrics: Dict]
    end
    
    ELR --> RE
    RSP --> PE
    TPS --> TE
    VRP --> RE
    
    RE --> MC
    PE --> MC
    TE --> MC
    
    MC --> PM
    MC --> RM
    MC --> QM
    MC --> Raw
```

### Error Categorization

```python
ERROR_CATEGORIES = {
    "timing": {
        "patterns": ["timing", "clock", "constraint", "slack"],
        "tips": [
            "Reduce clock frequency",
            "Increase folding factors",
            "Enable retiming optimizations"
        ]
    },
    "resource": {
        "patterns": ["utilization", "exceeded", "insufficient"],
        "tips": [
            "Increase folding to reduce parallelism",
            "Use different kernel implementations",
            "Target larger FPGA device"
        ]
    },
    "synthesis": {
        "patterns": ["synthesis", "vivado", "error"],
        "tips": [
            "Check RTL syntax",
            "Verify all IP cores are available",
            "Check license availability"
        ]
    }
}
```

## Integration Points

### Phase 2 Integration

```mermaid
graph LR
    subgraph "Phase 2 Output"
        BC[BuildConfig]
        BRF[build_runner_factory]
    end
    
    subgraph "Phase 3 Input"
        Factory[create_build_runner_factory]
        Runner[BuildRunner]
    end
    
    subgraph "Phase 3 Output"
        BR[BuildResult]
        Metrics[BuildMetrics]
        Artifacts[Artifacts Dict]
    end
    
    BC --> Runner
    BRF --> Factory
    Factory --> Runner
    Runner --> BR
    BR --> Metrics
    BR --> Artifacts
```

### FINN Integration

Phase 3 provides two integration paths:

```python
# Legacy Integration
class LegacyFINNBackend:
    def _create_dataflow_config(self, config: BuildConfig) -> Dict:
        """Convert to FINN DataflowBuildConfig format."""
        return {
            "output_dir": config.output_dir,
            "synth_clk_period_ns": config.config_flags.get("clock_period_ns", 10.0),
            "board": config.config_flags.get("board", "Pynq-Z1"),
            "steps": config.build_steps,
            "generate_outputs": self._map_output_stage(config.global_config.output_stage)
        }

# Future Integration (Stub)
class FutureBrainsmithBackend:
    def _prepare_finn_brainsmith_config(self, config: BuildConfig) -> Dict:
        """Marshal data for future FINN-Brainsmith API."""
        return {
            "kernels": [
                {
                    "name": kernel[0],
                    "parameters": kernel[1],
                    "metadata": {"index": i, "type": "hw_kernel"}
                }
                for i, kernel in enumerate(config.kernels)
            ],
            "transform_stages": config.transforms,
            "output_stage": config.global_config.output_stage.value,
            # ... additional configuration
        }
```

## Design Rationale

### 1. Dual Backend Architecture
- **Legacy Support**: Maintains compatibility with existing FINN infrastructure
- **Future Ready**: Stub implementation allows interface development without breaking changes
- **Clean Abstraction**: Interface ensures backends are interchangeable

### 2. Simplified Pipeline System
- **QONNX Transform Focused**: Pipelines expect QONNX transforms and provide handling logic
- **Placeholder Implementation**: Simple placeholder transforms for development and testing
- **Consistency**: Same pipeline execution regardless of backend
- **Modularity**: Each step is independent and optional
- **Future Ready**: Framework ready for real QONNX transform implementations

### 3. Comprehensive Metrics
- **Standardization**: All backends produce same metric structure
- **Flexibility**: Optional fields handle backend differences
- **Raw Data**: Preserves original metrics for debugging

### 4. Error Categorization
- **Actionable**: Each category has specific troubleshooting tips
- **Pattern-Based**: Automatic categorization from error messages
- **Detailed Reports**: Include configuration context for debugging

### 5. Factory Pattern
- **Late Binding**: Backend selection at runtime
- **Clean API**: Phase 2 doesn't need backend details
- **Testability**: Easy to inject mock backends

### 6. Robust Stub Implementation
- **Interface Development**: Allows API design without implementation
- **Realistic Testing**: Mock data correlates with complexity
- **Configuration Capture**: Saves all data for future use

## Performance Considerations

### 1. Resource Management
- Temporary file cleanup after builds
- Streaming file parsing for large outputs
- Lazy artifact collection

### 2. Scalability
- O(1) memory per build configuration
- Independent build execution
- No shared state between builds

### 3. Optimization Opportunities
- Parallel preprocessing steps
- Cached preprocessing results
- Incremental metrics extraction

## Output Structure

Phase 3 creates organized outputs within its assigned directory:

```
{config_output_dir}/                      # From BuildConfig.output_dir
├── preprocessing/                        # Preprocessing outputs
│   ├── graph_optimization.onnx
│   ├── input_normalization.onnx
│   └── quantize_model.onnx
├── finn_config.json                      # Backend configuration
├── finn_brainsmith_config.json          # Future backend config
├── build_dataflow.log                   # Build logs
├── report/                              # FINN reports
│   ├── estimate_layer_resources_hls.json
│   └── rtlsim_performance.json
├── postprocessing/                      # Analysis results
│   ├── performance_analysis.json
│   ├── resource_analysis.json
│   └── accuracy_validation.json
└── artifacts/                           # Build artifacts
    ├── stitched_ip/
    ├── rtl/
    └── deployment/
```

## Summary

Phase 3 provides a robust, extensible build execution system that:
- ✅ Executes builds through multiple backend implementations
- ✅ Applies consistent preprocessing and postprocessing
- ✅ Collects standardized metrics from diverse sources
- ✅ Handles errors with detailed categorization and tips
- ✅ Supports current FINN and future direct integration
- ✅ Integrates cleanly with Phase 2's exploration engine

The design prioritizes flexibility, maintainability, and future extensibility while providing immediate value through the legacy FINN integration and comprehensive testing support through the stub implementation.