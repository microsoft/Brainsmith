# Data Flow Through Brainsmith DSE v3

## Complete Data Flow Pipeline (ASCII Art)

```
                    DATA FLOW THROUGH BRAINSMITH DSE v3
    ═══════════════════════════════════════════════════════════════

    PyTorch Model                        Blueprint YAML
         │                                      │
         ▼                                      │
    ┌─────────────┐                            │
    │  Brevitas   │                            │
    │Quantization │                            │
    └──────┬──────┘                            │
           │                                    │
           ▼                                    ▼
    ┌─────────────┐                    ┌───────────────┐
    │ ONNX Model  │                    │ Blueprint     │
    │   (Q8/Q4)   │                    │ Specification │
    └──────┬──────┘                    └───────┬───────┘
           │                                    │
           └────────────┬───────────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │   PHASE 1       │
              │ Design Space    │──────► DesignSpace Object
              │ Constructor     │        • Valid configs
              └────────┬────────┘        • Plugin refs
                       │
                       ▼
              ┌─────────────────┐
              │   PHASE 2       │
              │ Design Space    │──────► BuildConfig[]
              │ Explorer        │        • Ranked configs
              └────────┬────────┘        • Ready to build
                       │
                       ▼
              ┌─────────────────┐
              │   PHASE 3       │
              │ Build Runner    │──────► BuildResult[]
              │                 │        • Metrics
              └────────┬────────┘        • Artifacts
                       │
                       ▼
              ┌─────────────────┐
              │ Results         │
              │ • Best Config   │
              │ • Pareto Set    │
              │ • Build Files   │
              └─────────────────┘
```

## Data Structure Evolution

### Input Stage
```python
# PyTorch Model
model = torch.nn.Module()

# After Brevitas Quantization
quantized_model = brevitas.export.onnx(model)

# Blueprint YAML
blueprint = """
version: "3.0"
name: "BERT Exploration"
hw_compiler:
  kernels: ["MatMul", "LayerNorm"]
  transforms: {...}
"""
```

### Phase 1 Output: DesignSpace
```python
@dataclass
class DesignSpace:
    # Model information
    model_path: Path
    model_info: ModelInfo
    
    # Hardware compiler space
    hw_compiler_space: HWCompilerSpace
    
    # Search configuration
    search_config: SearchConfig
    
    # Global settings
    global_config: GlobalConfig
    
    # Methods
    def get_total_combinations() -> int
    def validate() -> List[ValidationError]
```

### Phase 2 Output: BuildConfig
```python
@dataclass
class BuildConfig:
    # Unique identifier
    config_id: str  # e.g., "dse_abc123_config_00042"
    
    # Model reference
    model_path: Path
    
    # Selected options
    kernel_selections: Dict[str, str]
    transform_sequence: List[str]
    build_steps: List[str]
    
    # Backend selection
    backend: str
    
    # Configuration
    config_flags: Dict[str, Any]
    global_config: GlobalConfig
```

### Phase 3 Output: BuildResult
```python
@dataclass
class BuildResult:
    # Reference
    config_id: str
    
    # Status
    status: BuildStatus  # SUCCESS, FAILED, TIMEOUT
    
    # Timing
    start_time: datetime
    end_time: datetime
    duration: timedelta
    
    # Metrics
    metrics: BuildMetrics
    
    # Artifacts
    artifacts: Dict[str, Path]
    
    # Diagnostics
    error_message: Optional[str]
    warnings: List[str]
    logs: Path
```

## Transformation Examples

### 1. Model Transformation
```
PyTorch (float32) → Brevitas (int8) → ONNX (quantized)

Original: 512MB model
Quantized: 128MB model (4x reduction)
Hardware: Efficient INT8 operations
```

### 2. Configuration Expansion
```
Blueprint kernels:
  - ["MatMul", "MVAU"]
  - ["LayerNorm", "BatchNorm"]

Expanded combinations:
  1. MatMul + LayerNorm
  2. MatMul + BatchNorm
  3. MVAU + LayerNorm
  4. MVAU + BatchNorm
```

### 3. Build Artifact Generation
```
Input: BuildConfig
Output artifacts/
  ├── synthesis/
  │   ├── top_wrapper.v
  │   └── utilization.rpt
  ├── ip_blocks/
  │   ├── matmul_rtl.v
  │   └── layernorm_hls.cpp
  ├── bitstream/
  │   └── design.bit
  └── reports/
      ├── timing.rpt
      └── power.rpt
```

## Data Flow Optimizations

### 1. Caching Layer
```
┌─────────────┐     ┌──────────┐     ┌─────────────┐
│   Request   │ --> │  Cache   │ --> │   Builder   │
│ BuildConfig │     │  Check   │     │  (if miss)  │
└─────────────┘     └──────────┘     └─────────────┘
                          │
                          ▼
                    ┌──────────┐
                    │  Result  │
                    │  (cached) │
                    └──────────┘
```

### 2. Parallel Processing
```
Design Space (1000 configs)
         │
    ┌────┴────┬────┬────┐
    ▼         ▼    ▼    ▼
 Worker 1  Worker 2 ... Worker N
    │         │          │
    ▼         ▼          ▼
 Results   Results    Results
    └────┬────┴────┬────┘
         ▼
    Aggregated Results
```

### 3. Incremental Updates
```
Previous Run: 100 configs evaluated
Resume Point: config_00101
New Run: Continue from 101-1000
Final: Merged results (1000 total)
```

## Error Handling Flow

```
Build Attempt
     │
     ▼
┌─────────────┐
│ Try Build   │
└──────┬──────┘
       │
   ┌───┴───┐
   │Success│───► Store Result
   └───────┘
       │
   ┌───┴───┐
   │Failure│
   └───┬───┘
       │
   ┌───▼────────┐
   │Recoverable?│
   └───┬────┬───┘
      Yes   No
       │     │
   Retry   Mark Failed
```

## Performance Characteristics

### Data Volume
- **Input Model**: 10-500 MB
- **Design Space**: 10-100,000 configurations
- **Build Artifacts**: 1-10 GB per config
- **Final Results**: 100 MB - 1 GB

### Processing Time
- **Phase 1**: Seconds to minutes
- **Phase 2**: Minutes (generation)
- **Phase 3**: Hours to days (builds)
- **Total**: Hours to weeks

### Memory Usage
- **Design Space**: O(n) configs
- **Active Builds**: O(p) parallel
- **Results Storage**: O(n) compressed
- **Peak Usage**: 10-50 GB

## Best Practices

1. **Stream Processing**: Don't load all configs at once
2. **Lazy Evaluation**: Generate configs on demand
3. **Result Compression**: Store only essential metrics
4. **Artifact Management**: Clean intermediate files
5. **Checkpoint Frequently**: Enable resume capability
6. **Monitor Resources**: Track memory and disk usage