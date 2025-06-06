# AutoHWCustomOp Architecture Comparison

## Current Architecture (Problematic)

```mermaid
flowchart TD
    A[RTL Parser] --> B[HKG Template]
    B --> C[Generated AutoHWCustomOp]
    C --> D[Static Interface Dictionaries]
    C --> E[Placeholder Resource Methods]
    C --> F[Manual Parameter Handling]
    
    G[FINN Node Creation] --> H[onnx helper make node]
    H --> I[Sets qDim tDim dtype]
    
    J[FINN DSE] --> K[Transformations]
    K --> L[Sets iPar wPar]
    
    M[Constructor Call] --> N[Expects DataflowModel]
    N --> O[FAIL Model Not Available]
    
    style O fill:#ff9999
    style D fill:#ffcc99
    style E fill:#ffcc99
    style F fill:#ffcc99
```

## Proposed Architecture (Two-Phase Initialization)

```mermaid
flowchart TD
    A[RTL Parser] --> B[HKG Template]
    B --> C[Slim Generated AutoHWCustomOp]
    C --> D[Interface Specifications Only]
    
    E[FINN Node Creation] --> F[onnx helper make node]
    F --> G[Sets qDim tDim dtype]
    G --> H[Constructor Call]
    H --> I[Store Interface Specs]
    
    J[FINN DSE] --> K[Transformations]
    K --> L[Sets iPar wPar]
    L --> M[Method Call]
    M --> N[Lazy DataflowModel Build]
    N --> O[Dynamic Interface Creation]
    O --> P[Resource Estimation]
    
    Q[Attribute Updates] --> R[Invalidate Model]
    R --> S[Rebuild on Next Access]
    
    style I fill:#99ff99
    style N fill:#99ff99
    style O fill:#99ff99
    style P fill:#99ff99
    style D fill:#ccffcc
```

## Code Flow Comparison

### Current Flow (Broken)
```mermaid
sequenceDiagram
    participant F as FINN
    participant T as Template
    participant A as AutoHWCustomOp
    participant D as DataflowModel
    
    F->>T: Generate class
    T->>A: Create with static data
    Note over A: 300+ lines generated
    
    F->>F: onnx helper make node
    F->>A: Constructor call
    A->>D: Expects pre built model
    D-->>A: Not available
    
    Note over A: Falls back to static dictionaries
```

### Proposed Flow (Fixed)
```mermaid
sequenceDiagram
    participant F as FINN
    participant T as Template
    participant A as AutoHWCustomOp
    participant D as DataflowModel
    
    F->>T: Generate class
    T->>A: Create with interface specs
    Note over A: 50-80 lines generated
    
    F->>F: onnx helper make node
    F->>A: Constructor call
    A->>A: Store interface specs
    
    F->>F: DSE transformations
    F->>A: Method call get exp cycles
    A->>A: Check if model built
    A->>D: Build from current attributes
    D-->>A: Success
    A-->>F: Return result
```

## Resource Estimation Comparison

### Current Approach
```mermaid
flowchart LR
    A[Template] --> B[Generate Static Methods]
    B --> C[bram estimation 50 lines]
    B --> D[lut estimation 30 lines]
    B --> E[dsp estimation 20 lines]
    C --> F[Placeholder Logic]
    D --> F
    E --> F
    F --> G[Hard coded Return Values]
    
    style F fill:#ff9999
    style G fill:#ff9999
```

### Proposed Approach
```mermaid
flowchart LR
    A[Template] --> B[Generate Slim Methods]
    B --> C[bram estimation 3 lines]
    B --> D[lut estimation 3 lines]
    B --> E[dsp estimation 3 lines]
    
    C --> F[dataflow model get resource requirements]
    D --> F
    E --> F
    F --> G[Interface based Analysis]
    G --> H[Accurate Estimates]
    
    style F fill:#99ff99
    style G fill:#99ff99
    style H fill:#99ff99
```

## Template Complexity Reduction

### Before (Current Template)
```
Lines of Code: 298 total
├── Static Interface Specs: 83-104 (21 lines × N interfaces)
├── Resource Estimation: 136-232 (96 lines of placeholders)
├── Parameter Handling: 114-135 (21 lines)
├── Verification Logic: 236-256 (20 lines)
└── Boilerplate: 50+ lines

Complexity: High
Maintainability: Poor
Customization: Manual override required
```

### After (Proposed Template)
```
Lines of Code: 50-80 total
├── Interface Specs: 8-12 (JSON format)
├── Resource Estimation: 12-20 (delegates to DataflowModel)
├── Constructor: 5-8 lines
└── Optional Customization: 10-30 lines

Complexity: Low
Maintainability: Excellent  
Customization: DataflowModel-based
```

## Benefits Summary

| Aspect | Current | Proposed | Improvement |
|--------|---------|----------|-------------|
| Generated Code Lines | 300+ | 50-80 | 75-80% reduction |
| Template Complexity | High | Low | Major simplification |
| FINN Compatibility | Broken | Full + Enhanced Tensor Chunking | Complete fix + Enhancement |
| Interface Configuration | Manual qDim/tDim | Automatic Shape Extraction + Enhanced TDIM | Intuitive + Zero-config |
| Chunking Strategy | None | Index-based with Auto Shape Detection | Advanced chunking control |
| Resource Accuracy | Placeholder | Interface-based | Significant improvement |
| Maintenance Burden | High | Low | Major reduction |
| Customization | Manual | DataflowModel + Enhanced TDIM Pragma | Systematic + Zero-config |

## Implementation Timeline

```mermaid
gantt
    title AutoHWCustomOp Refactoring Timeline
    dateFormat YYYY-MM-DD
    
    section Phase1-BaseClass
    Refactor AutoHWCustomOp     :active, p1, 2024-01-01, 7d
    Two-phase initialization    :p1b, after p1, 7d
    Lazy model building        :p1c, after p1b, 7d
    
    section Phase2-Enhancement
    Resource estimation        :p2, after p1c, 7d
    DataflowModel methods      :p2b, after p2, 7d
    
    section Phase3-Templates
    Slim template design       :p3, after p2b, 7d
    Template context update    :p3b, after p3, 7d
    
    section Phase4-Testing
    Integration testing        :p4, after p3b, 7d
    FINN workflow validation   :p4b, after p4, 7d
    
    section Phase5-Migration
    Existing class migration   :p5, after p4b, 7d
    Documentation update       :p5b, after p5, 7d
```

## Architecture Benefits

This architectural refactoring resolves both the verbosity issue and the FINN workflow compatibility problem while significantly improving maintainability and accuracy of the generated AutoHWCustomOp classes.

### Key Improvements

1. **Code Reduction**: From 300+ lines to 50-80 lines per generated class
2. **FINN Compatibility**: Full support for onnx.helper.make_node workflow
3. **Resource Estimation**: Interface-aware algorithms replace placeholder logic
4. **Maintainability**: Simplified templates with DataflowModel integration
5. **Performance**: Lazy model building optimizes memory usage
6. **Extensibility**: Clean separation of concerns for future enhancements

## Enhanced Tensor Chunking Workflow (Final Solution)

```mermaid
flowchart TD
    A[RTL Analysis] --> B[Enhanced TDIM Pragma Detection]
    B --> C["@brainsmith TDIM intf_name index"]
    
    D[FINN Node Creation] --> E[Input Tensor Available]
    E --> F[Automatic Shape Extraction]
    F --> G[Layout Inference]
    
    H[AutoHWCustomOp Constructor] --> I[Store Interface Metadata]
    I --> J[Enhanced TDIM Pragma Present?]
    
    J -->|Yes| K[Extract Tensor Shape from Input]
    J -->|No| L[Use Default Layout-Based Chunking]
    
    K --> M[Apply Index-Based Chunking]
    M --> N[qDim/tDim Computed]
    
    L --> O[Infer Layout from Shape]
    O --> P[Standard Layout Chunking]
    P --> N
    
    N --> Q[DataflowModel Creation]
    Q --> R[Interface Configuration Complete]
    
    style F fill:#99ff99
    style G fill:#99ff99
    style K fill:#99ff99
    style M fill:#99ff99
    style C fill:#ccffcc
```

## Tensor Shape Extraction Process

```mermaid
sequenceDiagram
    participant F as FINN
    participant O as ONNX Node
    participant A as AutoHWCustomOp
    participant T as TensorChunking
    participant M as ModelWrapper
    
    F->>O: Create node with input tensors
    F->>A: Constructor call
    A->>A: Parse interface metadata
    
    A->>T: process_enhanced_tdim_pragma()
    T->>A: extract_tensor_shape_from_input()
    A->>M: get_tensor_shape(tensor_name)
    M-->>A: [1, 8, 32, 32]
    
    A->>T: infer_layout_from_shape()
    T-->>A: "NCHW"
    
    T->>T: apply_index_chunking_strategy()
    Note over T: chunk_index = -1
    Note over T: shape[3] = 32 (width)
    
    T-->>A: qDim=[1,8,32,1], tDim=[1,1,1,32]
    A->>A: Configure interfaces
    A-->>F: Ready for DataflowModel
```

## Enhanced TDIM Pragma Comparison

### Legacy Approach (Complex)
```systemverilog
// Manual qDim/tDim specification
// Requires deep HLS knowledge
qDim = [1, 8, 32, 1]
tDim = [1, 1, 1, 32]
```

### Final Enhanced Approach (Simple)
```systemverilog
// @brainsmith TDIM in0_V_data_V -1
// Automatic shape extraction + index-based chunking
// Zero configuration required
```

### Key Simplifications in Final Solution

1. **No Manual Shape Specification**: Removed `[:] ` and `[C, H, W]` syntax complexity
2. **Automatic Tensor Detection**: Shape extracted from ONNX input tensors
3. **Smart Layout Inference**: 4D→NCHW, 3D→CHW, 2D→NC, 1D→C
4. **Minimal Pragma Syntax**: Just interface name + chunk index
5. **Zero Configuration**: Works without any pragmas using sensible defaults