# BrainSmith Core Workflow Diagram

## Visual Flow Representation

```mermaid
graph TD
    %% Input
    YAML[Blueprint YAML<br/>• steps<br/>• kernels<br/>• config] --> FORGE[forge.py<br/>Entry Point]
    MODEL[ONNX Model] --> FORGE
    
    %% Parsing Phase
    FORGE --> PARSER[BlueprintParser<br/>• Load YAML<br/>• Handle inheritance<br/>• Validate steps]
    PARSER --> |validates| REGISTRY[Plugin Registry<br/>• 243 components<br/>• Steps, transforms<br/>• Kernels, backends]
    
    %% Design Space
    PARSER --> DS[DesignSpace<br/>• Resolved steps<br/>• Kernel→Backend map<br/>• Configurations]
    PARSER --> TREE[ExecutionTree<br/>• Segments<br/>• Branches<br/>• Efficient paths]
    
    %% Execution Phase
    DS --> EXPLORER[Explorer<br/>• Orchestrates<br/>• Creates adapter]
    TREE --> EXPLORER
    
    EXPLORER --> EXECUTOR[Executor<br/>• Traverse tree<br/>• Execute segments<br/>• Share artifacts]
    
    EXECUTOR --> |per segment| FINN[FINNAdapter<br/>• Handle quirks<br/>• Build models<br/>• Discover outputs]
    
    %% FINN Integration
    FINN --> |calls| FINNBUILD[FINN Build System<br/>• Execute steps<br/>• Transform model<br/>• Generate hardware]
    
    %% Output
    FINNBUILD --> OUTPUT[Output<br/>• ONNX models<br/>• RTL/HLS code<br/>• Bitstreams]
    
    %% Feedback
    OUTPUT --> |cached| EXECUTOR
    
    style YAML fill:#e1f5e1
    style MODEL fill:#e1f5e1
    style OUTPUT fill:#ffe1e1
    style REGISTRY fill:#fff4e1
```

## Component Interactions

### 1. Plugin Registry Usage
```
Blueprint Step Name → Registry Lookup → Step Function → Execution
         "cleanup" → has_step() ✓    → cleanup_step() → FINN
```

### 2. Tree Segmentation Example
```
Root
├─[cleanup, streamline]          # Segment 1 (shared prefix)
│  ├─[quantize_int8] → end       # Branch 1
│  └─[quantize_int4] → end       # Branch 2
└─[optimize_aggressive] → end     # Segment 2
```

### 3. Artifact Sharing
```
Parent Segment Output → Copy to Children → Continue Execution
    segment_1/output → segment_1/branch_1/
                     → segment_1/branch_2/
```

## Data Flow Summary

| Phase | Input | Process | Output | Used? |
|-------|-------|---------|--------|-------|
| Parse | YAML | Inheritance, validation | DesignSpace + Tree | ✅ |
| Build | DesignSpace | Create segments | ExecutionTree | ✅ |
| Execute | Tree + Model | FINN builds | Hardware models | ✅ |
| Registry | Plugin names | Lookup/validate | Classes/functions | Partial |

## Key Insights from Workflow

### ✅ What Works Well
1. **Clean API**: Simple forge() entry point
2. **Efficient Execution**: Segment-based prefix sharing
3. **Isolated Integration**: FINN quirks contained
4. **Flexible Structure**: Supports complex branching

### ⚠️ What's Overengineered  
1. **Plugin System**: 243 registered, <20 used
2. **Metadata**: Complex queries never used
3. **Namespacing**: Framework prefixes unused
4. **Statistics**: Calculated but not leveraged

### ❌ What's Missing
1. **Default Pipelines**: Must specify every step
2. **Progress Tracking**: Only print statements
3. **Kernel Inference**: Special case with no implementation
4. **Result Analysis**: No performance metrics

## Simplified Ideal Workflow

```mermaid
graph LR
    YAML[Blueprint] --> PARSE[Parse & Validate]
    PARSE --> EXECUTE[Execute Steps]
    EXECUTE --> HARDWARE[Hardware Output]
    
    REGISTRY[Step Registry] -.->|lookup| PARSE
    FINN[FINN] -.->|build| EXECUTE
```

The current workflow is functionally complete but could be 50% simpler by removing unused features and focusing on the actual usage patterns.