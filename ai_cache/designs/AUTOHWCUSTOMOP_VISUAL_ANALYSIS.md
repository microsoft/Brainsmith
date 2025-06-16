# AutoHWCustomOp Visual Architecture Analysis

## 1. AutoHWCustomOp Three-Tier Architecture

```mermaid
graph TB
    subgraph "Tier 1: Kernel Data (Static)"
        A1[Interface Metadata]
        A2[Chunking Strategies]
        A3[Datatype Constraints]
        A4[Block Shapes]
        A1 --> A2
        A2 --> A3
        A3 --> A4
    end
    
    subgraph "Tier 2: Model Data (Runtime)"
        B1[Tensor Dimensions]
        B2[Block Dimensions]
        B3[User Datatypes]
        B4[ONNX Context]
        B4 --> B1
        B1 --> B2
        B2 --> B3
    end
    
    subgraph "Tier 3: Parallelism (Dynamic)"
        C1[iPar Values]
        C2[wPar Values]
        C3[Stream Dimensions]
        C4[Performance Metrics]
        C1 --> C3
        C2 --> C3
        C3 --> C4
    end
    
    A4 --> B1
    B3 --> C1
    
    style A1 fill:#e1f5fe
    style B1 fill:#f3e5f5
    style C1 fill:#e8f5e8
```

## 2. Template Generation Flow

```mermaid
flowchart LR
    A[RTL Source] --> B[Tree-sitter Parser]
    B --> C[Interface Analysis]
    C --> D[Pragma Extraction]
    D --> E[Parameter Analysis]
    E --> F[Template Context]
    F --> G[Jinja2 Template Engine]
    G --> H[Generated AutoHWCustomOp]
    
    subgraph "Template Inputs"
        I[Interface Metadata]
        J[Parameter Definitions]
        K[Constraint Groups]
        L[Resource Hints]
    end
    
    F --> I
    F --> J
    F --> K
    F --> L
    
    style A fill:#ffcdd2
    style H fill:#c8e6c9
```

## 3. Method Coverage Comparison

```mermaid
graph TD
    subgraph "Legacy HWCustomOp (Manual)"
        L1[get_nodeattr_types - 50 lines]
        L2[get_input_datatype - 10 lines]
        L3[get_folded_input_shape - 15 lines]
        L4[get_instream_width - 20 lines]
        L5[minimize_accumulator_width - 30 lines]
        L6[get_hw_compatible_tensor - 40 lines]
        L7[execute_node - 25 lines]
        L8[verify_node - 15 lines]
        L9[calc_tmem - 5 lines]
        
        L1 -.-> L2
        L2 -.-> L3
        L3 -.-> L4
        L4 -.-> L5
        L5 -.-> L6
        L6 -.-> L7
        L7 -.-> L8
        L8 -.-> L9
    end
    
    subgraph "AutoHWCustomOp (Generated)"
        A1[get_nodeattr_types - Template]
        A2[get_input_datatype - Parent Class]
        A3[get_folded_input_shape - DataflowModel]
        A4[get_instream_width - DataflowModel]
        A5[minimize_accumulator_width - MISSING]
        A6[get_hw_compatible_tensor - MISSING]
        A7[execute_node - Basic Template]
        A8[verify_node - Enhanced Parent]
        A9[calc_tmem - MISSING]
        
        A1 --> A2
        A2 --> A3
        A3 --> A4
        A4 --> A5
        A5 --> A6
        A6 --> A7
        A7 --> A8
        A8 --> A9
    end
    
    %% Coverage indicators
    L1 -.->|✅ Covered| A1
    L2 -.->|✅ Covered| A2
    L3 -.->|✅ Covered| A3
    L4 -.->|✅ Covered| A4
    L5 -.->|❌ Missing| A5
    L6 -.->|❌ Missing| A6
    L7 -.->|⚠️ Basic| A7
    L8 -.->|✅ Enhanced| A8
    L9 -.->|❌ Missing| A9
    
    style A1 fill:#c8e6c9
    style A2 fill:#c8e6c9
    style A3 fill:#c8e6c9
    style A4 fill:#c8e6c9
    style A5 fill:#ffcdd2
    style A6 fill:#ffcdd2
    style A7 fill:#fff3e0
    style A8 fill:#c8e6c9
    style A9 fill:#ffcdd2
```

## 4. AutoHWCustomOp Class Structure

```mermaid
classDiagram
    class HWCustomOp {
        <<FINN Base Class>>
        +get_nodeattr_types()
        +execute_node()
        +infer_node_datatype()
        +verify_node()
        +get_rtlsim()
        +node_res_estimation()
    }
    
    class AutoHWCustomOp {
        <<Generated Base>>
        -_dataflow_model: DataflowModel
        -_current_parallelism: Dict
        +get_interface_metadata()* 
        +get_input_datatype()
        +get_folded_input_shape()
        +get_instream_width()
        +update_parallelism()
        +get_legacy_attr()
        +estimate_bram_usage()
    }
    
    class GeneratedKernelOp {
        <<Template Generated>>
        +get_interface_metadata()
        +get_nodeattr_types()
        +verify_node()
        +execute_node()
    }
    
    class DataflowModel {
        +input_interfaces: List
        +output_interfaces: List
        +weight_interfaces: List
        +calculate_initiation_intervals()
        +get_resource_requirements()
    }
    
    class InterfaceMetadata {
        +name: str
        +interface_type: InterfaceType
        +datatype_constraints: List
        +chunking_strategy: BlockChunkingStrategy
    }
    
    HWCustomOp <|-- AutoHWCustomOp
    AutoHWCustomOp <|-- GeneratedKernelOp
    AutoHWCustomOp --> DataflowModel
    DataflowModel --> InterfaceMetadata
```

## 5. Data Flow Through AutoHWCustomOp

```mermaid
sequenceDiagram
    participant RTL as RTL Analysis
    participant Template as Template Engine
    participant Generated as Generated Class
    participant ONNX as ONNX Node
    participant DataflowModel as DataflowModel
    
    RTL->>Template: Interface Metadata
    RTL->>Template: Parameter Definitions
    Template->>Generated: get_interface_metadata()
    Template->>Generated: get_nodeattr_types()
    
    ONNX->>Generated: __init__(onnx_node)
    Generated->>DataflowModel: _build_dataflow_model_from_node()
    DataflowModel->>Generated: Unified Interface System
    
    ONNX->>Generated: get_input_datatype(0)
    Generated->>DataflowModel: Query interface[0].dtype
    DataflowModel->>Generated: Validated DataType
    
    ONNX->>Generated: get_folded_input_shape(0)
    Generated->>DataflowModel: Calculate with parallelism
    DataflowModel->>Generated: Folded shape
    
    ONNX->>Generated: update_parallelism({iPar: 4})
    Generated->>DataflowModel: Apply new parallelism
    DataflowModel->>Generated: Updated stream_dims
```

## 6. Coverage Heat Map

```mermaid
graph LR
    subgraph "Core FINN Methods"
        direction TB
        A1[get_nodeattr_types] 
        A2[get_input_datatype]
        A3[get_output_datatype]
        A4[get_normal_input_shape]
        A5[get_folded_input_shape]
        A6[get_instream_width]
        A7[get_outstream_width]
        A8[get_number_output_values]
        A9[get_exp_cycles]
        A10[verify_node]
    end
    
    subgraph "Advanced Methods"
        direction TB
        B1[minimize_accumulator_width]
        B2[get_hw_compatible_tensor]
        B3[execute_node - Complex]
        B4[infer_node_datatype - Advanced]
        B5[operation_specific_utils]
    end
    
    subgraph "Resource Estimation"
        direction TB
        C1[bram_estimation]
        C2[lut_estimation]
        C3[dsp_estimation]
        C4[uram_estimation]
    end
    
    %% Coverage coloring
    style A1 fill:#4caf50,color:#fff
    style A2 fill:#4caf50,color:#fff
    style A3 fill:#4caf50,color:#fff
    style A4 fill:#4caf50,color:#fff
    style A5 fill:#4caf50,color:#fff
    style A6 fill:#4caf50,color:#fff
    style A7 fill:#4caf50,color:#fff
    style A8 fill:#4caf50,color:#fff
    style A9 fill:#4caf50,color:#fff
    style A10 fill:#8bc34a,color:#fff
    
    style B1 fill:#f44336,color:#fff
    style B2 fill:#f44336,color:#fff
    style B3 fill:#ff9800,color:#fff
    style B4 fill:#ff9800,color:#fff
    style B5 fill:#f44336,color:#fff
    
    style C1 fill:#8bc34a,color:#fff
    style C2 fill:#8bc34a,color:#fff
    style C3 fill:#8bc34a,color:#fff
    style C4 fill:#8bc34a,color:#fff
```

**Legend:**
- 🟢 **Full Coverage** - Implemented in AutoHWCustomOp
- 🟡 **Partial Coverage** - Basic implementation, missing advanced features
- 🔴 **Missing** - Not implemented, requires template extensions

## 7. Weight Tensor Processing Gap (Selected Code)

```mermaid
graph TD
    subgraph "Legacy: get_hw_compatible_threshold_tensor"
        L1[Original Threshold Matrix]
        L2[Validate PE Divisibility]
        L3[Handle Unsigned Constraints]
        L4[Interleave Rows Between PEs]
        L5[Reshape to HW Format]
        L6[Return (1, PE, TMEM, steps)]
        
        L1 --> L2
        L2 --> L3
        L3 --> L4
        L4 --> L5
        L5 --> L6
    end
    
    subgraph "AutoHWCustomOp: Current State"
        A1[Weight Interface Definition]
        A2[Basic DataType Validation]
        A3[Generic Shape Calculation]
        A4[❌ No HW Formatting]
        A5[❌ No PE Interleaving]
        A6[❌ No Operation Logic]
        
        A1 --> A2
        A2 --> A3
        A3 --> A4
        A4 --> A5
        A5 --> A6
    end
    
    subgraph "Missing Implementation"
        M1[40+ lines of tensor formatting]
        M2[PE-based memory layout]
        M3[Hardware constraint validation]
        M4[Operation-specific reshaping]
    end
    
    style L1 fill:#e8f5e8
    style L6 fill:#c8e6c9
    style A4 fill:#ffcdd2
    style A5 fill:#ffcdd2
    style A6 fill:#ffcdd2
    style M1 fill:#fff3e0
    style M2 fill:#fff3e0
    style M3 fill:#fff3e0
    style M4 fill:#fff3e0
```

## 8. Template Extension Architecture Proposal

```mermaid
graph TB
    subgraph "Current Template System"
        T1[RTL Analysis]
        T2[Interface Metadata]
        T3[Basic Template]
        T4[Generated AutoHWCustomOp]
        
        T1 --> T2
        T2 --> T3
        T3 --> T4
    end
    
    subgraph "Proposed Extension System"
        E1[Operation Detection]
        E2[Specialized Templates]
        E3[Custom Method Injection]
        E4[Enhanced Generation]
        
        E1 --> E2
        E2 --> E3
        E3 --> E4
    end
    
    subgraph "Extension Points"
        P1[weight_tensor_formatting]
        P2[execute_node_logic]
        P3[datatype_optimization]
        P4[utility_methods]
        P5[advanced_validation]
    end
    
    T2 --> E1
    E2 --> P1
    E2 --> P2
    E2 --> P3
    E2 --> P4
    E2 --> P5
    
    P1 --> E4
    P2 --> E4
    P3 --> E4
    P4 --> E4
    P5 --> E4
    
    style T4 fill:#e1f5fe
    style E4 fill:#c8e6c9
    style P1 fill:#fff3e0
    style P2 fill:#fff3e0
    style P3 fill:#fff3e0
    style P4 fill:#fff3e0
    style P5 fill:#fff3e0
```

## Summary

The visual analysis reveals that AutoHWCustomOp provides excellent coverage of standard FINN methods through its three-tier architecture and DataflowModel integration. However, critical gaps exist in operation-specific functionality, particularly around weight tensor formatting, advanced execution logic, and utility methods. The template system provides a strong foundation for addressing these gaps through proposed extension points.