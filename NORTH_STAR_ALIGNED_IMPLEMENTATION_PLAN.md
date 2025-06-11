# BrainSmith Infrastructure Unification Implementation Plan
## Aligned with North Star Goals & Core Design Axioms

> **North Star**: *"Make FPGA accelerator design as simple as calling a function."*  
> **Promise**: `result = brainsmith.forge('model.onnx', 'blueprint.yaml')`

---

## 🎯 Strategic Alignment with North Star Goals

### Core Problem Analysis
The current `brainsmith/core` ↔ `brainsmith/infrastructure` split **violates our fundamental axioms**:

```mermaid
graph LR
    subgraph "Current Violations"
        V1[Complex Import Patterns]
        V2[Framework-like Structure] 
        V3[Cognitive Load Burden]
        V4[Enterprise Abstraction]
    end
    
    subgraph "North Star Axioms"
        A1[Simplicity Over Sophistication]
        A2[Functions Over Frameworks]
        A3[Focus Over Feature Creep]
        A4[Documentation Over Discovery]
    end
    
    V1 -.->|violates| A1
    V2 -.->|violates| A2
    V3 -.->|violates| A3
    V4 -.->|violates| A4
    
    style V1 fill:#ffcccc
    style V2 fill:#ffcccc
    style V3 fill:#ffcccc
    style V4 fill:#ffcccc
    style A1 fill:#90EE90
    style A2 fill:#90EE90
    style A3 fill:#90EE90
    style A4 fill:#90EE90
```

### Current vs North Star Target State

```mermaid
graph TB
    subgraph "Current State"
        C1[15+ Import Paths]
        C2[try/except Fallbacks]
        C3[Multiple Module Discovery]
        C4[Framework Complexity]
        C5[Learning Curve: Hours]
    end
    
    subgraph "North Star Target"
        T1[Single brainsmith.core Import]
        T2[forge + 12 Helper Functions]
        T3[Zero Configuration Objects]
        T4[Direct Function Calls]
        T5[Time to Success: 5 Minutes]
    end
    
    C1 ==>|Unify| T1
    C2 ==>|Eliminate| T2
    C3 ==>|Simplify| T3
    C4 ==>|Remove| T4
    C5 ==>|Accelerate| T5
    
    style T1 fill:#90EE90
    style T2 fill:#90EE90
    style T3 fill:#90EE90
    style T4 fill:#90EE90
    style T5 fill:#90EE90
```

---

## 🏗️ Three-Phase Implementation Strategy

### Phase 1: Core Unification (Week 1)
**Goal**: Eliminate artificial boundaries while maintaining all functionality

```mermaid
gantt
    title Phase 1: Core Unification Timeline
    dateFormat  YYYY-MM-DD
    section File Movement
    Move DSE Components       :move-dse, 2024-12-11, 2d
    Move FINN Components      :move-finn, after move-dse, 1d
    Move Hooks Components     :move-hooks, after move-finn, 1d
    Move Data Components      :move-data, after move-hooks, 1d
    section Import Cleanup  
    Update Import Paths       :imports, after move-data, 2d
    Eliminate try/except      :fallbacks, after imports, 1d
    Test All Functions        :test-p1, after fallbacks, 1d
```

#### 1.1 Directory Restructuring

```mermaid
graph LR
    subgraph "Before: Artificial Split"
        B1[brainsmith/core/api.py]
        B2[brainsmith/infrastructure/dse/]
        B3[brainsmith/infrastructure/hooks/]
        B4[brainsmith/infrastructure/finn/]
        B5[brainsmith/infrastructure/data/]
    end
    
    subgraph "After: Unified Core"
        A1[brainsmith/core/api.py]
        A2[brainsmith/core/dse/]
        A3[brainsmith/core/hooks/]
        A4[brainsmith/core/finn/]
        A5[brainsmith/core/data/]
    end
    
    B1 --> A1
    B2 ==> A2
    B3 ==> A3
    B4 ==> A4
    B5 ==> A5
    
    style A1 fill:#90EE90
    style A2 fill:#90EE90
    style A3 fill:#90EE90
    style A4 fill:#90EE90
    style A5 fill:#90EE90
```

#### 1.2 Import Pattern Transformation

```mermaid
graph TB
    subgraph "Current Complex Imports"
        I1["from ..infrastructure.hooks import log_optimization_event"]
        I2["from ..infrastructure.dse.interface import DSEInterface"]
        I3["from ..infrastructure.finn import build_accelerator"]
        I4["try/except ImportError patterns"]
    end
    
    subgraph "Target Simple Imports"
        S1["from .hooks import log_optimization_event"]
        S2["from .dse.interface import DSEInterface"]
        S3["from .finn import build_accelerator"]
        S4["Direct imports, no fallbacks"]
    end
    
    I1 ==> S1
    I2 ==> S2
    I3 ==> S3
    I4 ==> S4
    
    style S1 fill:#90EE90
    style S2 fill:#90EE90
    style S3 fill:#90EE90
    style S4 fill:#90EE90
```

### Phase 2: Registry Standardization (Week 2) 
**Goal**: Apply successful BaseRegistry pattern universally

```mermaid
graph TB
    subgraph "Registry Inconsistency Problem"
        P1[HooksRegistry: Custom Pattern]
        P2[AutomationRegistry: BaseRegistry Pattern ✓]
        P3[Different Discovery Interfaces]
        P4[Inconsistent Error Handling]
    end
    
    subgraph "Unified Registry Solution"
        S1[HooksRegistry: BaseRegistry Pattern ✓]
        S2[AutomationRegistry: BaseRegistry Pattern ✓]
        S3[Consistent .discover Interface]
        S4[Unified Error Handling]
    end
    
    P1 ==> S1
    P2 ==> S2
    P3 ==> S3
    P4 ==> S4
    
    style S1 fill:#90EE90
    style S2 fill:#90EE90
    style S3 fill:#90EE90
    style S4 fill:#90EE90
```

**Key Issue**: [`HooksRegistry`](brainsmith/infrastructure/hooks/registry.py:68) doesn't follow successful [`BaseRegistry`](brainsmith/libraries/automation/registry.py) pattern (99% success rate)

**Solution**: Standardize all registries to inherit from `BaseRegistry` with unified discovery interface.

### Phase 3: API Simplification (Week 3)
**Goal**: Achieve North Star promise and metrics

```mermaid
graph TB
    subgraph "API Evolution"
        CURRENT[Multiple Import Sources]
        TARGET[Single Core Import]
        
        subgraph "Current Complex API"
            C1["from brainsmith.core import forge"]
            C2["from brainsmith.infrastructure.dse import DesignSpace"]
            C3["from brainsmith.infrastructure.hooks import log_optimization_event"]
            C4["from brainsmith.libraries.automation import parameter_sweep"]
        end
        
        subgraph "Target Unified API"
            T1["from brainsmith.core import forge"]
            T2["from brainsmith.core import DesignSpace"]
            T3["from brainsmith.core import log_optimization_event"]
            T4["from brainsmith.core import parameter_sweep"]
        end
        
        CURRENT --> C1
        CURRENT --> C2
        CURRENT --> C3
        CURRENT --> C4
        
        TARGET --> T1
        TARGET --> T2
        TARGET --> T3
        TARGET --> T4
    end
    
    style TARGET fill:#90EE90
    style T1 fill:#90EE90
    style T2 fill:#90EE90
    style T3 fill:#90EE90
    style T4 fill:#90EE90
```

---

## 🔧 Detailed Technical Implementation

### Critical File Modifications

#### 1. [`brainsmith/core/api.py`](brainsmith/core/api.py:17-24) Changes
**Problem**: Complex try/except patterns violate "Simplicity Over Sophistication"

```python
# BEFORE (Lines 17-24): Complex fallback pattern
try:
    from ..infrastructure.hooks import log_optimization_event, log_strategy_decision, log_dse_event
    HOOKS_AVAILABLE = True
except ImportError:
    HOOKS_AVAILABLE = False
    log_optimization_event = lambda *args, **kwargs: None
    log_strategy_decision = lambda *args, **kwargs: None
    log_dse_event = lambda *args, **kwargs: None

# AFTER: Direct imports (axiom-aligned)
from .hooks import log_optimization_event, log_strategy_decision, log_dse_event
```

#### 2. [`brainsmith/core/__init__.py`](brainsmith/core/__init__.py:8-15) Updates
**Goal**: Single import location for all core functionality

```python
# Enhanced exports for unified core
from .api import forge, validate_blueprint
from .metrics import DSEMetrics
from .dse.design_space import DesignSpace
from .dse.interface import DSEInterface
from .hooks import log_optimization_event, register_event_handler
from .finn import build_accelerator
from .data import DataManager

__all__ = [
    # Primary function (North Star Promise)
    'forge', 'validate_blueprint',
    # Core classes
    'DesignSpace', 'DSEInterface', 'DSEMetrics',
    # Helper functions (12 total per axioms)
    'log_optimization_event', 'build_accelerator', 'DataManager',
    # Additional helpers...
]
```

#### 3. Registry Standardization
**Issue**: [`HooksRegistry`](brainsmith/infrastructure/hooks/registry.py:68) uses different pattern than successful [`BaseRegistry`](brainsmith/libraries/automation/registry.py)

```python
# BEFORE: Inconsistent pattern
class HooksRegistry:
    def discover_plugins(self, rescan: bool = False) -> Dict[str, PluginInfo]:

# AFTER: Consistent BaseRegistry pattern  
class HooksRegistry(BaseRegistry):
    def discover(self, rescan: bool = False) -> Dict[str, ComponentInfo]:
```

### New Core API Structure

```mermaid
graph TD
    subgraph "brainsmith.core - Unified API"
        FORGE[forge: Primary Function]
        
        subgraph "Essential Classes (3)"
            DS[DesignSpace]
            DSE[DSEInterface]  
            METRICS[DSEMetrics]
        end
        
        subgraph "Helper Functions (12)"
            H1[parameter_sweep]
            H2[find_best_result]
            H3[log_optimization_event]
            H4[build_accelerator]
            H5[get_analysis_data]
            H6[validate_blueprint]
            H7[batch_process]
            H8[aggregate_stats]
            H9[export_results]
            H10[load_design_space]
            H11[sample_design_space]
            H12[register_event_handler]
        end
        
        FORGE --> DS
        FORGE --> DSE
        FORGE --> METRICS
        
        DS --> H1
        DSE --> H2
        METRICS --> H3
    end
    
    style FORGE fill:#4CAF50
    style DS fill:#90EE90
    style DSE fill:#90EE90
    style METRICS fill:#90EE90
```

---

## ✅ Success Validation & Metrics

### North Star Compliance Check

```mermaid
graph LR
    subgraph "North Star Metrics"
        M1[Time to First Success: <5min]
        M2[Primary Function: forge]
        M3[Helper Functions: 12]
        M4[Configuration Objects: 0]
        M5[Core Concepts: 3]
    end
    
    subgraph "Implementation Results"
        R1[✓ Single import + forge call]
        R2[✓ One primary function]
        R3[✓ 12 composable helpers]  
        R4[✓ Direct parameter passing]
        R5[✓ Models, blueprints, parameters]
    end
    
    M1 --> R1
    M2 --> R2
    M3 --> R3
    M4 --> R4
    M5 --> R5
    
    style R1 fill:#90EE90
    style R2 fill:#90EE90
    style R3 fill:#90EE90
    style R4 fill:#90EE90
    style R5 fill:#90EE90
```

### Axiom Compliance Dashboard

```mermaid
graph TB
    subgraph "Axiom Compliance Results"
        A1[Simplicity Over Sophistication ✓]
        A2[Functions Over Frameworks ✓]
        A3[Focus Over Feature Creep ✓]
        A4[Hooks Over Implementation ✓]
        A5[Performance Over Purity ✓]
        A6[Documentation Over Discovery ✓]
    end
    
    subgraph "Measurable Improvements"
        I1[Import Paths: 15→1]
        I2[Fallback Patterns: 8→0]
        I3[Module Discovery: Complex→Direct]
        I4[Learning Curve: Hours→Minutes]
        I5[Configuration: Objects→Parameters]
    end
    
    A1 --> I1
    A2 --> I2
    A3 --> I3
    A4 --> I4
    A5 --> I5
    A6 --> I5
    
    style A1 fill:#90EE90
    style A2 fill:#90EE90
    style A3 fill:#90EE90
    style A4 fill:#90EE90
    style A5 fill:#90EE90
    style A6 fill:#90EE90
```

### Final User Experience

```mermaid
graph LR
    subgraph "The BrainSmith Promise Achieved"
        USER[User]
        
        subgraph "Simple API"
            S1["import brainsmith.core as bs"]
            S2["result = bs.forge('model.onnx', 'blueprint.yaml')"]
            S3["best = bs.find_best_result(results, 'throughput')"]
            S4["data = bs.get_analysis_data(results)"]
        end
        
        subgraph "External Tools Integration"
            E1[pandas DataFrames]
            E2[scipy optimization]
            E3[scikit-learn analysis]
            E4[matplotlib visualization]
        end
        
        USER --> S1
        S1 --> S2
        S2 --> S3
        S3 --> S4
        
        S4 --> E1
        S4 --> E2
        S4 --> E3
        S4 --> E4
    end
    
    style S1 fill:#90EE90
    style S2 fill:#4CAF50
    style S3 fill:#90EE90
    style S4 fill:#90EE90
```

---

## 🚫 Anti-Patterns We Will Eliminate

### What We Will NOT Do (Per North Star Goals)

```mermaid
graph LR
    subgraph "Anti-Patterns to Eliminate"
        X1[❌ Backward Compatibility Layers]
        X2[❌ Configuration Objects]
        X3[❌ Plugin Architectures for Core]
        X4[❌ Enterprise Abstractions]
        X5[❌ Custom Analysis vs scipy]
    end
    
    subgraph "Clean Break Principles"
        B1[✓ Force Migration to Better APIs]
        B2[✓ Direct Parameter Passing]
        B3[✓ Built-in Functionality Only]
        B4[✓ Function Composition]
        B5[✓ External Library Integration]
    end
    
    X1 ==> B1
    X2 ==> B2
    X3 ==> B3
    X4 ==> B4
    X5 ==> B5
    
    style B1 fill:#90EE90
    style B2 fill:#90EE90
    style B3 fill:#90EE90
    style B4 fill:#90EE90
    style B5 fill:#90EE90
```

**Clean Refactor Principle**: "When improving APIs, eliminate legacy methods entirely rather than creating backward compatibility layers"

---

## 🎉 End State Vision

### What Success Looks Like

```mermaid
graph TB
    subgraph "BrainSmith Promise Delivered"
        P1[New users get results in 5 minutes]
        P2[Expert users use simple function composition]
        P3[Maintainers focus on optimization vs complexity]
        P4[Ecosystem integration with existing tools]
    end
    
    subgraph "Technical Achievements"
        T1[Single import: brainsmith.core]
        T2[Zero configuration objects]
        T3[Direct function calls only]
        T4[External tool data exposure]
    end
    
    subgraph "Organizational Benefits"
        O1[Reduced support questions]
        O2[Faster onboarding]
        O3[Cleaner documentation]
        O4[Easier testing]
    end
    
    P1 --> T1
    P2 --> T2
    P3 --> T3
    P4 --> T4
    
    T1 --> O1
    T2 --> O2
    T3 --> O3
    T4 --> O4
    
    style P1 fill:#4CAF50
    style P2 fill:#4CAF50
    style P3 fill:#4CAF50
    style P4 fill:#4CAF50
```

### The Ultimate User Experience

```python
# This is ALL users need to learn (North Star achieved)
import brainsmith.core as bs

# Primary workflow (5 minutes to success)
result = bs.forge('model.onnx', 'blueprint.yaml')
best = bs.find_best_result(result, metric='throughput')

# Advanced workflows (simple composition)
params = {'batch_size': [1, 4, 8], 'frequency': [200, 250, 300]}
swept = bs.parameter_sweep('model.onnx', 'blueprint.yaml', params)
optimized = bs.find_best_result(swept, metric='efficiency')

# External tool integration (hooks over implementation)
import pandas as pd
data = bs.get_analysis_data(swept)
df = pd.DataFrame(data)  # Ready for pandas/scipy/sklearn
```

This implementation plan transforms BrainSmith from a framework users must learn into a set of functions they can immediately use, fully aligned with our North Star vision.