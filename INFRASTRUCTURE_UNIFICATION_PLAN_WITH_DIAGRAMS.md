# BrainSmith Infrastructure Unification Plan

## Overview

This plan consolidates the artificial split between `brainsmith/core` and `brainsmith/infrastructure` into a unified core module that aligns with the "Functions Over Frameworks" philosophy.

## Current Architecture Problems

### Current Directory Structure
```mermaid
graph TD
    subgraph "Current Split Architecture"
        BRAIN[brainsmith/]
        
        subgraph "Core Layer"
            CORE[core/]
            CORE_API[api.py]
            CORE_CLI[cli.py]
            CORE_METRICS[metrics.py]
            CORE_REG[registry/]
            CORE --> CORE_API
            CORE --> CORE_CLI
            CORE --> CORE_METRICS
            CORE --> CORE_REG
        end
        
        subgraph "Infrastructure Layer"
            INFRA[infrastructure/]
            INFRA_DSE[dse/]
            INFRA_FINN[finn/]
            INFRA_HOOKS[hooks/]
            INFRA_DATA[data/]
            INFRA --> INFRA_DSE
            INFRA --> INFRA_FINN
            INFRA --> INFRA_HOOKS
            INFRA --> INFRA_DATA
        end
        
        subgraph "Libraries Layer"
            LIB[libraries/]
            LIB_AUTO[automation/]
            LIB_BLUE[blueprints/]
            LIB_ANALYSIS[analysis/]
            LIB --> LIB_AUTO
            LIB --> LIB_BLUE
            LIB --> LIB_ANALYSIS
        end
        
        BRAIN --> CORE
        BRAIN --> INFRA
        BRAIN --> LIB
    end
    
    style CORE fill:#ffcccc
    style INFRA fill:#ccffcc
    style LIB fill:#ccccff
```

### Current Dependency Nightmare
```mermaid
graph LR
    subgraph "Core Dependencies"
        API[core/api.py]
        CLI[core/cli.py]
        METRICS[core/metrics.py]
        REG[core/registry/]
    end
    
    subgraph "Infrastructure Dependencies"
        DSE[infrastructure/dse/]
        FINN[infrastructure/finn/]
        HOOKS[infrastructure/hooks/]
        DATA[infrastructure/data/]
    end
    
    subgraph "Complex Import Patterns"
        I1["from ..infrastructure.hooks import log_optimization_event"]
        I2["from ..infrastructure.dse.interface import DSEInterface"]
        I3["from ..infrastructure.finn import build_accelerator"]
        I4["from ..infrastructure.dse.blueprint_manager import BlueprintManager"]
    end
    
    API -.->|"try/except ImportError"| DSE
    API -.->|"try/except ImportError"| FINN
    API -.->|"try/except ImportError"| HOOKS
    API -.->|"fallback patterns"| DATA
    
    API --> I1
    API --> I2
    API --> I3
    API --> I4
    
    style API fill:#ff9999
    style I1 fill:#ffcccc
    style I2 fill:#ffcccc
    style I3 fill:#ffcccc
    style I4 fill:#ffcccc
```

## Proposed Unified Architecture

### New Directory Structure
```mermaid
graph TD
    subgraph "Unified Core Architecture"
        BRAIN[brainsmith/]
        
        subgraph "Unified Core"
            CORE[core/]
            CORE_API[api.py]
            CORE_CLI[cli.py]
            CORE_METRICS[metrics.py]
            CORE_REG[registry/]
            CORE_DSE[dse/]
            CORE_FINN[finn/]
            CORE_HOOKS[hooks/]
            CORE_DATA[data/]
            
            CORE --> CORE_API
            CORE --> CORE_CLI
            CORE --> CORE_METRICS
            CORE --> CORE_REG
            CORE --> CORE_DSE
            CORE --> CORE_FINN
            CORE --> CORE_HOOKS
            CORE --> CORE_DATA
        end
        
        subgraph "Libraries Layer (Unchanged)"
            LIB[libraries/]
            LIB_AUTO[automation/]
            LIB_BLUE[blueprints/]
            LIB_ANALYSIS[analysis/]
            LIB --> LIB_AUTO
            LIB --> LIB_BLUE
            LIB --> LIB_ANALYSIS
        end
        
        BRAIN --> CORE
        BRAIN --> LIB
    end
    
    style CORE fill:#90EE90
    style LIB fill:#ccccff
```

### Clean Import Patterns
```mermaid
graph LR
    subgraph "Unified Core Module"
        API[core/api.py]
        DSE[core/dse/]
        FINN[core/finn/]
        HOOKS[core/hooks/]
        DATA[core/data/]
        INIT[core/__init__.py]
    end
    
    subgraph "Simple Import Patterns"
        I1["from .hooks import log_optimization_event"]
        I2["from .dse.interface import DSEInterface"]
        I3["from .finn import build_accelerator"]
        I4["from .dse.blueprint_manager import BlueprintManager"]
    end
    
    subgraph "Public API"
        P1["from brainsmith.core import forge"]
        P2["from brainsmith.core import DesignSpace"]
        P3["from brainsmith.core import log_optimization_event"]
    end
    
    API --> I1
    API --> I2
    API --> I3
    API --> I4
    
    INIT --> P1
    INIT --> P2
    INIT --> P3
    
    style API fill:#90EE90
    style I1 fill:#d4edda
    style I2 fill:#d4edda
    style I3 fill:#d4edda
    style I4 fill:#d4edda
```

## File Movement Plan

### Phase 1: Infrastructure → Core Migration
```mermaid
gantt
    title Infrastructure to Core Migration Timeline
    dateFormat  YYYY-MM-DD
    section Week 1: File Movement
    Move DSE Module                 :move-dse, 2024-12-11, 1d
    Move FINN Module               :move-finn, after move-dse, 1d
    Move Hooks Module              :move-hooks, after move-finn, 1d
    Move Data Module               :move-data, after move-hooks, 1d
    Update Import Statements       :update-imports, after move-data, 2d
    Test All Functionality         :test-phase1, after update-imports, 1d
    
    section Week 2: API Simplification
    Update core/__init__.py        :update-init, after test-phase1, 1d
    Remove Fallback Patterns       :remove-fallbacks, after update-init, 2d
    Consolidate Configuration      :config-consolidate, after remove-fallbacks, 2d
    Integration Testing            :test-phase2, after config-consolidate, 2d
    
    section Week 3: Final Polish
    Standardize Error Handling     :error-handling, after test-phase2, 2d
    Update Documentation          :update-docs, after error-handling, 2d
    Performance Optimization      :optimize, after update-docs, 2d
    Final Testing                 :final-test, after optimize, 1d
```

### Detailed File Movement Map
```mermaid
graph LR
    subgraph "Source (infrastructure/)"
        SRC_DSE[dse/]
        SRC_FINN[finn/]
        SRC_HOOKS[hooks/]
        SRC_DATA[data/]
        
        subgraph "DSE Files"
            DSE_INIT[__init__.py]
            DSE_TYPES[types.py]
            DSE_ENGINE[engine.py]
            DSE_INTERFACE[interface.py]
            DSE_HELPERS[helpers.py]
            DSE_BLUEPRINT[blueprint_manager.py]
            DSE_DESIGN[design_space.py]
        end
        
        subgraph "FINN Files"
            FINN_INIT[__init__.py]
            FINN_TYPES[types.py]
            FINN_INTERFACE[interface.py]
            FINN_FINN[finn_interface.py]
        end
        
        subgraph "Hooks Files"
            HOOKS_INIT[__init__.py]
            HOOKS_TYPES[types.py]
            HOOKS_EVENTS[events.py]
            HOOKS_REGISTRY[registry.py]
            HOOKS_PLUGINS[plugins/]
        end
        
        subgraph "Data Files"
            DATA_INIT[__init__.py]
            DATA_TYPES[types.py]
            DATA_COLLECTION[collection.py]
            DATA_EXPORT[export.py]
            DATA_MANAGEMENT[management.py]
        end
        
        SRC_DSE --> DSE_INIT
        SRC_DSE --> DSE_TYPES
        SRC_DSE --> DSE_ENGINE
        SRC_DSE --> DSE_INTERFACE
        SRC_DSE --> DSE_HELPERS
        SRC_DSE --> DSE_BLUEPRINT
        SRC_DSE --> DSE_DESIGN
        
        SRC_FINN --> FINN_INIT
        SRC_FINN --> FINN_TYPES
        SRC_FINN --> FINN_INTERFACE
        SRC_FINN --> FINN_FINN
        
        SRC_HOOKS --> HOOKS_INIT
        SRC_HOOKS --> HOOKS_TYPES
        SRC_HOOKS --> HOOKS_EVENTS
        SRC_HOOKS --> HOOKS_REGISTRY
        SRC_HOOKS --> HOOKS_PLUGINS
        
        SRC_DATA --> DATA_INIT
        SRC_DATA --> DATA_TYPES
        SRC_DATA --> DATA_COLLECTION
        SRC_DATA --> DATA_EXPORT
        SRC_DATA --> DATA_MANAGEMENT
    end
    
    subgraph "Destination (core/)"
        DEST_DSE[dse/]
        DEST_FINN[finn/]
        DEST_HOOKS[hooks/]
        DEST_DATA[data/]
    end
    
    SRC_DSE ==> DEST_DSE
    SRC_FINN ==> DEST_FINN
    SRC_HOOKS ==> DEST_HOOKS
    SRC_DATA ==> DEST_DATA
    
    style SRC_DSE fill:#ffcccc
    style SRC_FINN fill:#ffcccc
    style SRC_HOOKS fill:#ffcccc
    style SRC_DATA fill:#ffcccc
    style DEST_DSE fill:#90EE90
    style DEST_FINN fill:#90EE90
    style DEST_HOOKS fill:#90EE90
    style DEST_DATA fill:#90EE90
```

## Updated Module Dependencies

### Before: Complex Cross-Layer Dependencies
```mermaid
graph TB
    subgraph "Current Complex Dependencies"
        USER[User Code]
        
        subgraph "Core Layer"
            FORGE[forge()]
            CLI[CLI]
            METRICS[Metrics]
        end
        
        subgraph "Infrastructure Layer"
            DSE[DSE Engine]
            FINN[FINN Interface]
            HOOKS[Hooks System]
            DATA[Data Management]
        end
        
        subgraph "Libraries Layer"
            AUTO[Automation]
            BLUE[Blueprints]
            ANALYSIS[Analysis]
        end
        
        USER --> FORGE
        USER --> CLI
        
        FORGE -.->|"try/except"| DSE
        FORGE -.->|"try/except"| FINN
        FORGE -.->|"fallback"| HOOKS
        FORGE -.->|"fallback"| DATA
        
        CLI -.->|"import"| FORGE
        
        AUTO --> DSE
        BLUE --> DSE
        ANALYSIS --> DATA
    end
    
    style FORGE fill:#ff9999
    style DSE fill:#ffcccc
    style FINN fill:#ffcccc
    style HOOKS fill:#ffcccc
    style DATA fill:#ffcccc
```

### After: Clean Unified Dependencies
```mermaid
graph TB
    subgraph "Unified Clean Dependencies"
        USER[User Code]
        
        subgraph "Unified Core"
            FORGE[forge()]
            CLI[CLI]
            METRICS[Metrics]
            DSE[DSE Engine]
            FINN[FINN Interface]
            HOOKS[Hooks System]
            DATA[Data Management]
            REGISTRY[Registry System]
        end
        
        subgraph "Libraries Layer"
            AUTO[Automation]
            BLUE[Blueprints]
            ANALYSIS[Analysis]
        end
        
        USER --> FORGE
        USER --> CLI
        
        FORGE --> DSE
        FORGE --> FINN
        FORGE --> HOOKS
        FORGE --> DATA
        
        CLI --> FORGE
        
        DSE --> REGISTRY
        HOOKS --> REGISTRY
        
        AUTO --> FORGE
        BLUE --> FORGE
        ANALYSIS --> FORGE
    end
    
    style FORGE fill:#90EE90
    style DSE fill:#90EE90
    style FINN fill:#90EE90
    style HOOKS fill:#90EE90
    style DATA fill:#90EE90
```

## API Evolution

### Current Fragmented API
```mermaid
graph LR
    subgraph "Current Fragmented Imports"
        USER[User Code]
        
        subgraph "Multiple Import Sources"
            I1["from brainsmith.core import forge"]
            I2["from brainsmith.infrastructure.dse import DesignSpace"]
            I3["from brainsmith.infrastructure.hooks import log_optimization_event"]
            I4["from brainsmith.libraries.automation import parameter_sweep"]
            I5["from brainsmith.core.registry import BaseRegistry"]
        end
        
        USER --> I1
        USER --> I2
        USER --> I3
        USER --> I4
        USER --> I5
    end
    
    style I1 fill:#ffcccc
    style I2 fill:#ffcccc
    style I3 fill:#ffcccc
    style I4 fill:#ccccff
    style I5 fill:#ffcccc
```

### Proposed Unified API
```mermaid
graph LR
    subgraph "Unified Core API"
        USER[User Code]
        
        subgraph "Single Core Import"
            I1["from brainsmith.core import forge"]
            I2["from brainsmith.core import DesignSpace"]
            I3["from brainsmith.core import log_optimization_event"]
            I4["from brainsmith.core import BaseRegistry"]
            I5["from brainsmith.core import DSEMetrics"]
        end
        
        subgraph "Libraries (Specialized)"
            L1["from brainsmith.libraries.automation import parameter_sweep"]
            L2["from brainsmith.libraries.blueprints import BlueprintLibraryRegistry"]
        end
        
        USER --> I1
        USER --> I2
        USER --> I3
        USER --> I4
        USER --> I5
        USER --> L1
        USER --> L2
    end
    
    style I1 fill:#90EE90
    style I2 fill:#90EE90
    style I3 fill:#90EE90
    style I4 fill:#90EE90
    style I5 fill:#90EE90
    style L1 fill:#ccccff
    style L2 fill:#ccccff
```

## Updated Core __init__.py Structure

### New Export Strategy
```mermaid
graph TD
    subgraph "core/__init__.py Exports"
        INIT[__init__.py]
        
        subgraph "Primary Functions"
            FORGE[forge]
            VALIDATE[validate_blueprint]
        end
        
        subgraph "Core Classes"
            DESIGN_SPACE[DesignSpace]
            METRICS[DSEMetrics]
            BASE_REG[BaseRegistry]
        end
        
        subgraph "Infrastructure Components"
            DSE_INTERFACE[DSEInterface]
            FINN_INTERFACE[FINNInterface]
            HOOKS_LOG[log_optimization_event]
            DATA_MANAGER[DataManager]
        end
        
        subgraph "Types & Exceptions"
            DSE_CONFIG[DSEConfiguration]
            REGISTRY_ERROR[RegistryError]
            SERVICE_ERROR[ServiceError]
        end
        
        INIT --> FORGE
        INIT --> VALIDATE
        INIT --> DESIGN_SPACE
        INIT --> METRICS
        INIT --> BASE_REG
        INIT --> DSE_INTERFACE
        INIT --> FINN_INTERFACE
        INIT --> HOOKS_LOG
        INIT --> DATA_MANAGER
        INIT --> DSE_CONFIG
        INIT --> REGISTRY_ERROR
        INIT --> SERVICE_ERROR
    end
    
    style INIT fill:#90EE90
    style FORGE fill:#4CAF50
    style VALIDATE fill:#4CAF50
```

## Implementation Benefits

### Complexity Reduction
```mermaid
graph LR
    subgraph "Before: Complex"
        B_MODULES[2 Core Modules]
        B_IMPORTS[15+ Import Paths]
        B_FALLBACKS[8 Fallback Patterns]
        B_ERRORS[Multiple Error Types]
    end
    
    subgraph "After: Simple"
        A_MODULES[1 Unified Module]
        A_IMPORTS[5 Clean Imports]
        A_FALLBACKS[0 Fallback Patterns]
        A_ERRORS[Unified Error Handling]
    end
    
    B_MODULES ==>|"Consolidate"| A_MODULES
    B_IMPORTS ==>|"Simplify"| A_IMPORTS
    B_FALLBACKS ==>|"Eliminate"| A_FALLBACKS
    B_ERRORS ==>|"Standardize"| A_ERRORS
    
    style B_MODULES fill:#ffcccc
    style B_IMPORTS fill:#ffcccc
    style B_FALLBACKS fill:#ffcccc
    style B_ERRORS fill:#ffcccc
    style A_MODULES fill:#90EE90
    style A_IMPORTS fill:#90EE90
    style A_FALLBACKS fill:#90EE90
    style A_ERRORS fill:#90EE90
```

## Success Metrics

### Measurable Improvements
```mermaid
graph TB
    subgraph "Success Metrics Dashboard"
        subgraph "Code Metrics"
            C1[Import Statements: 15 → 5]
            C2[Fallback Patterns: 8 → 0]
            C3[Module Depth: 3 → 2]
            C4[Error Types: 5 → 2]
        end
        
        subgraph "Developer Experience"
            D1[Learning Curve: -50%]
            D2[Setup Complexity: -70%]
            D3[Debug Time: -40%]
            D4[Onboarding: -60%]
        end
        
        subgraph "Maintenance"
            M1[Test Complexity: -30%]
            M2[Documentation Pages: -40%]
            M3[Configuration Points: -60%]
            M4[Dependency Management: -50%]
        end
    end
    
    style C1 fill:#90EE90
    style C2 fill:#90EE90
    style C3 fill:#90EE90
    style C4 fill:#90EE90
    style D1 fill:#87CEEB
    style D2 fill:#87CEEB
    style D3 fill:#87CEEB
    style D4 fill:#87CEEB
    style M1 fill:#DDA0DD
    style M2 fill:#DDA0DD
    style M3 fill:#DDA0DD
    style M4 fill:#DDA0DD
```

This consolidation eliminates the artificial core/infrastructure split while maintaining all functionality, creating a truly unified core toolchain that aligns with BrainSmith's "Functions Over Frameworks" philosophy.