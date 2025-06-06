# Brainsmith Blueprint Architecture: From Monolithic to Modular

## The Problem: Current Blueprint System

Today, Brainsmith has a rigid, hardcoded blueprint system that makes it difficult to create new model pipelines or modify existing ones.

### Current Architecture

```mermaid
graph TB
    subgraph Current["Current System"]
        A[User] --> B["forge() function"]
        B --> C["REGISTRY dict"]
        C --> D["bert.py Blueprint"]
        D --> E["Hardcoded BUILD_STEPS list"]
        E --> F["19 Mixed Functions"]
        F --> G["FINN Execution"]
    end
    
    style D fill:#ffcccc
    style E fill:#ffcccc
    style F fill:#ffcccc
```

### Problems with Current Approach

**🔒 Inflexible**: Only one blueprint ("bert") with hardcoded steps
**🔧 Hard to Extend**: Adding new models requires deep FINN knowledge
**🍝 Mixed Responsibilities**: Blueprint contains both step definitions AND execution order
**🔄 Code Duplication**: Can't reuse steps across different model types
**🐛 Hard to Test**: Monolithic structure makes isolated testing difficult

## The Solution: Step Library + YAML Blueprints

Transform the blueprint system into a modular, composable architecture where:
- **Steps** are reusable building blocks stored in a central library
- **Blueprints** are simple YAML files that compose steps together
- **Users** can create new blueprints without writing Python code

### New Architecture Overview

```mermaid
graph TB
    subgraph Modular["New Modular System"]
        A[User] --> B[YAML Blueprint]
        B --> C[Blueprint Manager]
        C --> D[Step Library]
        
        subgraph Library["Step Library"]
            E[Common Steps]
            F[Transformer Steps]
            G[CNN Steps]
            H[FINN Steps]
        end
        
        D --> I[Step Registry]
        I --> J["forge() function"]
        J --> K[FINN Execution]
    end
    
    style B fill:#ccffcc
    style D fill:#ccffcc
    style I fill:#ccffcc
```

## Core Components

### 1. Step Library: The Building Blocks

Think of steps as LEGO blocks - each one does one specific job, and you can combine them in different ways.

```mermaid
graph LR
    subgraph Categories["Step Categories"]
        A["Common Steps<br/>cleanup<br/>validation<br/>analysis"]
        B["Transformer Steps<br/>remove_head<br/>qonnx_conversion<br/>streamlining"]
        C["CNN Steps<br/>conv_optimization<br/>pooling_ops"]
    end
    
    subgraph External["External Dependencies"]
        D["FINN Steps<br/>from finn.builder<br/>(dependency)"]
    end
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
```

**Key Benefits:**
- **Reusable**: Same cleanup step works for BERT, ResNet, etc.
- **Testable**: Each step can be tested independently
- **Discoverable**: Users can list available steps by category
- **Extensible**: Add new steps without touching existing code
- **Clean Dependencies**: FINN steps remain in FINN repository, referenced directly

**Architecture Note:** FINN build steps (like `step_create_dataflow_partition`, `step_hw_codegen`) remain in the FINN repository and are imported as dependencies. Brainsmith's step library focuses on model-specific transformations and common utilities, while leveraging FINN's existing build infrastructure.

### 2. YAML Blueprints: The Recipes

Blueprints become simple configuration files that list which steps to run in what order.

**Before (Python):**
```python
BUILD_STEPS = [
    custom_step_cleanup,
    custom_step_remove_head,
    custom_step_qonnx2finn,
    # ... 16 more hardcoded functions
]
```

**After (YAML):**
```yaml
name: "bert"
description: "BERT transformer compilation"
architecture: "transformer"

build_steps:
  - "common.cleanup"
  - "transformer.remove_head"
  - "transformer.qonnx_to_finn"
  - "step_create_dataflow_partition"    # Direct FINN step
  - "step_specialize_layers"            # Direct FINN step
  - "step_hw_codegen"                   # Direct FINN step
  # ... FINN steps imported as dependencies
```

### 3. Blueprint Inheritance

Create blueprint families that build on each other:

```mermaid
graph TB
    A["transformer_base.yaml<br/>Basic transformer pipeline"]
    A --> B["bert.yaml<br/>BERT-specific tweaks"]
    A --> C["gpt.yaml<br/>GPT-specific tweaks"]
    A --> D["t5.yaml<br/>T5-specific tweaks"]
    
    E["cnn_base.yaml<br/>Basic CNN pipeline"]
    E --> F["resnet.yaml<br/>ResNet optimizations"]
    E --> G["efficientnet.yaml<br/>EfficientNet tweaks"]
    
    style A fill:#e3f2fd
    style E fill:#e8f5e8
```

## User Experience Transformation

### For Existing Users: Zero Disruption

The system maintains complete backward compatibility. Existing code continues to work unchanged:

```python
# This still works exactly as before
result = forge("bert", model, args)
```

Behind the scenes, the system automatically:
1. Loads the `bert.yaml` blueprint
2. Resolves step names to functions
3. Executes the pipeline

### For New Users: Much Easier

Creating a new model pipeline becomes straightforward:

```mermaid
graph LR
    A["Start with base<br/>blueprint"] --> B["Customize steps<br/>in YAML"] --> C["Test & iterate"] --> D["Share with team"]
    
    style A fill:#e8f5e8
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e1f5fe
```

**Example: Creating a MobileBERT Blueprint**
1. Copy `transformer_base.yaml`
2. Add MobileBERT-specific steps
3. Save as `mobilebert.yaml`
4. Run: `forge("mobilebert", model, args)`

No Python knowledge required!

## Migration Strategy: Smooth Transition

### Phase 1: Build Foundation (Week 1)
```mermaid
graph LR
    A["Current System<br/>Working"] --> B["+ Step Library<br/>Parallel"] --> C["Both Systems<br/>Coexist"]
    
    style A fill:#e8f5e8
    style B fill:#fff3e0
    style C fill:#e1f5fe
```

- Create step library infrastructure
- No changes to existing blueprints
- Zero risk to current functionality

### Phase 2: Migrate Content (Week 2)
```mermaid
graph LR
    A["BERT Python<br/>Blueprint"] --> B["Extract Steps<br/>to Library"] --> C["Create BERT<br/>YAML Blueprint"]
    
    style A fill:#ffcccc
    style B fill:#fff3e0
    style C fill:#ccffcc
```

- Move BERT steps to library
- Create equivalent YAML blueprint
- Test both produce identical results

### Phase 3: Switch Over (Week 3)
```mermaid
graph LR
    A["forge() tries<br/>YAML first"] --> B["Falls back to<br/>Python if needed"] --> C["Gradual Migration<br/>Complete"]
    
    style A fill:#fff3e0
    style B fill:#e1f5fe
    style C fill:#ccffcc
```

- Update `forge()` to prefer YAML blueprints
- Maintain fallback to legacy system
- Users don't notice the change

## Benefits Summary

### For Developers
- **Modularity**: Easier to maintain and test individual components
- **Reusability**: Write once, use across multiple model types
- **Extensibility**: Add new steps without touching core code
- **Clarity**: Separate concerns (step logic vs. pipeline configuration)

### For Users
- **Simplicity**: Create new pipelines with YAML instead of Python
- **Discoverability**: List available steps and their documentation
- **Flexibility**: Mix and match steps for custom workflows
- **Consistency**: Standardized interface across all model types

### For the Project
- **Maintainability**: Cleaner, more organized codebase
- **Community**: Lower barrier for contributions
- **Evolution**: Foundation for future enhancements (search strategies, optimization)
- **Stability**: Backward compatibility ensures no disruption

## Example: Real-World Usage

### Today's Workflow (Complex)
1. User wants to compile a custom transformer
2. Must understand FINN internals
3. Copy and modify `bert.py` 
4. Debug Python code issues
5. Hope it works

### Tomorrow's Workflow (Simple)
1. User wants to compile a custom transformer
2. Copy `transformer_base.yaml`
3. Edit step list in text editor
4. Run `forge("my_transformer", model, args)`
5. It just works

## Technical Implementation Highlights

### Automatic Step Discovery
The system automatically finds and loads all steps, so adding new ones is as simple as dropping a file in the right folder.

### Dependency Validation
Blueprint validation ensures all required steps are present and in the correct order.

### Error Messages
Clear, actionable error messages help users fix blueprint issues quickly.

### Testing Strategy
Each step can be tested independently, making the system more reliable.

## Conclusion

This architecture transformation takes Brainsmith from a rigid, expert-only system to a flexible, user-friendly platform. The modular design makes it easier to maintain, extend, and use, while the migration strategy ensures a smooth transition with zero disruption to existing workflows.

The step library becomes the foundation for future enhancements like automated design space exploration, while YAML blueprints make the system accessible to a much broader audience.