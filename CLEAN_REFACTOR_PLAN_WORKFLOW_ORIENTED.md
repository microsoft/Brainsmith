# 🛡️ **BrainSmith Clean Refactor Plan: Workflow-Oriented API Structure**

## **Executive Summary**

This plan addresses critical technical debt in BrainSmith while achieving the North Star goal of simplicity through a **workflow-oriented API structure**. Instead of a flat list of exports, the API is organized by **user journey stages** that align with the **Time to First Success** goals from the North Star axioms.

---

## **🎯 Context & Strategy**

### **Problem Statement**
Current BrainSmith implementation violates North Star axioms:
- **533-line fallback maze** in main `__init__.py` violates "Simplicity Over Sophistication"
- **Registry inconsistencies** violate "Functions Over Frameworks" 
- **Flat export list** violates "Problem-focused organization"
- **Infrastructure backup directory** suggests abandoned architectural decisions

### **Strategic Approach**
- **Preserve BaseRegistry** for open source extensibility
- **Eliminate complexity violations** through clean refactoring
- **Organize by user workflow** rather than technical structure
- **Maintain all North Star promises** while improving structure

---

## **📋 Three-Phase Implementation Plan**

### **🔥 PHASE 1: Dependency Cleanup (Day 1)**
*Status: Critical - eliminates import dependency hell*

**Objective**: Remove 533-line fallback maze and implement explicit dependencies

**Current Problem**:
```python
# brainsmith/__init__.py (533 lines of try/except blocks)
try:
    from .core.config import BrainsmithConfig
except ImportError:
    BrainsmithConfig = None
# ... 500+ more lines of fallbacks
```

**Solution**:
```python
# NEW: brainsmith/dependencies.py
def check_installation():
    """Fast dependency check on import - fail immediately if critical deps missing"""
    missing = []
    try:
        import onnx
    except ImportError:
        missing.append("onnx")
    
    try:
        import yaml
    except ImportError:
        missing.append("pyyaml")
    
    if missing:
        raise ImportError(f"BrainSmith requires: {', '.join(missing)}\nInstall: pip install {' '.join(missing)}")

check_installation()  # Run on import
```

**Actions**:
1. **Create** explicit dependency checking module
2. **Replace** [`brainsmith/__init__.py`](brainsmith/__init__.py) with clean 50-line version
3. **Remove** all try/except ImportError fallback logic
4. **Delete** [`infrastructure_backup/`](brainsmith/infrastructure_backup/) entirely

---

### **🔧 PHASE 2: Registry Simplification & Contributor Enablement (Day 2)**
*Status: REVISED - Keep BaseRegistry, eliminate inconsistencies*

**Objective**: Maintain robust BaseRegistry for extensibility while eliminating interface inconsistencies

**Current Problem**:
```python
# brainsmith/core/hooks/registry.py - INCONSISTENT INTERFACE
class HooksRegistry(BaseRegistry[PluginInfo]):
    def discover_components(self, rescan: bool = False): ...  # Unified interface
    def discover_plugins(self, rescan: bool = False): ...     # DUPLICATE - violates axioms
```

**Solution**:
```python
# FIXED: Remove duplicate methods, enforce single interface
class HooksRegistry(BaseRegistry[PluginInfo]):
    def discover_components(self, rescan: bool = False):
        """Single unified discovery method - no duplicates"""
        # Implementation stays the same, just remove duplicate methods
```

**Contributor Experience**:
```python
# Clear documentation for open source contributors
"""
Adding Components to BrainSmith

1. Create your component info class:
   class MyToolInfo(ComponentInfo):
       @property
       def name(self) -> str: return self._name
       @property 
       def description(self) -> str: return self._desc

2. Create your registry:
   class MyToolRegistry(BaseRegistry[MyToolInfo]):
       def discover_components(self): ...
       def _get_default_dirs(self): ...
       def _extract_info(self): ...
       def _validate_component_implementation(self): ...

3. Register with BrainSmith:
   # Your components automatically discoverable
"""
```

**Actions**:
1. **KEEP** BaseRegistry as foundation for extensibility
2. **ENFORCE** single interface compliance across all registries
3. **ELIMINATE** duplicate methods (e.g., `discover_plugins()`)
4. **STANDARDIZE** contributor documentation

---

### **🌟 PHASE 3: Workflow-Oriented API Structure (Day 3)**
*Status: ENHANCED - Core cleanup + structured organization*

**Objective**: Transform API from technical module list into user journey guide

**Current Problem**: Flat export list without cognitive structure
```python
# Current: Flat list provides no user guidance
__all__ = [
    'forge', 'parameter_sweep', 'find_best', 'batch_process', 'aggregate_stats',
    'log_optimization_event', 'register_event_handler', 'build_accelerator',
    # ... continues without structure
]
```

**Solution: Workflow-Oriented Structure**
```python
# NEW: brainsmith/__init__.py (workflow-oriented structure)
"""
BrainSmith: Simple FPGA accelerator design space exploration

Organized by user workflow stages:
• Core DSE (5-minute success)
• Automation (15-minute success)  
• Analysis & Monitoring (30-minute success)
• Advanced Building (1-hour success)
• Extensibility (contributor-focused)
"""

# Explicit dependency check - fail fast if missing
from .dependencies import check_installation
check_installation()

# === 🎯 CORE DSE (5-minute success) ===
from .core.api import forge, validate_blueprint
from .core.dse.design_space import DesignSpace
from .core.dse.interface import DSEInterface
from .core.metrics import DSEMetrics

# === ⚡ AUTOMATION (15-minute success) ===
from .libraries.automation import (
    parameter_sweep,    # Explore parameter combinations
    batch_process,      # Process multiple models
    find_best,          # Find optimal results
    aggregate_stats     # Statistical summaries
)

# === 📊 ANALYSIS & MONITORING (30-minute success) ===
from .core.hooks import (
    log_optimization_event,     # Event tracking
    register_event_handler      # Custom monitoring
)
from .core.data import (
    collect_dse_metrics as get_analysis_data,  # Data extraction
    export_metrics as export_results           # Data export
)

# === 🔧 ADVANCED BUILDING (1-hour success) ===
from .core.finn import build_accelerator      # FINN integration
from .core.dse import sample_design_space     # Advanced sampling

# === 🔌 EXTENSIBILITY (contributor-focused) ===
from .core.registry import BaseRegistry, ComponentInfo
from .core.hooks.registry import HooksRegistry, get_hooks_registry

# === 📋 STRUCTURED EXPORTS ===
__all__ = [
    # === CORE DSE (Start here - 5 minutes to success) ===
    'forge',              # Primary function: model + blueprint → accelerator
    'validate_blueprint', # Validate configuration before DSE
    'DesignSpace',        # Design space representation  
    'DSEInterface',       # Design space exploration engine
    'DSEMetrics',         # Performance metrics collection
    
    # === AUTOMATION (Scale up - 15 minutes to success) ===
    'parameter_sweep',    # Explore parameter combinations automatically
    'batch_process',      # Process multiple model/blueprint pairs
    'find_best',          # Find optimal results by metric
    'aggregate_stats',    # Generate statistical summaries
    
    # === ANALYSIS & MONITORING (Integrate - 30 minutes to success) ===
    'log_optimization_event',   # Track optimization events
    'register_event_handler',   # Custom monitoring and callbacks
    'get_analysis_data',        # Extract data for external analysis
    'export_results',           # Export to pandas, CSV, JSON
    
    # === ADVANCED BUILDING (Master - 1 hour to success) ===
    'build_accelerator',        # Generate FINN accelerator
    'sample_design_space',      # Advanced design space sampling
    
    # === EXTENSIBILITY (Contributors) ===
    'BaseRegistry',             # Foundation for component discovery
    'ComponentInfo',            # Component metadata interface
    'HooksRegistry',            # Plugin and handler management
    'get_hooks_registry'        # Registry access
]

# === 🎯 WORKFLOW HELPERS ===
class workflows:
    """Common workflow patterns for quick access"""
    
    @staticmethod
    def quick_dse(model_path: str, blueprint_path: str):
        """5-minute workflow: Basic DSE"""
        return forge(model_path, blueprint_path)
    
    @staticmethod 
    def parameter_exploration(model_path: str, blueprint_path: str, params: dict):
        """15-minute workflow: Parameter sweep + optimization"""
        results = parameter_sweep(model_path, blueprint_path, params)
        return find_best(results, metric='throughput')
    
    @staticmethod
    def full_analysis(model_path: str, blueprint_path: str, params: dict, export_path: str = None):
        """30-minute workflow: Full DSE + analysis + export"""
        results = parameter_sweep(model_path, blueprint_path, params)
        best = find_best(results, metric='throughput')
        stats = aggregate_stats(results)
        data = get_analysis_data(results)
        
        if export_path:
            export_results(data, export_path)
            
        return {'best': best, 'stats': stats, 'data': data}

# === 📚 LEARNING PATH ===
def help():
    """Show learning path for new users"""
    return """
🎯 BrainSmith Learning Path

⏱️  5 minutes:  result = brainsmith.forge('model.onnx', 'blueprint.yaml')
⏱️  15 minutes: results = brainsmith.parameter_sweep(model, blueprint, params)
⏱️  30 minutes: data = brainsmith.get_analysis_data(results)
⏱️  1 hour:     accelerator = brainsmith.build_accelerator(model, blueprint)

📖 Full documentation: https://brainsmith.readthedocs.io
🔧 Examples: brainsmith.workflows.quick_dse(model, blueprint)
"""

# Convenience aliases
find_best_result = find_best
```

**Actions**:
1. **Implement** workflow-oriented structure
2. **Add** progressive learning helpers (`workflows` class)
3. **Create** structured documentation
4. **Maintain** all existing function signatures (zero breaking changes)

---

## **🎯 Benefits & Validation**

### **Benefits of Workflow-Oriented Structure**

#### **🎯 User-Centric Organization**
- **Core DSE**: Everything needed for first success (5 minutes)
- **Automation**: Scale up to parameter exploration (15 minutes)  
- **Analysis**: Integrate with external tools (30 minutes)
- **Advanced**: Full accelerator building (1 hour)
- **Extensibility**: Contributor-focused tools

#### **📚 Progressive Learning**
- Users see **logical progression** rather than flat list
- Each section builds on the previous one
- Clear **milestone goals** (5min → 15min → 30min → 1hr)
- **Workflow helpers** for common patterns

#### **🔍 Discoverability**
- `brainsmith.help()` shows learning path
- `brainsmith.workflows` provides pre-built patterns
- Comments explain **when and why** to use each function
- Grouped by **user intent**, not technical structure

#### **⚡ Maintains Simplicity**
- Same simple imports: `brainsmith.forge()`, `brainsmith.parameter_sweep()`
- No complex namespacing or configuration objects
- Zero setup required - all functions work immediately
- Preserves all North Star promises

### **Validation Path**

#### **Phase 1 Validation**:
```bash
# Import speed and predictability
time python -c "import brainsmith"  # <1s, clear errors if deps missing
python -c "import brainsmith; print(len(brainsmith.__all__))"  # Exactly 19
```

#### **Phase 2 Validation**:
```python
# Registry interface consistency
from brainsmith.libraries.automation.registry import AutomationRegistry
from brainsmith.core.hooks.registry import HooksRegistry

# Both should have ONLY discover_components, not duplicate methods
assert hasattr(AutomationRegistry, 'discover_components')
assert not hasattr(HooksRegistry, 'discover_plugins')  # Removed duplicate

# Contributor extensibility still works
from brainsmith.core.registry import BaseRegistry
class TestRegistry(BaseRegistry): pass  # Should work
```

#### **Phase 3 Validation**: 
```python
# Core API reliability + workflow structure
import brainsmith

# Test North Star promise
result = brainsmith.forge('model.onnx', 'blueprint.yaml')  # Works or fails clearly

# Test workflow organization
help_text = brainsmith.help()  # Shows learning path
quick_result = brainsmith.workflows.quick_dse('model.onnx', 'blueprint.yaml')

# Test structured exports - each section should be clearly organized
core_functions = ['forge', 'validate_blueprint', 'DesignSpace', 'DSEInterface', 'DSEMetrics']
automation_functions = ['parameter_sweep', 'batch_process', 'find_best', 'aggregate_stats']
assert all(func in brainsmith.__all__ for func in core_functions)
assert all(func in brainsmith.__all__ for func in automation_functions)
```

---

## **🚀 Implementation Timeline**

| Phase | Duration | Priority | Focus |
|-------|----------|----------|-------|
| **Phase 1** | Day 1 | Critical | Dependency cleanup, remove 533-line fallback maze |
| **Phase 2** | Day 2 | High | Registry standardization, preserve extensibility |
| **Phase 3** | Day 3 | Enhanced | Workflow structure, progressive learning |

---

## **📊 Success Metrics**

### **Before vs After**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Main init.py lines** | 533 lines | <100 lines | 80% reduction |
| **Import predictability** | Unpredictable fallbacks | Explicit dependencies | 100% predictable |
| **Registry interfaces** | Inconsistent duplicates | Single unified interface | Standardized |
| **API discoverability** | Flat list | Workflow-organized | User journey guided |
| **Time to first success** | Variable | <5 minutes | Guaranteed |
| **Contributor experience** | Fragmented | Standardized BaseRegistry | Unified |

### **North Star Axiom Compliance**

- ✅ **Axiom 1**: Simplicity Over Sophistication - 80% line reduction
- ✅ **Axiom 2**: Focus Over Feature Creep - Abandoned features deleted
- ✅ **Axiom 3**: Hooks Over Implementation - Registry extensibility preserved
- ✅ **Axiom 4**: Functions Over Frameworks - Direct function calls maintained
- ✅ **Axiom 5**: Performance Over Purity - No import overhead from fallbacks
- ✅ **Axiom 6**: Documentation Over Discovery - Workflow-oriented organization

---

## **🔧 Migration Strategy**

### **Zero Breaking Changes**
All existing function calls continue to work:
```python
# These all continue to work exactly the same
import brainsmith
result = brainsmith.forge('model.onnx', 'blueprint.yaml')
results = brainsmith.parameter_sweep(model, blueprint, params)
best = brainsmith.find_best(results, metric='throughput')
```

### **Enhanced Discoverability**
New users get better guidance:
```python
# New users get structured learning path
help_text = brainsmith.help()
quick_result = brainsmith.workflows.quick_dse(model, blueprint)
```

### **Contributor Benefits**
Open source contributors get standardized tooling:
```python
# Contributors get robust, standardized foundation
from brainsmith.core.registry import BaseRegistry
class MyCustomRegistry(BaseRegistry[MyComponentInfo]):
    # Standardized interface, validation, health checking built-in
```

---

## **📝 Documentation Structure**

```markdown
# BrainSmith API Reference

## 🎯 Core DSE (5-minute success)
Start here for your first FPGA accelerator design.

### forge(model_path, blueprint_path) → result
The North Star function: model + blueprint → optimized accelerator
**Example**: `result = brainsmith.forge('model.onnx', 'blueprint.yaml')`

### DesignSpace, DSEInterface, DSEMetrics
Core concepts for understanding design space exploration

## ⚡ Automation (15-minute success) 
Scale up to parameter exploration and batch processing.

### parameter_sweep(model, blueprint, params) → results
### batch_process(model_blueprint_pairs) → results
### find_best(results, metric) → best_result

## 📊 Analysis & Monitoring (30-minute success)
Integrate with your analysis workflows.

### get_analysis_data(results) → pandas_ready_data
### export_results(data, path) → file
### log_optimization_event(event, data)

## 🔧 Advanced Building (1-hour success)
Master-level accelerator generation.

### build_accelerator(model, blueprint) → finn_accelerator
### sample_design_space(space, strategy) → samples

## 🔌 Extensibility (contributors)
Add your own components to BrainSmith.

### BaseRegistry, ComponentInfo
Foundation for building custom registries
```

---

## **🎉 Final Result**

This clean refactor achieves the **North Star vision** through:

1. **Elimination of technical debt** (533-line fallback maze → explicit dependencies)
2. **Preservation of extensibility** (BaseRegistry foundation maintained)
3. **Workflow-oriented organization** (user journey → technical modules)
4. **Progressive learning structure** (5min → 15min → 30min → 1hr success)
5. **Zero breaking changes** (all existing code continues to work)

The result is a **simple, predictable, extensible** tool that embodies the North Star promise: **"Make FPGA accelerator design as simple as calling a function"** while providing clear growth paths for users and robust foundations for contributors.

---

*This plan transforms BrainSmith from a technically-organized module collection into a user-journey-guided workflow platform while maintaining all simplicity promises and extensibility capabilities.*