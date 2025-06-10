# 🔗 **Analysis Module Hooks Implementation Plan**
## BrainSmith API Simplification - Replace Analysis Framework with Hooks

**Date**: June 10, 2025  
**Implementation**: Hooks-Based Analysis Module  
**Goal**: Replace 2,100+ lines of custom analysis with ~100 lines of hooks  

---

## 🎯 **Implementation Overview**

Replace the entire `brainsmith/analysis` module with a lightweight hooks infrastructure that exposes structured data for external analysis tools instead of providing custom analysis implementations.

### **Current State:**
- 7 files, 2,100+ lines of custom analysis code
- 48+ exported classes/functions
- Complex academic features (ML, statistics, benchmarking)
- Heavy dependencies and maintenance burden

### **Target State:**
- 3 files, ~100 lines of hooks infrastructure
- 5-8 exported hook functions
- Data exposure for external tools
- Zero analysis algorithm maintenance

---

## 📋 **Implementation Steps**

### **Phase 1: Create Hooks Infrastructure** ⏱️ 30 minutes

#### **Step 1.1: Create New Module Structure**
```
brainsmith/analysis/
├── __init__.py          # 25 lines - hook exports
├── hooks.py            # 60 lines - data exposure hooks  
└── adapters.py         # 20 lines - external tool examples
```

#### **Step 1.2: Implement Core Hooks (`hooks.py`)**
```python
def expose_analysis_data(dse_results) -> Dict[str, Any]:
    """Expose structured data for external analysis tools."""

def register_analyzer(name: str, analyzer_func: Callable) -> None:
    """Register external analysis function."""

def get_raw_data(dse_results) -> Dict[str, np.ndarray]:
    """Get raw metric arrays for external processing."""

def export_to_dataframe(dse_results) -> 'pd.DataFrame':
    """Export to pandas-compatible format."""
```

#### **Step 1.3: Create External Tool Adapters (`adapters.py`)**
```python
def pandas_adapter(analysis_data) -> 'pd.DataFrame':
    """Convert to pandas DataFrame."""

def scipy_adapter(analysis_data) -> Dict[str, Any]:
    """Prepare data for scipy analysis."""
```

### **Phase 2: Remove Existing Files** ⏱️ 10 minutes

#### **Step 2.1: Delete Bloat Files**
- Delete `benchmarking.py` (553 lines)
- Delete `statistics.py` (422 lines) 
- Delete `prediction.py` (59 lines)
- Delete `models.py` (492 lines)
- Delete `engine.py` (528 lines)
- Keep `utils.py` but simplify drastically

#### **Step 2.2: Update Imports**
- Remove all complex exports from `__init__.py`
- Export only hook functions

### **Phase 3: Integration with Core API** ⏱️ 15 minutes

#### **Step 3.1: Update `forge()` Function**
```python
# In brainsmith/core/api.py forge() function
from ..analysis import expose_analysis_data

def forge(...) -> Dict[str, Any]:
    # ... existing implementation ...
    
    # Expose analysis data instead of custom analysis
    analysis_data = expose_analysis_data(dse_results)
    
    return {
        'dataflow_graph': dataflow_graph,
        'dataflow_core': dataflow_core if build_core else None,
        'metrics': basic_metrics,
        'analysis_data': analysis_data,  # NEW: Structured data for external tools
        'analysis_hooks': {              # NEW: Hook registration
            'register_analyzer': register_analyzer,
            'get_raw_data': lambda: get_raw_data(dse_results)
        }
    }
```

### **Phase 4: Documentation and Examples** ⏱️ 20 minutes

#### **Step 4.1: Create Usage Examples**
```python
# Examples for different external tools
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

# Get data from BrainSmith
results = brainsmith.forge(model, blueprint)
data = results['analysis_data']

# Use pandas for analysis
df = pd.DataFrame(data['solutions'])
summary = df.describe()

# Use scipy for statistics
throughput = df['throughput'].values
normality_test = stats.normaltest(throughput)

# Use scikit-learn for preprocessing
scaler = StandardScaler()
normalized = scaler.fit_transform(df[['throughput', 'latency']])
```

#### **Step 4.2: Update Documentation**
- Document hook functions in docstrings
- Create integration examples for popular libraries
- Show migration from old analysis API

---

## 📁 **Detailed File Implementations**

### **File: `brainsmith/analysis/__init__.py`**
```python
"""
BrainSmith Analysis Hooks

Provides hooks for external analysis tools instead of custom implementations.
Users can integrate pandas, scipy, scikit-learn, or any other analysis library.
"""

from .hooks import (
    expose_analysis_data,
    register_analyzer,
    get_raw_data,
    export_to_dataframe
)

from .adapters import (
    pandas_adapter,
    scipy_adapter
)

__version__ = "0.1.0"
__all__ = [
    'expose_analysis_data',
    'register_analyzer', 
    'get_raw_data',
    'export_to_dataframe',
    'pandas_adapter',
    'scipy_adapter'
]
```

### **File: `brainsmith/analysis/hooks.py`**
```python
"""
Core hooks for external analysis tool integration.
"""

import numpy as np
from typing import Dict, List, Any, Callable, Optional

# Registry for external analyzers
_analyzer_registry: Dict[str, Callable] = {}

def expose_analysis_data(dse_results) -> Dict[str, Any]:
    """
    Expose structured data for external analysis tools.
    
    Args:
        dse_results: DSE results from forge() function
        
    Returns:
        Structured data compatible with external analysis libraries
    """
    if not dse_results:
        return {'solutions': [], 'metrics': {}, 'pareto_frontier': []}
    
    # Extract solution data
    solutions = []
    for i, result in enumerate(dse_results):
        solution = {
            'id': i,
            'parameters': getattr(result, 'design_parameters', {}),
            'objectives': getattr(result, 'objective_values', []),
            'constraints': getattr(result, 'constraint_violations', []),
            'metadata': getattr(result, 'metadata', {})
        }
        solutions.append(solution)
    
    # Extract metric arrays
    metrics = {}
    if solutions:
        # Get all unique metric names
        metric_names = set()
        for sol in solutions:
            if 'objectives' in sol and sol['objectives']:
                for i, val in enumerate(sol['objectives']):
                    metric_names.add(f'objective_{i}')
        
        # Create metric arrays
        for metric in metric_names:
            values = []
            for sol in solutions:
                if 'objectives' in sol and sol['objectives']:
                    obj_idx = int(metric.split('_')[1])
                    if obj_idx < len(sol['objectives']):
                        values.append(sol['objectives'][obj_idx])
            if values:
                metrics[metric] = np.array(values)
    
    # Find Pareto frontier (simplified)
    pareto_indices = _find_pareto_frontier(solutions)
    
    return {
        'solutions': solutions,
        'metrics': metrics,
        'pareto_frontier': pareto_indices,
        'metadata': {
            'num_solutions': len(solutions),
            'num_metrics': len(metrics),
            'data_format': 'brainsmith_v1'
        }
    }

def register_analyzer(name: str, analyzer_func: Callable) -> None:
    """Register external analysis function."""
    _analyzer_registry[name] = analyzer_func

def get_registered_analyzers() -> Dict[str, Callable]:
    """Get all registered analyzers."""
    return _analyzer_registry.copy()

def get_raw_data(dse_results) -> Dict[str, np.ndarray]:
    """Get raw metric arrays for external processing."""
    analysis_data = expose_analysis_data(dse_results)
    return analysis_data['metrics']

def export_to_dataframe(dse_results) -> Optional['pd.DataFrame']:
    """Export to pandas-compatible format."""
    try:
        import pandas as pd
        analysis_data = expose_analysis_data(dse_results)
        
        # Flatten solution data for DataFrame
        flattened_data = []
        for sol in analysis_data['solutions']:
            row = {'solution_id': sol['id']}
            
            # Add parameters
            for param, value in sol.get('parameters', {}).items():
                row[f'param_{param}'] = value
            
            # Add objectives
            for i, obj_val in enumerate(sol.get('objectives', [])):
                row[f'objective_{i}'] = obj_val
            
            # Add constraints
            for i, const_val in enumerate(sol.get('constraints', [])):
                row[f'constraint_{i}'] = const_val
            
            flattened_data.append(row)
        
        return pd.DataFrame(flattened_data)
        
    except ImportError:
        return None

def _find_pareto_frontier(solutions: List[Dict[str, Any]]) -> List[int]:
    """Find Pareto frontier indices (simplified implementation)."""
    if not solutions:
        return []
    
    pareto_indices = []
    
    for i, sol_i in enumerate(solutions):
        is_pareto = True
        objectives_i = sol_i.get('objectives', [])
        
        if not objectives_i:
            continue
            
        for j, sol_j in enumerate(solutions):
            if i == j:
                continue
                
            objectives_j = sol_j.get('objectives', [])
            if not objectives_j or len(objectives_j) != len(objectives_i):
                continue
            
            # Check if sol_j dominates sol_i (assuming minimization)
            dominates = True
            for obj_i, obj_j in zip(objectives_i, objectives_j):
                if obj_j >= obj_i:  # Not better in this objective
                    dominates = False
                    break
            
            if dominates:
                is_pareto = False
                break
        
        if is_pareto:
            pareto_indices.append(i)
    
    return pareto_indices
```

### **File: `brainsmith/analysis/adapters.py`**
```python
"""
Adapters for external analysis tools.
"""

from typing import Dict, Any, Optional

def pandas_adapter(analysis_data: Dict[str, Any]) -> Optional['pd.DataFrame']:
    """Convert analysis data to pandas DataFrame."""
    try:
        import pandas as pd
        
        solutions = analysis_data.get('solutions', [])
        if not solutions:
            return pd.DataFrame()
        
        # Flatten data for DataFrame
        rows = []
        for sol in solutions:
            row = {'solution_id': sol['id']}
            
            # Add parameters with prefix
            for param, value in sol.get('parameters', {}).items():
                row[f'param_{param}'] = value
            
            # Add objectives
            for i, obj_val in enumerate(sol.get('objectives', [])):
                row[f'objective_{i}'] = obj_val
                
            rows.append(row)
        
        return pd.DataFrame(rows)
        
    except ImportError:
        return None

def scipy_adapter(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare data for scipy analysis."""
    metrics = analysis_data.get('metrics', {})
    
    # Return metrics in scipy-friendly format
    return {
        'arrays': metrics,
        'sample_size': len(next(iter(metrics.values()))) if metrics else 0,
        'metric_names': list(metrics.keys())
    }
```

---

## 🧪 **Testing Strategy**

### **Create Test File: `tests/test_analysis_hooks.py`**
```python
def test_expose_analysis_data():
    """Test data exposure functionality."""
    
def test_register_analyzer():
    """Test external analyzer registration."""
    
def test_pandas_integration():
    """Test pandas DataFrame export."""
    
def test_scipy_integration():
    """Test scipy data format."""
```

---

## 📊 **Migration Impact**

### **Before (Current State):**
- **Files**: 7 files, 2,100+ lines
- **Exports**: 48+ classes/functions
- **Dependencies**: scipy, sklearn (if fully implemented)
- **Maintenance**: High (custom algorithms)

### **After (Hooks Implementation):**
- **Files**: 3 files, ~105 lines total
- **Exports**: 6 hook functions
- **Dependencies**: None (optional pandas for export)
- **Maintenance**: Minimal (just data exposure)

### **Benefits:**
- **95% code reduction**
- **Zero algorithm maintenance**
- **Better external tool support**
- **User flexibility**
- **Future-proof architecture**

---

## ⏱️ **Implementation Timeline**

| Phase | Duration | Tasks |
|-------|----------|-------|
| Phase 1 | 30 min | Create hooks infrastructure |
| Phase 2 | 10 min | Remove existing files |
| Phase 3 | 15 min | Integrate with forge() function |
| Phase 4 | 20 min | Documentation and examples |
| **Total** | **75 min** | **Complete implementation** |

---

## 🎯 **Success Criteria**

1. ✅ **Reduced complexity**: 2,100+ lines → ~105 lines
2. ✅ **Simplified API**: 48+ exports → 6 hooks
3. ✅ **External tool support**: pandas, scipy, sklearn integration
4. ✅ **Zero maintenance burden**: No custom analysis algorithms
5. ✅ **Preserved functionality**: Data still accessible for analysis
6. ✅ **Better user experience**: Choice of analysis tools

**Ready to begin implementation!** 🚀