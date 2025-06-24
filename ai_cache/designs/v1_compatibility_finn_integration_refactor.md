# V1 Removal & FINN Architecture Refactoring Proposal

## Executive Summary

This proposal addresses two critical architectural issues in the Brainsmith core infrastructure:
1. Complete removal of V1 compatibility layer - no migration tools, no deprecation, just deletion
2. Deep FINN refactoring to separate 6-entrypoint and Legacy workflows into distinct paths

## Issue 1: V1 Compatibility Removal

### Current State
The V1 compatibility layer in `api.py` (lines 401-587) represents dead code that:
- Accepts parameters that are silently ignored
- Returns mock results on failure
- Duplicates result formatting logic
- Adds ~186 lines of maintenance burden

### Proposed Solution: Complete Removal

#### Files to Modify
1. **`brainsmith/core/api.py`**
   - Delete `forge_v1_compat()` function (lines 401-587)
   - Delete `_convert_v1_objectives()` helper
   - Delete `_format_v1_result()` helper
   - Remove V1-related imports

2. **`brainsmith/core/finn/legacy_conversion.py`**
   - Delete entire file if only used by V1 compat
   - Or remove V1-specific conversion methods

3. **Tests**
   - Delete all V1 compatibility tests
   - Update integration tests to use V2 API only

#### Clean API Surface
After removal, the API becomes:
```python
# brainsmith/core/api.py
def forge(
    model_path: str,
    blueprint_path: str,
    output_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    max_workers: Optional[int] = None
) -> ForgeResult:
    """Single, clean entry point for model compilation."""
    # Pure V2 implementation
```

No compatibility layers. No migration paths. Just clean, modern code.

## Issue 2: FINN Workflow Separation

### Current State Analysis

The current FINN integration mixes two fundamentally different workflows:
1. **6-Entrypoint Workflow**: Modern, component-based DSE approach
2. **Legacy Workflow**: Direct build step specification

These are conflated in a single `FINNEvaluationBridge` with complex conditionals.

### Proposed Solution: Workflow-Specific Backends

#### Architecture Overview
```
blueprint.yaml
    |
    v
Workflow Detector
    |
    ├─> 6-Entrypoint Backend (DSE-based)
    |
    └─> Legacy Backend (Direct steps)
```

#### Step 1: Workflow Detection
```python
# brainsmith/core/backends/workflow_detector.py
from enum import Enum
from typing import Dict, Any

class WorkflowType(Enum):
    SIX_ENTRYPOINT = "six_entrypoint"
    LEGACY = "legacy"

def detect_workflow(blueprint: Dict[str, Any]) -> WorkflowType:
    """Detect workflow type from blueprint structure."""
    
    # Legacy blueprints specify build steps directly
    if 'build_steps' in blueprint.get('finn_config', {}):
        return WorkflowType.LEGACY
    
    # Legacy blueprints have explicit step sequences
    if 'dataflow_steps' in blueprint.get('finn_config', {}):
        return WorkflowType.LEGACY
        
    # 6-entrypoint uses component spaces
    if all(key in blueprint for key in ['nodes', 'transforms']):
        return WorkflowType.SIX_ENTRYPOINT
    
    raise ValueError(
        "Unable to detect workflow type. Blueprint must either define "
        "'build_steps' for legacy workflow or 'nodes'/'transforms' for "
        "6-entrypoint workflow."
    )
```

#### Step 2: Abstract Backend Interface
```python
# brainsmith/core/backends/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class EvaluationRequest:
    """Request for hardware evaluation."""
    model_path: str
    combination: Dict[str, Any]  # Either components or build steps
    work_dir: str
    timeout: Optional[int] = None

@dataclass
class EvaluationResult:
    """Result of hardware evaluation."""
    success: bool
    metrics: Dict[str, float]
    reports: Dict[str, str]  # report_type -> path
    error: Optional[str] = None
    
class EvaluationBackend(ABC):
    """Abstract backend for hardware evaluation."""
    
    @abstractmethod
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Evaluate hardware configuration."""
        pass
```

#### Step 3: 6-Entrypoint Backend
```python
# brainsmith/core/backends/six_entrypoint.py
import json
import subprocess
from pathlib import Path
from typing import Dict, Any

class SixEntrypointBackend(EvaluationBackend):
    """Modern 6-entrypoint FINN workflow."""
    
    def __init__(self, blueprint_config: Dict[str, Any]):
        self.entrypoint_mapping = blueprint_config.get('entrypoint_mapping', {})
        self.device_config = blueprint_config.get('device', {})
        
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Evaluate using 6-entrypoint workflow."""
        
        try:
            # Generate entrypoint configuration
            entrypoint_config = self._generate_entrypoint_config(
                request.combination,
                request.work_dir
            )
            
            # Execute FINN with 6-entrypoint flow
            finn_result = self._execute_finn_six_entrypoint(
                model_path=request.model_path,
                config_path=entrypoint_config,
                work_dir=request.work_dir,
                timeout=request.timeout
            )
            
            # Extract metrics from structured output
            metrics = self._extract_metrics(finn_result)
            reports = self._collect_reports(request.work_dir)
            
            return EvaluationResult(
                success=True,
                metrics=metrics,
                reports=reports
            )
            
        except Exception as e:
            return EvaluationResult(
                success=False,
                metrics={},
                reports={},
                error=str(e)
            )
    
    def _generate_entrypoint_config(
        self, 
        combination: Dict[str, Any],
        work_dir: str
    ) -> str:
        """Generate 6-entrypoint configuration file."""
        
        config = {
            "version": "2.0",
            "entrypoints": {
                # Map combination components to entrypoints
                "canonical_ops": combination.get('node_components', {}).get('canonical_ops', []),
                "hw_kernels": combination.get('node_components', {}).get('hw_kernels', []),
                "model_topology": combination.get('transform_components', {}).get('model_topology', []),
                "dataflow_partitioning": combination.get('transform_components', {}).get('dataflow_partitioning', []),
                "hw_kernel_transforms": combination.get('transform_components', {}).get('hw_kernel_transforms', []),
                "board_deployment": combination.get('transform_components', {}).get('board_deployment', [])
            },
            "device": self.device_config,
            "output_dir": work_dir
        }
        
        config_path = Path(work_dir) / "entrypoint_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        return str(config_path)
    
    def _execute_finn_six_entrypoint(
        self,
        model_path: str,
        config_path: str,
        work_dir: str,
        timeout: Optional[int]
    ) -> Dict[str, Any]:
        """Execute FINN with 6-entrypoint workflow."""
        
        cmd = [
            "python", "-m", "finn.builder.build_dataflow_v2",
            "--model", model_path,
            "--entrypoint-config", config_path,
            "--work-dir", work_dir,
            "--json-metrics"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"FINN 6-entrypoint failed: {result.stderr}")
            
        return json.loads(result.stdout)
```

#### Step 4: Legacy Backend
```python
# brainsmith/core/backends/legacy_finn.py
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List

class LegacyFINNBackend(EvaluationBackend):
    """Legacy FINN workflow with direct build steps."""
    
    def __init__(self, blueprint_config: Dict[str, Any]):
        self.finn_config = blueprint_config.get('finn_config', {})
        self.build_steps = self.finn_config.get('build_steps', [])
        
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Evaluate using legacy FINN workflow."""
        
        try:
            # Generate legacy dataflow config
            dataflow_config = self._generate_dataflow_config(
                build_steps=self.build_steps,
                model_path=request.model_path,
                work_dir=request.work_dir
            )
            
            # Execute legacy FINN build
            finn_result = self._execute_finn_legacy(
                dataflow_config=dataflow_config,
                work_dir=request.work_dir,
                timeout=request.timeout
            )
            
            # Extract metrics from legacy output
            metrics = self._extract_legacy_metrics(finn_result, request.work_dir)
            reports = self._collect_legacy_reports(request.work_dir)
            
            return EvaluationResult(
                success=True,
                metrics=metrics,
                reports=reports
            )
            
        except Exception as e:
            return EvaluationResult(
                success=False,
                metrics={},
                reports={},
                error=str(e)
            )
    
    def _generate_dataflow_config(
        self,
        build_steps: List[str],
        model_path: str,
        work_dir: str
    ) -> str:
        """Generate legacy dataflow build config."""
        
        # Legacy config format
        config = {
            "model_file": model_path,
            "output_dir": work_dir,
            "steps": build_steps,
            "save_intermediate_models": True,
            **self.finn_config  # Include other legacy options
        }
        
        config_path = Path(work_dir) / "dataflow_build_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        return str(config_path)
    
    def _execute_finn_legacy(
        self,
        dataflow_config: str,
        work_dir: str,
        timeout: Optional[int]
    ) -> Dict[str, Any]:
        """Execute legacy FINN build."""
        
        cmd = [
            "python", "-m", "finn.builder.build_dataflow",
            "--config", dataflow_config
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Legacy FINN build failed: {result.stderr}")
            
        # Legacy FINN doesn't output JSON, parse from files
        return self._parse_legacy_output(work_dir)
```

#### Step 5: Factory Pattern for Backend Selection
```python
# brainsmith/core/backends/factory.py
from typing import Dict, Any
from .workflow_detector import detect_workflow, WorkflowType
from .six_entrypoint import SixEntrypointBackend
from .legacy_finn import LegacyFINNBackend
from .base import EvaluationBackend

def create_backend(blueprint: Dict[str, Any]) -> EvaluationBackend:
    """Create appropriate backend based on blueprint."""
    
    workflow_type = detect_workflow(blueprint)
    
    if workflow_type == WorkflowType.SIX_ENTRYPOINT:
        return SixEntrypointBackend(blueprint)
    elif workflow_type == WorkflowType.LEGACY:
        return LegacyFINNBackend(blueprint)
    else:
        raise ValueError(f"Unknown workflow type: {workflow_type}")
```

#### Step 6: Updated Evaluation Bridge
```python
# brainsmith/core/finn/evaluation_bridge.py
from typing import Dict, Any
import tempfile
from ..backends.factory import create_backend
from ..backends.base import EvaluationRequest

class FINNEvaluationBridge:
    """Bridge between DSE and backend evaluation."""
    
    def __init__(self, blueprint: Dict[str, Any]):
        self.backend = create_backend(blueprint)
        self.blueprint = blueprint
        
    def evaluate(
        self,
        combination: ComponentCombination,
        model_path: str
    ) -> Dict[str, Any]:
        """Evaluate combination using appropriate backend."""
        
        with tempfile.TemporaryDirectory() as work_dir:
            # Create evaluation request
            request = EvaluationRequest(
                model_path=model_path,
                combination=combination.to_dict(),
                work_dir=work_dir,
                timeout=self.blueprint.get('evaluation_timeout', 3600)
            )
            
            # Execute evaluation
            result = self.backend.evaluate(request)
            
            # No mock results - fail honestly
            if not result.success:
                raise RuntimeError(
                    f"Hardware evaluation failed: {result.error}"
                )
            
            return {
                'success': True,
                'metrics': result.metrics,
                'reports': result.reports
            }
```

## Benefits of This Approach

### V1 Removal Benefits
1. **186 lines of code deleted** - Immediate reduction in maintenance burden
2. **No confusion** - Single API with clear semantics
3. **No hidden behaviors** - No more ignored parameters or mock results
4. **Simplified testing** - Only one code path to test

### FINN Refactoring Benefits
1. **Clear separation** - Each workflow has its own backend
2. **No conditionals** - Workflow detection happens once at initialization
3. **Easier evolution** - Can update workflows independently
4. **Better error messages** - Each backend can provide workflow-specific errors
5. **Future proof** - Easy to add new workflow types

## Implementation Plan

### Week 1: V1 Removal
- Delete all V1 compatibility code
- Update tests to remove V1 coverage
- Verify demos work with V2 API only

### Week 2: Backend Architecture
- Implement abstract backend interface
- Create workflow detection logic
- Set up factory pattern

### Week 3: 6-Entrypoint Backend
- Implement modern workflow backend
- Test with existing blueprints
- Verify metrics extraction

### Week 4: Legacy Backend
- Implement legacy workflow backend
- Test with legacy blueprints
- Ensure backward compatibility for legacy blueprints only

## Risk Mitigation

1. **Breaking Changes**: V1 removal will break any code using old API
   - Mitigation: Clear documentation of V2 API
   - Benefit: Forces users to modern, supported API

2. **Workflow Detection**: Must correctly identify blueprint type
   - Mitigation: Clear error messages if detection fails
   - Benefit: Explicit blueprint requirements

3. **Backend Differences**: Workflows may produce different outputs
   - Mitigation: Standardized result format
   - Benefit: Clear separation of concerns

## Success Metrics

- 186+ lines of dead code removed
- Zero V1 API calls remaining in codebase
- Two distinct backend implementations
- 100% of blueprints correctly routed to appropriate backend
- No mock results ever returned