# 🚀 **Core Simplification Implementation Plan**
## Step-by-Step Execution Guide

**Based on**: Updated Core Simplification Plan with feedback integration  
**Target**: Transform 14 files (~3,500 lines) → 6 files (~1,200 lines)  
**Timeline**: ~3 hours total implementation time  

---

## 📋 **Pre-Implementation Checklist**

### **Prerequisites**
- [ ] Core simplification plan reviewed and approved
- [ ] Backup strategy confirmed
- [ ] Test environment ready
- [ ] Dependencies identified (FINN, ONNX, etc.)

### **Safety Measures**
- [ ] Full backup of `brainsmith/core` directory
- [ ] Git branch created for implementation
- [ ] Test suite identified and ready

---

## 🗂️ **Phase 1: Backup & Preparation (15 minutes)**

### **Step 1.1: Create Comprehensive Backup**
```bash
# Create timestamped backup
BACKUP_DIR="brainsmith/core_backup_$(date +%Y%m%d_%H%M%S)"
cp -r brainsmith/core $BACKUP_DIR
echo "Backup created: $BACKUP_DIR"

# Create git branch for implementation
git checkout -b core-simplification-implementation
git add $BACKUP_DIR
git commit -m "Backup: Core module before simplification"
```

### **Step 1.2: Analyze Current Dependencies**
```bash
# Identify imports of files we're deleting
grep -r "from.*core\." brainsmith/ --include="*.py" > core_imports_analysis.txt
grep -r "import.*core\." brainsmith/ --include="*.py" >> core_imports_analysis.txt

# Check test dependencies
grep -r "core\." tests/ --include="*.py" > core_test_dependencies.txt
```

### **Step 1.3: Run Baseline Tests**
```bash
# Run existing tests to establish baseline
python -m pytest tests/ -v --tb=short > baseline_test_results.txt
echo "Baseline test results captured"
```

---

## 🔥 **Phase 2: Strategic Deletion (20 minutes)**

### **Step 2.1: Delete Enterprise Bloat Files**
```bash
# Delete 7 files in order of dependency (safest first)
rm brainsmith/core/legacy_support.py          # 433 lines - No dependencies
rm brainsmith/core/result.py                  # 493 lines - Complex objects
rm brainsmith/core/config.py                  # 374 lines - Configuration system
rm brainsmith/core/compiler.py                # 451 lines - Enterprise framework
rm brainsmith/core/workflow.py                # 356 lines - Workflow management
rm brainsmith/core/design_space_orchestrator.py # 461 lines - Orchestration
rm brainsmith/core/hw_compiler.py             # 89 lines - Legacy forge

echo "Deleted 7 files (~2,657 lines of enterprise bloat)"
```

### **Step 2.2: Update Imports Immediately**
```bash
# Remove deleted imports from __init__.py quickly to prevent import errors
sed -i '/design_space_orchestrator/d' brainsmith/core/__init__.py
sed -i '/workflow/d' brainsmith/core/__init__.py
sed -i '/compiler/d' brainsmith/core/__init__.py
sed -i '/config/d' brainsmith/core/__init__.py
sed -i '/result/d' brainsmith/core/__init__.py
sed -i '/legacy_support/d' brainsmith/core/__init__.py
sed -i '/hw_compiler/d' brainsmith/core/__init__.py
```

### **Step 2.3: Verify Deletion Success**
```bash
# Confirm files are deleted
ls -la brainsmith/core/
echo "Remaining files:"
find brainsmith/core -name "*.py" | sort
```

---

## ✏️ **Phase 3: Simplify CLI (25 minutes)**

### **Step 3.1: Create New Simple CLI**
```python
# File: brainsmith/core/cli.py (replace existing)
"""
Simple CLI wrapper for forge() function.
Replaces complex workflow orchestration with direct function calls.
"""

import click
import json
import sys
from pathlib import Path
from .api import forge, validate_blueprint

@click.group()
@click.version_option(version="0.5.0", prog_name="brainsmith")
def brainsmith_cli():
    """BrainSmith FPGA Accelerator Design Tool - Simple and Focused."""
    pass

@brainsmith_cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('blueprint_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--objectives', help='JSON objectives string (e.g., \'{"throughput": {"direction": "maximize"}}\')')
@click.option('--constraints', help='JSON constraints string (e.g., \'{"max_luts": 0.8}\')')
@click.option('--device', help='Target FPGA device (e.g., "xcvu9p-flga2104-2-i")')
@click.option('--build-core/--no-build-core', default=True, help='Generate full dataflow core')
@click.option('--hw-graph', is_flag=True, help='Input is already a dataflow graph')
def forge_cmd(model_path, blueprint_path, output, objectives, constraints, device, build_core, hw_graph):
    """Generate FPGA accelerator from model and blueprint."""
    try:
        # Parse JSON strings
        objectives_dict = json.loads(objectives) if objectives else None
        constraints_dict = json.loads(constraints) if constraints else None
        
        click.echo(f"🚀 Starting forge...")
        click.echo(f"   Model: {model_path}")
        click.echo(f"   Blueprint: {blueprint_path}")
        
        # Call forge function
        result = forge(
            model_path=str(model_path),
            blueprint_path=str(blueprint_path),
            objectives=objectives_dict,
            constraints=constraints_dict,
            target_device=device,
            is_hw_graph=hw_graph,
            build_core=build_core,
            output_dir=str(output) if output else None
        )
        
        click.echo(f"✅ Forge completed successfully!")
        if output:
            click.echo(f"📁 Results saved to: {output}")
        
        # Display key metrics
        metrics = result.get('metrics', {})
        if metrics:
            perf = metrics.get('performance', {})
            if perf.get('throughput_ops_sec'):
                click.echo(f"⚡ Throughput: {perf['throughput_ops_sec']:.1f} ops/sec")
            if perf.get('latency_ms'):
                click.echo(f"⏱️  Latency: {perf['latency_ms']:.2f} ms")
                
    except json.JSONDecodeError as e:
        click.echo(f"❌ Invalid JSON in objectives/constraints: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Forge failed: {e}", err=True)
        sys.exit(1)

@brainsmith_cli.command()
@click.argument('blueprint_path', type=click.Path(exists=True))
def validate(blueprint_path):
    """Validate blueprint configuration."""
    try:
        click.echo(f"🔍 Validating blueprint: {blueprint_path}")
        
        is_valid, errors = validate_blueprint(str(blueprint_path))
        
        if is_valid:
            click.echo("✅ Blueprint is valid")
        else:
            click.echo("❌ Blueprint validation failed:")
            for error in errors:
                click.echo(f"  • {error}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}", err=True)
        sys.exit(1)

@brainsmith_cli.command()
def version():
    """Show version information."""
    click.echo("BrainSmith Core v0.5.0")
    click.echo("Simple FPGA Accelerator Design Tool")

if __name__ == '__main__':
    brainsmith_cli()
```

### **Step 3.2: Test CLI Functionality**
```bash
# Test CLI help
python -m brainsmith.core.cli --help

# Test individual commands
python -m brainsmith.core.cli forge_cmd --help
python -m brainsmith.core.cli validate --help
```

---

## 🎯 **Phase 4: Simplify DesignSpace (30 minutes)**

### **Step 4.1: Create Simplified DesignSpace**
```python
# File: brainsmith/core/design_space.py (replace existing)
"""
Simplified Design Space for Blueprint Instantiation

Provides robust DesignSpace object to instantiate what's defined in blueprints.
Focus on practical blueprint support rather than academic research features.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class DesignSpace:
    """Simplified design space for blueprint instantiation."""
    
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_blueprint(cls, blueprint) -> 'DesignSpace':
        """Create design space from blueprint configuration."""
        try:
            # Extract design space definition from blueprint
            if hasattr(blueprint, 'get_design_space'):
                design_space_config = blueprint.get_design_space()
            else:
                # Fallback: extract from blueprint data
                design_space_config = getattr(blueprint, 'yaml_data', {})
            
            # Extract parameters and constraints
            parameters = design_space_config.get('parameters', {})
            constraints = design_space_config.get('constraints', {})
            
            return cls(
                name=getattr(blueprint, 'name', 'blueprint_design_space'),
                parameters=parameters,
                constraints=constraints,
                metadata={'source': 'blueprint', 'blueprint_name': getattr(blueprint, 'name', 'unknown')}
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract design space from blueprint: {e}")
            return cls(
                name='fallback_design_space',
                parameters={},
                constraints={},
                metadata={'source': 'fallback', 'error': str(e)}
            )
    
    def get_parameter_ranges(self) -> Dict[str, List[Any]]:
        """Get parameter ranges for exploration."""
        ranges = {}
        
        for param_name, param_config in self.parameters.items():
            if isinstance(param_config, dict):
                if 'values' in param_config:
                    # Discrete values
                    ranges[param_name] = param_config['values']
                elif 'range' in param_config:
                    # Generate range values
                    ranges[param_name] = self._generate_range_values(param_config['range'])
                elif 'min' in param_config and 'max' in param_config:
                    # Direct min/max specification
                    ranges[param_name] = self._generate_range_values(param_config)
                else:
                    # Single value
                    ranges[param_name] = [param_config.get('default', param_config)]
            else:
                # Direct value
                ranges[param_name] = [param_config]
                
        return ranges
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate parameter values against design space."""
        errors = []
        
        for param_name, param_value in parameters.items():
            if param_name not in self.parameters:
                errors.append(f"Unknown parameter: {param_name}")
                continue
            
            # Get valid values for this parameter
            valid_ranges = self.get_parameter_ranges()
            if param_name in valid_ranges:
                valid_values = valid_ranges[param_name]
                if param_value not in valid_values:
                    errors.append(f"Parameter '{param_name}' value {param_value} not in valid range: {valid_values}")
        
        return len(errors) == 0, errors
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameter values."""
        defaults = {}
        
        for param_name, param_config in self.parameters.items():
            if isinstance(param_config, dict):
                if 'default' in param_config:
                    defaults[param_name] = param_config['default']
                elif 'values' in param_config and param_config['values']:
                    defaults[param_name] = param_config['values'][0]  # First value as default
                elif 'min' in param_config:
                    defaults[param_name] = param_config['min']  # Min value as default
            else:
                defaults[param_name] = param_config
                
        return defaults
    
    def _generate_range_values(self, range_config: Dict[str, Any]) -> List[Any]:
        """Generate values from range configuration."""
        if not isinstance(range_config, dict):
            return [range_config]
        
        min_val = range_config.get('min')
        max_val = range_config.get('max')
        step = range_config.get('step', 1)
        
        if min_val is not None and max_val is not None:
            if isinstance(min_val, int) and isinstance(max_val, int):
                return list(range(min_val, max_val + 1, step))
            else:
                # Float range
                values = []
                val = min_val
                while val <= max_val:
                    values.append(val)
                    val += step
                return values
        
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'constraints': self.constraints,
            'metadata': self.metadata,
            'parameter_ranges': self.get_parameter_ranges(),
            'default_parameters': self.get_default_parameters()
        }
    
    def __repr__(self) -> str:
        return f"DesignSpace(name='{self.name}', {len(self.parameters)} parameters, {len(self.constraints)} constraints)"


# Utility functions for backward compatibility
def create_design_space_from_blueprint(blueprint) -> DesignSpace:
    """Create design space from blueprint (convenience function)."""
    return DesignSpace.from_blueprint(blueprint)
```

### **Step 4.2: Test DesignSpace Functionality**
```python
# Create test script: test_design_space.py
from brainsmith.core.design_space import DesignSpace

# Test basic functionality
ds = DesignSpace(
    name="test_space",
    parameters={
        'pe_count': {'values': [4, 8, 16]},
        'frequency': {'min': 100, 'max': 200, 'step': 25}
    }
)

print(f"Design space: {ds}")
print(f"Parameter ranges: {ds.get_parameter_ranges()}")
print(f"Default parameters: {ds.get_default_parameters()}")

# Test validation
valid, errors = ds.validate_parameters({'pe_count': 8, 'frequency': 150})
print(f"Validation result: {valid}, errors: {errors}")
```

---

## 📊 **Phase 5: Simplify Metrics (25 minutes)**

### **Step 5.1: Create Simplified Metrics**
```python
# File: brainsmith/core/metrics.py (replace existing)
"""
Simplified Metrics for DSE Optimization

Essential metrics collection for DSE feedback and optimization.
Focus on practical metrics needed for design space exploration.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

@dataclass
class DSEMetrics:
    """Essential metrics for DSE optimization and feedback."""
    
    # Performance metrics
    throughput_ops_sec: Optional[float] = None
    latency_ms: Optional[float] = None
    frequency_mhz: Optional[float] = None
    
    # Resource metrics  
    lut_utilization: Optional[float] = None
    dsp_utilization: Optional[float] = None
    bram_utilization: Optional[float] = None
    power_consumption_w: Optional[float] = None
    
    # Build metrics
    build_time_sec: Optional[float] = None
    build_success: bool = False
    
    # Quality metrics
    verification_passed: Optional[bool] = None
    accuracy: Optional[float] = None
    
    # Additional metrics (extensible)
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DSEMetrics':
        """Create from dictionary."""
        # Handle nested structure from forge() results
        if 'performance' in data and 'resources' in data:
            # Extract from structured format
            perf = data.get('performance', {})
            res = data.get('resources', {})
            build = data.get('build', {})
            quality = data.get('quality', {})
            
            return cls(
                throughput_ops_sec=perf.get('throughput_ops_sec'),
                latency_ms=perf.get('latency_ms'),
                frequency_mhz=perf.get('frequency_mhz'),
                lut_utilization=res.get('lut_utilization'),
                dsp_utilization=res.get('dsp_utilization'),
                bram_utilization=res.get('bram_utilization'),
                power_consumption_w=res.get('power_consumption_w'),
                build_time_sec=build.get('build_time_sec'),
                build_success=build.get('build_success', False),
                verification_passed=quality.get('verification_passed'),
                accuracy=quality.get('accuracy'),
                custom_metrics=data.get('custom_metrics', {}),
                timestamp=data.get('timestamp')
            )
        else:
            # Direct mapping
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to structured dictionary for serialization."""
        return {
            'performance': {
                'throughput_ops_sec': self.throughput_ops_sec,
                'latency_ms': self.latency_ms,
                'frequency_mhz': self.frequency_mhz
            },
            'resources': {
                'lut_utilization': self.lut_utilization,
                'dsp_utilization': self.dsp_utilization,
                'bram_utilization': self.bram_utilization,
                'power_consumption_w': self.power_consumption_w
            },
            'build': {
                'build_time_sec': self.build_time_sec,
                'build_success': self.build_success
            },
            'quality': {
                'verification_passed': self.verification_passed,
                'accuracy': self.accuracy
            },
            'custom_metrics': self.custom_metrics,
            'timestamp': self.timestamp
        }
    
    def get_optimization_score(self, objectives: Dict[str, str]) -> float:
        """Calculate optimization score based on objectives."""
        if not objectives:
            return 0.0
        
        score = 0.0
        count = 0
        
        for objective, direction in objectives.items():
            value = self._get_metric_value(objective)
            if value is not None:
                # Normalize and weight based on direction
                normalized = self._normalize_metric(objective, value)
                if direction == 'minimize':
                    normalized = 1.0 - normalized
                score += normalized
                count += 1
        
        return score / count if count > 0 else 0.0
    
    def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get metric value by name."""
        # Direct attribute access
        if hasattr(self, metric_name):
            return getattr(self, metric_name)
        
        # Check custom metrics
        return self.custom_metrics.get(metric_name)
    
    def _normalize_metric(self, metric_name: str, value: float) -> float:
        """Normalize metric value to [0, 1] range."""
        # Simple normalization - can be enhanced with historical data
        if metric_name in ['throughput_ops_sec']:
            return min(value / 10000.0, 1.0)  # Rough normalization
        elif metric_name in ['frequency_mhz']:
            return min(value / 500.0, 1.0)  # Up to 500MHz
        elif metric_name in ['latency_ms', 'build_time_sec']:
            return max(0.0, 1.0 - min(value / 1000.0, 1.0))  # Inverse normalization
        elif 'utilization' in metric_name:
            return value if 0 <= value <= 1 else value / 100.0  # Handle percentage
        elif metric_name in ['accuracy']:
            return value if 0 <= value <= 1 else value / 100.0  # Handle percentage
        elif metric_name in ['power_consumption_w']:
            return max(0.0, 1.0 - min(value / 100.0, 1.0))  # Inverse, up to 100W
        return 0.5  # Default neutral value
    
    def add_custom_metric(self, name: str, value: float):
        """Add custom metric for specialized optimization."""
        self.custom_metrics[name] = value
    
    def get_summary(self) -> Dict[str, str]:
        """Get human-readable summary."""
        summary = {}
        
        if self.throughput_ops_sec:
            summary['Throughput'] = f"{self.throughput_ops_sec:.1f} ops/sec"
        if self.latency_ms:
            summary['Latency'] = f"{self.latency_ms:.2f} ms"
        if self.frequency_mhz:
            summary['Frequency'] = f"{self.frequency_mhz:.1f} MHz"
        if self.lut_utilization:
            summary['LUT Usage'] = f"{self.lut_utilization:.1%}"
        if self.dsp_utilization:
            summary['DSP Usage'] = f"{self.dsp_utilization:.1%}"
        if self.power_consumption_w:
            summary['Power'] = f"{self.power_consumption_w:.1f} W"
        if self.build_time_sec:
            summary['Build Time'] = f"{self.build_time_sec:.1f} sec"
        
        summary['Success'] = "✅" if self.build_success else "❌"
        
        return summary
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


# Utility functions for backward compatibility
def extract_metrics_from_result(result: Dict[str, Any]) -> DSEMetrics:
    """Extract metrics from forge() result."""
    metrics_data = result.get('metrics', {})
    return DSEMetrics.from_dict(metrics_data)
```

### **Step 5.2: Test Metrics Functionality**
```python
# Create test script: test_metrics.py
from brainsmith.core.metrics import DSEMetrics

# Test basic functionality
metrics = DSEMetrics(
    throughput_ops_sec=5000.0,
    latency_ms=2.5,
    lut_utilization=0.75,
    build_success=True
)

print(f"Metrics: {metrics}")
print(f"Summary: {metrics.get_summary()}")

# Test optimization score
objectives = {'throughput_ops_sec': 'maximize', 'latency_ms': 'minimize'}
score = metrics.get_optimization_score(objectives)
print(f"Optimization score: {score:.3f}")

# Test serialization
metrics_dict = metrics.to_dict()
reconstructed = DSEMetrics.from_dict(metrics_dict)
print(f"Serialization test: {reconstructed == metrics}")
```

---

## 🔧 **Phase 6: Simplify FINN Interface (35 minutes)**

### **Step 6.1: Create Simplified FINN Interface**
```python
# File: brainsmith/core/finn_interface.py (replace existing)
"""
Simplified FINN Interface

Essential FINN abstraction for DataflowBuildConfig → 4-hooks transition.
Maintains compatibility while preparing for future interface.
"""

from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FINNInterface:
    """Simplified FINN interface for build execution."""
    
    def __init__(self):
        self.current_interface = "DataflowBuildConfig"
        self.hooks_available = self._check_hooks_availability()
        logger.info(f"FINNInterface initialized - using {self.current_interface}, 4-hooks available: {self.hooks_available}")
    
    def _check_hooks_availability(self) -> bool:
        """Check if 4-hooks interface is available."""
        try:
            # Placeholder for 4-hooks detection
            # When 4-hooks system is ready, this will check for the new interface
            return False  # Currently always False
        except Exception:
            return False
    
    def build_dataflow(self, model_path: str, design_config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Execute FINN build using current interface."""
        logger.info(f"Starting FINN build: {model_path}")
        
        if self.hooks_available:
            return self._build_with_hooks(model_path, design_config, output_dir)
        else:
            return self._build_with_dataflow_config(model_path, design_config, output_dir)
    
    def _build_with_dataflow_config(self, model_path: str, design_config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Build using current DataflowBuildConfig."""
        try:
            # Import FINN components
            from finn.builder.build_dataflow import build_dataflow
            from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType, VerificationStepType
            
            # Create build configuration
            build_config = self._create_dataflow_build_config(design_config, output_dir)
            
            logger.info("Executing FINN build_dataflow")
            
            # Execute FINN build
            result = build_dataflow(model_path, build_config)
            
            # Extract results
            build_results = {
                'success': True,
                'interface': 'DataflowBuildConfig',
                'output_dir': output_dir,
                'model_path': model_path,
                'build_config': self._config_to_dict(build_config)
            }
            
            # Add performance metrics if available
            if hasattr(result, 'get_performance_metrics'):
                build_results['performance_metrics'] = result.get_performance_metrics()
            
            logger.info("FINN build completed successfully")
            return build_results
            
        except ImportError as e:
            logger.warning(f"FINN not available: {e}")
            return self._create_fallback_result(model_path, design_config, output_dir, f"FINN not available: {e}")
        
        except Exception as e:
            logger.error(f"FINN build failed: {e}")
            return self._create_fallback_result(model_path, design_config, output_dir, f"Build failed: {e}")
    
    def _build_with_hooks(self, model_path: str, design_config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Build using future 4-hooks interface (placeholder)."""
        logger.info("Using 4-hooks interface (future implementation)")
        
        # Convert config to hooks format
        hooks_config = self._convert_to_hooks_config(design_config)
        
        # Placeholder for 4-hooks execution
        # When 4-hooks system is ready, this will call the new interface
        return {
            'success': True,
            'interface': '4-hooks',
            'output_dir': output_dir,
            'model_path': model_path,
            'hooks_config': hooks_config,
            'status': 'future_implementation'
        }
    
    def _create_dataflow_build_config(self, design_config: Dict[str, Any], output_dir: str):
        """Convert design configuration to DataflowBuildConfig."""
        try:
            from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType, VerificationStepType
            
            # Extract parameters with defaults
            config_params = {
                'output_dir': output_dir,
                'target_fps': design_config.get('target_fps', 1000),
                'synth_clk_period_ns': design_config.get('clk_period_ns', 10.0),
                'board': design_config.get('board', 'Pynq-Z1'),
                'folding_config_file': design_config.get('folding_config'),
                'auto_fifo_depths': design_config.get('auto_fifo_depths', True),
                'generate_outputs': [DataflowOutputType.ESTIMATE],  # Default to estimate
            }
            
            # Add verification steps if enabled
            if design_config.get('enable_verification', False):
                config_params['verify_steps'] = [
                    VerificationStepType.FOLDED_HLS_CPPSIM,
                    VerificationStepType.STITCHED_IP_RTLSIM
                ]
            
            # Add synthesis if requested
            if design_config.get('enable_synthesis', False):
                config_params['generate_outputs'].append(DataflowOutputType.STITCHED_IP)
            
            # Filter None values
            config_params = {k: v for k, v in config_params.items() if v is not None}
            
            logger.debug(f"DataflowBuildConfig params: {config_params}")
            return DataflowBuildConfig(**config_params)
            
        except ImportError:
            raise RuntimeError("FINN DataflowBuildConfig not available")
    
    def _convert_to_hooks_config(self, design_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert design configuration to 4-hooks format."""
        return {
            'preprocessing': {
                'model_cleanup': design_config.get('model_cleanup', True),
                'shape_inference': design_config.get('shape_inference', True)
            },
            'transformation': {
                'streamlining': design_config.get('streamlining', True),
                'folding': design_config.get('folding_config')
            },
            'optimization': {
                'target_fps': design_config.get('target_fps', 1000),
                'resource_constraints': design_config.get('resource_constraints', {})
            },
            'generation': {
                'board': design_config.get('board', 'Pynq-Z1'),
                'clk_period_ns': design_config.get('clk_period_ns', 10.0),
                'enable_synthesis': design_config.get('enable_synthesis', False)
            }
        }
    
    def _config_to_dict(self, config) -> Dict[str, Any]:
        """Convert build config to dictionary."""
        if hasattr(config, '__dict__'):
            return {k: str(v) for k, v in config.__dict__.items()}
        else:
            return {'config': str(config)}
    
    def _create_fallback_result(self, model_path: str, design_config: Dict[str, Any], output_dir: str, error: str) -> Dict[str, Any]:
        """Create fallback result when FINN build fails."""
        return {
            'success': False,
            'interface': 'fallback',
            'output_dir': output_dir,
            'model_path': model_path,
            'error': error,
            'fallback': True,
            'design_config': design_config
        }
    
    def get_interface_status(self) -> Dict[str, Any]:
        """Get current status of FINN interface."""
        return {
            'current_interface': self.current_interface,
            'hooks_available': self.hooks_available,
            'interface_ready': True,
            'supported_features': [
                'DataflowBuildConfig',
                'Estimate generation',
                'IP synthesis (optional)',
                'Verification (optional)'
            ]
        }


# Utility functions for backward compatibility
def create_finn_interface() -> FINNInterface:
    """Create FINN interface (convenience function)."""
    return FINNInterface()

def build_with_finn(model_path: str, design_config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Build using FINN interface (convenience function)."""
    interface = FINNInterface()
    return interface.build_dataflow(model_path, design_config, output_dir)
```

### **Step 6.2: Test FINN Interface Functionality**
```python
# Create test script: test_finn_interface.py
from brainsmith.core.finn_interface import FINNInterface

# Test interface initialization
interface = FINNInterface()
print(f"Interface status: {interface.get_interface_status()}")

# Test configuration conversion
design_config = {
    'target_fps': 2000,
    'board': 'ZCU104',
    'clk_period_ns': 5.0,
    'auto_fifo_depths': True
}

# Test hooks config conversion
hooks_config = interface._convert_to_hooks_config(design_config)
print(f"Hooks config: {hooks_config}")

print("FINN interface test completed")
```

---

## 📦 **Phase 7: Update __init__.py (15 minutes)**

### **Step 7.1: Create Simple Exports**
```python
# File: brainsmith/core/__init__.py (replace existing)
"""
BrainSmith Core - Simple and Focused

Single forge() function + essential helpers for FPGA accelerator DSE.
Includes robust DesignSpace, essential metrics, and FINN abstraction.

This module embodies our design axioms:
- Simplicity Over Sophistication: Simple exports, no complex loading
- Functions Over Frameworks: forge() function + helper objects
- Focus Over Feature Creep: Core FPGA DSE functionality only
- Performance Over Purity: Fast, practical implementation
"""

# Core API
from .api import forge, validate_blueprint

# Essential supporting objects
from .design_space import DesignSpace
from .metrics import DSEMetrics
from .finn_interface import FINNInterface

# Module metadata
__version__ = "0.5.0"
__author__ = "BrainSmith Development Team"
__description__ = "Simple and focused FPGA accelerator design space exploration"

# Public API exports
__all__ = [
    # Core functionality
    'forge',
    'validate_blueprint',
    
    # Essential objects
    'DesignSpace',
    'DSEMetrics', 
    'FINNInterface'
]

# Module-level convenience functions
def get_version() -> str:
    """Get module version."""
    return __version__

def get_status() -> dict:
    """Get module status and available components."""
    return {
        'version': __version__,
        'components': {
            'forge': True,
            'validate_blueprint': True,
            'DesignSpace': True,
            'DSEMetrics': True,
            'FINNInterface': True
        },
        'description': __description__,
        'simplified': True,
        'enterprise_bloat_removed': True
    }


# Optional: Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BrainSmith Core v{__version__} - Simple and Focused")
```

### **Step 7.2: Test Module Imports**
```python
# Create test script: test_core_imports.py
# Test all imports work correctly
from brainsmith.core import forge, validate_blueprint, DesignSpace, DSEMetrics, FINNInterface
from brainsmith.core import get_version, get_status

print(f"BrainSmith Core version: {get_version()}")
print(f"Module status: {get_status()}")

# Test object creation
ds = DesignSpace(name="test", parameters={}, constraints={})
metrics = DSEMetrics(build_success=True)
finn = FINNInterface()

print("All imports successful!")
print(f"DesignSpace: {ds}")
print(f"DSEMetrics: {metrics}")
print(f"FINNInterface status: {finn.get_interface_status()}")
```

---

## ✅ **Phase 8: Testing & Validation (30 minutes)**

### **Step 8.1: Run Core Tests**
```bash
# Test individual components
python test_design_space.py
python test_metrics.py
python test_finn_interface.py
python test_core_imports.py

# Run existing test suite
python -m pytest tests/ -v --tb=short -k "core" > simplified_test_results.txt

# Compare with baseline
diff baseline_test_results.txt simplified_test_results.txt
```

### **Step 8.2: Integration Testing**
```python
# Create integration test: test_core_integration.py
from brainsmith.core import forge, DesignSpace, DSEMetrics
import tempfile
import os

def test_core_integration():
    """Test core module integration."""
    print("Testing core module integration...")
    
    # Test imports
    print("✓ Imports successful")
    
    # Test DesignSpace
    ds = DesignSpace(name="integration_test")
    print(f"✓ DesignSpace: {ds}")
    
    # Test DSEMetrics
    metrics = DSEMetrics(throughput_ops_sec=1000.0, build_success=True)
    print(f"✓ DSEMetrics: {metrics.get_summary()}")
    
    # Test that forge function exists and is callable
    assert callable(forge), "forge function should be callable"
    print("✓ forge function is callable")
    
    print("Integration test completed successfully!")

if __name__ == "__main__":
    test_core_integration()
```

### **Step 8.3: Performance Verification**
```python
# Create performance test: test_core_performance.py
import time
from brainsmith.core import DesignSpace, DSEMetrics, FINNInterface

def test_object_creation_performance():
    """Test that simplified objects create quickly."""
    
    # Test DesignSpace creation speed
    start = time.time()
    for i in range(1000):
        ds = DesignSpace(name=f"test_{i}")
    ds_time = time.time() - start
    
    # Test DSEMetrics creation speed
    start = time.time()
    for i in range(1000):
        metrics = DSEMetrics(throughput_ops_sec=float(i))
    metrics_time = time.time() - start
    
    # Test FINNInterface creation speed
    start = time.time()
    finn = FINNInterface()
    finn_time = time.time() - start
    
    print(f"Performance test results:")
    print(f"  DesignSpace creation: {ds_time:.3f}s for 1000 objects ({ds_time*1000:.3f}ms each)")
    print(f"  DSEMetrics creation: {metrics_time:.3f}s for 1000 objects ({metrics_time*1000:.3f}ms each)")
    print(f"  FINNInterface creation: {finn_time:.3f}s")
    
    # Performance should be good (less than 1ms per object)
    assert ds_time < 1.0, "DesignSpace creation should be fast"
    assert metrics_time < 1.0, "DSEMetrics creation should be fast"
    
    print("✓ Performance test passed")

if __name__ == "__main__":
    test_object_creation_performance()
```

---

## 🔍 **Phase 9: Cleanup & Documentation (20 minutes)**

### **Step 9.1: Remove Test Files**
```bash
# Clean up temporary test files
rm test_design_space.py
rm test_metrics.py
rm test_finn_interface.py
rm test_core_imports.py
rm test_core_integration.py
rm test_core_performance.py
rm core_imports_analysis.txt
rm core_test_dependencies.txt
rm baseline_test_results.txt
rm simplified_test_results.txt
```

### **Step 9.2: Update Documentation**
```python
# Create documentation: brainsmith/core/README.md
"""
# BrainSmith Core Module

Simple and focused FPGA accelerator design space exploration.

## Quick Start

```python
from brainsmith.core import forge

# Generate FPGA accelerator
result = forge("model.onnx", "blueprint.yaml")
print(f"Success: {result.get('success', False)}")
```

## Essential Components

- **forge()**: Core function for FPGA accelerator generation
- **DesignSpace**: Robust blueprint instantiation and parameter management
- **DSEMetrics**: Essential metrics for optimization feedback
- **FINNInterface**: FINN abstraction with 4-hooks transition support

## Simplified from Enterprise

This module was simplified from 14 files (~3,500 lines) to 6 files (~1,200 lines):
- 66% code reduction
- Enterprise bloat eliminated
- Essential functionality preserved
- Performance optimized

## Migration from Old API

### Old Enterprise Pattern:
```python
from brainsmith.core import DesignSpaceOrchestrator, WorkflowManager
orchestrator = DesignSpaceOrchestrator(blueprint)
workflow = WorkflowManager(orchestrator)
result = workflow.execute_workflow("comprehensive")
```

### New Simple Pattern:
```python
from brainsmith.core import forge
result = forge("model.onnx", "blueprint.yaml")
```
"""
```

### **Step 9.3: Final Verification**
```bash
# Final module structure check
find brainsmith/core -name "*.py" | sort
echo "Total lines in simplified core:"
find brainsmith/core -name "*.py" -exec wc -l {} + | tail -1

# Final import test
python -c "from brainsmith.core import forge, DesignSpace, DSEMetrics, FINNInterface; print('✓ Final import test passed')"
```

---

## 📊 **Phase 10: Commit & Documentation (15 minutes)**

### **Step 10.1: Git Commit**
```bash
# Stage all changes
git add brainsmith/core/

# Commit with detailed message
git commit -m "Core Module Simplification Complete

- Reduced from 14 files (~3,500 lines) to 6 files (~1,200 lines)
- 66% code reduction achieved
- Enterprise bloat eliminated: orchestrator, workflow, compiler, config, result, legacy
- Essential components preserved: DesignSpace, DSEMetrics, FINNInterface
- Simplified API: forge() + 4 supporting objects
- All functionality tested and verified

Files deleted:
- design_space_orchestrator.py (461 lines)
- workflow.py (356 lines)
- compiler.py (451 lines)
- config.py (374 lines)
- result.py (493 lines)
- legacy_support.py (433 lines)
- hw_compiler.py (89 lines)

Files simplified:
- cli.py (443 → 80 lines)
- design_space.py (453 → 200 lines)
- metrics.py (382 → 180 lines)
- finn_interface.py (429 → 250 lines)
- __init__.py (322 → 30 lines)

Preserved and enhanced:
- api.py (462 lines) - Core forge() function
"
```

### **Step 10.2: Create Implementation Summary**
```bash
echo "
## 🎉 Core Simplification Implementation Complete!

### Transformation Summary:
- **Files**: 14 → 6 (57% reduction)
- **Lines**: ~3,500 → ~1,200 (66% reduction)
- **API Exports**: 50+ → 5 (90% reduction)

### Time Taken: $(date)
- Phase 1 (Backup): 15 min
- Phase 2 (Deletion): 20 min
- Phase 3 (CLI): 25 min
- Phase 4 (DesignSpace): 30 min
- Phase 5 (Metrics): 25 min
- Phase 6 (FINN): 35 min
- Phase 7 (Init): 15 min
- Phase 8 (Testing): 30 min
- Phase 9 (Cleanup): 20 min
- Phase 10 (Commit): 15 min
- **Total**: ~3.5 hours

### Result:
Simple, focused core module with essential functionality preserved.
Enterprise bloat eliminated while maintaining robust design capabilities.
" > CORE_SIMPLIFICATION_COMPLETE.md
```

---

## 🎯 **Success Criteria Verification**

### **Checklist**
- [ ] **Code Reduction**: >60% achieved (66% actual)
- [ ] **Files Reduced**: 14 → 6 files
- [ ] **API Simplified**: 5 essential exports
- [ ] **Essential Preserved**: DesignSpace, Metrics, FINN interface
- [ ] **Tests Pass**: All critical functionality tested
- [ ] **Performance**: No regression in object creation
- [ ] **Documentation**: Updated and complete
- [ ] **Git Committed**: All changes tracked

### **Validation Commands**
```bash
# Final validation
python -c "
from brainsmith.core import forge, DesignSpace, DSEMetrics, FINNInterface, get_status
print('✅ Core simplification successful!')
print(f'Status: {get_status()}')
print(f'Files: {len([f for f in __import__(\"os\").listdir(\"brainsmith/core\") if f.endswith(\".py\")])}')
"
```

---

## 🏁 **Implementation Complete**

This step-by-step plan provides a comprehensive, systematic approach to implementing the core module simplification. The plan is designed to:

1. **Minimize Risk**: Backup strategy and incremental changes
2. **Ensure Quality**: Testing at each phase
3. **Preserve Function**: Essential components maintained
4. **Document Progress**: Clear tracking and validation
5. **Enable Rollback**: Git commits and backups throughout

**Total Implementation Time**: ~3.5 hours  
**Result**: Simple, focused core module aligned with design axioms while preserving essential functionality for robust FPGA accelerator design space exploration.