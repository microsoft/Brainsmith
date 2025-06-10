# HWKG Simplification Implementation Plan

## Executive Summary

This plan implements the aggressive simplification of the Hardware Kernel Generator (HWKG) from ~6000 lines of enterprise bloat to ~750 lines of clean, maintainable code. The implementation will be done incrementally to ensure functionality is preserved.

## Implementation Strategy

### Phase 1: Create Simplified Implementation (Parallel Development)
- Create new `hw_kernel_gen_simple/` directory
- Implement minimal core in parallel with existing system
- Preserve all existing functionality during development
- Test against existing examples to ensure compatibility

### Phase 2: Feature Parity and Validation
- Run comprehensive tests comparing old vs new outputs
- Performance benchmarking
- Integration testing with dataflow components

### Phase 3: Migration and Cleanup
- Switch references to use simplified implementation
- Remove enterprise bloat
- Update documentation

## Detailed Implementation Steps

### Step 1: Directory Structure Setup (30 minutes)

Create new simplified directory structure:

```
brainsmith/tools/hw_kernel_gen_simple/
├── __init__.py                   # Package initialization
├── cli.py                        # Command-line interface (~150 lines)
├── config.py                     # Simple configuration (~50 lines)
├── data.py                       # Data structures (~50 lines)
├── errors.py                     # Error definitions (~30 lines)
├── generators/
│   ├── __init__.py
│   ├── base.py                   # Simple base class (~50 lines)
│   ├── hw_custom_op.py          # HW custom op generator (~150 lines)
│   ├── rtl_backend.py           # RTL backend generator (~150 lines)
│   └── test_suite.py            # Test suite generator (~100 lines)
├── rtl_parser/                   # Port existing parser
│   ├── __init__.py
│   ├── parser.py
│   ├── interface_scanner.py
│   ├── data.py
│   └── grammar.py
└── templates/                    # Port existing templates
    ├── hw_custom_op_slim.py.j2
    ├── rtl_backend.py.j2
    ├── rtl_wrapper.v.j2
    └── test_suite.py.j2
```

### Step 2: Core Components Implementation (2 hours)

#### 2.1 Simple Configuration (`config.py`)

```python
"""Simple configuration for HWKG."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class Config:
    """Simple configuration for Hardware Kernel Generator."""
    rtl_file: Path
    compiler_data_file: Path
    output_dir: Path
    template_dir: Optional[Path] = None
    debug: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.rtl_file.exists():
            raise FileNotFoundError(f"RTL file not found: {self.rtl_file}")
        if not self.compiler_data_file.exists():
            raise FileNotFoundError(f"Compiler data file not found: {self.compiler_data_file}")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments."""
        return cls(
            rtl_file=Path(args.rtl_file),
            compiler_data_file=Path(args.compiler_data),
            output_dir=Path(args.output),
            template_dir=Path(args.template_dir) if args.template_dir else None,
            debug=args.debug if hasattr(args, 'debug') else False
        )
```

#### 2.2 Simple Data Structures (`data.py`)

```python
"""Simple data structures for HWKG."""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

@dataclass
class HWKernel:
    """Simple hardware kernel representation."""
    name: str
    class_name: str
    interfaces: List[Dict[str, Any]]
    rtl_parameters: List[Dict[str, Any]]
    source_file: Path
    compiler_data: Dict[str, Any]
    
    @property
    def kernel_name(self) -> str:
        """Get kernel name for templates."""
        return self.name
    
    @property
    def generation_timestamp(self) -> str:
        """Get timestamp for templates."""
        from datetime import datetime
        return datetime.now().isoformat()

@dataclass  
class GenerationResult:
    """Result of code generation."""
    generated_files: List[Path]
    success: bool
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
```

#### 2.3 Simple CLI Interface (`cli.py`)

```python
"""Simple command-line interface for HWKG."""
import argparse
import sys
from pathlib import Path
from typing import List

from .config import Config
from .data import HWKernel, GenerationResult
from .rtl_parser import parse_rtl_file
from .generators import HWCustomOpGenerator, RTLBackendGenerator, TestSuiteGenerator
from .errors import HWKGError

def create_hw_kernel(config: Config) -> HWKernel:
    """Create HWKernel from RTL file and compiler data."""
    # Parse RTL file
    rtl_data = parse_rtl_file(config.rtl_file)
    
    # Load compiler data
    import importlib.util
    spec = importlib.util.spec_from_file_location("compiler_data", config.compiler_data_file)
    compiler_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiler_module)
    
    # Create kernel object
    return HWKernel(
        name=rtl_data.module_name,
        class_name=f"{rtl_data.module_name.title()}",
        interfaces=rtl_data.interfaces,
        rtl_parameters=rtl_data.parameters,
        source_file=config.rtl_file,
        compiler_data=compiler_module.compiler_data
    )

def generate_all(hw_kernel: HWKernel, config: Config) -> GenerationResult:
    """Generate all output files."""
    generators = [
        HWCustomOpGenerator(config.template_dir),
        RTLBackendGenerator(config.template_dir),
        TestSuiteGenerator(config.template_dir)
    ]
    
    generated_files = []
    errors = []
    
    for generator in generators:
        try:
            output_file = generator.generate(hw_kernel, config.output_dir)
            generated_files.append(output_file)
            if config.debug:
                print(f"Generated: {output_file}")
        except Exception as e:
            error_msg = f"Failed to generate {generator.__class__.__name__}: {e}"
            errors.append(error_msg)
            if config.debug:
                print(f"Error: {error_msg}")
    
    return GenerationResult(
        generated_files=generated_files,
        success=len(errors) == 0,
        errors=errors
    )

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hardware Kernel Generator - Simple Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m brainsmith.tools.hw_kernel_gen_simple thresholding.sv compiler_data.py -o output/
  python -m brainsmith.tools.hw_kernel_gen_simple input.sv data.py -o generated/ --debug
        """
    )
    
    parser.add_argument('rtl_file', type=str, help='SystemVerilog RTL file to process')
    parser.add_argument('compiler_data', type=str, help='Python file containing compiler data')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output directory')
    parser.add_argument('--template-dir', type=str, help='Custom template directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = Config.from_args(args)
        
        if config.debug:
            print(f"Processing RTL file: {config.rtl_file}")
            print(f"Using compiler data: {config.compiler_data_file}")
            print(f"Output directory: {config.output_dir}")
        
        # Create hardware kernel representation
        hw_kernel = create_hw_kernel(config)
        
        if config.debug:
            print(f"Created kernel: {hw_kernel.name} ({hw_kernel.class_name})")
            print(f"Interfaces: {len(hw_kernel.interfaces)}")
        
        # Generate all outputs
        result = generate_all(hw_kernel, config)
        
        if result.success:
            print(f"✅ Successfully generated {len(result.generated_files)} files in {config.output_dir}")
            for file_path in result.generated_files:
                print(f"   - {file_path.name}")
        else:
            print(f"❌ Generation completed with {len(result.errors)} errors:")
            for error in result.errors:
                print(f"   - {error}")
            sys.exit(1)
            
    except HWKGError as e:
        print(f"❌ HWKG Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
```

### Step 3: Simple Generator Pattern (1.5 hours)

#### 3.1 Generator Base Class (`generators/base.py`)

```python
"""Simple base class for all generators."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import jinja2

from ..data import HWKernel

class GeneratorBase(ABC):
    """Simple base class for all HWKG generators."""
    
    def __init__(self, template_name: str, template_dir: Optional[Path] = None):
        self.template_name = template_name
        self.template_env = self._setup_jinja_env(template_dir)
    
    def _setup_jinja_env(self, template_dir: Optional[Path] = None) -> jinja2.Environment:
        """Setup Jinja2 environment."""
        if template_dir and template_dir.exists():
            loader = jinja2.FileSystemLoader(template_dir)
        else:
            # Use package templates
            import importlib.resources
            templates_path = importlib.resources.files(__package__.split('.')[0]) / 'templates'
            loader = jinja2.FileSystemLoader(templates_path)
        
        env = jinja2.Environment(
            loader=loader,
            trim_blocks=True,
            lstrip_blocks=True
        )
        return env
    
    def generate(self, hw_kernel: HWKernel, output_dir: Path) -> Path:
        """Generate output file for the given hardware kernel."""
        template = self.template_env.get_template(self.template_name)
        content = template.render(hw_kernel=hw_kernel, **self._get_template_context(hw_kernel))
        
        output_file = output_dir / self._get_output_filename(hw_kernel)
        output_file.write_text(content)
        
        return output_file
    
    @abstractmethod
    def _get_output_filename(self, hw_kernel: HWKernel) -> str:
        """Get output filename for the kernel."""
        pass
    
    def _get_template_context(self, hw_kernel: HWKernel) -> dict:
        """Get additional template context. Override in subclasses."""
        return {}
```

#### 3.2 Concrete Generator Implementations

```python
# generators/hw_custom_op.py
"""HWCustomOp generator implementation."""
from pathlib import Path
from .base import GeneratorBase
from ..data import HWKernel

class HWCustomOpGenerator(GeneratorBase):
    """Generates HWCustomOp Python classes."""
    
    def __init__(self, template_dir: Path = None):
        super().__init__('hw_custom_op_slim.py.j2', template_dir)
    
    def _get_output_filename(self, hw_kernel: HWKernel) -> str:
        return f"{hw_kernel.class_name.lower()}.py"
    
    def _get_template_context(self, hw_kernel: HWKernel) -> dict:
        return {
            'class_name': hw_kernel.class_name,
            'kernel_name': hw_kernel.kernel_name,
            'source_file': hw_kernel.source_file.name,
            'interfaces': hw_kernel.interfaces,
            'rtl_parameters': hw_kernel.rtl_parameters
        }
```

### Step 4: Port RTL Parser (30 minutes)

Copy existing working RTL parser with minimal modifications:

```python
# rtl_parser/__init__.py
"""Simple RTL parser interface."""
from .parser import parse_rtl_file

__all__ = ['parse_rtl_file']
```

### Step 5: Port Templates (15 minutes)

Copy existing Jinja2 templates with no modifications - they already work.

### Step 6: Testing and Validation (1 hour)

Create test script to validate simplified implementation:

```python
# test_simplified_hwkg.py
"""Test script for simplified HWKG implementation."""
import shutil
import tempfile
from pathlib import Path

def test_thresholding_example():
    """Test with existing thresholding example."""
    # Use existing test files
    rtl_file = Path("examples/thresholding/thresholding.sv")
    compiler_data = Path("examples/thresholding/dummy_compiler_data.py")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "output"
        
        # Run simplified HWKG
        from brainsmith.tools.hw_kernel_gen_simple.cli import main
        import sys
        
        # Mock command line args
        sys.argv = [
            'hwkg_simple',
            str(rtl_file),
            str(compiler_data),
            '-o', str(output_dir),
            '--debug'
        ]
        
        main()
        
        # Verify outputs
        assert (output_dir / "thresholding.py").exists()
        assert (output_dir / "thresholding_rtlbackend.py").exists()
        assert (output_dir / "test_thresholding.py").exists()
        
        print("✅ Simplified HWKG test passed!")

if __name__ == '__main__':
    test_thresholding_example()
```

## Implementation Timeline

- **Step 1**: Directory setup (30 min)
- **Step 2**: Core components (2 hours)  
- **Step 3**: Generator pattern (1.5 hours)
- **Step 4**: Port RTL parser (30 min)
- **Step 5**: Port templates (15 min)
- **Step 6**: Testing (1 hour)

**Total Implementation Time: ~5.5 hours**

## Success Criteria

1. ✅ Simplified implementation generates identical outputs to current HWKG
2. ✅ CLI interface works with existing examples
3. ✅ Code volume reduced from ~6000 to ~750 lines
4. ✅ All tests pass
5. ✅ Performance is equal or better than current implementation

## Next Phase (After Implementation)

1. **Integration Testing**: Test with all existing examples
2. **Performance Benchmarking**: Measure improvement over current HWKG
3. **Documentation Update**: Update tutorials to use simplified interface
4. **Migration**: Switch CI/CD and examples to use simplified version
5. **Cleanup**: Remove enterprise bloat from original HWKG

Let's get started with the implementation!