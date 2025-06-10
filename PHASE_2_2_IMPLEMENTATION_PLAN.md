# Phase 2.2 Unified Implementation Plan

## Strategic Approach

Based on comprehensive feature analysis, we will **enhance hw_kernel_gen_simple** rather than creating a complex unification. The analysis revealed that the "simple" system already provides superior architecture, UX, and data modeling.

## Implementation Strategy

### Core Principle: Enhancement over Unification
- **Foundation**: hw_kernel_gen_simple (proven superior UX and architecture)
- **Enhancement**: Add optional BDIM pragma sophistication via feature flags
- **Compatibility**: Maintain template reuse and error resilience
- **Experience**: Preserve simple CLI while enabling advanced features

## Detailed Implementation Plan

### Step 1: Create hw_kernel_gen_unified Module Structure

```
brainsmith/tools/hw_kernel_gen_unified/
├── __init__.py
├── __main__.py                 # Module entry point
├── cli.py                      # Enhanced CLI with feature flags
├── config.py                   # Configuration with complexity levels
├── data.py                     # Enhanced HWKernel from simple system
├── errors.py                   # Error handling from simple system
├── generators/
│   ├── __init__.py
│   ├── base.py                 # GeneratorBase from simple system
│   ├── hw_custom_op.py         # Enhanced with optional BDIM
│   ├── rtl_backend.py          # From simple system
│   └── test_suite.py           # From simple system
├── rtl_parser/
│   ├── __init__.py
│   └── unified_parser.py       # Enhanced simple_parser with BDIM
└── pragma_integration/
    ├── __init__.py
    ├── bdim_processor.py       # Optional BDIM sophistication
    └── strategy_converter.py   # From full system
```

### Step 2: Data Model Enhancement

**Base: hw_kernel_gen_simple/data.py HWKernel class**

**Enhancements:**
```python
@dataclass
class UnifiedHWKernel(HWKernel):
    """Enhanced HWKernel with optional BDIM pragma support."""
    
    # Inherit all smart properties from simple system
    # Add optional BDIM metadata
    bdim_metadata: Optional[Dict[str, Any]] = None
    pragma_sophistication_level: str = "simple"  # simple | advanced
    
    @property
    def has_enhanced_bdim(self) -> bool:
        """Check if kernel has enhanced BDIM pragma information."""
        return (self.pragma_sophistication_level == "advanced" 
                and self.bdim_metadata is not None)
    
    @property
    def dataflow_interfaces(self) -> List[Dict[str, Any]]:
        """Get interfaces with dataflow type classification."""
        # Enhanced logic for dataflow type determination
        
    @property
    def chunking_strategies(self) -> Dict[str, Any]:
        """Get chunking strategies from BDIM pragmas if available."""
        if self.has_enhanced_bdim:
            return self.bdim_metadata.get('chunking_strategies', {})
        return {}
```

### Step 3: Enhanced CLI Design

**Philosophy**: Simple by default, powerful when needed

```python
# Simple mode (default - maintains hw_kernel_gen_simple UX)
python -m brainsmith.tools.hw_kernel_gen_unified \
    thresholding.sv \
    compiler_data.py \
    -o output/

# Advanced mode (enables BDIM pragma sophistication)
python -m brainsmith.tools.hw_kernel_gen_unified \
    thresholding.sv \
    compiler_data.py \
    -o output/ \
    --advanced-pragmas

# Debug mode with multi-phase execution (optional complexity)
python -m brainsmith.tools.hw_kernel_gen_unified \
    thresholding.sv \
    compiler_data.py \
    -o output/ \
    --advanced-pragmas \
    --multi-phase \
    --debug
```

**CLI Enhancement Implementation:**
```python
def main():
    parser = argparse.ArgumentParser(
        description="Unified Hardware Kernel Generator",
        epilog="Simple by default, powerful when needed."
    )
    
    # Core arguments (from simple system)
    parser.add_argument('rtl_file', help='SystemVerilog RTL file')
    parser.add_argument('compiler_data', help='Compiler data file')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    
    # Feature flags for complexity levels
    parser.add_argument('--advanced-pragmas', action='store_true',
                       help='Enable enhanced BDIM pragma processing')
    parser.add_argument('--multi-phase', action='store_true',
                       help='Enable multi-phase execution with debugging')
    parser.add_argument('--debug', action='store_true', help='Debug output')
    
    # Template and configuration options
    parser.add_argument('--template-dir', help='Custom template directory')
    parser.add_argument('--stop-after', help='Stop after specific phase (requires --multi-phase)')
```

### Step 4: Enhanced Parser Integration

**Base: hw_kernel_gen_simple/rtl_parser/simple_parser.py**

**Enhancement Strategy:**
```python
class UnifiedRTLParser:
    """Enhanced RTL parser with optional BDIM sophistication."""
    
    def __init__(self, advanced_pragmas: bool = False):
        self.advanced_pragmas = advanced_pragmas
        self.simple_parser = SimpleRTLParser()  # Wrap existing simple parser
        
        if advanced_pragmas:
            from ..pragma_integration import BDimProcessor
            self.bdim_processor = BDimProcessor()
    
    def parse_rtl_file(self, rtl_file: Path) -> UnifiedHWKernel:
        # Always use simple parser as foundation (error resilience)
        rtl_data = self.simple_parser.parse_rtl_file(rtl_file)
        
        # Convert to unified format
        unified_kernel = self._convert_to_unified(rtl_data)
        
        # Enhance with BDIM processing if enabled
        if self.advanced_pragmas:
            unified_kernel = self._enhance_with_bdim(unified_kernel, rtl_file)
        
        return unified_kernel
    
    def _enhance_with_bdim(self, kernel: UnifiedHWKernel, rtl_file: Path) -> UnifiedHWKernel:
        """Add enhanced BDIM pragma processing."""
        try:
            # Use sophisticated RTL parser for BDIM extraction
            from brainsmith.tools.hw_kernel_gen.rtl_parser import RTLParser
            full_parser = RTLParser()
            hw_kernel_full = full_parser.parse_file(str(rtl_file))
            
            # Extract BDIM metadata using existing sophisticated logic
            bdim_metadata = self.bdim_processor.extract_bdim_metadata(hw_kernel_full)
            
            kernel.bdim_metadata = bdim_metadata
            kernel.pragma_sophistication_level = "advanced"
            
        except Exception as e:
            # Graceful degradation - log warning but continue
            print(f"Warning: Advanced BDIM processing failed: {e}")
            print("Continuing with simple pragma processing...")
        
        return kernel
```

### Step 5: Generator Enhancement

**Base: hw_kernel_gen_simple/generators/base.py GeneratorBase**

**Enhancement for HWCustomOp Generator:**
```python
class UnifiedHWCustomOpGenerator(GeneratorBase):
    """Enhanced HWCustomOp generator with optional BDIM support."""
    
    def _get_template_context(self, hw_kernel: UnifiedHWKernel) -> dict:
        # Start with simple system context (proven reliable)
        context = super()._get_template_context(hw_kernel)
        
        # Enhance with BDIM sophistication if available
        if hw_kernel.has_enhanced_bdim:
            context.update({
                'enhanced_bdim_interfaces': self._build_enhanced_interfaces(hw_kernel),
                'chunking_strategies': hw_kernel.chunking_strategies,
                'advanced_pragma_mode': True
            })
        else:
            context.update({
                'advanced_pragma_mode': False
            })
        
        return context
    
    def _build_enhanced_interfaces(self, hw_kernel: UnifiedHWKernel) -> List[Dict]:
        """Build sophisticated interface data when BDIM processing is enabled."""
        # Use logic from hw_kernel_gen HWCustomOpGenerator
        # But with error handling from simple system
```

### Step 6: Configuration Management

```python
@dataclass
class UnifiedConfig:
    """Configuration for unified HWKG with complexity levels."""
    
    # Core configuration (from simple system)
    rtl_file: Path
    compiler_data_file: Path
    output_dir: Path
    template_dir: Optional[Path] = None
    debug: bool = False
    
    # Complexity level controls
    advanced_pragmas: bool = False
    multi_phase_execution: bool = False
    stop_after: Optional[str] = None
    
    @property
    def complexity_level(self) -> str:
        """Determine complexity level from flags."""
        if self.advanced_pragmas and self.multi_phase_execution:
            return "expert"
        elif self.advanced_pragmas:
            return "advanced"
        else:
            return "simple"
    
    @classmethod
    def from_args(cls, args) -> 'UnifiedConfig':
        """Create config from CLI arguments."""
        return cls(
            rtl_file=Path(args.rtl_file),
            compiler_data_file=Path(args.compiler_data),
            output_dir=Path(args.output),
            template_dir=Path(args.template_dir) if args.template_dir else None,
            debug=args.debug,
            advanced_pragmas=args.advanced_pragmas,
            multi_phase_execution=args.multi_phase,
            stop_after=args.stop_after
        )
```

### Step 7: Template Compatibility

**Key Insight**: Both systems already use identical templates

**Strategy**: 
- Use existing templates without modification
- Enhance template context based on complexity level
- Maintain backward compatibility

**Template Context Enhancement:**
```python
def build_template_context(hw_kernel: UnifiedHWKernel) -> dict:
    """Build context compatible with existing templates."""
    
    # Base context from simple system (always present)
    context = {
        'hw_kernel': hw_kernel,
        'class_name': hw_kernel.class_name,
        'kernel_name': hw_kernel.kernel_name,
        'interfaces': hw_kernel.interfaces,
        'kernel_complexity': hw_kernel.kernel_complexity,
        'kernel_type': hw_kernel.kernel_type
    }
    
    # Enhanced context for advanced mode
    if hw_kernel.has_enhanced_bdim:
        context.update({
            'interfaces': enhance_interfaces_with_bdim(hw_kernel.interfaces, hw_kernel.bdim_metadata),
            'enhanced_bdim_available': True,
            'chunking_strategies': hw_kernel.chunking_strategies
        })
    else:
        context.update({
            'enhanced_bdim_available': False
        })
    
    return context
```

## Implementation Timeline

### Week 1: Foundation Setup
- **Day 1-2**: Create unified module structure
- **Day 3-4**: Implement enhanced data model (UnifiedHWKernel)
- **Day 5**: Design and implement unified CLI

### Week 2: Core Enhancement
- **Day 1-2**: Implement unified RTL parser with optional BDIM
- **Day 3-4**: Enhance generators with feature flag support
- **Day 5**: Implement configuration management

### Week 3: Integration and Testing
- **Day 1-2**: Template compatibility and context enhancement
- **Day 3-4**: Integration testing and error handling validation
- **Day 5**: Performance testing and optimization

## Backward Compatibility Strategy

### For hw_kernel_gen_simple Users
```bash
# Existing command (still works identically)
python -m brainsmith.tools.hw_kernel_gen_simple input.sv data.py -o output/

# Unified equivalent (identical behavior by default)
python -m brainsmith.tools.hw_kernel_gen_unified input.sv data.py -o output/
```

### For hw_kernel_gen Users
```bash
# Existing complex command
python -m brainsmith.tools.hw_kernel_gen.hkg input.sv data.py -o output/ --stop-after generate_rtl_template

# Unified equivalent with same sophistication
python -m brainsmith.tools.hw_kernel_gen_unified input.sv data.py -o output/ --advanced-pragmas --multi-phase --stop-after generate_rtl_template
```

## Quality Assurance

### Testing Strategy
1. **Simple Mode Testing**: Ensure identical behavior to hw_kernel_gen_simple
2. **Advanced Mode Testing**: Validate BDIM pragma processing
3. **Template Compatibility**: Verify all templates work with both modes
4. **Error Resilience**: Test graceful degradation when advanced features fail
5. **Migration Testing**: Validate user migration paths

### Success Criteria
- [ ] Simple mode produces identical output to hw_kernel_gen_simple
- [ ] Advanced mode provides BDIM sophistication from hw_kernel_gen
- [ ] All existing templates work without modification
- [ ] Error handling maintains robustness of simple system
- [ ] CLI provides clear complexity level choices
- [ ] Migration requires zero code changes for simple users

## Risk Mitigation

### Technical Risks
- **Complexity Creep**: Maintain simple system as foundation, add only optional enhancements
- **Template Breakage**: Use existing templates without modification
- **Performance Regression**: Profile and optimize unified parser

### User Experience Risks  
- **Confusion**: Default to simple mode, make advanced features opt-in
- **Migration Friction**: Ensure zero changes needed for simple system users
- **Documentation Gap**: Create clear complexity level documentation

This implementation plan leverages the superior foundation of hw_kernel_gen_simple while selectively adding the valuable BDIM sophistication from hw_kernel_gen, resulting in a truly unified system that serves all users effectively.