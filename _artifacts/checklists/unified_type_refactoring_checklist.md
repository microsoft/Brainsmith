# Unified Type Refactoring Implementation Checklist

## Phase 1: Move Core Types to Dataflow [Est: 2-3 hours]

### 1.1 Extend dataflow/types.py
- [ ] Back up current dataflow/types.py
- [ ] Add InterfaceType enum definition from kernel_integrator/data.py
- [ ] Add ShapeExpr type alias: `Union[int, str]`
- [ ] Add ShapeSpec type alias: `List[ShapeExpr]`
- [ ] Run dataflow tests to ensure no breakage
- [ ] Commit: "feat(dataflow): add InterfaceType and shape expression types"

### 1.2 Update kernel_integrator imports
- [ ] Find all usages of InterfaceType in kernel_integrator (use grep/rg)
- [ ] Update imports to use `from brainsmith.core.dataflow.types import InterfaceType`
- [ ] Remove InterfaceType definition from kernel_integrator/data.py
- [ ] Run kernel_integrator tests
- [ ] Commit: "refactor(kernel_integrator): use InterfaceType from dataflow"

### Verification Point 1
- [ ] All tests pass
- [ ] No circular import errors
- [ ] InterfaceType accessible from both modules

## Phase 2: Create Kernel Integrator Type Structure [Est: 4-6 hours]

### 2.1 Create types directory structure
- [ ] Create `brainsmith/tools/kernel_integrator/types/` directory
- [ ] Create `__init__.py` with proper exports
- [ ] Create empty module files:
  - [ ] `core.py`
  - [ ] `rtl.py`
  - [ ] `metadata.py`
  - [ ] `generation.py`
  - [ ] `binding.py`
  - [ ] `config.py`

### 2.2 Implement core types
- [ ] Copy PortDirection enum to types/core.py
- [ ] Implement DatatypeSpec dataclass
- [ ] Implement DimensionSpec with ShapeSpec integration
- [ ] Add validation methods
- [ ] Write unit tests for core types

### 2.3 Implement RTL types
- [ ] Move Port, Parameter dataclasses to types/rtl.py
- [ ] Move ParsedModule dataclass
- [ ] Move ValidationError, ValidationResult
- [ ] Update imports in rtl_parser modules
- [ ] Run rtl_parser tests

### 2.4 Implement metadata types
- [ ] Create streamlined InterfaceMetadata dataclass
- [ ] Create focused KernelMetadata dataclass
- [ ] Add computed properties for common queries
- [ ] Remove circular dependencies with data.py
- [ ] Test metadata creation and access patterns

### 2.5 Implement generation types
- [ ] Create GeneratedFile dataclass
- [ ] Create GenerationContext dataclass
- [ ] Create simplified GenerationResult
- [ ] Move file I/O operations to GeneratedFile.write()
- [ ] Update generator modules to use new types

### 2.6 Implement binding types
- [ ] Create IOSpec dataclass
- [ ] Create AttributeBinding dataclass
- [ ] Create CodegenBinding dataclass
- [ ] Add helper methods for common queries
- [ ] Update codegen_binding.py to use new types

### 2.7 Implement config types
- [ ] Move Config dataclass to types/config.py
- [ ] Add validation in __post_init__
- [ ] Add helper methods (to_camel_case, etc.)
- [ ] Update CLI to use new config location

### Verification Point 2
- [ ] All type modules import without errors
- [ ] No circular dependencies between type modules
- [ ] All existing tests still pass

## Phase 3: Migrate Existing Code [Est: 6-8 hours]

### 3.1 Update imports systematically
- [ ] Create import mapping file (old → new)
- [ ] Update kernel_integrator.py main module
- [ ] Update all generator modules
- [ ] Update all rtl_parser modules
- [ ] Update template modules
- [ ] Update CLI module

### 3.2 Refactor kitchen sink classes
- [ ] Break up TemplateContext into focused contexts
- [ ] Split KernelMetadata responsibilities
- [ ] Simplify GenerationResult to remove file I/O
- [ ] Create adapter functions for backward compatibility

### 3.3 Remove old type definitions
- [ ] Remove types from data.py (keep only essential functions)
- [ ] Remove types from metadata.py (keep only builders)
- [ ] Remove redundant type definitions
- [ ] Clean up TYPE_CHECKING imports

### Verification Point 3
- [ ] Full test suite passes
- [ ] No import errors
- [ ] No missing type definitions

## Phase 4: Create Integration Layer [Est: 3-4 hours]

### 4.1 Create converters module
- [ ] Create `brainsmith/tools/kernel_integrator/converters.py`
- [ ] Implement metadata_to_kernel_definition function
- [ ] Add interface type mapping logic
- [ ] Add dimension conversion methods
- [ ] Write comprehensive tests

### 4.2 Create constraint builders
- [ ] Create `constraint_builder.py` module
- [ ] Implement build_datatype_constraints function
- [ ] Add dimension relationship builders
- [ ] Add parameter constraint builders
- [ ] Test constraint generation

### 4.3 Add validation layer
- [ ] Create validation for conversions
- [ ] Add type compatibility checks
- [ ] Implement error reporting
- [ ] Test edge cases

### Verification Point 4
- [ ] Converters handle all interface types
- [ ] Constraint builders produce valid dataflow constraints
- [ ] Round-trip conversion maintains data integrity

## Phase 5: Update Documentation [Est: 2-3 hours]

### 5.1 Update architecture documentation
- [ ] Update kernel_integrator ARCHITECTURE.md
- [ ] Document new type hierarchy
- [ ] Add dependency diagrams
- [ ] Update code examples

### 5.2 Update API documentation
- [ ] Document all public types
- [ ] Add usage examples
- [ ] Document migration from old types
- [ ] Update docstrings

### 5.3 Create migration guide
- [ ] Document breaking changes
- [ ] Provide before/after examples
- [ ] List deprecated types
- [ ] Add troubleshooting section

### Verification Point 5
- [ ] Documentation builds without errors
- [ ] All examples run correctly
- [ ] Migration guide covers all changes

## Phase 6: Final Testing and Cleanup [Est: 2-3 hours]

### 6.1 Comprehensive testing
- [ ] Run full test suite with coverage
- [ ] Test all example scripts
- [ ] Run integration tests
- [ ] Check for performance regressions

### 6.2 Code cleanup
- [ ] Remove all deprecated code
- [ ] Clean up unused imports
- [ ] Run linters and formatters
- [ ] Check for TODO comments

### 6.3 Final verification
- [ ] No circular dependencies
- [ ] All tests pass
- [ ] Documentation is complete
- [ ] Code follows project standards

## Rollback Plan

If issues arise:
1. [ ] Keep old type definitions in place (marked deprecated)
2. [ ] Use adapter pattern for gradual migration
3. [ ] Maintain backward compatibility layer
4. [ ] Document known issues

## Success Criteria

- [ ] Zero circular dependencies in type system
- [ ] All tests pass without modifications
- [ ] Clear separation between dataflow and kernel_integrator
- [ ] Improved code maintainability metrics
- [ ] No performance degradation

## Total Estimated Time: 20-28 hours

### Risk Factors
- Unexpected dependencies on old type structures
- Test suite modifications needed
- Performance impact from additional abstraction layers

### Mitigation Strategies
- Incremental migration with adapters
- Comprehensive testing at each phase
- Performance profiling before/after

Arete.