# HWKG Rectification Implementation Plan

## Overview
This plan addresses critical architectural issues in the Hardware Kernel Generator (HWKG) system identified during axiom analysis. The goal is to create a unified, robust implementation that properly integrates with Interface-Wise Dataflow Modeling.

## Issues to Address
1. **Dual-Architecture Anti-Pattern**: Eliminate parallel `hw_kernel_gen` and `hw_kernel_gen_simple` implementations
2. **Weak Dataflow Integration**: Make Interface-Wise Dataflow Modeling the mandatory core foundation
3. **Terminology Inconsistency**: Update TDIM→BDIM pragma and align all terminology with dataflow axioms

## Implementation Phases

### Phase 1: Terminology Unification (Week 1)
**Goal**: Align all terminology with Interface-Wise Dataflow Modeling axioms

#### 1.1 Pragma Renaming
- [ ] **Search for all TDIM pragma references in codebase**
- [ ] **Update pragma parser in `rtl_parser/pragma.py` to recognize BDIM**
- [ ] **Add backward compatibility for TDIM with deprecation warning**
- [ ] **Update pragma type enum to include BDIM**
- [ ] **Update all test files using TDIM pragma**
- [ ] **Update documentation examples**

#### 1.2 Code Terminology Update
- [ ] **Replace "tensor dimension" with "block dimension" in pragma contexts**
- [ ] **Ensure consistent use of tensor_dims, block_dims, stream_dims**
- [ ] **Update variable names in pragma_to_strategy.py**
- [ ] **Update comments and docstrings**
- [ ] **Update error messages**

#### 1.3 Template Updates
- [ ] **Update Jinja2 templates to use correct terminology**
- [ ] **Ensure generated code uses dataflow terminology consistently**
- [ ] **Update template context variable names**

### Phase 2: Architecture Unification (Week 2-3)
**Goal**: Eliminate dual-architecture anti-pattern

#### 2.1 Feature Analysis
- [ ] **Audit differences between hw_kernel_gen vs hw_kernel_gen_simple**
- [ ] **Identify essential features that must be preserved**
- [ ] **Map migration path for simple system users**
- [ ] **Document feature compatibility matrix**

#### 2.2 Unified Implementation Design
- [ ] **Create unified architecture specification**
- [ ] **Design feature flags for complexity levels**
- [ ] **Plan clean CLI that abstracts complexity choices**
- [ ] **Design migration strategy for existing users**

#### 2.3 Implementation
- [ ] **Create new unified hw_kernel_gen core**
- [ ] **Implement configurable complexity via feature flags**
- [ ] **Migrate essential features from simple system**
- [ ] **Update CLI interface**
- [ ] **Create adapter layer for backward compatibility**

### Phase 3: Dataflow Integration Strengthening (Week 4-5)
**Goal**: Make Interface-Wise Dataflow Modeling the core foundation

#### 3.1 Mandatory Dataflow Integration
- [ ] **Remove optional dataflow integration logic**
- [ ] **Make DataflowInterface objects central to all processing**
- [ ] **Update generators to require dataflow objects**
- [ ] **Remove graceful degradation code**

#### 3.2 Enhanced Generation
- [ ] **Update templates to fully utilize tensor_dims, block_dims, stream_dims**
- [ ] **Implement proper chunking strategy application**
- [ ] **Ensure generated HWCustomOp classes use dataflow calculations**
- [ ] **Update RTLBackend generation to use dataflow metadata**

#### 3.3 Runtime Configuration
- [ ] **Enhance determine_chunking_from_layout() integration**
- [ ] **Ensure all dimension extraction happens at runtime**
- [ ] **Remove any compile-time dimension hardcoding**
- [ ] **Update templates for dynamic dimension handling**

### Phase 4: Testing and Validation (Week 6)
**Goal**: Ensure rectified system maintains all functionality

#### 4.1 Comprehensive Testing
- [ ] **Create test suite covering unified architecture**
- [ ] **Validate terminology consistency across system**
- [ ] **Test dataflow integration end-to-end**
- [ ] **Create regression tests for existing functionality**

#### 4.2 Migration Testing
- [ ] **Test migration path from simple system**
- [ ] **Validate backward compatibility for TDIM→BDIM**
- [ ] **Ensure existing generated components still work**
- [ ] **Test CLI compatibility**

#### 4.3 Documentation Update
- [ ] **Update all documentation to reflect unified architecture**
- [ ] **Create migration guides for users**
- [ ] **Update axioms documentation**
- [ ] **Update README files**

## Implementation Tracking

### Current Status: **STARTING PHASE 1**

### Phase 1 Progress: 5/12 tasks complete
- Pragma Renaming: 4/6 complete ✓
- Code Terminology: 1/5 complete  
- Template Updates: 0/3 complete

### Phase 2 Progress: 0/11 tasks complete
- Feature Analysis: 0/4 complete
- Unified Implementation Design: 0/4 complete
- Implementation: 0/5 complete

### Phase 3 Progress: 0/12 tasks complete
- Mandatory Dataflow Integration: 0/4 complete
- Enhanced Generation: 0/4 complete
- Runtime Configuration: 0/4 complete

### Phase 4 Progress: 0/12 tasks complete
- Comprehensive Testing: 0/4 complete
- Migration Testing: 0/4 complete
- Documentation Update: 0/4 complete

## Success Criteria
- [ ] **Single HWKG implementation (no dual architecture)**
- [ ] **100% terminology consistency with dataflow axioms**
- [ ] **All generated components use full dataflow capabilities**
- [ ] **Zero hardcoded dimensions in generated code**
- [ ] **Successful migration of existing users**
- [ ] **All tests passing**

## Risk Mitigation
- Maintain backward compatibility during transition
- Provide clear migration guides
- Extensive testing before deprecating legacy systems
- Gradual rollout with feature flags

## Timeline: 6 Weeks Total
- **Week 1**: Terminology unification (Phase 1)
- **Week 2-3**: Architecture unification (Phase 2)
- **Week 4-5**: Dataflow integration strengthening (Phase 3)
- **Week 6**: Testing and validation (Phase 4)

---

**Next Steps**: Begin Phase 1.1 - Search for all TDIM pragma references in codebase