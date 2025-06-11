# HWKG Rectification Plan

## Issues Identified

### 1. **Dual-Architecture Anti-Pattern**
**Problem**: Maintaining both `hw_kernel_gen` (full) and `hw_kernel_gen_simple` (simple) creates:
- Code duplication and maintenance burden
- Confusion about which implementation to use
- Inconsistent behavior between implementations
- Architectural complexity without clear benefit

**Root Cause**: Evolution from complex enterprise system toward simplified approach without removing legacy architecture.

### 2. **Interface-Wise Dataflow Integration Weakness**
**Problem**: Dataflow modeling treated as "optional" rather than core foundation:
- Graceful degradation logic suggests dataflow is secondary
- Generated components don't fully leverage dataflow capabilities
- Inconsistent use of tensor_dims, block_dims, stream_dims terminology

**Root Cause**: HWKG predates full dataflow framework integration, retaining legacy approaches.

### 3. **Terminology Inconsistency**
**Problem**: Using legacy `TDIM` pragma name instead of `BDIM`:
- Conflicts with Interface-Wise Dataflow Modeling terminology
- Creates confusion between tensor_dims and block_dims concepts
- Inconsistent with core axioms

**Root Cause**: Pragma system implemented before dataflow terminology was standardized.

## Rectification Strategy

### Phase 1: Terminology Unification ✅ COMPLETED (Week 1)
**Goal**: Align all terminology with Interface-Wise Dataflow Modeling axioms

#### 1.1 Pragma Renaming ✅ COMPLETED
- [x] Rename `TDIM` pragma to `BDIM` in all code
- [x] Update pragma parser to recognize `BDIM` syntax
- [x] Maintain backward compatibility with `TDIM` (with deprecation warning)
- [x] Update BDimPragma class with correct terminology
- [x] Test BDIM/TDIM pragma functionality
- [x] Update all test files and examples
- [ ] Update documentation examples (IN PROGRESS)

#### 1.2 Code Terminology Update ✅ COMPLETED
- [x] Update BDimPragma error messages and metadata keys
- [x] Replace "tensor dimension" with "block dimension" in relevant contexts
- [x] Ensure consistent use of tensor_dims, block_dims, stream_dims
- [x] Update variable names and comments to match axioms
- [x] Update error messages

#### 1.3 Template Updates ✅ COMPLETED
- [x] Update Jinja2 templates to use correct terminology
- [x] Ensure generated code uses dataflow terminology consistently
- [x] Update template context variable names

### Phase 2: Architecture Unification (Week 2-3) 🚧 IN PROGRESS
**Goal**: Eliminate dual-architecture anti-pattern

#### 2.1 Feature Analysis ✅ COMPLETED
- [x] Audit `hw_kernel_gen` vs `hw_kernel_gen_simple` feature differences
- [x] Identify essential features that must be preserved
- [x] Map migration path for simple system users
- [x] Created comprehensive feature comparison analysis (HWKG_FEATURE_COMPARISON.md)

**Key Findings from Analysis**:
1. **Architectural Paradox**: The "simple" system provides superior UX and richer data modeling
2. **Template Compatibility**: Both systems use identical Jinja2 templates
3. **Foundation Recommendation**: Use hw_kernel_gen_simple as base, enhance with optional BDIM sophistication
4. **Migration Strategy**: Enhancement approach rather than complex unification

**Essential Features Identified**:
- **Priority 1 (Critical)**: Rich HWKernel data class, safe extraction methods, simple CLI, enhanced BDIM pragma integration
- **Priority 2 (Important)**: Template reuse pattern, error handling, multi-phase debugging (optional)

**Migration Path**: Minimal migration needed - enhance simple system with feature flags for complexity levels

#### 2.2 Unified Implementation 🚧 IN PROGRESS
- [ ] Create new `hw_kernel_gen_unified` based on hw_kernel_gen_simple foundation
- [ ] Implement optional BDIM pragma sophistication via feature flags
- [ ] Add complexity levels (simple/advanced modes) with backward compatibility
- [ ] Design unified CLI that maintains simple UX while enabling advanced features
- [ ] Preserve template compatibility and error resilience
- [ ] Create detailed implementation plan (PHASE_2_2_IMPLEMENTATION_PLAN.md)

#### 2.3 Legacy System Deprecation 🚧 IN PROGRESS
- [ ] Mark `hw_kernel_gen_simple` as deprecated
- [ ] Mark `hw_kernel_gen` as deprecated  
- [ ] Provide migration guide for existing users
- [ ] Plan removal timeline (6-month deprecation period)
- [ ] Add deprecation warnings to legacy CLIs
- [ ] Update documentation to recommend unified system

### Phase 3: Dataflow Integration Strengthening (Week 4-5)
**Goal**: Make Interface-Wise Dataflow Modeling the core foundation

#### 3.1 Mandatory Dataflow Integration
- [ ] Remove "optional" dataflow integration logic
- [ ] Make DataflowInterface objects central to all processing
- [ ] Ensure all generated components leverage full dataflow capabilities

#### 3.2 Enhanced Generation
- [ ] Update templates to fully utilize tensor_dims, block_dims, stream_dims
- [ ] Implement proper chunking strategy application
- [ ] Ensure generated HWCustomOp classes use dataflow calculations

#### 3.3 Runtime Configuration
- [ ] Enhance `determine_chunking_from_layout()` integration
- [ ] Ensure all dimension extraction happens at runtime
- [ ] Remove any compile-time dimension hardcoding

### Phase 4: Testing and Validation (Week 6)
**Goal**: Ensure rectified system maintains all functionality

#### 4.1 Comprehensive Testing
- [ ] Create test suite covering unified architecture
- [ ] Validate terminology consistency across system
- [ ] Test dataflow integration end-to-end

#### 4.2 Migration Testing
- [ ] Test migration path from simple system
- [ ] Validate backward compatibility for TDIM→BDIM
- [ ] Ensure existing generated components still work

#### 4.3 Documentation Update
- [ ] Update all documentation to reflect unified architecture
- [ ] Create migration guides for users
- [ ] Update axioms documentation

## Implementation Priorities

### High Priority (Must Fix)
1. **Terminology unification** - Critical for consistency
2. **Dataflow integration strengthening** - Core architectural requirement
3. **BDIM pragma implementation** - Immediate terminology fix

### Medium Priority (Should Fix)
1. **Architecture unification** - Reduces maintenance burden
2. **Legacy system deprecation** - Cleanups codebase

### Low Priority (Nice to Have)
1. **Enhanced testing** - Improves reliability
2. **Documentation updates** - Improves usability

## Success Metrics

### Technical Metrics
- [ ] Single HWKG implementation (no dual architecture)
- [ ] 100% terminology consistency with dataflow axioms
- [ ] All generated components use full dataflow capabilities
- [ ] Zero hardcoded dimensions in generated code

### Quality Metrics
- [ ] Reduced codebase size (eliminate duplication)
- [ ] Improved test coverage for unified system
- [ ] Enhanced integration test suite
- [ ] Clear migration documentation

### User Experience Metrics
- [ ] Simplified CLI interface
- [ ] Clear feature documentation
- [ ] Successful migration of existing users
- [ ] Reduced user confusion about which system to use

## Timeline: 6 Weeks Total
- **Week 1**: Terminology unification
- **Week 2-3**: Architecture unification  
- **Week 4-5**: Dataflow integration strengthening
- **Week 6**: Testing and validation

## Risk Mitigation
- Maintain backward compatibility during transition
- Provide clear migration guides
- Extensive testing before deprecating legacy systems
- Gradual rollout with feature flags