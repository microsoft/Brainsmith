# Phase 2 Week 1: Foundation Refactoring - Implementation Plan

## Overview

Week 1 establishes the foundational infrastructure for the new architecture by creating the configuration framework and extracting the template context builder. This provides the base components needed for subsequent weeks.

**Duration**: 3 days  
**Risk Level**: Low (foundation work)  
**Dependencies**: Phase 1 complete ✅

## Daily Task Breakdown

### Day 1: Configuration Framework Foundation (Morning)

#### Task 1.1: Create Base Configuration Classes
**Time**: 2-3 hours  
**Objective**: Establish centralized configuration management

**Deliverables**:
- `brainsmith/tools/hw_kernel_gen/config.py` - Base configuration classes
- Configuration validation framework
- Default configuration generation

**Implementation**:
```python
# Key classes to implement:
- PipelineConfig (main configuration)
- GenerationConfig (generation-specific)
- TemplateConfig (template-specific)
- AnalysisConfig (analysis-specific)
- ValidationConfig (validation-specific)
```

#### Task 1.2: Configuration Validation Framework
**Time**: 1-2 hours  
**Objective**: Ensure configuration consistency

**Deliverables**:
- Configuration validation methods
- Error handling for invalid configurations
- Default value resolution

### Day 1: Configuration Integration (Afternoon)

#### Task 1.3: Configuration Factory Methods
**Time**: 2-3 hours  
**Objective**: Provide convenient configuration creation

**Deliverables**:
- `from_args()` factory methods
- `from_defaults()` factory methods
- Configuration serialization/deserialization

#### Task 1.4: Basic Configuration Tests
**Time**: 1 hour  
**Objective**: Validate configuration framework

**Deliverables**:
- `tests/test_config.py` - Configuration tests
- Test coverage for validation logic

### Day 2: Template Context Builder (Morning)

#### Task 2.1: Base Context Data Structures
**Time**: 2-3 hours  
**Objective**: Define template context data structures

**Deliverables**:
- `brainsmith/tools/hw_kernel_gen/template_context.py`
- BaseContext, HWCustomOpContext, RTLBackendContext classes
- Context inheritance hierarchy

#### Task 2.2: Template Context Builder Class
**Time**: 2-3 hours  
**Objective**: Centralize template context building

**Deliverables**:
- TemplateContextBuilder class
- Context caching mechanism
- Context building methods for each generator type

### Day 2: Template Infrastructure (Afternoon)

#### Task 2.3: Template Manager
**Time**: 2-3 hours  
**Objective**: Centralized template management with caching

**Deliverables**:
- TemplateManager class
- Jinja2 environment optimization
- Template caching system

#### Task 2.4: Template Context Tests
**Time**: 1 hour  
**Objective**: Validate template context building

**Deliverables**:
- `tests/test_template_context.py`
- Context building validation tests

### Day 3: Integration and Shared Infrastructure (Full Day)

#### Task 3.1: Generator Base Interface
**Time**: 2-3 hours  
**Objective**: Define common generator interface

**Deliverables**:
- Generator abstract base class
- GeneratedArtifact data structure
- Generator interface standardization

#### Task 3.2: Data Structure Definitions
**Time**: 2-3 hours  
**Objective**: Define pipeline data structures

**Deliverables**:
- PipelineInputs, PipelineResults classes
- ParsedRTLData, AnalyzedInterfaces structures
- Result validation and error handling

#### Task 3.3: Validation Framework Integration
**Time**: 1-2 hours  
**Objective**: Integrate validation with new structures

**Deliverables**:
- ValidationResult integration
- Error handling consistency
- Validation framework tests

#### Task 3.4: Week 1 Integration Testing
**Time**: 1-2 hours  
**Objective**: Validate Week 1 components work together

**Deliverables**:
- Integration tests for Week 1 components
- End-to-end configuration and context building test
- Performance baseline measurement

## Success Criteria

### Functional Requirements
- [ ] Configuration framework created and validated
- [ ] Template context builder eliminates duplication
- [ ] Template manager provides caching
- [ ] Generator base interface defined
- [ ] All tests passing (>90% coverage)

### Quality Requirements
- [ ] Clean separation of configuration concerns
- [ ] Proper error handling and validation
- [ ] Comprehensive test coverage
- [ ] Performance baseline established
- [ ] Documentation for new components

### Integration Requirements
- [ ] New components integrate with Phase 1 error framework
- [ ] Backward compatibility preserved
- [ ] No breaking changes to existing APIs
- [ ] Clean interfaces for Week 2 development

## File Structure After Week 1

```
brainsmith/tools/hw_kernel_gen/
├── config.py                    # ✅ New configuration framework
├── template_context.py          # ✅ New template context building
├── template_manager.py          # ✅ New template management
├── data_structures.py          # ✅ New pipeline data structures
├── generator_base.py           # ✅ New generator interface
├── errors.py                   # ✅ Phase 1 error framework
└── hkg.py                      # Existing (unchanged in Week 1)

tests/
├── test_config.py              # ✅ New configuration tests
├── test_template_context.py    # ✅ New context tests
├── test_template_manager.py    # ✅ New template tests
├── test_generator_base.py      # ✅ New generator tests
└── test_error_handling.py      # ✅ Phase 1 tests
```

## Testing Strategy

### Unit Tests
```bash
# Each new component has dedicated tests
pytest tests/test_config.py -v
pytest tests/test_template_context.py -v
pytest tests/test_template_manager.py -v
pytest tests/test_generator_base.py -v
```

### Integration Tests
```bash
# Test components work together
pytest tests/test_week1_integration.py -v

# Ensure no regressions
pytest tests/test_error_handling.py -v
```

### Performance Tests
```bash
# Baseline performance measurement
pytest tests/test_week1_performance.py -v
```

## Risk Mitigation

### Technical Risks
1. **Template Caching Issues**: Comprehensive cache testing and invalidation logic
2. **Configuration Complexity**: Simple, flat configuration structure with validation
3. **Performance Regression**: Baseline measurement and optimization

### Implementation Risks
1. **Scope Creep**: Stick to foundation components only
2. **Over-Engineering**: Focus on simple, working solutions
3. **Integration Issues**: Continuous integration testing

## Daily Validation Checklist

### End of Day 1
- [ ] Configuration framework created and tested
- [ ] Basic validation working
- [ ] No breaking changes to existing code
- [ ] All existing tests still pass

### End of Day 2  
- [ ] Template context builder created
- [ ] Template manager implemented
- [ ] Context caching working
- [ ] Template tests passing

### End of Day 3
- [ ] Generator interface defined
- [ ] Data structures created
- [ ] Integration tests passing
- [ ] Performance baseline established
- [ ] Ready for Week 2 development

## Dependencies for Week 2

Week 1 creates these components that Week 2 will use:

1. **PipelineConfig** - Used by PipelineOrchestrator
2. **TemplateContextBuilder** - Used by GeneratorFactory
3. **TemplateManager** - Used by all generators
4. **Generator base interface** - Implemented by specific generators
5. **Data structures** - Used throughout pipeline

## Acceptance Criteria

### Code Quality
- All new code follows established patterns from Phase 1
- Comprehensive error handling using Phase 1 framework
- Clean separation of concerns
- Proper documentation and type hints

### Functionality
- Configuration framework supports all identified use cases
- Template context building eliminates code duplication
- Template caching improves performance
- Generator interface enables clean separation

### Testing
- >90% test coverage for new components
- Integration tests validate component interaction
- Performance tests establish baseline
- No regressions in existing functionality

This Week 1 plan provides a solid foundation for the architectural refactoring while maintaining system stability and backward compatibility.