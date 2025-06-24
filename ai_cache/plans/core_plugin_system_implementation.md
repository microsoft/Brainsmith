# Core Plugin System Implementation Plan

**Version**: 1.0  
**Date**: December 2024  
**Objective**: Implement the core plugin system for transforms, kernels, backends, and hw_transforms

## Phase 1: Foundation (Day 1-2)

### Tasks:

- [ ] Create plugin package structure
  - [ ] Create `brainsmith/plugin/__init__.py`
  - [ ] Create `brainsmith/plugin/decorators.py`
  - [ ] Create `brainsmith/plugin/registry.py`
  - [ ] Create `brainsmith/plugin/discovery.py`
  - [ ] Create `brainsmith/plugin/exceptions.py`

- [ ] Implement core decorators
  - [ ] Implement `@transform` decorator with validation
  - [ ] Implement `@kernel` decorator (stub)
  - [ ] Implement `@backend` decorator (stub)
  - [ ] Implement `@hw_transform` decorator (stub)
  - [ ] Add metadata validation for each decorator

- [ ] Implement PluginRegistry singleton
  - [ ] Basic registration functionality
  - [ ] Duplicate detection
  - [ ] Get methods for each plugin type
  - [ ] List methods with filtering
  - [ ] Requirements validation logic

### Deliverables:
- `brainsmith/plugin/` package with core functionality
- Unit tests for decorators and registry
- Basic error handling

## Phase 2: Discovery System (Day 3-4)

### Tasks:

- [ ] Implement PluginDiscovery class
  - [ ] Built-in plugin discovery from `brainsmith/libraries/transforms/operations/`
  - [ ] User plugin discovery from `~/.brainsmith/plugins/`
  - [ ] Project plugin discovery from `./brainsmith_plugins/`
  - [ ] Module loading with error handling
  - [ ] Logging for discovery process

- [ ] Create plugin loader utilities
  - [ ] Safe module importing
  - [ ] Dependency checking before load
  - [ ] Plugin validation after load
  - [ ] Error recovery mechanisms

- [ ] Test with existing transforms
  - [ ] Ensure ExpandNorms can be discovered
  - [ ] Test other existing transforms
  - [ ] Verify no conflicts with current system

### Deliverables:
- Complete discovery system
- Integration tests for plugin loading
- Documentation for plugin paths

## Phase 3: Transform Integration (Day 5-6)

### Tasks:

- [ ] Create TransformExecutor
  - [ ] Stage-based transform execution
  - [ ] Plugin registry integration
  - [ ] Error handling and logging
  - [ ] Configuration passing

- [ ] Update existing transforms
  - [ ] Add `@transform` decorator to ExpandNorms
  - [ ] Update other transforms in `brainsmith/libraries/transforms/operations/`
  - [ ] Ensure backward compatibility
  - [ ] Add appropriate metadata

- [ ] Create example plugin transforms
  - [ ] Simple graph cleanup transform
  - [ ] Example optimization transform
  - [ ] Documentation transform example

### Deliverables:
- Working transform execution system
- Updated existing transforms
- Example transforms for documentation

## Phase 4: Integration with FINN V2 (Day 7-8)

### Tasks:

- [ ] Update CompilationStrategy
  - [ ] Support plugin transform names
  - [ ] Validation of plugin availability
  - [ ] Fallback mechanisms

- [ ] Update StageExecutor
  - [ ] Use TransformExecutor for plugin transforms
  - [ ] Mix built-in and plugin transforms
  - [ ] Performance optimization

- [ ] Create compatibility layer
  - [ ] Map old transform names to plugins
  - [ ] Handle legacy configuration
  - [ ] Migration utilities

### Deliverables:
- Integrated plugin system with FINN V2
- Backward compatibility layer
- Migration guide

## Phase 5: Testing & Documentation (Day 9-10)

### Tasks:

- [ ] Comprehensive testing
  - [ ] Unit tests for all components
  - [ ] Integration tests for discovery
  - [ ] End-to-end compilation tests
  - [ ] Performance benchmarks

- [ ] Documentation
  - [ ] Plugin developer guide
  - [ ] API reference
  - [ ] Example plugins
  - [ ] Troubleshooting guide

- [ ] Validation tools
  - [ ] Plugin validator script
  - [ ] Dependency checker
  - [ ] Compatibility tester

### Deliverables:
- Complete test suite
- Developer documentation
- Validation tools

## File Structure

```
brainsmith/
├── plugin/
│   ├── __init__.py          # Package exports
│   ├── decorators.py        # @transform, @kernel, etc.
│   ├── registry.py          # PluginRegistry singleton
│   ├── discovery.py         # Plugin discovery logic
│   ├── exceptions.py        # Custom exceptions
│   └── validators.py        # Plugin validation utilities
├── core/
│   └── finn_v2/
│       └── transform_executor.py  # Transform execution with plugins
├── kernels/                 # NEW: Plugin kernels
│   ├── __init__.py
│   ├── matmul.py
│   └── layernorm.py
└── transforms/              # NEW: Plugin transforms organized by stage
    ├── __init__.py
    ├── graph_cleanup/
    │   ├── __init__.py
    │   └── remove_identity.py
    ├── topology_optimization/
    │   ├── __init__.py
    │   └── expand_norms.py
    ├── kernel_mapping/
    │   ├── __init__.py
    │   └── infer_hardware.py
    ├── kernel_optimization/
    │   ├── __init__.py
    │   └── set_folding.py
    └── graph_optimization/
        ├── __init__.py
        └── set_fifo_depths.py
```

**Note**: The existing `brainsmith/libraries/` remains untouched. We're creating parallel structures for the new plugin-based system.

## Implementation Guidelines

### 1. Decorator Implementation Pattern

```python
def transform(name, stage, **kwargs):
    def decorator(cls):
        # Validate class inheritance
        if not issubclass(cls, Transformation):
            raise TypeError(...)
        
        # Add metadata
        cls._plugin_metadata = {
            "type": "transform",
            "name": name,
            "stage": stage,
            **kwargs
        }
        
        # Auto-register
        PluginRegistry.register(cls)
        
        return cls
    return decorator
```

### 2. Registry Pattern

```python
class PluginRegistry:
    _instance = None
    
    def __new__(cls):
        # Singleton pattern
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
```

### 3. Discovery Pattern

```python
def discover_directory(path):
    for py_file in path.rglob("*.py"):
        try:
            # Dynamic import
            spec = importlib.util.spec_from_file_location(...)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            logger.warning(f"Failed to load {py_file}: {e}")
```

## Testing Strategy

### Unit Tests
- Test each decorator in isolation
- Test registry operations
- Test discovery with mock files

### Integration Tests
- Test full discovery process
- Test transform execution
- Test with real transforms

### System Tests
- End-to-end compilation with plugins
- Performance comparison
- Memory usage analysis

## Risk Mitigation

### Technical Risks
1. **Import conflicts**: Use isolated namespaces
2. **Circular dependencies**: Lazy loading where needed
3. **Performance impact**: Cache discovered plugins

### Compatibility Risks
1. **Breaking existing code**: Maintain backward compatibility
2. **QONNX version conflicts**: Version checking in requires
3. **Missing dependencies**: Graceful degradation

## Success Criteria

1. All existing transforms work through plugin system
2. No performance regression
3. Easy to add new transforms
4. Clear error messages
5. Comprehensive documentation

## Timeline

- **Week 1**: Phase 1-3 (Core implementation)
- **Week 2**: Phase 4-5 (Integration and polish)

Total estimated time: 10 working days

## Next Steps

1. Review and approve plan
2. Create feature branch
3. Set up test infrastructure
4. Begin Phase 1 implementation