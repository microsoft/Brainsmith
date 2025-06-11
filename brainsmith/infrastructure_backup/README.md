# Brainsmith Infrastructure

This directory contains the platform services and infrastructure components that support the core API and libraries.

## Overview

The infrastructure layer provides essential platform services including design space exploration, FINN integration, blueprint management, hooks system, and data management.

## Infrastructure Components

### 🎯 Design Space Exploration (`dse/`)
Multi-objective optimization and design space exploration engine.
- **Core Engine**: DesignSpace class and parameter management
- **Optimization**: Multi-objective optimization strategies
- **Integration**: Seamless integration with core API

### 🔧 FINN Integration (`finn/`)
Interface layer for FINN framework integration with 4-hooks preparation.
- **Interface**: Clean abstraction for FINN operations
- **Hooks**: Preparation for 4-hooks FINN integration
- **Compatibility**: Maintains FINN workflow compatibility

### 📋 Blueprint System (`blueprint/`)
YAML-based configuration system for design space definitions and compilation recipes.
- **Configuration**: Blueprint loading and validation
- **Recipes**: Compilation workflow definitions
- **Validation**: Blueprint schema validation

### 🪝 Hooks System (`hooks/`)
Event system for extensibility, monitoring, and plugin support.
- **Events**: Comprehensive event system
- **Types**: Hook type definitions and interfaces  
- **Plugins**: Example plugin implementations

### 💾 Data Management (`data/`)
Centralized data handling, caching, and persistence layer.
- **Caching**: Intelligent result caching
- **Persistence**: Data storage and retrieval
- **Management**: Centralized data lifecycle management

## Usage Patterns

### Direct Infrastructure Access
```python
from brainsmith.infrastructure.dse import DesignSpace
from brainsmith.infrastructure.hooks import events
```

### Integration Through Core API
```python
from brainsmith.core.api import forge

# Infrastructure components are automatically used
result = forge(
    model=model,
    blueprint="efficient_inference",  # Uses blueprint system
    hooks=["progress_monitor"],       # Uses hooks system
    cache=True                        # Uses data management
)
```

## Architecture Principles

1. **Service Layer**: Provides platform services to core and libraries
2. **Abstraction**: Clean interfaces hiding implementation complexity
3. **Extensibility**: Plugin and hook systems for customization
4. **Performance**: Optimized for high-performance operations
5. **Reliability**: Robust error handling and recovery

## Component Details

### DSE Engine
- Multi-objective optimization support
- Efficient parameter space sampling
- Integration with external optimization libraries
- Result analysis and Pareto frontier computation

### FINN Integration
- Clean abstraction layer for FINN operations
- Preparation for 4-hooks integration pattern
- Maintains compatibility with existing FINN workflows
- Extensible for future FINN enhancements

### Blueprint System
- YAML-based configuration management
- Schema validation and error reporting
- Template system for common configurations
- Integration with DSE for parameter definitions

### Hooks System
- Event-driven architecture support
- Plugin system for extensibility
- Monitoring and progress reporting
- Integration points for external tools

### Data Management
- Intelligent caching for performance
- Persistent storage for results
- Data lifecycle management
- Integration with analysis tools

## Development Guidelines

- Follow clean architecture principles
- Maintain clear interface boundaries
- Provide comprehensive error handling
- Include extensive documentation
- Design for extensibility and plugin support