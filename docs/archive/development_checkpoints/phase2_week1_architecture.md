# HWKG Phase 2 Week 1: Foundation Architecture

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [Class Diagrams](#class-diagrams)
6. [Sequence Diagrams](#sequence-diagrams)
7. [Integration Points](#integration-points)
8. [File Structure](#file-structure)

## Overview

The Phase 2 Week 1 implementation establishes the foundation architecture for the Hardware Kernel Generator (HWKG) refactoring. This document provides detailed architectural views of the new system components and their interactions.

### Key Architectural Principles
- **Separation of Concerns**: Each component has a single, well-defined responsibility
- **Dependency Injection**: Components receive dependencies rather than creating them
- **Interface Segregation**: Clean interfaces between components
- **Open/Closed Principle**: Extensible for new generator types without modifying core code

## System Architecture

```mermaid
graph TB
    subgraph "Entry Points"
        CLI[CLI Entry Point<br/>hkg.py]
        API[API Entry Point<br/>future]
    end

    subgraph "Configuration Layer"
        PC[PipelineConfig<br/>config.py]
        TC[TemplateConfig<br/>config.py]
        GC[GenerationConfig<br/>config.py]
        AC[AnalysisConfig<br/>config.py]
        VC[ValidationConfig<br/>config.py]
    end

    subgraph "Template System"
        TM[TemplateManager<br/>template_manager.py]
        TCB[TemplateContextBuilder<br/>template_context.py]
        Cache[TemplateCache<br/>template_manager.py]
    end

    subgraph "Generator Framework"
        GB[GeneratorBase<br/>generator_base.py]
        GA[GeneratedArtifact<br/>generator_base.py]
        GR[GenerationResult<br/>generator_base.py]
    end

    subgraph "Data Structures"
        DS[Data Structures<br/>data_structures.py]
        RTL[RTL Structures]
        PD[Pipeline Data]
    end

    CLI --> PC
    API --> PC
    PC --> TC
    PC --> GC
    PC --> AC
    PC --> VC
    
    GB --> TM
    GB --> TCB
    TM --> Cache
    
    GB --> GA
    GA --> GR
    
    GB --> DS
    DS --> RTL
    DS --> PD
```

## Component Architecture

### 1. Configuration Framework ([`config.py`](../brainsmith/tools/hw_kernel_gen/config.py))

The configuration framework provides centralized configuration management with validation and factory methods.

```mermaid
classDiagram
    class PipelineConfig {
        +generator_type: GeneratorType
        +template_config: TemplateConfig
        +generation_config: GenerationConfig
        +analysis_config: AnalysisConfig
        +validation_config: ValidationConfig
        +from_args(args) PipelineConfig
        +from_defaults(generator_type) PipelineConfig
        +from_file(path) PipelineConfig
        +from_dict(data) PipelineConfig
        +to_dict() dict
        +to_file(path) None
        +validate() None
    }

    class TemplateConfig {
        +base_dirs: List[Path]
        +custom_templates: Dict[str, Path]
        +cache_templates: bool
        +cache_size: int
        +cache_ttl: int
        +validate() None
    }

    class GenerationConfig {
        +output_dir: Path
        +overwrite: bool
        +generate_tb: bool
        +tb_type: str
        +target_language: str
        +indent_size: int
        +line_width: int
        +validate() None
    }

    PipelineConfig --> TemplateConfig
    PipelineConfig --> GenerationConfig
    PipelineConfig --> AnalysisConfig
    PipelineConfig --> ValidationConfig
```

**Key Features:**
- **Factory Methods**: Multiple ways to create configurations
- **Validation**: Built-in validation for all settings
- **Serialization**: JSON serialization support
- **Type Safety**: Enum-based configuration options

### 2. Template Context System ([`template_context.py`](../brainsmith/tools/hw_kernel_gen/template_context.py))

The template context system builds structured data for template rendering.

```mermaid
classDiagram
    class TemplateContextBuilder {
        -config: Optional[PipelineConfig]
        -_cache: Dict[str, Any]
        +build_hw_custom_op_context() HWCustomOpContext
        +build_rtl_backend_context() RTLBackendContext
        +clear_cache() None
        +get_cache_stats() Dict
    }

    class BaseContext {
        +module_name: str
        +file_name: str
        +timestamp: str
        +generator_version: str
        +validate() None
        +to_dict() Dict
    }

    class HWCustomOpContext {
        +top_module: str
        +class_name: str
        +interfaces: List[InterfaceInfo]
        +parameters: List[ParameterInfo]
        +has_axi_interfaces: bool
        +get_axi_interfaces() List
        +get_control_interfaces() List
        +validate() None
    }

    class InterfaceInfo {
        +name: str
        +direction: str
        +width: int
        +is_clock: bool
        +is_reset: bool
        +is_control: bool
        +is_axi: bool
    }

    TemplateContextBuilder --> BaseContext
    BaseContext <|-- HWCustomOpContext
    BaseContext <|-- RTLBackendContext
    HWCustomOpContext --> InterfaceInfo
    HWCustomOpContext --> ParameterInfo
```

**Key Features:**
- **Context Caching**: Efficient reuse of computed contexts
- **Signal Classification**: Automatic signal type detection
- **Validation**: Context validation before use
- **Extensibility**: Easy to add new context types

### 3. Template Management ([`template_manager.py`](../brainsmith/tools/hw_kernel_gen/template_manager.py))

The template manager provides optimized Jinja2 template handling with caching.

```mermaid
classDiagram
    class TemplateManager {
        -config: TemplateConfig
        -env: Environment
        -cache: Optional[TemplateCache]
        +get_template(name) Template
        +render_template(name, context) str
        +render_string(template_str, context) str
        +list_templates() List[str]
        +template_exists(name) bool
        +clear_cache() None
        +reload_templates() None
    }

    class TemplateCache {
        -max_size: int
        -ttl: int
        -cache: OrderedDict
        -timestamps: Dict
        +get(key) Optional[Template]
        +put(key, template) None
        +clear() None
        +get_stats() Dict
    }

    TemplateManager --> TemplateCache
    TemplateManager --> TemplateConfig
```

**Key Features:**
- **LRU Caching**: Efficient template caching with size limits
- **TTL Support**: Time-based cache expiration
- **Custom Filters**: Jinja2 filter extensions
- **Template Discovery**: Automatic template finding

### 4. Generator Framework ([`generator_base.py`](../brainsmith/tools/hw_kernel_gen/generator_base.py))

The generator framework provides the abstract base for all code generators.

```mermaid
classDiagram
    class GeneratorBase {
        <<abstract>>
        #config: PipelineConfig
        #template_manager: Optional[TemplateManager]
        #context_builder: Optional[TemplateContextBuilder]
        +generate(inputs)* GenerationResult
        +get_generator_info() Dict
        #create_artifact() GeneratedArtifact
        #render_template() str
        #build_context() Any
    }

    class GeneratedArtifact {
        +file_name: str
        +content: str
        +artifact_type: str
        +metadata: Dict
        +validate() bool
        +write_to_file() None
        +to_dict() Dict
    }

    class GenerationResult {
        +success: bool
        +artifacts: List[GeneratedArtifact]
        +errors: List[str]
        +warnings: List[str]
        +metrics: Dict
        +add_artifact() None
        +validate_all_artifacts() bool
        +write_all_artifacts() None
    }

    GeneratorBase --> GenerationResult
    GenerationResult --> GeneratedArtifact
```

**Key Features:**
- **Abstract Interface**: Clean contract for generators
- **Artifact Management**: Structured output handling
- **Metrics Tracking**: Performance and result metrics
- **Validation**: Built-in artifact validation

### 5. Data Structures ([`data_structures.py`](../brainsmith/tools/hw_kernel_gen/data_structures.py))

The data structures module provides comprehensive data models for the pipeline.

```mermaid
classDiagram
    class RTLSignal {
        +name: str
        +direction: str
        +width: int
        +signal_type: str
        +attributes: Dict
        +is_clock() bool
        +is_reset() bool
        +is_control() bool
    }

    class RTLInterface {
        +name: str
        +interface_type: str
        +signals: List[RTLSignal]
        +parameters: Dict
        +get_signals_by_direction() Dict
        +to_dict() Dict
    }

    class RTLModule {
        +name: str
        +parameters: Dict
        +interfaces: List[RTLInterface]
        +internal_signals: List[RTLSignal]
        +get_interface() Optional[RTLInterface]
        +get_all_signals() List[RTLSignal]
    }

    class ParsedRTLData {
        +modules: List[RTLModule]
        +top_module: Optional[str]
        +metadata: Dict
        +validate() None
        +get_top_module() Optional[RTLModule]
    }

    RTLModule --> RTLInterface
    RTLInterface --> RTLSignal
    ParsedRTLData --> RTLModule
```

## Data Flow

The following diagram shows how data flows through the Week 1 components:

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Config
    participant Generator
    participant Context
    participant Template
    participant Result

    User->>CLI: Execute command
    CLI->>Config: Create PipelineConfig
    Config->>Config: Validate settings
    
    CLI->>Generator: Initialize with config
    Generator->>Context: Request context build
    Context->>Context: Build and cache context
    Context-->>Generator: Return context
    
    Generator->>Template: Request template render
    Template->>Template: Check cache
    Template->>Template: Render with context
    Template-->>Generator: Return rendered content
    
    Generator->>Result: Create artifact
    Result->>Result: Validate artifact
    Generator->>Result: Add to results
    
    Generator-->>CLI: Return GenerationResult
    CLI->>Result: Write artifacts
    CLI-->>User: Display results
```

## Sequence Diagrams

### Configuration Loading Sequence

```mermaid
sequenceDiagram
    participant App
    participant PipelineConfig
    participant TemplateConfig
    participant FileSystem

    App->>PipelineConfig: from_file("config.json")
    PipelineConfig->>FileSystem: Read JSON file
    FileSystem-->>PipelineConfig: JSON data
    
    PipelineConfig->>PipelineConfig: Parse JSON
    PipelineConfig->>TemplateConfig: Create from dict
    TemplateConfig->>TemplateConfig: Validate paths
    TemplateConfig->>FileSystem: Check directories exist
    FileSystem-->>TemplateConfig: Directory status
    
    TemplateConfig-->>PipelineConfig: Valid config
    PipelineConfig->>PipelineConfig: validate()
    PipelineConfig-->>App: Configured pipeline
```

### Template Rendering Sequence

```mermaid
sequenceDiagram
    participant Generator
    participant TemplateManager
    participant Cache
    participant Jinja2
    participant Context

    Generator->>Context: Build context
    Context-->>Generator: Context data
    
    Generator->>TemplateManager: render_template("hw_custom_op.py.j2", context)
    TemplateManager->>Cache: get(template_name)
    
    alt Template in cache
        Cache-->>TemplateManager: Cached template
    else Template not cached
        TemplateManager->>Jinja2: get_template(name)
        Jinja2-->>TemplateManager: Template object
        TemplateManager->>Cache: put(name, template)
    end
    
    TemplateManager->>Jinja2: render(context)
    Jinja2-->>TemplateManager: Rendered content
    TemplateManager-->>Generator: Final output
```

## Integration Points

### 1. Configuration Integration
- **Entry Point**: [`PipelineConfig.from_args()`](../brainsmith/tools/hw_kernel_gen/config.py#L272)
- **Validation**: [`PipelineConfig.validate()`](../brainsmith/tools/hw_kernel_gen/config.py#L308)
- **Persistence**: [`PipelineConfig.to_file()`](../brainsmith/tools/hw_kernel_gen/config.py#L304)

### 2. Template System Integration
- **Manager Creation**: [`create_template_manager()`](../brainsmith/tools/hw_kernel_gen/template_manager.py#L391)
- **Context Building**: [`TemplateContextBuilder.build_hw_custom_op_context()`](../brainsmith/tools/hw_kernel_gen/template_context.py#L273)
- **Rendering**: [`TemplateManager.render_template()`](../brainsmith/tools/hw_kernel_gen/template_manager.py#L186)

### 3. Generator Integration
- **Base Class**: [`GeneratorBase`](../brainsmith/tools/hw_kernel_gen/generator_base.py#L225)
- **Result Creation**: [`create_generation_result()`](../brainsmith/tools/hw_kernel_gen/generator_base.py#L437)
- **Artifact Management**: [`GeneratedArtifact.write_to_file()`](../brainsmith/tools/hw_kernel_gen/generator_base.py#L85)

### 4. Data Structure Integration
- **RTL Parsing Output**: [`ParsedRTLData`](../brainsmith/tools/hw_kernel_gen/data_structures.py#L168)
- **Pipeline Input**: [`PipelineInputs`](../brainsmith/tools/hw_kernel_gen/data_structures.py#L223)
- **Pipeline Results**: [`PipelineResults`](../brainsmith/tools/hw_kernel_gen/data_structures.py#L266)

## File Structure

```
brainsmith/tools/hw_kernel_gen/
├── config.py                    # Configuration framework (336 lines)
│   ├── GeneratorType           # Enum for generator types
│   ├── ValidationLevel         # Enum for validation levels  
│   ├── TemplateConfig          # Template system configuration
│   ├── GenerationConfig        # Generation settings
│   ├── AnalysisConfig          # RTL analysis settings
│   ├── ValidationConfig        # Validation settings
│   └── PipelineConfig          # Main configuration class
│
├── template_context.py          # Context building (369 lines)
│   ├── BaseContext             # Base context class
│   ├── InterfaceInfo           # Interface metadata
│   ├── ParameterInfo           # Parameter metadata
│   ├── HWCustomOpContext       # HW Custom Op context
│   ├── RTLBackendContext       # RTL Backend context
│   └── TemplateContextBuilder  # Context builder with caching
│
├── template_manager.py          # Template management (406 lines)
│   ├── TemplateCache           # LRU cache implementation
│   ├── TemplateManager         # Jinja2 template manager
│   └── Factory functions       # Helper functions
│
├── generator_base.py            # Generator framework (449 lines)
│   ├── GeneratedArtifact       # Output artifact class
│   ├── GenerationResult        # Generation result container
│   ├── GeneratorBase           # Abstract generator base
│   └── Factory functions       # Helper functions
│
└── data_structures.py           # Pipeline data structures (434 lines)
    ├── RTLSignal               # Signal representation
    ├── RTLInterface            # Interface representation
    ├── RTLModule               # Module representation
    ├── ParsedRTLData           # Parser output
    ├── PipelineInputs          # Pipeline input container
    ├── PipelineResults         # Pipeline results container
    └── PipelineStage           # Pipeline stage enum
```

## Testing Architecture

The testing architecture validates all components:

```mermaid
graph LR
    subgraph "Unit Tests"
        TC[test_config.py<br/>34 tests]
        TTC[test_template_context.py<br/>42 tests]
        TTM[test_template_manager.py<br/>34 tests]
        TGB[test_generator_base.py<br/>37 tests]
    end

    subgraph "Integration Tests"
        TI[test_week1_integration.py<br/>10 tests]
        TE[test_error_handling.py<br/>9 tests]
    end

    subgraph "Components"
        C[config.py]
        TCX[template_context.py]
        TM[template_manager.py]
        GB[generator_base.py]
        DS[data_structures.py]
    end

    TC --> C
    TTC --> TCX
    TTM --> TM
    TGB --> GB
    TGB --> DS
    
    TI --> C
    TI --> TCX
    TI --> TM
    TI --> GB
    TI --> DS
    
    TE --> C
    TE --> TCX
    TE --> TM
    TE --> GB
```

## Summary

The Week 1 architecture establishes a solid foundation for the HWKG refactoring:

1. **Configuration Framework**: Centralized, validated configuration management
2. **Template System**: Optimized template handling with caching
3. **Generator Framework**: Standardized generator implementation pattern
4. **Data Structures**: Comprehensive data modeling for the pipeline
5. **Integration**: Clean interfaces between all components

This architecture supports the Phase 2 goals of eliminating code duplication, improving maintainability, and enabling extensibility for future generator types.