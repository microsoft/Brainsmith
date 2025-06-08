# Brainsmith-2: System Architecture

## High-Level Architecture Overview

Brainsmith-2 implements a **layered architecture** that transforms neural network models into optimized FPGA implementations through a series of specialized frameworks and tools. The system operates on the principle of **progressive refinement**, where high-level model descriptions are systematically transformed into hardware-specific implementations.

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  BERT Demo Pipeline  │  Custom Model Workflows  │  Examples │
├─────────────────────────────────────────────────────────────┤
│                     Blueprint System                        │
│               (Configurable Build Processes)                │
├─────────────────────────────────────────────────────────────┤
│                 Hardware Compiler Core                      │
│                (Main Orchestration Engine)                  │
├─────────────────────────────────────────────────────────────┤
│ Dataflow Modeling │ HW Kernel Generator │ Custom Operations │
│    Framework      │      Pipeline       │     Library      │
├─────────────────────────────────────────────────────────────┤
│              External Framework Integration                  │
│          FINN • QONNX • Brevitas • Tree-sitter             │
└─────────────────────────────────────────────────────────────┘
```

## Core System Components

### 1. Hardware Compiler Core

**Location**: `brainsmith/core/hw_compiler.py`  
**Role**: Main orchestration engine for the entire compilation pipeline

The Hardware Compiler Core serves as the **central coordinator** that manages the transformation of ONNX models into FPGA implementations. It implements a configurable pipeline architecture through the Blueprint System.

#### Key Functions

**`forge(blueprint, model, args)`** - Primary compilation entry point
- Orchestrates the complete model-to-hardware transformation
- Manages intermediate representations and validation
- Coordinates with FINN's dataflow building system
- Handles error recovery and progress reporting

**Blueprint Integration**
- Loads configurable build processes from the Blueprint Registry
- Supports custom transformation steps for different model architectures
- Provides extensible hooks for new model types and optimization strategies

#### Integration Points
- **Upstream**: Receives ONNX models from ML training frameworks
- **Downstream**: Produces FINN dataflow graphs and RTL implementations
- **Lateral**: Coordinates with Dataflow Framework and Custom Operations

### 2. Interface-Wise Dataflow Modeling Framework

**Location**: `brainsmith/dataflow/`  
**Role**: Unified abstraction layer for hardware interface modeling

This framework represents Brainsmith-2's **core innovation**, providing a mathematical foundation for hardware interface representation and optimization.

#### Architecture Components

**DataflowInterface** (`core/dataflow_interface.py:142`)
```python
# Three-tier dimension system enabling mathematical optimization
class DataflowInterface:
    qDim: int  # Query dimension (original tensor shape)
    tDim: int  # Tensor dimension (processing granularity)
    sDim: int  # Stream dimension (hardware parallelism)
    # Mathematical relationship: sDim ≤ tDim ≤ qDim × tDim = original_tensor_shape
```

**DataflowModel** (`core/dataflow_model.py:39`)
```python
# Unified computational model for performance analysis
class DataflowModel:
    def calculate_initiation_intervals(self, iPar: int, wPar: int) -> Dict[str, int]
    def get_parallelism_bounds(self) -> Dict[str, Tuple[int, int]]
    def optimize_parallelism(self, constraints: Dict) -> Dict[str, int]
```

**AutoHWCustomOp/AutoRTLBackend** (Base Classes)
- Eliminate 80%+ of generated code through intelligent inheritance
- Provide template-friendly base implementations
- Enable rapid prototyping and customization

#### Mathematical Foundation

The framework implements a **dimensional analysis system** where:
- **qDim**: Represents the original tensor query dimension
- **tDim**: Defines the processing tensor granularity
- **sDim**: Specifies hardware streaming parallelism

**Constraint Relationships**:
```
sDim ≤ tDim ≤ qDim
qDim × tDim = original_tensor_shape
Parallelism = f(sDim, tDim, hardware_constraints)
```

### 3. Hardware Kernel Generator Pipeline

**Location**: `brainsmith/tools/hw_kernel_gen/`  
**Role**: Automated generation of FINN integration components from RTL specifications

The HKG Pipeline implements a **multi-phase architecture** that transforms RTL specifications into complete FINN-compatible software stacks.

#### Pipeline Phases

**Phase 1: RTL Analysis** (`rtl_parser/`)
```
SystemVerilog Input → Tree-sitter Parser → AST Analysis → Interface Detection
```
- Tree-sitter based parsing with custom SystemVerilog grammar
- Automated port and signal analysis
- Pragma extraction for dataflow metadata

**Phase 2: Interface Modeling** (`analysis/`)
```
RTL Interfaces → Dataflow Converter → DataflowInterface Objects → Validation
```
- `InterfaceAnalyzer`: Dedicated interface analysis and classification
- `PragmaProcessor`: Pragma-driven configuration and optimization
- `EnhancedInterfaceAnalyzer`: Dataflow-aware processing with optimization

**Phase 3: Code Generation** (`generators/`)
```
Dataflow Model → Template Engine → FINN Components → Validation
```
- `HWCustomOpGenerator`: Generates FINN HWCustomOp implementations
- `RTLBackendGenerator`: Creates RTL backend with resource estimation
- Template-based approach with comprehensive customization

**Phase 4: Orchestration** (`orchestration/`)
```
Generator Coordination → Pipeline Management → Artifact Collection → Documentation
```
- `PipelineOrchestrator`: Pure orchestration without generation logic
- `GeneratorFactory`: Centralized generator management
- Clean separation of concerns architecture

#### Key Innovations

**Intelligent Template Selection**
- Automatic selection between full generation and base class inheritance
- Context-aware template rendering based on complexity analysis
- Hybrid generation strategies for optimal code size and performance

**Dataflow Integration**
- Native integration with Interface-Wise Dataflow Modeling
- Automatic performance optimization through mathematical relationships
- Resource estimation with dataflow-aware analysis

### 4. Custom Operations Library

**Location**: `brainsmith/custom_op/`  
**Role**: Hardware-accelerated implementations of neural network operations

The Custom Operations Library provides **production-ready implementations** of common neural network operations optimized for FPGA deployment.

#### FPGA Dataflow Operations (`fpgadataflow/`)

**LayerNorm** (`layernorm.py`)
- Quantized layer normalization with configurable precision
- HLS backend implementation for optimal resource utilization
- Integration with FINN's dataflow optimization passes

**HWSoftmax** (`hwsoftmax.py`)
- Hardware-optimized softmax implementation
- Configurable input/output precisions
- Support for both streaming and batch processing modes

**Shuffle Operations** (`shuffle.py`)
- Data reorganization and tensor reshaping operations
- Optimized memory access patterns for FPGA architectures
- Support for complex tensor transformations

#### HLS Kernel Implementations (`hw_kernels/hls/`)

**Optimized C++ Kernels**
- `layernorm.hpp`: High-level synthesis layer normalization
- `softmax.hpp`: Optimized softmax with configurable precision
- `bs_utils.hpp`: Utility functions for bit manipulation and data types

### 5. Blueprint System

**Location**: `brainsmith/blueprints/`  
**Role**: Configurable build processes for different model architectures

The Blueprint System provides **modular build orchestration** that can be customized for different model types and optimization strategies.

#### BERT Blueprint (`bert.py`)

**Custom Transformation Steps** (40+ specialized transformations)
- `custom_step_qonnx2finn()`: QONNX to FINN conversion with SoftMax handling
- `custom_step_streamlining()`: Graph optimization for quantized operations  
- `custom_step_infer_hardware()`: Hardware inference for custom operations
- `custom_step_remove_head/tail()`: Model surgery for deployment optimization

**Integration Architecture**
```python
# Blueprint Registry enables extensible build processes
@register_blueprint("bert")
def bert_build_process(model, args):
    return [
        custom_step_qonnx2finn,
        custom_streamlining_step,
        custom_step_infer_hardware,
        # ... additional transformation steps
    ]
```

## Data Flow Through the Platform

### Model Compilation Pipeline

```
ONNX Model Input
       ↓
Blueprint Selection (model type detection)
       ↓
Hardware Compiler Core (forge() orchestration)
       ↓
QONNX Preprocessing (graph optimization)
       ↓
FINN Dataflow Building (hardware mapping)
       ↓
Custom Operation Integration (hardware-specific optimizations)
       ↓
RTL Generation & Synthesis
       ↓
FPGA Bitstream Output
```

### Hardware Kernel Generation Pipeline

```
RTL Specification
       ↓
Tree-sitter Parsing (SystemVerilog → AST)
       ↓
Interface Analysis (port detection & classification)
       ↓
Dataflow Modeling (dimension analysis & optimization)
       ↓
Template Rendering (code generation)
       ↓
FINN Integration Components
```

## Integration Architecture

### External Framework Integration

**FINN Framework**
- **Base Classes**: Inherit from `HWCustomOp` and `RTLBackend`
- **Dataflow Building**: Native integration with `finn.builder.build_dataflow`
- **Optimization**: Leverages FINN's parallelization and resource optimization

**QONNX Integration**
- **Model Preprocessing**: Graph optimization and cleanup
- **Quantization Support**: Multi-precision datatype handling
- **Transformation Pipeline**: Seamless conversion to FINN format

**Tree-sitter Integration**
- **SystemVerilog Parsing**: Grammar-based AST generation with custom rules
- **Incremental Parsing**: Efficient handling of large RTL files
- **Error Recovery**: Robust parsing with meaningful error reporting

### Internal Component Relationships

**Dataflow Framework ↔ HKG Pipeline**
```python
# RTL interfaces converted to DataflowInterface objects
converter = RTLInterfaceConverter(onnx_metadata)
dataflow_interfaces = converter.convert_interfaces(rtl_interfaces)
```

**Dataflow Framework ↔ FINN**
```python
# DataflowModel provides optimization bounds for FINN
bounds = model.get_parallelism_bounds()
optimal_config = model.optimize_parallelism(constraints)
```

**Template Generation ↔ Base Classes**
```python
# Templates inherit from base classes instead of full generation
class AutoThresholdingAxi(AutoHWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        dataflow_model = create_thresholding_model()
        super().__init__(onnx_node, dataflow_model, **kwargs)
```

## Performance and Scalability

### Optimization Strategies

**Lazy Evaluation and Caching**
- Dataflow models are built on-demand with intelligent caching
- Template rendering cached with content-based invalidation
- Interface analysis results cached across generation cycles

**Parallelism-Aware Design**
- Mathematical optimization of parallelism parameters
- Resource-constrained optimization with configurable bounds
- Performance estimation integrated into the optimization loop

**Memory Efficiency**
- Streaming-first design for large model handling
- Configurable memory limits with graceful degradation
- Efficient intermediate representation management

### Scalability Characteristics

**Model Size Scalability**
- Tested with BERT models up to 1536 intermediate dimensions
- Streaming architecture supports arbitrarily large models
- Memory usage scales linearly with model complexity

**Hardware Target Scalability**
- Configurable resource constraints for different FPGA families
- Template system adapts to different hardware capabilities
- Performance estimation scales across hardware targets

**Development Team Scalability**
- Modular architecture supports parallel development
- Clean separation of concerns enables team specialization
- Comprehensive testing framework supports large team coordination

## Extensibility Model

### Blueprint Extension
```python
@register_blueprint("custom_model")
def custom_model_pipeline(model, args):
    return [
        custom_preprocessing_step,
        custom_optimization_step,
        custom_hardware_mapping_step
    ]
```

### Custom Operation Development
```python
class CustomOperation(AutoHWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        # Minimal implementation using base class
        dataflow_model = create_custom_dataflow_model()
        super().__init__(onnx_node, dataflow_model, **kwargs)
```

### Template Customization
```jinja2
{# Custom template extending base functionality #}
{% extends "base_hwcustomop.py.j2" %}
{% block custom_methods %}
    # Custom hardware-specific implementations
{% endblock %}
```

## Security and Reliability

### Validation Framework
- **Multi-level validation**: Syntax, semantic, and functional validation
- **Resource estimation**: Automatic resource usage analysis and bounds checking
- **Template validation**: Automated testing of generated code quality

### Error Handling
- **Graceful degradation**: Fallback strategies for parsing and generation failures
- **Comprehensive logging**: Detailed error reporting with context
- **Recovery mechanisms**: Automatic retry logic for transient failures

### Testing Architecture
- **575+ automated tests** covering all system components
- **Integration testing** with real FPGA hardware
- **Performance regression testing** for optimization validation

This architecture represents a mature, production-ready platform that combines innovative technical approaches with proven engineering practices to deliver reliable, high-performance FPGA AI acceleration capabilities.