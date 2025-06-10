# Hardware Kernel Generator (HWKG): Core Design Axioms

## 1. Interface-Wise Dataflow Foundation
```
RTL Input → RTL Parser → Dataflow Interface Model → FINN Components
```
Interface-Wise Dataflow Modeling is the core framework that drives all HWKG generation. RTL interfaces are converted to DataflowInterface objects with tensor_dims, block_dims, and stream_dims relationships.

## 2. Multi-Phase Pipeline
```
Parse RTL → Parse Compiler Data → Build Dataflow Model → Generate Templates → Generate Components
```
Generation follows strict phases that can be executed selectively for debugging. Each phase is independently testable and can be stopped for analysis.

## 3. Template-Driven Code Generation
All code generation uses Jinja2 templates with rich context objects:
- **HWCustomOp**: Slim Python classes (50-80 lines) with runtime configuration
- **RTLBackend**: FINN integration components
- **RTL Wrapper**: Parameterized Verilog instantiation templates
- **Test Suite**: Comprehensive validation frameworks

## 4. Pragma-to-Chunking Conversion
RTL pragmas automatically convert to chunking strategies using Interface-Wise Dataflow terminology:
- **BDIM pragmas** → block_dims chunking strategies for tensor dimension handling
- **DATATYPE pragmas** → interface datatype constraints
- **WEIGHT pragmas** → interface classification metadata

## 5. Runtime Dimension Extraction
Generated components extract tensor_dims, block_dims, and stream_dims at runtime (not compile-time) to support FINN's dynamic tensor handling via `determine_chunking_from_layout()` and ModelWrapper integration.

## 6. Metadata-Driven Generation
All generation decisions are driven by extracted metadata rather than hardcoded rules:
- **Interface metadata**: Type classification, constraints, chunking strategies
- **Kernel metadata**: Complexity estimation, resource requirements
- **Template context**: Consolidated metadata for consistent generation

## 7. Hierarchical Error Handling
Structured error handling with context, severity levels, and actionable suggestions:
```
BrainsmithError → ParserError | GenerationError | ValidationError
```
All errors provide rich debugging context and resolution guidance.

## 8. Configuration Layering
Configuration precedence (highest to lowest):
1. **RTL Pragmas** → 2. **Compiler Data** → 3. **CLI Arguments** → 4. **System Defaults**

## 9. Generator Factory Pattern
Specialized generators (HWCustomOp, RTLBackend, TestSuite) implement common interface with dedicated generation logic for each output type.

## 10. Unified Architecture Principle
HWKG maintains a single, robust yet streamlined implementation that integrates deeply with Interface-Wise Dataflow Modeling rather than maintaining parallel architectures.