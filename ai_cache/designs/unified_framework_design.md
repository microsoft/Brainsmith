# Unified Framework Design Document

## Overview

This document describes the design of a new unified implementation that integrates the Unified Kernel Modeling Framework with FINN's HWCustomOp infrastructure. The design creates a clean, optimal implementation that runs in parallel with the existing dataflow framework.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        ONNX Model                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   FINN Framework                            │
│  ┌─────────────────────────┐  ┌──────────────────────────┐ │
│  │   Legacy Implementation  │  │  Unified Implementation   │ │
│  │  - AutoHWCustomOp       │  │  - UnifiedHWCustomOp      │ │
│  │  - DataflowModel        │  │  - Kernel + DataflowGraph │ │
│  │  - AutoRTLBackend       │  │  - DSE Integration        │ │
│  └─────────────────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                      │                        │
                      ▼                        ▼
              ┌──────────────┐         ┌──────────────┐
              │   RTL Code   │         │   RTL Code   │
              │  (Legacy)    │         │  (Optimized) │
              └──────────────┘         └──────────────┘
```

## Core Components

### 1. Kernel Definition System

The kernel definition is the foundation of the unified design, providing a clean abstraction for hardware operators.

```python
@dataclass
class KernelDefinition:
    """
    Complete definition of a hardware kernel.
    This is the single source of truth for kernel behavior.
    """
    # Basic metadata
    name: str
    version: str = "1.0"
    
    # Interface definitions
    interfaces: List[Interface]
    
    # Behavioral constraints
    constraints: List[Constraint]
    
    # RTL metadata (from parser or manual)
    rtl_parameters: Dict[str, Any]
    exposed_parameters: Set[str]
    
    # Performance model
    performance_model: Optional[PerformanceModel] = None
    
    # Resource model
    resource_model: Optional[ResourceModel] = None
```

#### Interface Definition
```python
@dataclass
class Interface:
    """Enhanced interface with full metadata."""
    name: str
    type: InterfaceType
    
    # Shape specifications
    tensor_shape: Shape      # Full tensor dimensions
    block_shape: Shape       # Processing block size
    stream_shape: Shape      # Parallelism (elements per cycle)
    
    # Datatype constraints
    datatype_constraints: List[DatatypeConstraint]
    
    # Protocol information
    protocol: Protocol = Protocol.AXI_STREAM
    protocol_config: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata for code generation
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 2. UnifiedHWCustomOp Base Class

The core class that bridges FINN with the new framework.

```python
class UnifiedHWCustomOp(HWCustomOp):
    """
    Base class for unified hardware custom operators.
    Provides clean integration between FINN and Unified Kernel Framework.
    """
    
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        
        # Get kernel definition (subclasses override)
        self.kernel_def = self.get_kernel_definition()
        
        # Create kernel instance
        self.kernel = Kernel(
            name=self.kernel_def.name,
            interfaces=self.kernel_def.interfaces
        )
        
        # Add constraints
        for constraint in self.kernel_def.constraints:
            self.kernel.add_constraint(constraint)
        
        # Create dataflow graph with single kernel
        self.graph = DataflowGraph()
        self.graph.add_kernel(self.kernel)
        
        # Initialize DSE components
        self.dse_constraints = DSEConstraints()
        self.config_space = ConfigurationSpace(self.graph)
        self.evaluator = PerformanceEvaluator()
        
        # State management
        self._current_config = None
        self._optimized = False
        
        # Initialize from ONNX attributes
        self._initialize_from_onnx()
    
    @abstractmethod
    def get_kernel_definition(self) -> KernelDefinition:
        """Subclasses provide kernel definition."""
        pass
    
    def _initialize_from_onnx(self):
        """Initialize kernel from ONNX node attributes."""
        # Set tensor shapes from ONNX inputs/outputs
        for i, intf in enumerate(self.kernel.input_interfaces):
            onnx_shape = self.get_input_shape(i)
            intf.tensor_shape = Shape(onnx_shape)
            
        for i, intf in enumerate(self.kernel.output_interfaces):
            onnx_shape = self.get_output_shape(i)
            intf.tensor_shape = Shape(onnx_shape)
        
        # Set block shapes from attributes or defaults
        self._set_block_shapes()
        
        # Initialize parallelism
        self._initialize_parallelism()
    
    def _set_block_shapes(self):
        """Set block shapes from node attributes or compute defaults."""
        for intf in self.kernel.interfaces:
            block_shape_attr = f"{intf.name}_block_shape"
            if self.has_nodeattr(block_shape_attr):
                block_shape = self.get_nodeattr(block_shape_attr)
                intf.block_shape = Shape(block_shape)
            else:
                # Compute default block shape
                intf.block_shape = self._compute_default_block_shape(intf)
    
    def _initialize_parallelism(self):
        """Initialize parallelism from attributes or defaults."""
        config = ParallelismConfig()
        
        for intf in self.kernel.interfaces:
            par_attr = f"{intf.name}_parallelism"
            if self.has_nodeattr(par_attr):
                parallelism = self.get_nodeattr(par_attr)
                config.set_interface_parallelism(intf.name, parallelism)
            else:
                # Default parallelism = 1
                config.set_interface_parallelism(intf.name, 1)
        
        # Apply configuration
        self._current_config = config.apply_to_graph(self.graph)
```

### 3. DSE Integration

Automatic optimization using the DSE framework.

```python
class UnifiedDSEMixin:
    """Mixin for DSE capabilities in unified operators."""
    
    def optimize_for_target(self, target_spec: Dict[str, Any]):
        """
        Optimize operator for target specification.
        
        Args:
            target_spec: Dictionary containing:
                - target_throughput: Desired throughput (optional)
                - max_resources: Resource constraints (optional)
                - optimization_objective: "throughput" | "latency" | "balanced"
        """
        # Set DSE constraints from target spec
        constraints = DSEConstraints()
        
        if "max_resources" in target_spec:
            constraints.max_luts = target_spec["max_resources"].get("LUT", float('inf'))
            constraints.max_brams = target_spec["max_resources"].get("BRAM", float('inf'))
            constraints.max_dsps = target_spec["max_resources"].get("DSP", float('inf'))
        
        if "target_throughput" in target_spec:
            constraints.min_throughput = target_spec["target_throughput"]
        
        # Create explorer
        explorer = DesignSpaceExplorer(
            self.graph,
            constraints,
            strategy="pareto" if target_spec.get("optimization_objective") == "balanced" else "single"
        )
        
        # Explore design space
        results = explorer.explore()
        
        # Select best configuration
        if target_spec.get("optimization_objective") == "throughput":
            best = max(results, key=lambda r: r.performance.throughput)
        elif target_spec.get("optimization_objective") == "latency":
            best = min(results, key=lambda r: r.performance.latency)
        else:
            # Balanced - select from Pareto frontier
            best = self._select_balanced_config(results)
        
        # Apply best configuration
        self._current_config = best.config
        self._optimized = True
        
        # Update node attributes
        self._update_attributes_from_config()
    
    def _select_balanced_config(self, results: List[DSEResult]) -> DSEResult:
        """Select balanced configuration from Pareto frontier."""
        # Find Pareto optimal configurations
        pareto_configs = find_pareto_optimal(
            results,
            objectives=['throughput', 'resource_usage'],
            directions=['maximize', 'minimize']
        )
        
        # Select middle point from Pareto frontier
        return pareto_configs[len(pareto_configs) // 2]
    
    def _update_attributes_from_config(self):
        """Update node attributes from current configuration."""
        for kernel_name, kernel_config in self._current_config.kernel_configs.items():
            for intf_name, parallelism in kernel_config.interface_parallelism.items():
                self.set_nodeattr(f"{intf_name}_parallelism", parallelism)
```

### 4. RTL Generation

Clean RTL generation using the new framework.

```python
class UnifiedRTLGenerator:
    """RTL generation for unified operators."""
    
    def __init__(self, kernel_def: KernelDefinition):
        self.kernel_def = kernel_def
        self.template_engine = TemplateEngine()
    
    def generate(self, config: Configuration, 
                 target_part: str, clock_freq: float) -> Dict[str, str]:
        """
        Generate RTL code for given configuration.
        
        Returns:
            Dictionary mapping filename to file content
        """
        files = {}
        
        # Generate main module
        main_module = self._generate_main_module(config)
        files[f"{self.kernel_def.name}.v"] = main_module
        
        # Generate wrapper if needed
        if self._needs_wrapper(config):
            wrapper = self._generate_wrapper(config)
            files[f"{self.kernel_def.name}_wrapper.v"] = wrapper
        
        # Generate configuration package
        if self.kernel_def.exposed_parameters:
            config_pkg = self._generate_config_package(config)
            files[f"{self.kernel_def.name}_config.sv"] = config_pkg
        
        # Copy support files
        support_files = self._get_support_files()
        files.update(support_files)
        
        return files
    
    def _generate_main_module(self, config: Configuration) -> str:
        """Generate main RTL module."""
        # Prepare template context
        context = {
            'module_name': self.kernel_def.name,
            'interfaces': [],
            'parameters': {},
            'config': config
        }
        
        # Add interface information
        for intf in self.kernel_def.interfaces:
            intf_config = config.get_interface_config(intf.name)
            context['interfaces'].append({
                'name': intf.name,
                'type': intf.type.value,
                'width': np.prod(intf_config.stream_shape),
                'protocol': intf.protocol.value,
                'direction': 'input' if intf.type in [InterfaceType.INPUT, InterfaceType.WEIGHT] else 'output'
            })
        
        # Add parameters
        for param_name in self.kernel_def.exposed_parameters:
            context['parameters'][param_name] = config.get_parameter(param_name)
        
        # Render template
        return self.template_engine.render('main_module.v.j2', context)
```

### 5. Example Implementation

Example of a Thresholding operator using the unified framework.

```python
class UnifiedThresholding(UnifiedHWCustomOp, UnifiedDSEMixin):
    """Thresholding operator using unified framework."""
    
    def get_kernel_definition(self) -> KernelDefinition:
        """Define thresholding kernel."""
        return KernelDefinition(
            name="thresholding",
            interfaces=[
                Interface(
                    name="input",
                    type=InterfaceType.INPUT,
                    tensor_shape=Shape([1]),  # Set from ONNX
                    block_shape=Shape([1]),   # Set from attributes
                    datatype_constraints=[
                        DatatypeConstraint("INT", 4, 8),
                        DatatypeConstraint("UINT", 4, 8)
                    ]
                ),
                Interface(
                    name="threshold",
                    type=InterfaceType.WEIGHT,
                    tensor_shape=Shape([1]),
                    block_shape=Shape([1]),
                    datatype_constraints=[
                        DatatypeConstraint("INT", 8, 32)
                    ]
                ),
                Interface(
                    name="output",
                    type=InterfaceType.OUTPUT,
                    tensor_shape=Shape([1]),
                    block_shape=Shape([1]),
                    datatype_constraints=[
                        DatatypeConstraint("UINT", 1, 8)
                    ]
                )
            ],
            constraints=[
                EqualityConstraint("output.tensor_shape", "input.tensor_shape"),
                EqualityConstraint("threshold.tensor_shape[0]", "input.tensor_shape[-1]")
            ],
            rtl_parameters={
                "N_INPUTS": "input.block_shape[0]",
                "N_THRESHOLDS": "threshold.block_shape[0]",
                "BIAS": 0
            },
            exposed_parameters={"BIAS"}
        )
    
    def get_nodeattr_types(self):
        """Define node attributes."""
        attrs = super().get_nodeattr_types()
        attrs.update({
            # Parallelism attributes (from base class)
            "input_parallelism": ("i", False, 1),
            "threshold_parallelism": ("i", False, 1),
            
            # Datatype attributes
            "input_datatype": ("s", False, "UINT8"),
            "threshold_datatype": ("s", False, "INT16"),
            "output_datatype": ("s", False, "UINT1"),
            
            # Algorithm parameters
            "bias": ("i", False, 0),
            "activation": ("s", False, "binary", {"binary", "relu", "custom"}),
            
            # Optimization attributes
            "optimization_objective": ("s", False, "balanced"),
            "auto_optimize": ("i", False, 0, {0, 1})
        })
        return attrs
    
    def execute_node(self, context, graph):
        """Execute thresholding operation."""
        # Auto-optimize if requested
        if self.get_nodeattr("auto_optimize"):
            self.optimize_for_target({
                "optimization_objective": self.get_nodeattr("optimization_objective")
            })
        
        # Get input data
        input_data = context[self.onnx_node.input[0]]
        threshold_data = context[self.onnx_node.input[1]]
        
        # Apply thresholding using current configuration
        config = self._current_config
        output_data = self._execute_thresholding(
            input_data, 
            threshold_data,
            config,
            bias=self.get_nodeattr("bias"),
            activation=self.get_nodeattr("activation")
        )
        
        # Store output
        context[self.onnx_node.output[0]] = output_data
```

## Key Design Decisions

### 1. Kernel Definition as Source of Truth
- Single definition captures all kernel information
- Used for both simulation and RTL generation
- Enables consistency across all tools

### 2. Clean Separation of Concerns
- Kernel definition: What the hardware does
- Configuration: How parallel it is
- DSE: Finding optimal configuration
- RTL generation: Creating the implementation

### 3. FINN Integration Strategy
- Inherit from HWCustomOp for compatibility
- Use node attributes for configuration
- Support both manual and automatic optimization
- Maintain FINN's execution model

### 4. Parallel Implementation Approach
- New operators have different names (e.g., "UnifiedThresholding")
- Can coexist with legacy operators
- Users opt-in through model conversion
- Enables gradual migration

## Configuration Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   ONNX Model    │────▶│ Kernel Instance  │────▶│  Configuration  │
│   Attributes    │     │  (from def)      │     │   (from DSE)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                           │
                               ▼                           ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │   Constraints    │     │  RTL Generator  │
                        │  (from pragmas)  │     │                 │
                        └──────────────────┘     └─────────────────┘
```

## Performance Optimization

### Automatic Optimization
```python
# User specifies target
node.set_attribute("auto_optimize", 1)
node.set_attribute("target_throughput", 1000.0)  # MHz
node.set_attribute("max_luts", 10000)

# Framework automatically:
# 1. Explores design space
# 2. Finds Pareto optimal configurations
# 3. Selects best match for constraints
# 4. Applies configuration
```

### Manual Configuration
```python
# User specifies exact parallelism
node.set_attribute("input_parallelism", 8)
node.set_attribute("threshold_parallelism", 4)

# Framework validates and applies
```

## Migration Strategy

### Phase 1: Core Operators
- Thresholding
- StreamingFIFO
- DWC (Data Width Converter)

### Phase 2: Complex Operators
- MVU (Matrix Vector Unit)
- ConvolutionInputGenerator
- Pool_batch

### Phase 3: Full Coverage
- All remaining operators
- Deprecated legacy implementations

## Testing and Validation

### Functional Testing
```python
def test_unified_operator(op_class, test_vectors):
    """Test unified operator implementation."""
    # Create operator
    op = op_class(onnx_node)
    
    # Test with various configurations
    for config in generate_test_configs():
        op.apply_config(config)
        
        # Verify outputs
        outputs = op.execute(test_vectors.inputs)
        assert np.allclose(outputs, test_vectors.expected_outputs)
```

### Performance Validation
```python
def validate_performance(op, constraints):
    """Validate performance meets constraints."""
    # Optimize for constraints
    op.optimize_for_target(constraints)
    
    # Get actual performance
    perf = op.get_performance_metrics()
    
    # Verify constraints met
    assert perf.throughput >= constraints.get("min_throughput", 0)
    assert perf.resources <= constraints.get("max_resources", inf)
```

## Conclusion

This unified design provides:
1. **Clean Architecture**: Clear separation between definition, configuration, and implementation
2. **Optimal Performance**: Full DSE integration for automatic optimization
3. **FINN Compatibility**: Seamless integration with existing infrastructure
4. **Migration Path**: Parallel implementation enables gradual adoption
5. **Extensibility**: Easy to add new operators and optimizations

The design leverages the full power of the Unified Kernel Modeling Framework while maintaining practical compatibility with FINN's ecosystem.