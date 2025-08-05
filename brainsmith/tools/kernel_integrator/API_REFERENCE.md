# Kernel Integrator API Reference

This document provides a comprehensive reference for all public types and functions in the kernel integrator module.

## Table of Contents

1. [Core Types](#core-types)
2. [RTL Types](#rtl-types)
3. [Metadata Types](#metadata-types)
4. [Generation Types](#generation-types)
5. [Binding Types](#binding-types)
6. [Config Types](#config-types)
7. [Integration Layer](#integration-layer)

## Core Types

### PortDirection

Enumeration for RTL port directions.

```python
from brainsmith.tools.kernel_integrator.types.core import PortDirection

class PortDirection(Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"
```

**Usage Example:**
```python
port = Port(name="data_in", direction=PortDirection.INPUT, width=32)
```

### DatatypeSpec

Represents a datatype specification with width, signedness, and constraints.

```python
from brainsmith.tools.kernel_integrator.types.core import DatatypeSpec

@dataclass
class DatatypeSpec:
    name: str
    width: Optional[int] = None
    signed: bool = False
    min_width: Optional[int] = None
    max_width: Optional[int] = None
    
    def validate(self) -> None:
        """Validate datatype constraints."""
```

**Usage Example:**
```python
# Fixed-width datatype
int16 = DatatypeSpec(name="INT16", width=16, signed=True)

# Variable-width datatype with constraints
custom = DatatypeSpec(
    name="CUSTOM",
    min_width=8,
    max_width=32,
    signed=False
)
```

### DimensionSpec

Represents dimension specifications supporting both concrete and symbolic shapes.

```python
from brainsmith.tools.kernel_integrator.types.core import DimensionSpec
from brainsmith.core.dataflow.types import ShapeSpec

@dataclass
class DimensionSpec:
    name: str
    shape: ShapeSpec  # List[Union[int, str]]
    parameters: List[str] = field(default_factory=list)
    
    def is_symbolic(self) -> bool:
        """Check if any dimension is symbolic."""
    
    def get_concrete_shape(self, param_values: Dict[str, int]) -> Tuple[int, ...]:
        """Resolve symbolic dimensions to concrete values."""
```

**Usage Example:**
```python
# Concrete dimensions
bdim = DimensionSpec(name="BDIM", shape=[1, 784])

# Symbolic dimensions
sdim = DimensionSpec(
    name="SDIM", 
    shape=["N", "SIMD"],
    parameters=["N", "SIMD"]
)

# Resolve symbolic to concrete
concrete = sdim.get_concrete_shape({"N": 10, "SIMD": 8})  # Returns (10, 8)
```

## RTL Types

### Port

Represents an RTL port with name, direction, and width.

```python
from brainsmith.tools.kernel_integrator.types.rtl import Port

@dataclass
class Port:
    name: str
    direction: PortDirection
    width: Optional[int] = None
    width_expr: Optional[str] = None
```

**Usage Example:**
```python
# Fixed-width port
clk = Port(name="clk", direction=PortDirection.INPUT, width=1)

# Parameterized width
data = Port(
    name="data", 
    direction=PortDirection.INPUT, 
    width_expr="DATA_WIDTH"
)
```

### Parameter

Represents an RTL parameter.

```python
from brainsmith.tools.kernel_integrator.types.rtl import Parameter

@dataclass
class Parameter:
    name: str
    value: Optional[str] = None
    datatype: Optional[str] = None
```

**Usage Example:**
```python
param = Parameter(name="DATA_WIDTH", value="32", datatype="integer")
```

### ParsedModule

Complete parsed RTL module structure.

```python
from brainsmith.tools.kernel_integrator.types.rtl import ParsedModule

@dataclass
class ParsedModule:
    name: str
    ports: List[Port]
    parameters: List[Parameter]
    pragmas: Dict[str, List[Any]] = field(default_factory=dict)
    source_file: Optional[Path] = None
    line_number: Optional[int] = None
```

**Usage Example:**
```python
module = ParsedModule(
    name="my_kernel",
    ports=[
        Port(name="clk", direction=PortDirection.INPUT, width=1),
        Port(name="data_in", direction=PortDirection.INPUT, width=32),
        Port(name="data_out", direction=PortDirection.OUTPUT, width=32)
    ],
    parameters=[
        Parameter(name="DATA_WIDTH", value="32")
    ]
)
```

## Metadata Types

### InterfaceMetadata

Complete interface specification including type, dimensions, and datatypes.

```python
from brainsmith.tools.kernel_integrator.types.metadata import InterfaceMetadata

@dataclass
class InterfaceMetadata:
    # Core properties
    compiler_name: str
    interface_type: InterfaceType
    pragmas: Dict[str, Any] = field(default_factory=dict)
    
    # Port information
    port_map: Dict[str, str] = field(default_factory=dict)
    
    # Datatype information
    datatype_name: Optional[str] = None
    datatype_width: Optional[int] = None
    datatype_signed: Optional[bool] = None
    datatype_spec: Optional[str] = None
    datatype_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Dimension information
    block_dimensions: Optional[List[Union[int, str]]] = None
    stream_dimensions: Optional[List[Union[int, str]]] = None
    bdim_params: List[str] = field(default_factory=list)
    sdim_params: List[str] = field(default_factory=list)
    
    # Parameter relationships
    parameter_map: Dict[str, str] = field(default_factory=dict)
    
    def get_dimension_params(self) -> List[str]:
        """Get all dimension parameters."""
    
    def has_stream_dimensions(self) -> bool:
        """Check if interface has streaming dimensions."""
```

**Usage Example:**
```python
# Input interface with streaming
input_interface = InterfaceMetadata(
    compiler_name="input0",
    interface_type=InterfaceType.INPUT,
    datatype_name="UINT8",
    datatype_width=8,
    datatype_signed=False,
    block_dimensions=[1, 784],
    stream_dimensions=[1, "SIMD"],
    sdim_params=["SIMD"],
    port_map={
        "data": "s_axis_tdata",
        "valid": "s_axis_tvalid",
        "ready": "s_axis_tready"
    }
)

# Weight interface without streaming
weight_interface = InterfaceMetadata(
    compiler_name="weight0",
    interface_type=InterfaceType.WEIGHT,
    datatype_spec="ap_fixed<16,6>",
    block_dimensions=["N", 768],
    bdim_params=["N"]
)
```

### KernelMetadata

Full kernel specification with interfaces, parameters, and pragmas.

```python
from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata

@dataclass
class KernelMetadata:
    name: str
    source_file: Path
    interfaces: List[InterfaceMetadata]
    parameters: Dict[str, Any] = field(default_factory=dict)
    pragmas: Dict[str, Any] = field(default_factory=dict)
    parsing_warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate kernel has required global control interface."""
    
    def get_interface(self, name: str) -> Optional[InterfaceMetadata]:
        """Get interface by compiler name."""
    
    def get_interfaces_by_type(self, interface_type: InterfaceType) -> List[InterfaceMetadata]:
        """Get all interfaces of a specific type."""
    
    def get_input_interfaces(self) -> List[InterfaceMetadata]:
        """Get all input interfaces."""
    
    def get_output_interfaces(self) -> List[InterfaceMetadata]:
        """Get all output interfaces."""
```

**Usage Example:**
```python
kernel = KernelMetadata(
    name="MatMul",
    source_file=Path("matmul.sv"),
    interfaces=[
        InterfaceMetadata(
            compiler_name="input0",
            interface_type=InterfaceType.INPUT,
            datatype_width=8,
            block_dimensions=[1, 784],
            stream_dimensions=[1, "SIMD"]
        ),
        InterfaceMetadata(
            compiler_name="weight0",
            interface_type=InterfaceType.WEIGHT,
            datatype_width=8,
            block_dimensions=[784, 128]
        ),
        InterfaceMetadata(
            compiler_name="output0",
            interface_type=InterfaceType.OUTPUT,
            datatype_width=32,
            block_dimensions=[1, 128]
        ),
        InterfaceMetadata(
            compiler_name="global",
            interface_type=InterfaceType.CONTROL
        )
    ],
    parameters={
        "SIMD": 8,
        "PE": 16
    }
)

# Query interfaces
inputs = kernel.get_input_interfaces()
first_input = kernel.get_interface("input0")
```

## Generation Types

### GeneratedFile

Represents a single generated file with content and metadata.

```python
from brainsmith.tools.kernel_integrator.types.generation import GeneratedFile

@dataclass
class GeneratedFile:
    filename: str
    content: str
    generator_name: str
    template_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def write(self, output_dir: Path) -> Path:
        """Write file to disk."""
```

**Usage Example:**
```python
file = GeneratedFile(
    filename="matmul_hw_custom_op.py",
    content="class MatMul(HWCustomOp):\n    ...",
    generator_name="hwcustomop",
    template_name="hw_custom_op.j2",
    metadata={"lines": 250, "classes": 1}
)

# Write to disk
output_path = file.write(Path("/output/dir"))
```

### GenerationContext

Context for template rendering.

```python
from brainsmith.tools.kernel_integrator.types.generation import GenerationContext

@dataclass
class GenerationContext:
    kernel: KernelMetadata
    config: Config
    codegen_binding: Optional[CodegenBinding] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_template_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for template rendering."""
```

### GenerationResult

Collection of all generated artifacts.

```python
from brainsmith.tools.kernel_integrator.types.generation import GenerationResult

@dataclass
class GenerationResult:
    success: bool
    files: List[GeneratedFile]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def write_all(self, output_dir: Path) -> List[Path]:
        """Write all files to disk."""
```

## Binding Types

### IOSpec

Input/output specification for interfaces.

```python
from brainsmith.tools.kernel_integrator.types.binding import IOSpec

@dataclass
class IOSpec:
    name: str
    shape: List[Union[int, str]]
    datatype: str
    pragma_map: Dict[str, Any] = field(default_factory=dict)
```

### AttributeBinding

Node attribute to RTL parameter binding.

```python
from brainsmith.tools.kernel_integrator.types.binding import AttributeBinding

@dataclass
class AttributeBinding:
    rtl_name: str
    attribute_name: str
    binding_type: str  # "NODEATTR", "INTERFACE_DATATYPE", etc.
    interface_name: Optional[str] = None
    property_name: Optional[str] = None
    transform: Optional[str] = None
```

**Usage Example:**
```python
# Bind PE parameter to node attribute
pe_binding = AttributeBinding(
    rtl_name="PE",
    attribute_name="PE",
    binding_type="NODEATTR"
)

# Bind width from interface datatype
width_binding = AttributeBinding(
    rtl_name="INPUT_WIDTH",
    attribute_name="inputDataType",
    binding_type="INTERFACE_DATATYPE",
    interface_name="input0",
    property_name="bitwidth"
)
```

### CodegenBinding

Complete codegen binding specification.

```python
from brainsmith.tools.kernel_integrator.types.binding import CodegenBinding

@dataclass
class CodegenBinding:
    inputs: List[IOSpec]
    outputs: List[IOSpec]
    weights: List[IOSpec]
    attribute_bindings: List[AttributeBinding]
    internal_datatypes: Dict[str, str] = field(default_factory=dict)
    derived_parameters: Dict[str, str] = field(default_factory=dict)
    
    def get_all_node_attributes(self) -> Set[str]:
        """Get all unique node attribute names."""
    
    def get_binding_for_param(self, param_name: str) -> Optional[AttributeBinding]:
        """Find binding for a specific RTL parameter."""
```

## Config Types

### Config

CLI and generation configuration.

```python
from brainsmith.tools.kernel_integrator.types.config import Config

@dataclass
class Config:
    output_dir: Path
    kernel_name: str
    module_name: Optional[str] = None
    overwrite: bool = False
    verbose: bool = False
    generators: Optional[List[str]] = None
    
    def to_camel_case(self) -> str:
        """Convert kernel name to CamelCase."""
    
    def to_snake_case(self) -> str:
        """Convert kernel name to snake_case."""
```

## Integration Layer

### Converters

Functions for converting between kernel integrator and dataflow types.

```python
from brainsmith.tools.kernel_integrator.converters import (
    metadata_to_kernel_definition,
    kernel_definition_to_metadata
)

# Convert KernelMetadata to dataflow KernelDefinition
kernel_def = metadata_to_kernel_definition(
    kernel_metadata,
    kernel_path=Path("kernels/matmul.sv")  # Optional
)

# Convert back to KernelMetadata
metadata = kernel_definition_to_metadata(
    kernel_def,
    source_file=Path("kernels/matmul.sv")
)
```

### Constraint Builder

Build constraints for dataflow integration.

```python
from brainsmith.tools.kernel_integrator.constraint_builder import (
    build_dimension_constraints,
    build_parameter_constraints,
    DimensionConstraint,
    ParameterConstraint
)

# Build dimension constraints
dim_constraints = build_dimension_constraints(
    shape_spec=["N", 768],
    param_names=["N"]
)

# Build parameter constraints
param_constraints = build_parameter_constraints(
    kernel_metadata,
    parameter_name="SIMD"
)
```

## Usage Examples

### Complete Workflow Example

```python
from pathlib import Path
from brainsmith.tools.kernel_integrator import KernelIntegrator
from brainsmith.tools.kernel_integrator.types.config import Config
from brainsmith.tools.kernel_integrator.converters import metadata_to_kernel_definition

# 1. Parse RTL file
config = Config(
    output_dir=Path("output"),
    kernel_name="matmul",
    verbose=True
)

integrator = KernelIntegrator(config)
kernel_metadata = integrator.parse_rtl(Path("matmul.sv"))

# 2. Convert to dataflow types if needed
kernel_def = metadata_to_kernel_definition(kernel_metadata)

# 3. Generate artifacts
result = integrator.generate(kernel_metadata)

# 4. Write files
if result.success:
    paths = result.write_all(config.output_dir)
    print(f"Generated {len(paths)} files")
else:
    print(f"Generation failed: {result.errors}")
```

### Custom Generator Example

```python
from brainsmith.tools.kernel_integrator.generators.base import GeneratorBase
from brainsmith.tools.kernel_integrator.types.generation import GeneratedFile

class MyCustomGenerator(GeneratorBase):
    name = "custom"
    template_file = "custom_template.j2"
    output_pattern = "{kernel_name}_custom.txt"
    
    def generate(self, context: GenerationContext) -> GeneratedFile:
        # Custom generation logic
        content = self.render_template(context)
        
        return GeneratedFile(
            filename=self.get_output_filename(context.kernel.name),
            content=content,
            generator_name=self.name,
            template_name=self.template_file
        )
```

## Error Handling

All type classes include validation in their `__post_init__` methods. Common exceptions:

```python
# ValueError for invalid configurations
try:
    datatype = DatatypeSpec(name="INT", width=7, min_width=8)
except ValueError as e:
    print(f"Invalid datatype: {e}")

# Missing required interfaces
try:
    kernel = KernelMetadata(
        name="test",
        source_file=Path("test.sv"),
        interfaces=[]  # Missing global control
    )
except ValueError as e:
    print(f"Invalid kernel: {e}")
```

## Best Practices

1. **Use Type Hints**: Always use proper type hints when working with these types
2. **Validate Early**: Call validation methods before processing
3. **Handle Symbolic Dimensions**: Always check `is_symbolic()` before assuming concrete values
4. **Preserve Metadata**: Use the integration layer to maintain full fidelity conversions
5. **Check Interface Types**: Use `get_interfaces_by_type()` for type-safe interface access

Arete.