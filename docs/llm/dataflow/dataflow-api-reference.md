# Dataflow API Reference

## Core Classes

### BaseDefinition
Abstract base class for all definition types.

```python
class BaseDefinition(ABC):
    @abstractmethod
    def validate(self) -> List[str]
        """Validate definition consistency. Returns list of errors."""
    
    @abstractmethod
    def create_model(self, **params) -> BaseModel
        """Create runtime model from definition."""
```

### BaseModel
Abstract base class for all model types.

```python
class BaseModel(ABC):
    definition: Optional[BaseDefinition]
    
    @abstractmethod
    def calculate_performance_metrics(self) -> Dict[str, Any]
        """Calculate performance metrics."""
```

## Definition Classes

### KernelDefinition

Container for kernel interface definitions and relationships.

```python
@dataclass
class KernelDefinition(BaseDefinition):
    name: str
    input_definitions: List[InputDefinition] = field(default_factory=list)
    output_definitions: List[OutputDefinition] = field(default_factory=list)
    relationships: List[DimensionRelationship] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
```

**Methods:**

```python
def add_input(self, input_def: InputDefinition) -> None
    """Add input definition. Raises ValueError if name exists."""

def add_output(self, output_def: OutputDefinition) -> None
    """Add output definition. Raises ValueError if name exists."""

def add_relationship(self,
                    source_name: str,
                    target_name: str,
                    relationship_type: RelationType,
                    source_dim: Optional[int] = None,
                    target_dim: Optional[int] = None,
                    **kwargs) -> None
    """Add relationship between interfaces."""

def get_input(self, name: str) -> Optional[InputDefinition]
    """Get input definition by name."""

def get_output(self, name: str) -> Optional[OutputDefinition]
    """Get output definition by name."""

def get_required_parameters(self) -> Dict[str, str]
    """Get all parameters used in tiling expressions.
    Returns dict mapping parameter names to usage context."""

def has_weights(self) -> bool
    """Check if kernel has weight inputs."""

def get_regular_inputs(self) -> List[InputDefinition]
    """Get non-weight inputs."""

def get_weight_inputs(self) -> List[InputDefinition]
    """Get weight inputs."""

def create_model(self,
                input_specs: Dict[str, Tuple[Shape, BaseDataType]],
                output_specs: Dict[str, Tuple[Shape, BaseDataType]],
                parameter_binding: Optional[Union[Dict[str, int], ParameterBinding]] = None
                ) -> KernelModel
    """Create runtime model with concrete types."""

def validate(self) -> List[str]
    """Validate kernel consistency."""
```

### InputDefinition

Schema for input interfaces with tiling and constraints.

```python
@dataclass
class InputDefinition(BaseDefinition):
    name: str
    datatype_constraints: List[DatatypeConstraintGroup] = field(default_factory=list)
    block_tiling: Optional[List[Union[int, str]]] = None
    stream_tiling: Optional[List[Union[int, str]]] = None
    optional: bool = False
    is_weight: bool = False
```

**Methods:**

```python
def create_model(self,
                tensor_dims: Shape,
                datatype: BaseDataType,
                parameter_binding: Optional[Union[Dict[str, int], ParameterBinding]] = None,
                config: Optional[Dict[str, Any]] = None) -> InputInterface
    """Create runtime input model."""

def validates_datatype(self, datatype: BaseDataType) -> bool
    """Check if datatype satisfies constraints."""

def derive_block_dims(self,
                     tensor_dims: Shape,
                     parameter_binding: Optional[ParameterBinding] = None,
                     config: Optional[Dict[str, Any]] = None) -> Shape
    """Derive block dimensions from tensor shape."""

def derive_stream_dims(self,
                      block_dims: Shape,
                      parameter_binding: Optional[ParameterBinding] = None,
                      config: Optional[Dict[str, Any]] = None) -> Shape
    """Derive stream dimensions from block shape."""

def get_tiling_parameters(self) -> Dict[str, str]
    """Get parameters used in tiling expressions."""

def validate(self) -> List[str]
    """Validate definition consistency."""
```

### OutputDefinition

Schema for output interfaces.

```python
@dataclass
class OutputDefinition(BaseDefinition):
    name: str
    datatype_constraints: List[DatatypeConstraintGroup] = field(default_factory=list)
    block_tiling: Optional[List[Union[int, str]]] = None
    # No stream_tiling - outputs compute their rates
```

**Methods:**

Similar to InputDefinition but without stream dimension configuration.

## Model Classes

### KernelModel

Runtime kernel with concrete types and dimensions.

```python
@dataclass
class KernelModel(BaseModel):
    input_models: List[InputInterface] = field(default_factory=list)
    output_models: List[OutputInterface] = field(default_factory=list)
    parameter_binding: ParameterBinding = field(default_factory=lambda: ParameterBinding({}))
    
    # Timing characteristics
    latency_cycles: Tuple[int, int] = (1, 1)
    calculation_ii: Optional[int] = None
    execution_ii: Optional[int] = None
    
    # Pipeline characteristics
    priming_cycles: int = 0
    flush_cycles: int = 0
    pipeline_depth: int = 1
    
    # Performance characteristics
    clock_freq_mhz: float = 100.0
    actual_efficiency: float = 1.0
```

**Methods:**

```python
def get_input_model(self, name: str) -> Optional[InputInterface]
    """Get input model by name."""

def get_output_model(self, name: str) -> Optional[OutputInterface]
    """Get output model by name."""

def get_sdim_parameters(self) -> Dict[str, SDIMParameterInfo]
    """Get SDIM parameters that need configuration."""

def configure_sdim(self, config: Dict[str, Union[int, List[int], Dict[int, int]]]) -> None
    """Configure SDIM for input interfaces.
    
    Args:
        config: Maps input names to SDIM specs:
            - int: Uniform for all dimensions
            - List[int]: Per-dimension values
            - Dict[int, int]: Sparse specification
    """

def compute_output_rates(self) -> None
    """Compute output streaming rates from inputs and kernel behavior."""

def get_sdim_state(self) -> Dict[str, Shape]
    """Get current SDIM values for all inputs."""

def initiation_interval(self) -> int
    """Compute kernel initiation interval."""

def throughput_fps(self) -> float
    """Compute throughput in inferences/second."""

def calculate_performance_metrics(self, frequency_mhz: float = 100.0) -> Dict[str, Any]
    """Calculate comprehensive performance metrics."""

def clear_cache(self) -> None
    """Clear performance metric caches."""
```

### InputInterface

Runtime input with configurable streaming.

```python
@dataclass
class InputInterface(BaseModel):
    tensor_dims: Shape
    block_dims: Shape
    stream_dims: Shape
    datatype: BaseDataType
    definition: Optional[InputDefinition] = None
    parameter_binding: Optional[ParameterBinding] = None
    
    # Internal state
    _sdim: Optional[Shape] = None
```

**Properties:**

```python
@property
def sdim(self) -> Shape
    """Get/set streaming dimensions."""

@property
def streaming_bandwidth(self) -> int
    """Elements transferred per cycle."""

@property
def bandwidth_bits(self) -> int
    """Bits transferred per cycle."""

@property
def initiation_interval(self) -> int
    """Cycles to process one block."""
```

**Methods:**

```python
def calculate_performance_metrics(self) -> Dict[str, Any]
    """Calculate interface performance metrics."""

def validate_sdim(self, sdim_values: Shape) -> List[str]
    """Validate SDIM against block dimensions."""
```

### OutputInterface

Runtime output with computed streaming.

```python
@dataclass
class OutputInterface(BaseModel):
    tensor_dims: Shape
    block_dims: Shape
    datatype: BaseDataType
    definition: Optional[OutputDefinition] = None
    
    # Computed streaming rate
    _streaming_rate: Optional[int] = None
```

**Properties:**

```python
@property
def streaming_rate(self) -> int
    """Elements produced per cycle (computed)."""

@property
def bandwidth_bits(self) -> int
    """Bits produced per cycle."""
```

**Methods:**

```python
def set_streaming_rate(self, rate: int) -> None
    """Set streaming rate (called by kernel)."""

def calculate_performance_metrics(self) -> Dict[str, Any]
    """Calculate output performance metrics."""
```

## Type System

### DatatypeConstraintGroup

Constraint specification for allowed datatypes.

```python
@dataclass
class DatatypeConstraintGroup:
    base_type: str      # "INT", "UINT", "FIXED", "FLOAT"
    min_width: int      # Minimum bit width
    max_width: int      # Maximum bit width
```

### BaseDataType (from QONNX)

```python
# Re-exported QONNX types
from qonnx.core.datatype import DataType

# Usage
DataType["INT8"]    # 8-bit signed integer
DataType["UINT16"]  # 16-bit unsigned integer
DataType["FIXED<16,8>"]  # 16-bit fixed, 8 fractional
```

## Tiling System

### TilingExpr

Single tiling expression.

```python
class TilingExprType(Enum):
    SINGLETON = "singleton"  # Value: 1
    FULL = "full"           # Value: ":"
    LITERAL = "literal"     # Value: integer
    PARAMETER = "parameter" # Value: string

@dataclass
class TilingExpr:
    expr_type: TilingExprType
    value: Optional[Union[int, str]] = None
    
    @classmethod
    def from_value(cls, value: Union[int, str]) -> 'TilingExpr'
        """Create from value (1, ":", int, or string)."""
    
    @property
    def is_static(self) -> bool
        """Check if has static value."""
    
    @property
    def is_parameter(self) -> bool
        """Check if is parameter."""
    
    @property
    def parameter_name(self) -> Optional[str]
        """Get parameter name if parameter type."""
```

### TilingSpec

Collection of tiling expressions.

```python
@dataclass
class TilingSpec:
    expressions: List[TilingExpr]
    
    def __init__(self, values: List[Union[int, str]])
        """Initialize from list of values."""
    
    @property
    def ndim(self) -> int
        """Number of dimensions."""
    
    def get_parameters(self) -> Set[str]
        """Get all parameter names."""
    
    def validate_against_shape(self, shape: List[int]) -> List[str]
        """Validate against tensor shape."""
    
    def resolve(self, shape: List[int], parameters: dict) -> List[int]
        """Resolve to concrete sizes."""
    
    def to_list(self) -> List[Union[int, str]]
        """Convert back to list form."""
```

### TilingStrategy

Coordinates block and stream tiling.

```python
class TilingStrategy:
    def __init__(self,
                 block_spec: Optional[TilingSpec],
                 stream_spec: Optional[TilingSpec],
                 order: TilingOrder = TilingOrder.ROW_MAJOR)
    
    def get_required_parameters(self) -> Dict[str, str]
        """Get all required parameters."""
    
    def apply_block_tiling(self,
                          tensor_shape: Shape,
                          parameters: Dict[str, int]) -> TilingResult
        """Apply block tiling to tensor."""
    
    def apply_stream_tiling(self,
                           block_shape: Shape,
                           parameters: Dict[str, int]) -> TilingResult
        """Apply stream tiling to block."""
```

## Relationships

### RelationType

Types of interface relationships.

```python
class RelationType(Enum):
    EQUAL = "equal"              # Full equality
    MULTIPLE = "multiple"        # Factor relationship
    DIVISIBLE = "divisible"      # Divisibility constraint
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    COUPLED = "coupled"          # Custom function
    DEPENDENT = "dependent"      # Dimension dependency
```

### DimensionRelationship

Relationship between interface dimensions.

```python
@dataclass(frozen=True)
class DimensionRelationship:
    source_interface: str
    target_interface: str
    relation: RelationType
    source_dim: Optional[int] = None  # None = total size
    target_dim: Optional[int] = None  # None = total size
    factor: Optional[Union[int, float]] = None
    coupling_func: Optional[Callable[[int, int], bool]] = None
    dependency_type: Optional[str] = None  # "copy", "scaled", "min"
    description: str = ""
    
    def describe(self) -> str
        """Human-readable description."""
    
    def evaluate(self, interfaces: Dict[str, Any]) -> bool
        """Evaluate relationship satisfaction."""
```

## Utility Classes

### ParameterBinding

Runtime parameter values.

```python
@dataclass
class ParameterBinding:
    parameters: Dict[str, Union[int, float, str]]
    constants: Dict[str, Union[int, float]] = None
    
    def get_value(self, name: str, default: Any = None) -> Any
        """Get parameter value."""
    
    def update(self, **kwargs) -> 'ParameterBinding'
        """Create new binding with updates."""
```

### Shape

Type alias for dimension lists.

```python
Shape = Union[List[int], Tuple[int, ...]]
```

### SDIMParameterInfo

SDIM configuration information.

```python
@dataclass
class SDIMParameterInfo:
    interface_name: str
    total_dimensions: int
    free_dimensions: List[int]
    constrained_dimensions: Dict[int, str]
    block_dims: Shape
```

## Validation Functions

```python
def validate_datatype_against_constraints(
    datatype: BaseDataType,
    constraints: List[DatatypeConstraintGroup]
) -> bool
    """Check if datatype satisfies any constraint group."""
```

## Example Usage

### Complete Kernel Definition

```python
from brainsmith.core.dataflow import *
from qonnx.core.datatype import DataType

# 1. Define kernel
kernel_def = KernelDefinition(name="conv2d")

# 2. Add inputs
kernel_def.add_input(InputDefinition(
    name="input",
    datatype_constraints=[DatatypeConstraintGroup("INT", 8, 8)],
    block_tiling=[1, "CH_IN", 14, 14],
    stream_tiling=[1, "SIMD", 1, 1]
))

kernel_def.add_input(InputDefinition(
    name="weights",
    datatype_constraints=[DatatypeConstraintGroup("INT", 8, 8)],
    block_tiling=["CH_OUT", "CH_IN", 3, 3],
    stream_tiling=["PE", "SIMD", 1, 1],
    is_weight=True
))

# 3. Add output
kernel_def.add_output(OutputDefinition(
    name="output",
    datatype_constraints=[DatatypeConstraintGroup("INT", 32, 32)],
    block_tiling=[1, "CH_OUT", 12, 12]
))

# 4. Add relationships
kernel_def.add_relationship(
    "input", "weights",
    RelationType.DEPENDENT,
    source_dim=1, target_dim=1
)

# 5. Create model
model = kernel_def.create_model(
    input_specs={
        "input": ((1, 64, 224, 224), DataType["INT8"]),
        "weights": ((128, 64, 3, 3), DataType["INT8"])
    },
    output_specs={
        "output": ((1, 128, 222, 222), DataType["INT32"])
    },
    parameter_binding={
        "CH_IN": 16, "CH_OUT": 32,
        "SIMD": 4, "PE": 8
    }
)

# 6. Configure and analyze
model.configure_sdim({"input": 4, "weights": [8, 4, 1, 1]})
metrics = model.calculate_performance_metrics(frequency_mhz=200)
```