# Brainsmith-2: Advanced Topics

## Dataflow Modeling Deep Dive

### Three-Tier Dimension System (qDim/tDim/sDim)

The **Interface-Wise Dataflow Modeling Framework** implements a novel three-tier dimension system that provides mathematical foundations for hardware optimization. This system enables automatic parallelism optimization and resource estimation through dimensional relationships.

#### Dimensional Hierarchy

**qDim (Query Dimension)**
- Represents the **original tensor query dimension** from the neural network
- Corresponds to the semantic meaning in the neural network context
- Examples: sequence length in transformers, batch size in CNNs, feature dimensions

**tDim (Tensor Dimension)**  
- Defines the **processing tensor granularity** for hardware implementation
- Represents the size of tensor chunks processed in parallel
- Must satisfy: `tDim ≤ qDim`

**sDim (Stream Dimension)**
- Specifies **hardware streaming parallelism** and data organization
- Determines the number of parallel processing elements
- Must satisfy: `sDim ≤ tDim`

#### Mathematical Relationships

**Fundamental Constraint**
```
sDim ≤ tDim ≤ qDim
qDim × tDim = original_tensor_shape
```

**Parallelism Calculations**
```python
# Streaming parallelism
stream_parallelism = sDim

# Tensor processing parallelism  
tensor_parallelism = tDim / sDim

# Total effective parallelism
total_parallelism = qDim / (tDim / sDim) = (qDim × sDim) / tDim
```

**Performance Optimization**
```python
from brainsmith.dataflow.core.dataflow_model import DataflowModel

class AdvancedDataflowModel(DataflowModel):
    def optimize_dimensional_configuration(self, constraints):
        """Optimize qDim/tDim/sDim configuration for performance."""
        
        # Objective function: minimize latency while respecting constraints
        def objective(dims):
            q_dim, t_dim, s_dim = dims
            
            # Calculate performance metrics
            initiation_interval = self.calculate_ii(t_dim, s_dim)
            latency = (q_dim / s_dim) * initiation_interval
            throughput = 1.0 / initiation_interval
            
            # Resource utilization
            lut_usage = self.estimate_lut_usage(t_dim, s_dim)
            dsp_usage = self.estimate_dsp_usage(s_dim)
            
            # Multi-objective optimization
            return {
                'latency': latency,
                'throughput': throughput,
                'resource_efficiency': (lut_usage + dsp_usage) / constraints['max_resources']
            }
        
        # Constraint satisfaction
        def constraints_satisfied(dims):
            q_dim, t_dim, s_dim = dims
            
            # Dimensional constraints
            if not (s_dim <= t_dim <= q_dim):
                return False
            
            # Resource constraints  
            if self.estimate_total_resources(t_dim, s_dim) > constraints['max_resources']:
                return False
                
            # Frequency constraints
            if self.estimate_max_frequency(s_dim) < constraints['min_frequency']:
                return False
                
            return True
        
        # Optimization algorithm (simplified)
        optimal_dims = self._pareto_optimization(objective, constraints_satisfied)
        return optimal_dims
```

#### Practical Example: BERT Attention

```python
# BERT Multi-Head Attention configuration
attention_interface = DataflowInterface(
    name="attention_input",
    interface_type="INPUT",
    qDim=512,      # Sequence length
    tDim=64,       # Processing chunk size (efficient for attention)
    sDim=8,        # 8-way parallel processing
    dtype="INT8"
)

# Mathematical relationships for this configuration:
# - Process 8 elements in parallel (sDim=8)
# - Each processing cycle handles 64-element chunks (tDim=64)  
# - Total sequence length is 512 (qDim=512)
# - Cycles required: 512 / 8 = 64 cycles
# - Processing efficiency: 64/8 = 8 elements per cycle
```

### Custom Interface Development

#### Advanced Interface Types

**Custom Protocol Interfaces**
```python
from brainsmith.dataflow.core.dataflow_interface import DataflowInterface, DataflowInterfaceType

class CustomProtocolInterface(DataflowInterface):
    """Custom interface for specialized hardware protocols."""
    
    def __init__(self, name, protocol_spec, **kwargs):
        super().__init__(name, "CUSTOM", **kwargs)
        self.protocol_spec = protocol_spec
        self._validate_protocol()
    
    def _validate_protocol(self):
        """Validate custom protocol specification."""
        required_signals = ['valid', 'ready', 'data']
        
        for signal in required_signals:
            if signal not in self.protocol_spec:
                raise ValueError(f"Missing required signal: {signal}")
    
    def generate_interface_logic(self):
        """Generate hardware-specific interface logic."""
        return {
            'handshake_logic': self._generate_handshake(),
            'data_path': self._generate_datapath(),
            'flow_control': self._generate_flow_control()
        }
    
    def _generate_handshake(self):
        """Generate protocol-specific handshake logic."""
        return f"""
        // Custom protocol handshake for {self.name}
        wire {self.name}_transaction = {self.name}_valid && {self.name}_ready;
        wire {self.name}_stall = {self.name}_valid && !{self.name}_ready;
        """
```

**Memory Interface Modeling**
```python
class MemoryInterface(DataflowInterface):
    """Specialized interface for memory access patterns."""
    
    def __init__(self, name, memory_type, access_pattern, **kwargs):
        super().__init__(name, "MEMORY", **kwargs)
        self.memory_type = memory_type  # BRAM, URAM, DDR
        self.access_pattern = access_pattern  # SEQUENTIAL, RANDOM, BURST
    
    def calculate_memory_bandwidth(self):
        """Calculate required memory bandwidth."""
        data_rate = self.sDim * self.calculate_frequency()
        data_width = self.get_data_width()
        
        return {
            'bandwidth_gbps': (data_rate * data_width) / 1e9,
            'access_pattern': self.access_pattern,
            'memory_efficiency': self._calculate_efficiency()
        }
    
    def optimize_memory_access(self, memory_constraints):
        """Optimize memory access pattern for performance."""
        if self.memory_type == "DDR":
            return self._optimize_ddr_access(memory_constraints)
        elif self.memory_type == "BRAM":
            return self._optimize_bram_access(memory_constraints)
        else:
            return self._optimize_generic_access(memory_constraints)
```

## Template System Architecture

### Template Engine Deep Dive

#### Jinja2 Integration and Optimization

**Advanced Template Context Building**
```python
from brainsmith.tools.hw_kernel_gen.enhanced_template_context import TemplateContextBuilder

class AdvancedTemplateContextBuilder(TemplateContextBuilder):
    """Advanced template context with optimization-aware building."""
    
    def build_optimized_context(self, dataflow_model, optimization_level='balanced'):
        """Build context with optimization-specific information."""
        
        base_context = self.build_base_context(dataflow_model)
        
        # Add optimization-specific context
        optimization_context = {
            'optimization_level': optimization_level,
            'parallelism_config': self._optimize_parallelism(dataflow_model),
            'resource_allocation': self._allocate_resources(dataflow_model),
            'memory_optimization': self._optimize_memory_usage(dataflow_model),
            'pipeline_configuration': self._configure_pipeline(dataflow_model)
        }
        
        return {**base_context, **optimization_context}
    
    def _optimize_parallelism(self, model):
        """Generate parallelism configuration for templates."""
        bounds = model.get_parallelism_bounds()
        
        return {
            'input_parallelism': bounds['input']['optimal'],
            'compute_parallelism': bounds['compute']['optimal'], 
            'output_parallelism': bounds['output']['optimal'],
            'pipeline_depth': self._calculate_pipeline_depth(bounds)
        }
```

**Template Inheritance and Composition**
```jinja2
{# base_hwcustomop.py.j2 - Base template with common functionality #}
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp

class {{ class_name }}(AutoHWCustomOp):
    """Auto-generated {{ operation_type }} implementation."""
    
    def __init__(self, onnx_node, **kwargs):
        # Create dataflow model
        dataflow_model = self._create_dataflow_model()
        super().__init__(onnx_node, dataflow_model, **kwargs)
    
    def _create_dataflow_model(self):
        """Create operation-specific dataflow model."""
        {% block dataflow_model_creation %}
        # Default implementation
        return DataflowModel(
            interfaces={{ interfaces | tojson }},
            operation_type="{{ operation_type }}"
        )
        {% endblock %}
    
    {% block custom_methods %}
    # Custom operation-specific methods
    {% endblock %}
    
    {% block performance_methods %}
    # Performance-specific implementations
    {% if optimization_level == 'aggressive' %}
    def get_exp_cycles(self):
        # Optimized cycle calculation
        return {{ optimized_cycles }}
    {% endif %}
    {% endblock %}
```

**Specialized Template Generation**
```jinja2
{# specialized_attention.py.j2 - BERT attention-specific template #}
{% extends "base_hwcustomop.py.j2" %}

{% block dataflow_model_creation %}
# BERT attention-specific dataflow model
interfaces = [
    DataflowInterface(
        name="query", interface_type="INPUT",
        qDim={{ sequence_length }}, tDim={{ attention_chunk }}, sDim={{ parallelism.query }}
    ),
    DataflowInterface(
        name="key", interface_type="INPUT", 
        qDim={{ sequence_length }}, tDim={{ attention_chunk }}, sDim={{ parallelism.key }}
    ),
    DataflowInterface(
        name="value", interface_type="INPUT",
        qDim={{ sequence_length }}, tDim={{ attention_chunk }}, sDim={{ parallelism.value }}
    ),
    DataflowInterface(
        name="attention_output", interface_type="OUTPUT",
        qDim={{ sequence_length }}, tDim={{ output_chunk }}, sDim={{ parallelism.output }}
    )
]

return DataflowModel(
    interfaces=interfaces,
    operation_type="multi_head_attention",
    attention_heads={{ num_heads }},
    head_dimension={{ head_dim }}
)
{% endblock %}

{% block custom_methods %}
def calculate_attention_weights(self, query, key):
    """Attention-specific weight calculation."""
    # Hardware-optimized attention computation
    return self.dataflow_model.compute_attention_matrix(query, key)

def apply_attention_mask(self, attention_weights, mask):
    """Apply attention mask in hardware-efficient manner."""
    return self.dataflow_model.apply_mask(attention_weights, mask)
{% endblock %}
```

### Custom Template Development

#### Template Development Workflow

**1. Template Structure Definition**
```python
# Define template metadata
template_metadata = {
    'name': 'custom_operation',
    'version': '1.0',
    'target_operations': ['custom_conv', 'custom_pooling'],
    'optimization_targets': ['latency', 'throughput', 'resource_efficiency'],
    'required_context': ['input_shape', 'kernel_config', 'parallelism']
}
```

**2. Context Schema Definition**
```python
from pydantic import BaseModel
from typing import List, Dict, Any

class CustomOperationContext(BaseModel):
    """Schema for custom operation template context."""
    
    operation_name: str
    input_shapes: List[List[int]]
    kernel_configuration: Dict[str, Any]
    parallelism_config: Dict[str, int]
    resource_constraints: Dict[str, int]
    optimization_objectives: List[str]
    
    class Config:
        extra = "allow"  # Allow additional context fields
```

**3. Template Implementation**
```jinja2
{# custom_operation.py.j2 #}
"""
Custom {{ operation_name }} implementation
Generated for {{ target_device }} with {{ optimization_objectives | join(', ') }} optimization
"""

from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.dataflow import DataflowInterface, DataflowModel

class {{ class_name }}(AutoHWCustomOp):
    """{{ operation_name }} with custom optimizations."""
    
    def __init__(self, onnx_node, **kwargs):
        # Build dataflow model with custom configuration
        interfaces = self._build_custom_interfaces()
        dataflow_model = DataflowModel(
            interfaces=interfaces,
            operation_type="{{ operation_name.lower() }}",
            custom_config={{ kernel_configuration | tojson }}
        )
        
        super().__init__(onnx_node, dataflow_model, **kwargs)
    
    def _build_custom_interfaces(self):
        """Build operation-specific interfaces."""
        interfaces = []
        
        {% for input_shape in input_shapes %}
        interfaces.append(DataflowInterface(
            name="input_{{ loop.index0 }}",
            interface_type="INPUT",
            qDim={{ input_shape[0] }},
            tDim={{ input_shape[1] if input_shape|length > 1 else input_shape[0] }},
            sDim={{ parallelism_config.get('input_' + loop.index0|string, 1) }}
        ))
        {% endfor %}
        
        return interfaces
    
    {% if 'latency' in optimization_objectives %}
    def get_exp_cycles(self):
        """Latency-optimized cycle calculation."""
        base_cycles = {{ kernel_configuration.get('base_cycles', 100) }}
        parallelism_factor = {{ parallelism_config.get('compute_parallelism', 1) }}
        return max(1, base_cycles // parallelism_factor)
    {% endif %}
    
    {% if 'resource_efficiency' in optimization_objectives %}
    def lut_estimation(self):
        """Resource-efficient LUT estimation."""
        base_luts = {{ kernel_configuration.get('base_luts', 1000) }}
        efficiency_factor = {{ resource_constraints.get('efficiency_factor', 1.0) }}
        return int(base_luts * efficiency_factor)
    {% endif %}
```

## Performance Optimization

### Parallelism Analysis and Optimization

#### Mathematical Optimization Framework

**Parallelism Optimization Algorithm**
```python
from scipy.optimize import minimize
import numpy as np

class ParallelismOptimizer:
    """Advanced parallelism optimization using mathematical programming."""
    
    def __init__(self, dataflow_model, constraints):
        self.model = dataflow_model
        self.constraints = constraints
    
    def optimize_multi_objective(self, objectives=['latency', 'throughput', 'resources']):
        """Multi-objective optimization of parallelism parameters."""
        
        def objective_function(params):
            """Combined objective function for optimization."""
            parallelism_config = self._params_to_config(params)
            
            # Calculate individual objectives
            latency = self._calculate_latency(parallelism_config)
            throughput = self._calculate_throughput(parallelism_config)
            resource_usage = self._calculate_resource_usage(parallelism_config)
            
            # Normalize objectives (0-1 scale)
            normalized_latency = latency / self.constraints['max_latency']
            normalized_throughput = self.constraints['min_throughput'] / throughput
            normalized_resources = resource_usage / self.constraints['max_resources']
            
            # Weighted combination
            weights = {'latency': 0.4, 'throughput': 0.4, 'resources': 0.2}
            
            return (weights['latency'] * normalized_latency + 
                   weights['throughput'] * normalized_throughput + 
                   weights['resources'] * normalized_resources)
        
        def constraint_function(params):
            """Constraint satisfaction function."""
            config = self._params_to_config(params)
            
            constraints = []
            
            # Resource constraints
            total_resources = self._calculate_resource_usage(config)
            constraints.append(self.constraints['max_resources'] - total_resources)
            
            # Performance constraints
            throughput = self._calculate_throughput(config)
            constraints.append(throughput - self.constraints['min_throughput'])
            
            # Dimensional constraints (qDim/tDim/sDim relationships)
            for interface in self.model.interfaces:
                constraints.append(interface.tDim - interface.sDim)
                constraints.append(interface.qDim - interface.tDim)
            
            return np.array(constraints)
        
        # Initial guess (conservative parallelism)
        initial_params = self._config_to_params(self._get_conservative_config())
        
        # Optimization bounds (parallelism ranges)
        bounds = [(1, 64) for _ in initial_params]  # 1 to 64-way parallelism
        
        # Constraint specification
        constraint_spec = {'type': 'ineq', 'fun': constraint_function}
        
        # Solve optimization problem
        result = minimize(
            objective_function,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_spec,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            optimal_config = self._params_to_config(result.x)
            return {
                'config': optimal_config,
                'performance': self._analyze_performance(optimal_config),
                'resource_usage': self._calculate_resource_usage(optimal_config)
            }
        else:
            raise OptimizationError(f"Optimization failed: {result.message}")
```

#### Performance Analysis Tools

**Initiation Interval Analysis**
```python
class InitiationIntervalAnalyzer:
    """Analyze and optimize initiation intervals for dataflow operations."""
    
    def __init__(self, dataflow_model):
        self.model = dataflow_model
    
    def analyze_critical_path(self):
        """Identify critical path limitations for II optimization."""
        
        # Build dependency graph
        dependencies = self._build_dependency_graph()
        
        # Calculate critical path
        critical_path = self._find_critical_path(dependencies)
        
        # Analyze bottlenecks
        bottlenecks = self._identify_bottlenecks(critical_path)
        
        return {
            'critical_path_length': len(critical_path),
            'critical_operations': critical_path,
            'bottlenecks': bottlenecks,
            'optimization_opportunities': self._suggest_optimizations(bottlenecks)
        }
    
    def optimize_pipeline_depth(self, target_ii=1):
        """Optimize pipeline depth to achieve target initiation interval."""
        
        current_ii = self.model.calculate_initiation_intervals()
        
        if current_ii <= target_ii:
            return {'status': 'optimal', 'current_ii': current_ii}
        
        # Calculate required pipeline depth
        required_depth = self._calculate_required_depth(target_ii)
        
        # Analyze resource implications
        resource_impact = self._analyze_resource_impact(required_depth)
        
        return {
            'status': 'optimization_needed',
            'current_ii': current_ii,
            'target_ii': target_ii,
            'required_pipeline_depth': required_depth,
            'resource_impact': resource_impact,
            'feasibility': self._check_feasibility(resource_impact)
        }
```

### Resource Estimation

#### Advanced Resource Modeling

**Multi-Level Resource Estimation**
```python
class AdvancedResourceEstimator:
    """Comprehensive resource estimation with accuracy modeling."""
    
    def __init__(self, target_device='ultrascale_plus'):
        self.device_characteristics = self._load_device_characteristics(target_device)
        self.estimation_models = self._load_estimation_models()
    
    def estimate_comprehensive_resources(self, dataflow_model, parallelism_config):
        """Comprehensive resource estimation with confidence intervals."""
        
        estimates = {}
        
        # LUT estimation with multiple models
        lut_estimates = self._estimate_luts_multi_model(dataflow_model, parallelism_config)
        estimates['luts'] = {
            'conservative': lut_estimates['worst_case'],
            'optimistic': lut_estimates['best_case'],
            'realistic': lut_estimates['expected'],
            'confidence': lut_estimates['confidence']
        }
        
        # DSP estimation
        dsp_estimates = self._estimate_dsps(dataflow_model, parallelism_config)
        estimates['dsps'] = dsp_estimates
        
        # Memory estimation (BRAM/URAM)
        memory_estimates = self._estimate_memory(dataflow_model, parallelism_config)
        estimates['memory'] = memory_estimates
        
        # Routing estimation
        routing_estimates = self._estimate_routing(dataflow_model, parallelism_config)
        estimates['routing'] = routing_estimates
        
        # Power estimation
        power_estimates = self._estimate_power(estimates, parallelism_config)
        estimates['power'] = power_estimates
        
        return estimates
    
    def _estimate_luts_multi_model(self, model, config):
        """LUT estimation using multiple models for accuracy."""
        
        # Model 1: Linear scaling model
        linear_estimate = self._lut_linear_model(model, config)
        
        # Model 2: Complexity-based model
        complexity_estimate = self._lut_complexity_model(model, config)
        
        # Model 3: Historical data model
        historical_estimate = self._lut_historical_model(model, config)
        
        # Ensemble prediction
        estimates = [linear_estimate, complexity_estimate, historical_estimate]
        weights = [0.3, 0.4, 0.3]  # Based on historical accuracy
        
        expected = sum(est * weight for est, weight in zip(estimates, weights))
        variance = sum(weight * (est - expected)**2 for est, weight in zip(estimates, weights))
        
        return {
            'best_case': min(estimates),
            'worst_case': max(estimates),
            'expected': expected,
            'confidence': 1.0 - (variance / expected**2)  # Confidence based on agreement
        }
```

## Extension Development

### Custom Blueprint Creation

#### Blueprint Development Framework

**Blueprint Structure**
```python
from brainsmith.blueprints import BlueprintBase, register_blueprint
from brainsmith.core.build_steps import BuildStep

@register_blueprint("custom_vision_transformer")
class VisionTransformerBlueprint(BlueprintBase):
    """Custom blueprint for Vision Transformer models."""
    
    def __init__(self):
        super().__init__()
        self.model_requirements = {
            'model_type': 'vision_transformer',
            'required_ops': ['patch_embedding', 'multi_head_attention', 'mlp'],
            'quantization': 'optional'
        }
    
    def validate_model(self, model):
        """Validate that model is suitable for this blueprint."""
        # Check model structure
        if not self._has_patch_embedding(model):
            raise ValueError("Model missing patch embedding layer")
        
        if not self._has_transformer_blocks(model):
            raise ValueError("Model missing transformer blocks")
        
        return True
    
    def build_pipeline(self, model, args):
        """Define custom build pipeline for Vision Transformers."""
        
        pipeline = [
            # Preprocessing steps
            BuildStep("preprocess_model", self._preprocess_vit_model),
            BuildStep("optimize_patch_embedding", self._optimize_patch_embedding),
            
            # Custom ViT transformations
            BuildStep("optimize_attention_patterns", self._optimize_attention_patterns),
            BuildStep("optimize_mlp_blocks", self._optimize_mlp_blocks),
            BuildStep("fuse_layer_norms", self._fuse_layer_norms),
            
            # Hardware mapping
            BuildStep("map_attention_to_hardware", self._map_attention_hardware),
            BuildStep("map_mlp_to_hardware", self._map_mlp_hardware),
            
            # Optimization passes
            BuildStep("optimize_memory_access", self._optimize_memory_access),
            BuildStep("balance_compute_pipeline", self._balance_pipeline),
            
            # Validation
            BuildStep("validate_performance", self._validate_performance),
            BuildStep("generate_performance_report", self._generate_report)
        ]
        
        return pipeline
    
    def _optimize_attention_patterns(self, model, state):
        """Custom attention optimization for ViT models."""
        
        # Analyze attention patterns
        attention_analysis = self._analyze_attention_patterns(model)
        
        # Apply ViT-specific optimizations
        if attention_analysis['sparsity'] > 0.7:
            model = self._apply_sparse_attention_optimization(model)
        
        if attention_analysis['locality'] > 0.8:
            model = self._apply_local_attention_optimization(model)
        
        return model
```

### New Generator Development

#### Generator Framework

**Custom Generator Implementation**
```python
from brainsmith.tools.hw_kernel_gen.enhanced_generator_base import EnhancedGeneratorBase
from brainsmith.tools.hw_kernel_gen.enhanced_data_structures import GeneratedArtifact

class CustomVerificationGenerator(EnhancedGeneratorBase):
    """Generator for formal verification artifacts."""
    
    def __init__(self, template_manager, config):
        super().__init__(template_manager, config)
        self.generator_type = "formal_verification"
    
    def generate(self, context):
        """Generate formal verification suite."""
        
        # Build verification-specific context
        verification_context = self._build_verification_context(context)
        
        # Generate multiple verification artifacts
        artifacts = []
        
        # Generate property specifications
        property_spec = self._generate_property_specifications(verification_context)
        artifacts.append(property_spec)
        
        # Generate assertion library
        assertion_lib = self._generate_assertion_library(verification_context)
        artifacts.append(assertion_lib)
        
        # Generate verification testbench
        testbench = self._generate_verification_testbench(verification_context)
        artifacts.append(testbench)
        
        # Generate verification script
        verification_script = self._generate_verification_script(verification_context)
        artifacts.append(verification_script)
        
        return artifacts
    
    def _build_verification_context(self, base_context):
        """Build verification-specific template context."""
        
        dataflow_model = base_context['dataflow_model']
        
        verification_context = {
            **base_context,
            'properties': self._extract_properties(dataflow_model),
            'invariants': self._extract_invariants(dataflow_model),
            'coverage_targets': self._define_coverage_targets(dataflow_model),
            'formal_methods': self._select_formal_methods(dataflow_model)
        }
        
        return verification_context
    
    def _generate_property_specifications(self, context):
        """Generate formal property specifications."""
        
        template = self.template_manager.get_template('formal_properties.psl.j2')
        content = template.render(context)
        
        return GeneratedArtifact(
            file_name=f"{context['kernel_name']}_properties.psl",
            content=content,
            artifact_type="formal_verification",
            metadata={
                'property_count': len(context['properties']),
                'coverage_targets': context['coverage_targets']
            }
        )
```

### RTL Parser Extensions

#### Custom Grammar Development

**Extending SystemVerilog Grammar**
```javascript
// Custom tree-sitter grammar extension for specialized constructs
module.exports = grammar(require('./base_systemverilog'), {
  name: 'systemverilog_extended',
  
  rules: {
    // Extend base grammar with custom pragma support
    custom_pragma: $ => seq(
      '(*',
      choice(
        $.dataflow_pragma,
        $.performance_pragma,
        $.resource_pragma
      ),
      '*)'
    ),
    
    dataflow_pragma: $ => seq(
      'dataflow',
      field('interface_name', $.identifier),
      field('dimension_spec', $.dimension_specification)
    ),
    
    dimension_specification: $ => seq(
      'qDim', '=', field('q_dim', $.number),
      'tDim', '=', field('t_dim', $.number),
      'sDim', '=', field('s_dim', $.number)
    ),
    
    performance_pragma: $ => seq(
      'performance',
      field('target', choice('latency', 'throughput', 'power')),
      field('value', $.number),
      optional(field('unit', $.identifier))
    )
  }
});
```

**Custom Parser Implementation**
```python
from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser

class ExtendedRTLParser(RTLParser):
    """Extended RTL parser with custom pragma support."""
    
    def __init__(self):
        super().__init__()
        self.grammar_extensions = ['dataflow_pragmas', 'performance_hints']
    
    def parse_extended_pragmas(self, ast_node):
        """Parse extended pragma information."""
        
        pragmas = {}
        
        # Find custom pragma nodes
        pragma_nodes = self._find_nodes_by_type(ast_node, 'custom_pragma')
        
        for pragma_node in pragma_nodes:
            pragma_type = self._get_pragma_type(pragma_node)
            
            if pragma_type == 'dataflow':
                pragmas.update(self._parse_dataflow_pragma(pragma_node))
            elif pragma_type == 'performance':
                pragmas.update(self._parse_performance_pragma(pragma_node))
            elif pragma_type == 'resource':
                pragmas.update(self._parse_resource_pragma(pragma_node))
        
        return pragmas
    
    def _parse_dataflow_pragma(self, pragma_node):
        """Parse dataflow-specific pragma information."""
        
        interface_name = self._get_field_value(pragma_node, 'interface_name')
        
        # Extract dimension specification
        dim_spec = self._get_field_value(pragma_node, 'dimension_spec')
        q_dim = self._get_field_value(dim_spec, 'q_dim')
        t_dim = self._get_field_value(dim_spec, 't_dim')
        s_dim = self._get_field_value(dim_spec, 's_dim')
        
        return {
            f'{interface_name}_dataflow': {
                'qDim': int(q_dim),
                'tDim': int(t_dim),
                'sDim': int(s_dim),
                'pragma_type': 'dataflow'
            }
        }
```

This advanced topics documentation provides deep technical insight into Brainsmith-2's most sophisticated features, enabling expert users to leverage the platform's full capabilities and extend it for specialized use cases.