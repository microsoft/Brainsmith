# PE and SIMD Parallelism Variables in FINN: Comprehensive Analysis

## Executive Summary

This document provides a comprehensive analysis of PE (Processing Elements) and SIMD (Single Instruction Multiple Data) parallelism variables implementation in the FINN neural network acceleration framework. Based on extensive exploration of the FINN codebase, this analysis covers the architectural principles, implementation details, optimization strategies, and hardware generation patterns.

## Table of Contents

1. [Core Architectural Concepts](#core-architectural-concepts)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Implementation Layers](#implementation-layers)
4. [Hardware Architecture](#hardware-architecture)
5. [Optimization Strategies](#optimization-strategies)
6. [Resource Estimation and Trade-offs](#resource-estimation-and-trade-offs)
7. [Practical Examples](#practical-examples)
8. [Advanced Features](#advanced-features)
9. [Key Insights and Recommendations](#key-insights-and-recommendations)

## Core Architectural Concepts

### Fundamental Definitions

**PE (Processing Elements)**: Controls **output parallelism** - determines how many output elements are computed simultaneously in parallel. In matrix operations, this corresponds to processing multiple rows/outputs concurrently.

**SIMD (Single Instruction Multiple Data)**: Controls **input parallelism** - determines how many input elements are processed simultaneously. In matrix operations, this corresponds to processing multiple columns/inputs concurrently.

### Dataflow Architecture Principles

FINN implements a **streaming dataflow architecture** where:
- Each layer is allocated proportionate compute resources
- Parallelism is exploited both within layers (PE/SIMD) and across layers (pipelining)
- Data flows continuously through dedicated hardware blocks without intermediate memory storage

```
Input → [Layer1: PE₁×SIMD₁] → [Layer2: PE₂×SIMD₂] → ... → Output
```

### Folding Factor Concept

**Total Folding** = Time multiplexing factor that determines how many clock cycles are needed:
- For MVAU: `Total_Folding = (MH/PE) × (MW/SIMD)`
- Lower folding = Higher parallelism = Fewer cycles = More resources
- Higher folding = Lower parallelism = More cycles = Fewer resources

## Mathematical Foundations

### MVAU (Matrix Vector Activation Unit)

The core matrix-vector multiplication operation with constraints and calculations:

#### Constraints
```python
MW % SIMD == 0  # Input width must be divisible by SIMD
MH % PE == 0    # Output height must be divisible by PE
```

#### Cycle Calculation
```python
exp_cycles = (MH / PE) * (MW / SIMD) * np.prod(num_input_vectors) / mmv
```

Where:
- `MH`: Matrix Height (output features)
- `MW`: Matrix Width (input features)
- `PE`: Processing Elements (output parallelism)
- `SIMD`: Input parallelism
- `mmv`: Memory macroprocessing value (currently fixed at 1)

#### Memory Organization
```python
omega = (MW * MH) / (PE * SIMD)  # Number of weight words stored
mem_width = SIMD * weight_bitwidth * PE  # Memory interface width
```

### VVAU (Vector Vector Activation Unit)

Used for depthwise convolutions and elementwise operations:

#### Constraints
```python
Channels % PE == 0  # Channel dimension must be divisible by PE
# SIMD constraints depend on kernel size and operation type
```

#### Cycle Calculation
```python
exp_cycles = ((channels * k_h * k_w) / PE / SIMD) * batch_size * (dim_h * dim_w) / mmv
```

Where:
- `channels`: Number of channels
- `k_h, k_w`: Kernel dimensions
- `dim_h, dim_w`: Feature map dimensions

### Stream Width Calculations

#### Data Stream Widths
```systemverilog
// From hardware implementation
WEIGHT_STREAM_WIDTH = PE * SIMD * WEIGHT_WIDTH;
INPUT_STREAM_WIDTH = (IS_MVU ? 1 : PE) * SIMD * ACTIVATION_WIDTH;
OUTPUT_STREAM_WIDTH = PE * ACCU_WIDTH;
```

#### Folding Factors
```systemverilog
SF = MW/SIMD;  // SIMD folding factor (number of SIMD cycles)
NF = MH/PE;    // PE folding factor (number of PE cycles)
```

## Implementation Layers

### Python Layer (High-Level Transformations)

#### SetFolding Transformation
The `SetFolding` transformation implements an intelligent optimization algorithm:

```python
# Algorithm for MVAU optimization:
# 1. First phase: Increase SIMD while weight stream width ≤ mvau_wwidth_max
for simd_val in divisors(max_simd):
    if (input_datatype.bitwidth() * SIMD > mvau_wwidth_max):
        break  # Revert if weight stream too wide
    if cycles < target_cycles_per_frame:
        break

# 2. Second phase: Increase PE until target met or max PE reached
optimize_attribute_val(node_inst, max_pe, "PE")
```

#### Node Attribute Management
```python
# Each HW node has PE/SIMD attributes
node_inst = getCustomOp(node)
pe = node_inst.get_nodeattr("PE")
simd = node_inst.get_nodeattr("SIMD")
node_inst.set_nodeattr("PE", new_pe_value)
node_inst.set_nodeattr("SIMD", new_simd_value)
```

### SystemVerilog RTL Layer (Hardware Implementation)

#### Core Compute Modules
- **mvu_4sx4u.sv**: 4-bit weights with DSP48, achieving 4 MACs/DSP
- **mvu_8sx8u_dsp48.sv**: 8-bit implementation with DSP48, 2 MACs/DSP
- **mvu_vvu_8sx9_dsp58.sv**: 9-bit signed support with DSP58, 3 MACs/DSP

#### Parameterization
```systemverilog
module mvu_vvu_axi #(
    int unsigned PE,
    int unsigned SIMD,
    int unsigned MW,     // Matrix width
    int unsigned MH,     // Matrix height
    int unsigned ACTIVATION_WIDTH,
    int unsigned WEIGHT_WIDTH,
    int unsigned ACCU_WIDTH,
    // ... other parameters
)(
    // Interface signals
);
```

#### Pipeline Architecture
```systemverilog
// Pipeline depth calculation
localparam int unsigned PIPELINE_DEPTH = 3 + $clog2(SIMD+1) + (SIMD == 1);

// Cross-SIMD reduction tree
// Binary reduction trees with SIMD leaf nodes
// Balanced tree construction with pipelined levels
```

## Hardware Architecture

### DSP Utilization Patterns

#### DSP48E1/E2 (Xilinx 7-series, UltraScale)
- **4-bit operations**: 4 MACs per DSP
- **8-bit operations**: 2 MACs per DSP
- **Resource efficiency**: Good for moderate precision networks

#### DSP58 (Xilinx UltraScale+)
- **8×9-bit operations**: 3 MACs per DSP
- **Enhanced precision**: Support for 9-bit signed activations
- **Higher efficiency**: Better resource utilization for modern networks

### Memory Architecture

#### Weight Memory Organization
```python
# Memory words required
omega = (input_features * output_features) / (PE * SIMD)

# Memory interface width
mem_width = PE * SIMD * weight_bitwidth

# Memory efficiency
efficiency = actual_bits_used / allocated_memory_bits
```

#### Memory Modes
1. **internal_embedded**: Weights embedded directly in RTL (fast, large area)
2. **internal_decoupled**: Streaming weights with internal streamer (balanced)
3. **external**: External weight streaming (flexible, complex interface)

#### RAM Style Selection
```python
ram_style_options = {
    "auto": "Let Vivado decide",
    "block": "Use BRAM",
    "distributed": "Use LUTRAM", 
    "ultra": "Use UltraRAM (requires runtime_writeable_weights=1)"
}
```

### Compute Core Architecture

#### Matrix-Vector Multiplication Flow
```
Input Vector → [SIMD Parallelism] → Partial Products → [PE Parallelism] → Outputs
     ↓              ↓                      ↓                ↓              ↓
   [A₀...A_{SIMD-1}] → [W₀₀...W_{PE,SIMD}] → [PP₀...PP_{PE}] → [Y₀...Y_{PE-1}]
```

#### Reduction Tree Implementation
```systemverilog
// Cross-SIMD adder tree for accumulation
// Balanced binary tree with configurable pipeline stages
for (genvar i = 0; i < PE; i++) begin : genPE
    for (genvar j = 0; j < SIMD; j++) begin : genSIMD
        // Multiply-accumulate operations
        assign partial_product[i][j] = activation[j] * weight[i][j];
    end
    // Tree reduction across SIMD dimension
    assign output[i] = reduce_tree(partial_product[i]);
end
```

## Optimization Strategies

### SetFolding Algorithm Details

#### Target-Driven Optimization
```python
class SetFolding(Transformation):
    def __init__(self, target_cycles_per_frame=1000, mvau_wwidth_max=36):
        self.target_cycles_per_frame = target_cycles_per_frame
        self.mvau_wwidth_max = mvau_wwidth_max
```

#### Multi-Pass Relaxation
```python
if self.two_pass_relaxation:
    perf_dict = model.analysis(dataflow_performance)
    if perf_dict["max_cycles"] > self.target_cycles_per_frame:
        # Run second pass with achievable target
        model = model.transform(SetFolding(
            target_cycles_per_frame=perf_dict["max_cycles"],
            two_pass_relaxation=False
        ))
```

#### Operation-Specific Strategies

**PE Operations** (output parallelism):
```python
pe_ops = [
    "AddStreams_hls", "ChannelwiseOp_hls", "DuplicateStreams_hls",
    "GlobalAccPool_hls", "Thresholding_hls", "Thresholding_rtl"
]
# Constraint: max_pe = NumChannels
```

**SIMD Operations** (input parallelism):
```python
simd_ops = [
    "DownSampler_hls", "FMPadding_hls", "ConvolutionInputGenerator_hls",
    "ConvolutionInputGenerator_rtl", "QuantSoftmax_hls", "Shuffle_hls"
]
# Constraint: max_simd = NumChannels or IFMChannels
```

**Depthwise Special Cases**:
```python
depthwise_op_exceptions = ["VVAU_hls", "VVAU_rtl", "Pool_hls"]
# Special coupling: SWG SIMD = VVAU PE for depthwise operations
```

### Pipeline Balancing Strategy

#### Bottleneck Analysis
```python
def dataflow_performance(model):
    max_cycles = 0
    critical_path_cycles = 0
    
    for node in model.graph.node:
        node_cycles = inst.get_nodeattr("cycles_estimate")
        if node_cycles > max_cycles:
            max_cycles = node_cycles
            bottleneck_node = node.name
    
    return {
        "max_cycles": max_cycles,
        "max_cycles_node_name": bottleneck_node,
        "critical_path_cycles": critical_path_cycles
    }
```

#### Target Selection Heuristics
1. **Initial target**: User-specified or derived from requirements
2. **Bottleneck detection**: Identify layers that cannot meet target
3. **Relaxed target**: Use achievable bottleneck as new target
4. **Pipeline rebalancing**: Redistribute resources across all layers

## Resource Estimation and Trade-offs

### BRAM Estimation

#### Calculation Method
```python
def bram_estimation(self):
    P = self.get_nodeattr("PE")
    Q = self.get_nodeattr("SIMD") 
    W = weight_datatype.bitwidth()
    omega = self.calc_wmem()  # Number of weight words
    mem_width = Q * W * P
    
    # BRAM utilization based on width and depth
    if mem_width == 1:
        return math.ceil(omega / 16384)
    elif mem_width == 2:
        return math.ceil(omega / 8192)
    elif mem_width <= 4:
        return math.ceil(omega / 4096) * math.ceil(mem_width / 4)
    # ... additional cases
```

#### Memory Efficiency
```python
def bram_efficiency_estimation(self):
    wbits = weight_bitwidth * input_features * output_features
    bram_capacity = bram_estimation() * 36 * 512  # bits per BRAM18
    return wbits / bram_capacity
```

### LUT Estimation

#### Empirical Model
```python
def lut_estimation(self):
    P = self.get_nodeattr("PE")
    Q = self.get_nodeattr("SIMD")
    
    # Base LUT cost
    lut_cost = base_lut_per_dsp * num_dsps
    
    # Scaling factors
    lut_cost += pe_scaling_factor * P
    lut_cost += simd_scaling_factor * Q
    lut_cost += interconnect_overhead(P, Q)
    
    return int(lut_cost)
```

### DSP Estimation

#### DSP Utilization
```python
def dsp_estimation(self):
    P = self.get_nodeattr("PE")
    Q = self.get_nodeattr("SIMD")
    
    # DSPs required depend on bit widths and core type
    if compute_core == "mvu_4sx4u":
        macs_per_dsp = 4
    elif compute_core == "mvu_8sx8u_dsp48":
        macs_per_dsp = 2
    elif compute_core == "mvu_vvu_8sx9_dsp58":
        macs_per_dsp = 3
    
    total_macs = P * Q
    return math.ceil(total_macs / macs_per_dsp)
```

### Performance vs Resource Trade-offs

#### Pareto Frontier Analysis
```
High Performance (Low Latency)     High Resource Efficiency
         ↑                                    ↑
    High PE/SIMD  ←→ Trade-off Space ←→   Low PE/SIMD
         ↓                                    ↓
   High Resources                        Low Performance
```

#### Optimization Objectives
1. **Latency minimization**: Maximize PE/SIMD subject to resource constraints
2. **Resource minimization**: Minimize PE/SIMD subject to performance constraints  
3. **Throughput maximization**: Balance PE/SIMD for pipeline equilibrium
4. **Energy efficiency**: Optimize for operations per joule

## Practical Examples

### Folding Optimization Example

#### Original Configuration
```python
# From cybersecurity MLP example
layer_params = {
    "MW": 600,    # Input features
    "MH": 64,     # Output features  
    "PE": 1,      # Single output per cycle
    "SIMD": 1     # Single input per cycle
}

cycles_original = (64/1) * (600/1) = 38,400 cycles
```

#### Optimized Configuration
```python
# After SetFolding optimization
layer_params_optimized = {
    "MW": 600,
    "MH": 64,
    "PE": 2,      # Process 2 outputs in parallel
    "SIMD": 5     # Process 5 inputs in parallel
}

cycles_optimized = (64/2) * (600/5) = 32 * 120 = 3,840 cycles
speedup = 38,400 / 3,840 = 10x improvement
```

### Test Configuration Patterns

#### Common Parameter Ranges
```python
# From test suite analysis
typical_configurations = {
    "PE_values": [1, 2, 4, 8, 16, 32],
    "SIMD_values": [1, 3, 6, 9, 15, 30, 60],
    "matrix_sizes": {
        "small": {"MW": 64, "MH": 32},
        "medium": {"MW": 256, "MH": 128}, 
        "large": {"MW": 600, "MH": 256}
    }
}
```

#### Depthwise Convolution Example
```python
# VVAU with depthwise constraints
vvau_config = {
    "Channels": 64,
    "Kernel": [3, 3], 
    "PE": 4,          # Process 4 channels in parallel
    "SIMD": 9         # Process full 3×3 kernel in parallel
}

# Coupled SWG configuration
swg_config = {
    "SIMD": 4,        # Must equal VVAU PE for depthwise
    "parallel_window": 1  # Enable window parallelism
}
```

### Hardware Generation Examples

#### RTL Parameter Mapping
```systemverilog
// Generated hardware instance
mvu_vvu_axi #(
    .IS_MVU(1),
    .MW(600),
    .MH(64), 
    .PE(2),
    .SIMD(5),
    .ACTIVATION_WIDTH(8),
    .WEIGHT_WIDTH(4),
    .ACCU_WIDTH(16)
) inst_mvau (
    .ap_clk(clk),
    .ap_rst_n(rst_n),
    // ... interface signals
);
```

## Advanced Features

### Pumped Compute

#### Concept
Pumped compute uses a faster clock to effectively double the SIMD parallelism:

```systemverilog
// Clock pumping implementation
localparam int unsigned EFFECTIVE_SIMD = SIMD_UNEVEN && PUMPED_COMPUTE ? SIMD+1 : SIMD;
localparam int unsigned DSP_SIMD = EFFECTIVE_SIMD/(PUMPED_COMPUTE+1);

// Dual-clock operation
input logic ap_clk,     // Standard clock
input logic ap_clk2x,   // Double-speed clock for pumped compute
```

#### Benefits and Constraints
- **Benefits**: Higher effective parallelism without proportional resource increase
- **Constraints**: Requires even SIMD values, complex timing closure
- **Use cases**: High-throughput applications with relaxed area constraints

### Dynamic Reconfiguration

#### Runtime Weight Loading
```python
mem_mode_options = {
    "internal_embedded": "Static weights, fastest access",
    "internal_decoupled": "Runtime loadable, balanced performance",  
    "external": "Fully dynamic, highest flexibility"
}
```

#### Adaptive Parallelism
```python
# Future capability: Runtime PE/SIMD adjustment
def set_runtime_parallelism(self, new_pe, new_simd):
    if self.supports_dynamic_reconfiguration():
        self.reconfigure_compute_core(new_pe, new_simd)
        self.update_stream_widths()
        self.retile_memory_interfaces()
```

### Multi-Core Scaling

#### Spatial Parallelism
```python
# Multiple compute cores for higher throughput
multi_core_config = {
    "num_cores": 4,
    "pe_per_core": 8,
    "simd_per_core": 16,
    "effective_pe": 32,      # 4 × 8
    "effective_simd": 64     # 4 × 16  
}
```

## Key Insights and Recommendations

### Architectural Insights

1. **Unified Design Principle**: PE and SIMD parameters create a unified abstraction from high-level Python transformations down to low-level RTL implementation

2. **Hardware-Software Co-design**: The same parallelism parameters control optimization at every level of the stack, enabling efficient design space exploration

3. **Constraint-Driven Optimization**: All optimizations must respect mathematical divisibility constraints and hardware resource limitations

4. **Pipeline-Centric Performance**: Overall system performance is determined by the slowest layer, making balanced parallelism allocation critical

### Optimization Recommendations

#### For Design Space Exploration
1. **Start with SetFolding**: Use automated folding as initial baseline
2. **Identify bottlenecks**: Analyze cycles per layer to find limiting factors
3. **Manual fine-tuning**: Adjust critical layers beyond automated optimization
4. **Validate constraints**: Ensure all PE/SIMD values respect divisibility requirements

#### For Resource Efficiency
1. **Memory width limits**: Keep weight stream width ≤ 36 bits for optimal BRAM utilization
2. **PE vs SIMD trade-offs**: Prefer SIMD increases over PE for better critical path timing
3. **Pipeline balancing**: Target similar cycle counts across all layers
4. **Memory mode selection**: Choose appropriate memory mode for performance/flexibility trade-offs

#### For Performance Optimization
1. **Critical path analysis**: Focus optimization effort on bottleneck layers
2. **Datatype optimization**: Use minimum precision required for accuracy
3. **Spatial vs temporal trade-offs**: Balance parallelism with memory bandwidth
4. **Multi-objective optimization**: Consider latency, throughput, and energy simultaneously

### Implementation Best Practices

#### For Kernel Development
1. **Interface standardization**: Follow established PE/SIMD conventions for consistency
2. **Constraint validation**: Implement robust checking of parallelism constraints
3. **Resource modeling**: Provide accurate estimation functions for design space exploration
4. **Test coverage**: Include varied PE/SIMD configurations in test suites

#### For System Integration
1. **Modular design**: Separate parallelism logic from functional implementation
2. **Parameterizable interfaces**: Support runtime or compile-time parallelism configuration
3. **Performance monitoring**: Include instrumentation for cycle counting and resource utilization
4. **Documentation standards**: Clearly specify constraints and optimization guidelines

### Future Development Directions

#### Emerging Requirements
1. **Heterogeneous parallelism**: Support for different PE/SIMD per layer type
2. **Dynamic reconfiguration**: Runtime adjustment of parallelism parameters
3. **Multi-precision support**: Mixed-precision operations within single kernels
4. **Advanced memory hierarchies**: Support for HBM, DDR, and on-chip memory tiers

#### Research Opportunities
1. **Automated DSE**: Machine learning-driven design space exploration
2. **Cross-layer optimization**: Global optimization across multiple layers simultaneously
3. **Energy-aware parallelism**: PE/SIMD selection optimized for energy efficiency
4. **Fault-tolerant designs**: Graceful degradation with reduced parallelism

---

## Conclusion

The PE and SIMD parallelism variables in FINN represent a sophisticated and well-engineered approach to neural network acceleration. The unified abstraction spans from high-level Python transformations to low-level hardware generation, enabling efficient design space exploration and optimization. The mathematical foundations are solid, the implementation is robust, and the optimization strategies are intelligent.

This analysis reveals that FINN's parallelism system is not just a set of parameters, but a comprehensive framework for hardware-software co-design that enables scalable, efficient neural network acceleration across a wide range of applications and target platforms.

The key to effective use of this system lies in understanding the fundamental trade-offs between performance, resources, and constraints, and applying this knowledge systematically through the provided tools and transformations.
