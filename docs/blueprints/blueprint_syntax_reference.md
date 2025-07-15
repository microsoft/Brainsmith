# Blueprint Syntax Reference

A comprehensive example demonstrating all possible input types in a Brainsmith blueprint, kept as concise as possible.

```yaml
version: "3.0"
name: "Complete Syntax Demo"
description: "Demonstrates all blueprint input types and syntax options"

hw_compiler:
  # Kernel syntax variations
  kernels:
    - "MatMul"                                 # Simple string - auto-discovers backends
    - ["LayerNorm", ["hls", "rtl"]]            # Explicit backends - tuple format
    - ["GELU", "ReLU", "SiLU"]                 # Mutually exclusive group - choose one
    - "~Dropout"                               # Optional kernel with ~ prefix
    - ["~", ["Conv2D", "DepthwiseConv2D"]]     # Nested groups (optional group of exclusive options)
    
  # Transform syntax variations
  transforms:
    cleanup:                                   # Stage-based organization (dict of lists)
      - "RemoveIdentityOps"                    # Simple transform
      - "~FoldConstants"                       # Optional transform
    optimization:
      - ["Streamline", "StreamlineLight", "~"] # Mutually exclusive transforms
    
  # Build steps (for legacy backend only)
  build_steps:
    - "cleanup"                                # Custom step
    - "step_tidy_up"                           # FINN step (with prefix)
    - "finn:qonnx_to_finn"                     # Framework-qualified step
    - "brainsmith:custom_step"                 # Explicit framework
    
# Search configuration
search:
  strategy: "exhaustive"                       # Enum: exhaustive, random, genetic, bayesian
  
  constraints:                                 # List of constraint objects
    - metric: "lut_utilization"                # Required fields
      operator: "<="
      value: 0.85                              # Float constraint
      
    - metric: "latency"
      operator: "<-"
      value: 10                                # Integer constraint
  
  max_evaluations: 5000                        # Optional int
  timeout_minutes: 720                         # Optional int
  parallel_builds: 4                           # Optional with default: 1
  
  early_stopping:                              # Optional nested dict
    enabled: true
    patience: 50
    min_delta: 0.001

# Global configuration
global:
  output_stage: "stitched_ip"                  # Enum: dataflow_graph, rtl, stitched_ip
  working_directory: "./builds"                # Path
  log_file: "dse.log"                          # Filename
  log_level: "INFO"                            # Enum: DEBUG, INFO, WARNING, ERROR
  
  cache_results: true                          # Boolean with default: true
  save_artifacts: false                        # Boolean with default: true
  
  max_combinations: 1000                       # Optional int
  timeout_minutes: null                        # Explicit null
  start_step: "step_one"                       # Optional string
  stop_step: "step_five"                       # Optional string

# Processing configuration (optional section)
processing:
  preprocessing:                               # List of module configs
    - module: "quantization"
      config:                                  # Arbitrary dict
        bits: 8
        symmetric: true
        
  postprocessing:
    - module: "optimization"
      enabled: true                            # Module-specific config

# Test configuration (optional section)
test_config:
  unit_tests: ["test1", "test2"]               # String list
  
  validation:                                  # Mixed type dict
    enabled: true
    threshold: 0.95
    datasets: ["val1", "val2"]

# Custom sections (allowed for extensions)
experimental:
  feature_flags:
    new_optimizer: false
    beta_kernels: true
  
  advanced_config:                             # Complex nested structure
    level1:
      level2:
        setting: "value"
        number: 42
        
# Metadata (optional)
metadata:
  author: "DSE Team"
  version: "1.0.0"
  tags: ["production", "bert", "optimized"]
  created: "2024-01-15"                        # Date string
```

## Input Type Reference

### Basic Types
- **String**: `"value"` - Text values, paths, identifiers
- **Integer**: `42` - Whole numbers
- **Float**: `3.14` - Decimal numbers  
- **Boolean**: `true`/`false` - Binary flags
- **Null**: `null` - Explicit absence of value

### Collection Types
- **List**: `[item1, item2]` - Ordered sequences
- **Dict**: `{key: value}` - Key-value mappings

### Special Syntax
- **Optional prefix**: `~` - Makes element optional (e.g., `~Transform`)
- **Mutually exclusive**: `[option1, option2]` - Choose one from group
- **Framework qualification**: `framework:name` - Disambiguate names
- **Tuple format**: `["kernel", ["backend1", "backend2"]]` - Kernel with backends

### Enums (Restricted Values)
- `strategy`: exhaustive, random, genetic, bayesian
- `output_stage`: dataflow_graph, rtl, stitched_ip
- `log_level`: DEBUG, INFO, WARNING, ERROR
- `operator`: <=, >=, ==, <, >

### Required vs Optional
- **Required sections**: version, name, hw_compiler, global
- **Optional sections**: description, model_config, finn_config, search, processing, test_config
- **Section-specific requirements**: Varies by section (e.g., constraints require metric, operator, value)