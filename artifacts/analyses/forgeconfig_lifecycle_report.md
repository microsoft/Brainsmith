# ForgeConfig Lifecycle Report

## Overview

ForgeConfig is the central configuration object in Brainsmith that controls the build process from blueprint parsing through FPGA synthesis. This report traces every configuration parameter from its origin to its final use.

## Executive Summary

### Configuration Sources (in precedence order)
1. **Blueprint YAML** (highest precedence)
   - Top-level fields override `global_config` section
   - `finn_config` section for FINN-specific parameters
   - Legacy format support (platform, target_clk)
2. **Environment Variables** 
   - `BRAINSMITH_MAX_COMBINATIONS` (default: 100000)
   - `BRAINSMITH_TIMEOUT_MINUTES` (default: 60)
   - `BSMITH_BUILD_DIR` (for output location)
3. **Dataclass Defaults** (lowest precedence)

### Critical Configuration Parameters
- **output_stage**: Controls synthesis depth (reports → RTL → bitfile)
- **finn_params['board']**: Target FPGA platform (required for synthesis)
- **finn_params['synth_clk_period_ns']**: Target clock period (required for synthesis)
- **fail_fast**: Stop on first error vs. explore all paths
- **max_combinations**: Safety limit for design space explosion

### Known Issues
- **Critical Bug**: output_stage configuration ignored due to parameter name mismatch
- **Unused Parameters**: working_directory, timeout_minutes appear unused
- **Missing Validation**: finn_params passed through without deep validation

## ForgeConfig Fields

### 1. `output_stage: OutputStage`
- **Default**: `OutputStage.COMPILE_AND_PACKAGE`
- **Type**: Enum with values:
  - `COMPILE_AND_PACKAGE = "compile_and_package"`
  - `SYNTHESIZE_BITSTREAM = "synthesize_bitstream"`
  - `GENERATE_REPORTS = "generate_reports"`
- **Sources**:
  - Blueprint YAML: `output_stage` (top-level or in `global_config`)
- **Usage**:
  - Validation: Determines if FINN hardware synthesis config is required
  - Explorer: Maps to FINN build products:
    - `GENERATE_REPORTS` → `["ESTIMATE_REPORTS"]`
    - `SYNTHESIZE_BITSTREAM` → `["ESTIMATE_REPORTS", "RTLSIM_PERFORMANCE", "STITCHED_IP"]`
    - `COMPILE_AND_PACKAGE` → `["ESTIMATE_REPORTS", "RTLSIM_PERFORMANCE", "STITCHED_IP", "BITFILE", "DEPLOYMENT_PACKAGE"]`

### 2. `working_directory: str`
- **Default**: `"work"`
- **Sources**:
  - Blueprint YAML: `working_directory` (top-level or in `global_config`)
- **Usage**:
  - Passed to executor but appears unused in current implementation
  - Likely intended for FINN working directory control

### 3. `save_intermediate_models: bool`
- **Default**: `False`
- **Sources**:
  - Blueprint YAML: `save_intermediate_models` (top-level or in `global_config`)
- **Usage**:
  - Passed to executor but implementation not visible in core
  - Likely controls whether intermediate ONNX models are preserved

### 4. `fail_fast: bool`
- **Default**: `False`
- **Sources**:
  - Blueprint YAML: `fail_fast` (top-level or in `global_config`)
- **Usage**:
  - Executor: Controls whether exploration stops on first failure
  - When `True`, exploration terminates immediately on error
  - When `False`, continues exploring other paths

### 5. `max_combinations: int`
- **Default**: `100000` (or from env var `BRAINSMITH_MAX_COMBINATIONS`)
- **Sources**:
  - Environment: `BRAINSMITH_MAX_COMBINATIONS`
  - Blueprint YAML: `max_combinations` (top-level or in `global_config`)
- **Usage**:
  - DesignSpace validation: Prevents creation of trees exceeding this limit
  - Safety mechanism to avoid combinatorial explosion

### 6. `timeout_minutes: int`
- **Default**: `60` (or from env var `BRAINSMITH_TIMEOUT_MINUTES`)
- **Sources**:
  - Environment: `BRAINSMITH_TIMEOUT_MINUTES`
  - Blueprint YAML: `timeout_minutes` (top-level or in `global_config`)
- **Usage**:
  - Passed to executor but timeout implementation not visible
  - Likely controls per-segment or total exploration timeout

### 7. `finn_params: Dict[str, Any]`
- **Default**: `{}` (empty dict)
- **Sources**:
  - Blueprint YAML: `finn_config` section (passed through directly)
  - Legacy mappings:
    - `platform` → `finn_params['board']`
    - `target_clk` → `finn_params['synth_clk_period_ns']` (with unit parsing)
- **Required Fields** (for hardware synthesis):
  - `board`: Target FPGA board (e.g., "Pynq-Z1")
  - `synth_clk_period_ns`: Clock period in nanoseconds
- **Optional Fields**:
  - `kernel_selections`: List of (kernel, backend) tuples
  - Any other FINN DataflowBuildConfig parameters
- **Usage**:
  - Passed directly to FINN's DataflowBuildConfig
  - No validation except for required fields when `output_stage != GENERATE_REPORTS`

## Configuration Flow

### 1. Blueprint Loading
```yaml
# All these locations are supported:
# Option 1: In global_config
global_config:
  output_stage: "synthesize_bitstream"
  max_combinations: 5000
  
# Option 2: Top-level
output_stage: "synthesize_bitstream"
max_combinations: 5000

# Option 3: Mixed
global_config:
  output_stage: "synthesize_bitstream"
max_combinations: 5000  # Top-level overrides global_config

# FINN parameters
finn_config:
  board: "Pynq-Z1"
  synth_clk_period_ns: 5.0
  
# Legacy format still supported
platform: "Pynq-Z1"  # Becomes finn_params['board']
target_clk: "5ns"    # Parsed to finn_params['synth_clk_period_ns'] = 5.0
```

### 2. Extraction Process
1. `BlueprintParser._extract_config_and_mappings()`:
   - Merges `global_config` with top-level fields
   - Excludes `design_space` and `extends`
   - Uses dataclass introspection to extract ForgeConfig fields
   - Handles OutputStage string → enum conversion
   - Processes legacy parameter mappings

### 3. Time Unit Parsing
The `target_clk` field supports multiple time units:
- `"5"` or `5` → 5.0 ns
- `"5ns"` → 5.0 ns
- `"5000ps"` → 5.0 ns  
- `"0.005us"` → 5.0 ns
- `"0.000005ms"` → 5.0 ns

### 4. Validation
- If `output_stage != GENERATE_REPORTS`:
  - Requires `synth_clk_period_ns` in finn_params
  - Requires `board` in finn_params
- Design space size must not exceed `max_combinations`

### 5. Exploration Usage
The explorer converts ForgeConfig into separate dictionaries:
- `global_config`: Contains forge control parameters
- `finn_config`: Contains finn_params plus kernel_selections from design space

## Environment Variables

### Core ForgeConfig Variables

#### `BSMITH_BUILD_DIR`
- Used by: `forge()` function
- Purpose: Default parent directory for forge output
- Default: `"./build"`
- Usage: `{BSMITH_BUILD_DIR}/forge_YYYYMMDD_HHMMSS`

#### `BRAINSMITH_MAX_COMBINATIONS`
- Used by: ForgeConfig default
- Purpose: Default limit for design space size
- Default: `"100000"`
- Override: Blueprint can specify lower value

#### `BRAINSMITH_TIMEOUT_MINUTES`
- Used by: ForgeConfig default
- Purpose: Default timeout for exploration
- Default: `"60"`
- Override: Blueprint can specify different value

### Additional System Variables

#### `BSMITH_PLUGINS_STRICT`
- Used by: Plugin framework adapters
- Purpose: Controls plugin loading behavior
- Default: `"false"`
- Values: `"true"` = strict mode (fail on plugin errors), `"false"` = permissive mode
- Usage: Set to `"true"` for production environments

#### `VITIS_PATH`
- Used by: HLS kernel implementations (crop, shuffle, layernorm, softmax)
- Purpose: Path to Xilinx Vitis installation
- Required for: Hardware kernel compilation
- Example: `/opt/Xilinx/Vitis/2023.2`

#### `HLS_PATH` (deprecated)
- Used by: Some legacy kernel implementations
- Purpose: Path to HLS installation
- Status: Being replaced by VITIS_PATH in newer kernels

## Key Insights

1. **Dual Configuration Sources**: ForgeConfig unifies both build control (Brainsmith-specific) and FINN parameters into a single object.

2. **Legacy Support**: The system maintains backward compatibility with older blueprint formats through parameter mapping.

3. **Environment Defaults**: Critical safety parameters (max_combinations, timeout) have environment variable defaults to prevent runaway processes.

4. **Pass-Through Design**: The `finn_params` dict is passed through without deep validation, allowing flexibility for FINN configuration evolution.

5. **Output Stage Mapping**: The OutputStage enum provides a high-level abstraction over FINN's detailed build products list.

## Critical Bug Found

**Issue**: Parameter name mismatch between explorer and executor causes output stage configuration to be ignored.

**Details**:
- `explorer.py:38` passes `'output_stage': forge_config.output_stage` (an OutputStage enum)
- `executor.py:93` expects `global_config.get("output_products", "df")`
- The executor DOES have mappings for the enum string values:
  - `"generate_reports"` → `["ESTIMATE_REPORTS"]`
  - `"synthesize_bitstream"` → `["ESTIMATE_REPORTS", "RTLSIM_PERFORMANCE", "STITCHED_IP"]`
  - `"compile_and_package"` → `["ESTIMATE_REPORTS", "RTLSIM_PERFORMANCE", "STITCHED_IP", "BITFILE", "DEPLOYMENT_PACKAGE"]`

**Impact**: Users cannot generate RTL or bitfiles regardless of blueprint configuration - it always defaults to "df" (dataflow estimates only).

**Fix Required**: 
```python
# In explorer.py, change line 38 from:
'output_stage': forge_config.output_stage,
# To:
'output_products': forge_config.output_stage.value,
```

## Recommendations

1. **Document finn_params**: Create comprehensive documentation of supported FINN parameters.

2. **Validate timeout implementation**: The timeout_minutes parameter appears unused - either implement or remove.

3. **Clarify working_directory**: This parameter's purpose and usage should be documented or removed if obsolete.

4. **Type finn_params**: Consider creating a typed FinnConfig dataclass instead of Dict[str, Any] for better validation and documentation.