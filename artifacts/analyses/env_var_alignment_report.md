# Environment Variable Alignment Report: Smithy vs FINN vs ForgeConfig

## Executive Summary

This report analyzes the alignment between environment variables handled by:
1. **Smithy** - Brainsmith's container management script
2. **FINN's run-docker.sh** - FINN's native Docker script (not used by Brainsmith)
3. **ForgeConfig** - Brainsmith's configuration system

### Key Findings
- **Naming Mismatch**: Smithy uses `BSMITH_*` prefix while FINN uses `FINN_*`
- **Redundant Mappings**: Several environment variables are set but never used
- **Missing Links**: Some FINN requirements are not exposed through ForgeConfig
- **Broken Chain**: Build directory configuration doesn't flow through to FINN execution

## Environment Variable Comparison

### 1. Build Directory Configuration

| Variable | Smithy | FINN | ForgeConfig | Status |
|----------|--------|------|-------------|---------|
| Build Dir | `BSMITH_BUILD_DIR` | `FINN_BUILD_DIR` | Uses `BSMITH_BUILD_DIR` | ⚠️ Partial |

**Issue**: Smithy sets both internally but ForgeConfig only reads `BSMITH_BUILD_DIR`:
```bash
# Smithy line 81
: ${BSMITH_BUILD_DIR="/tmp/$DOCKER_INST_NAME"}

# Smithy line 361 - passed to container
DOCKER_CMD+=" -e BSMITH_BUILD_DIR=$BSMITH_BUILD_DIR"

# FINN expects (line 197)
DOCKER_EXEC+="-e FINN_BUILD_DIR=$FINN_HOST_BUILD_DIR"
```

**Impact**: FINN tools inside container may not find the correct build directory.

### 2. Xilinx Tool Paths

| Variable | Smithy | FINN | ForgeConfig | Status |
|----------|--------|------|-------------|---------|
| Xilinx Base | `BSMITH_XILINX_PATH` | `FINN_XILINX_PATH` | ❌ Not used | ⚠️ Missing |
| Xilinx Version | `BSMITH_XILINX_VERSION` | `FINN_XILINX_VERSION` | ❌ Not used | ⚠️ Missing |
| Vivado Path | Set by Smithy | `XILINX_VIVADO` | ❌ Not used | ✅ Auto-set |
| Vitis Path | `VITIS_PATH` | Same | Used by kernels | ✅ OK |
| HLS Path | `HLS_PATH` | Same | Used by kernels | ✅ OK |

**Issue**: ForgeConfig doesn't expose Xilinx configuration, relying on Smithy defaults.

### 3. Worker Configuration

| Variable | Smithy | FINN | ForgeConfig | Status |
|----------|--------|------|-------------|---------|
| Workers | `NUM_DEFAULT_WORKERS` | Same | ❌ Not used | ❌ Unused |

**Issue**: Set by both scripts (default: 4) but never consumed by ForgeConfig or passed to FINN builds.

### 4. Platform Repository

| Variable | Smithy | FINN | ForgeConfig | Status |
|----------|--------|------|-------------|---------|
| Platform Paths | `PLATFORM_REPO_PATHS` | Same | ❌ Not used | ⚠️ Missing |

**Issue**: Required for Alveo cards but not configurable through ForgeConfig.

### 5. Docker/Container Settings

| Variable | Smithy | FINN | ForgeConfig | Status |
|----------|--------|------|-------------|---------|
| Docker Tag | `BSMITH_DOCKER_TAG` | `FINN_DOCKER_TAG` | ❌ N/A | ✅ OK |
| Run as Root | `BSMITH_DOCKER_RUN_AS_ROOT` | `FINN_DOCKER_RUN_AS_ROOT` | ❌ N/A | ✅ OK |
| Skip Deps | `BSMITH_SKIP_DEP_REPOS` | `FINN_SKIP_DEP_REPOS` | ❌ N/A | ✅ OK |

These are container management variables, correctly outside ForgeConfig scope.

### 6. Plugin Strictness

| Variable | Smithy | FINN | ForgeConfig | Status |
|----------|--------|------|-------------|---------|
| Strict Mode | `BSMITH_PLUGINS_STRICT` | ❌ N/A | ❌ Not used | ⚠️ Inconsistent |

**Issue**: Smithy sets this to `true` by default, but ForgeConfig doesn't read it. The plugin system uses it directly from environment.

### 7. Network Ports

| Variable | Smithy | FINN | ForgeConfig | Status |
|----------|--------|------|-------------|---------|
| Jupyter Port | ❌ Not set | `JUPYTER_PORT` | ❌ N/A | ✅ OK |
| Netron Port | `NETRON_PORT` | Same | ❌ N/A | ✅ OK |

These are for interactive use, correctly outside ForgeConfig scope.

## Redundant/Unused Variables

### Set but Never Used
1. **`NUM_DEFAULT_WORKERS`** - Set to 4, passed to container, but no consumer found
2. **`LOCALHOST_URL`** - Set to "localhost", passed around but purpose unclear
3. **`working_directory`** in ForgeConfig - Defaults to "work" but never used
4. **`timeout_minutes`** in ForgeConfig - Defaults to 60 but implementation missing

### Legacy/Deprecated
1. **`OHMYXILINX`** - Points to oh-my-xilinx dependency (board files)
2. **`VIVADO_HLS_LOCAL`** - Redundant with VIVADO_PATH
3. **`VIVADO_IP_CACHE`** - Set but actual caching mechanism unclear

## Critical Breaks in the Chain

### 1. Build Directory Naming Mismatch
```python
# ForgeConfig reads BSMITH_BUILD_DIR
build_dir = Path(os.environ.get("BSMITH_BUILD_DIR", "./build"))

# But FINN expects FINN_BUILD_DIR internally
# This could cause FINN to use wrong directories
```

### 2. Output Stage Bug (Previously Identified)
```python
# Explorer passes 'output_stage'
# Executor expects 'output_products'
# Result: Always defaults to estimates only
```

### 3. Missing FINN Environment Setup
FINN's build system expects certain environment variables that aren't set:
- `FINN_ROOT` - Not set by Smithy (FINN sets to $SCRIPTPATH)
- `FINN_BUILD_DIR` - Not mapped from BSMITH_BUILD_DIR
- Board file setup assumes different paths

### 4. Xilinx Tool Discovery
ForgeConfig can't configure Xilinx paths, relying entirely on:
- User's ~/.bashrc exports
- Smithy's defaults
- No validation that tools exist

## Recommendations

### Immediate Fixes
1. **Add FINN_BUILD_DIR mapping** in Smithy:
   ```bash
   DOCKER_CMD+=" -e FINN_BUILD_DIR=$BSMITH_BUILD_DIR"
   ```

2. **Fix output_stage parameter** (already identified)

3. **Remove unused ForgeConfig fields**:
   - `working_directory`
   - `timeout_minutes` (or implement)

### Medium Term
1. **Expose Xilinx configuration** in ForgeConfig:
   ```python
   @dataclass
   class ForgeConfig:
       # ... existing fields ...
       xilinx_path: Optional[str] = None
       xilinx_version: Optional[str] = None
       platform_repo_paths: Optional[str] = None
   ```

2. **Consolidate environment variables**:
   - Use consistent naming (BSMITH_* everywhere)
   - Document which are container-only vs build configuration

3. **Add validation** for required tools:
   - Check Xilinx tools exist before build
   - Validate platform files for target board

### Long Term
1. **Decouple from FINN's environment expectations**:
   - Create explicit FINN configuration object
   - Map Brainsmith concepts to FINN requirements clearly

2. **Document environment variable flow**:
   - Which variables are for container management
   - Which affect build behavior
   - Which are passed through to FINN

## Variable Flow Diagram

```
Environment → Smithy → Docker Container → ForgeConfig → Explorer → Executor → FINN
     ↓           ↓            ↓               ↓           ↓          ↓         ↓
BSMITH_*     Set env     Pass to       Read from env   Extract    Use      Expect
             vars        container      or defaults     config     config   FINN_*
```

### Example: Build Directory Flow
1. User sets: `BSMITH_BUILD_DIR=/workspace/builds`
2. Smithy: Passes as `-e BSMITH_BUILD_DIR=/workspace/builds`
3. ForgeConfig: Reads from environment for output path
4. Explorer: Uses for output directory
5. FINN: Expects `FINN_BUILD_DIR` (not set!) - may use wrong directory

## Conclusion

The current system works but has several inefficiencies and potential failure points:
- **Redundant variables** that serve no purpose (NUM_DEFAULT_WORKERS, working_directory)
- **Naming mismatches** between BSMITH_* and FINN_* conventions
- **Missing configuration options** force reliance on environment defaults
- **No validation** of Xilinx tool availability before builds
- **Broken parameter chains** (output_stage → output_products)

The most critical issues are:
1. **Output stage bug** - prevents generating RTL/bitfiles
2. **Build directory mismatch** - FINN may write to unexpected locations
3. **Missing FINN_BUILD_DIR** - FINN tools may not find build artifacts

These issues stem from Brainsmith wrapping FINN without fully adapting to its environment expectations.