# Import Hooks Module

## ⚠️ Temporary Solution Notice

This module contains import hooks that are a **temporary workaround** for environment configuration issues with external dependencies, particularly FINN.

## Why This Exists

FINN (and several other FPGA-related tools) rely heavily on environment variables for configuration:
- `FINN_ROOT` - Base directory of FINN installation
- `XILINX_VIVADO` - Path to Vivado installation
- `XILINX_VITIS` - Path to Vitis installation
- And many others...

Currently, users must remember to `import brainsmith` before `import finn` to ensure these environment variables are properly set. This is non-intuitive and error-prone.

## What It Does

The import hook intercepts `import finn` (and related imports) and automatically configures the environment before FINN loads. This means:

```python
# This now works without importing brainsmith first
import finn
from finn.core import dataflow
```

## Known Limitations

1. **Not PyPI-compliant**: Import hooks with global side effects are inappropriate for public PyPI packages
2. **Debugging complexity**: Makes import behavior "magical" and harder to debug
3. **Potential conflicts**: Could interfere with other packages or testing frameworks
4. **Security concerns**: Modifying import behavior globally has security implications

## Future Plans

This hook will be removed once FINN adopts a modern configuration system. The proposed solution is to:

1. Implement Pydantic Settings in FINN for type-safe configuration
2. Use configuration files instead of environment variables
3. Provide clear APIs for configuration management

See `finn_pydantic_config_analysis.md` in the project root for detailed analysis of the proposed solution.

## Disabling the Hook

If you experience issues with the import hook, you can disable it:

```python
# Option 1: Environment variable
os.environ['BSMITH_DISABLE_IMPORT_HOOKS'] = '1'
import brainsmith  # Hook won't be installed

# Option 2: Manual uninstall
from brainsmith.hooks.finn_import_hook import uninstall_finn_hook
uninstall_finn_hook()
```

## For Developers

When distributing Brainsmith as a PyPI package, consider:
1. Making the import hook opt-in rather than automatic
2. Providing clear documentation about the hook's purpose
3. Including warnings about potential side effects
4. Offering alternative solutions (like the import wrapper pattern)

## Timeline

- **Current**: Import hook provides automatic configuration (with limitations)
- **Next 3-6 months**: Work with FINN team to implement Pydantic Settings
- **Future**: Remove import hook entirely once FINN configuration is modernized