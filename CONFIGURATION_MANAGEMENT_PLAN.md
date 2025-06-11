# Configuration Management with Environment Profiles Implementation Plan

## Overview
Implement a centralized configuration management system through a clean refactor that replaces scattered configuration throughout the codebase with environment profiles. This is a breaking change that establishes unified configuration management across all components.

## 1. Current State Analysis

### Configuration Scatter Pattern
Based on the codebase analysis, configuration is currently scattered across:

**Core Layer:**
- `BrainsmithConfig` in core API
- Various test configurations in `conftest.py`
- CLI argument parsing configurations

**Infrastructure Layer:**
- `DSEConfiguration` for design space exploration
- `FINNConfig` for FINN interface settings
- Hooks system configurations

**Libraries Layer:**
- Blueprint configurations (YAML-based)
- Registry configurations for search directories
- Analysis tool configurations

### Current Problems
1. **No Central Authority**: Configuration logic duplicated across components
2. **Environment Blindness**: No support for dev/test/prod environments
3. **Dependency Chaos**: Missing dependencies handled inconsistently
4. **Validation Scatter**: Configuration validation logic spread everywhere
5. **Override Complexity**: No clear precedence for configuration sources

## 2. Proposed Architecture

### Configuration Hierarchy
```python
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path

class Environment(Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    CUSTOM = "custom"

@dataclass
class ConfigurationProfile:
    """Environment-specific configuration profile."""
    name: str
    environment: Environment
    base_config: Dict[str, Any] = field(default_factory=dict)
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    dependency_config: Dict[str, Any] = field(default_factory=dict)
    performance_config: Dict[str, Any] = field(default_factory=dict)
    
    def merge_with(self, other: 'ConfigurationProfile') -> 'ConfigurationProfile':
        """Merge this profile with another, other takes precedence."""
        pass

class ConfigurationManager:
    """Central configuration management system."""
    
    def __init__(self, 
                 environment: Union[Environment, str] = Environment.DEVELOPMENT,
                 config_paths: Optional[List[Path]] = None,
                 enable_cache: bool = True):
        self.environment = Environment(environment) if isinstance(environment, str) else environment
        self.config_paths = config_paths or self._get_default_config_paths()
        self.enable_cache = enable_cache
        
        self._profiles: Dict[str, ConfigurationProfile] = {}
        self._active_profile: Optional[ConfigurationProfile] = None
        self._config_cache: Dict[str, Any] = {}
        
        self._load_profiles()
        self._activate_environment_profile()
    
    # Core Methods
    def get_config(self, key: str, default: Any = None, component: str = "global") -> Any:
        """Get configuration value with environment-aware fallback."""
        pass
    
    def set_config(self, key: str, value: Any, component: str = "global", persist: bool = False):
        """Set configuration value with optional persistence."""
        pass
    
    def get_component_config(self, component: str) -> Dict[str, Any]:
        """Get all configuration for a specific component."""
        pass
    
    def validate_configuration(self) -> List[str]:
        """Validate current configuration and return list of errors."""
        pass
    
    def get_dependency_status(self) -> Dict[str, bool]:
        """Check availability of all configured dependencies."""
        pass
    
    def create_environment_profile(self, name: str, base_environment: Environment) -> ConfigurationProfile:
        """Create new environment profile based on existing one."""
        pass
```

### Component-Specific Configuration Interfaces
```python
class ComponentConfigMixin:
    """Mixin for components that need configuration access."""
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        self.config_manager = config_manager or get_global_config_manager()
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration for this component."""
        component_name = self.__class__.__module__.split('.')[-2]  # Extract from module path
        return self.config_manager.get_config(key, default, component_name)
    
    def get_feature_flag(self, flag_name: str, default: bool = False) -> bool:
        """Check if a feature flag is enabled."""
        return self.config_manager.get_config(f"features.{flag_name}", default, "features")

# Registry Base Class Integration
class BaseRegistry(ComponentConfigMixin, Generic[T], ABC):
    """Updated base registry with configuration support."""
    
    def __init__(self, 
                 search_dirs: Optional[List[str]] = None, 
                 config_manager: Optional[ConfigurationManager] = None):
        ComponentConfigMixin.__init__(self, config_manager)
        
        # Get search directories from config if not provided
        if search_dirs is None:
            search_dirs = self.get_config("search_dirs", self._get_default_dirs())
        
        self.search_dirs = search_dirs
        self._cache = {}
        self._enable_caching = self.get_config("enable_caching", True)
        self._cache_ttl = self.get_config("cache_ttl_seconds", 300)
```

## 3. Configuration Structure

### Profile Definitions
```yaml
# config/profiles/development.yaml
name: "development"
environment: "development"

base_config:
  logging:
    level: "DEBUG"
    console_output: true
    file_output: false
  
  performance:
    max_parallel_workers: 2
    timeout_seconds: 300
    enable_profiling: true
  
  paths:
    temp_directory: "./tmp/dev"
    output_directory: "./output/dev"
    cache_directory: "./cache/dev"

feature_flags:
  strict_validation: false
  experimental_features: true
  debug_mode: true
  mock_external_dependencies: true

dependency_config:
  required_packages:
    - "pyyaml"
    - "numpy"
  optional_packages:
    qonnx: 
      fallback_behavior: "warn_and_continue"
      affected_features: ["transforms", "model_conversion"]
    model_profiling:
      fallback_behavior: "disable_feature"
      affected_features: ["analysis_tools", "roofline_analysis"]
  
  external_tools:
    vivado:
      required: false
      fallback: "simulation_mode"

components:
  registries:
    search_dirs_relative: true
    enable_caching: true
    cache_ttl_seconds: 60
    auto_refresh: true
  
  dse:
    max_evaluations_default: 20
    enable_parallel: true
    continue_on_failure: true
  
  finn:
    use_legacy_interface: true
    enable_mock_mode: true
    validate_configurations: false

---
# config/profiles/testing.yaml
name: "testing"
environment: "testing"

base_config:
  logging:
    level: "INFO"
    console_output: false
    file_output: true
    file_path: "./logs/test.log"
  
  performance:
    max_parallel_workers: 1
    timeout_seconds: 120
    enable_profiling: false

feature_flags:
  strict_validation: true
  experimental_features: false
  debug_mode: false
  mock_external_dependencies: true

dependency_config:
  required_packages:
    - "pytest"
    - "pyyaml"
    - "numpy"
  
  fallback_behavior_default: "raise_error"
  
components:
  registries:
    enable_caching: false
    auto_refresh: false
  
  dse:
    max_evaluations_default: 5
    enable_parallel: false
    
---
# config/profiles/production.yaml
name: "production"
environment: "production"

base_config:
  logging:
    level: "WARNING"
    console_output: false
    file_output: true
    file_path: "/var/log/brainsmith/app.log"
    rotation: true
  
  performance:
    max_parallel_workers: -1  # Use all available cores
    timeout_seconds: 3600
    enable_profiling: false

feature_flags:
  strict_validation: true
  experimental_features: false
  debug_mode: false
  mock_external_dependencies: false

dependency_config:
  fallback_behavior_default: "raise_error"
  
components:
  registries:
    enable_caching: true
    cache_ttl_seconds: 3600
  
  dse:
    max_evaluations_default: 100
    enable_parallel: true
    continue_on_failure: false
```

## 4. Integration Strategy

### Phase 1: Core Configuration Infrastructure (Week 1-2)
```python
# Step 1: Create core configuration classes
brainsmith/core/config/
├── __init__.py
├── manager.py          # ConfigurationManager class
├── profiles.py         # ConfigurationProfile and Environment
├── validation.py       # Configuration validation logic
├── dependency.py       # Dependency checking and fallbacks
└── utils.py           # Configuration utilities

# Step 2: Create default profiles
config/
├── profiles/
│   ├── development.yaml
│   ├── testing.yaml
│   ├── production.yaml
│   └── custom.yaml.template
└── schema/
    └── profile_schema.yaml

# Step 3: Environment detection and initialization
class EnvironmentDetector:
    @staticmethod
    def detect_environment() -> Environment:
        """Auto-detect environment from various sources."""
        # Check environment variable
        if env_val := os.getenv("BRAINSMITH_ENV"):
            return Environment(env_val.lower())
        
        # Check for testing context
        if "pytest" in sys.modules or "unittest" in sys.modules:
            return Environment.TESTING
        
        # Check for development indicators
        if Path(".git").exists() or Path("setup.py").exists():
            return Environment.DEVELOPMENT
        
        # Default to production
        return Environment.PRODUCTION
```

### Phase 2: Registry Integration (Week 2-3)
```python
# Update all registry base classes
class BaseRegistry(ComponentConfigMixin, Generic[T], ABC):
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        ComponentConfigMixin.__init__(self, config_manager)
        
        # Configuration-driven initialization
        self.search_dirs = self.get_config("search_dirs", self._get_default_dirs())
        self._enable_caching = self.get_config("enable_caching", True)
        self._auto_refresh = self.get_config("auto_refresh", False)
        self._cache_ttl = self.get_config("cache_ttl_seconds", 300)
        
        if self._auto_refresh:
            self._setup_auto_refresh()

# Example: KernelRegistry updates
class KernelRegistry(BaseRegistry[KernelPackage]):
    def discover_components(self, rescan: bool = False) -> Dict[str, KernelPackage]:
        # Use configuration for cache behavior
        if not rescan and self._enable_caching and self.kernel_cache:
            return self.kernel_cache
        
        # Configuration-driven discovery
        strict_validation = self.get_feature_flag("strict_validation", False)
        mock_dependencies = self.get_feature_flag("mock_external_dependencies", False)
        
        # ... rest of discovery logic with config-driven behavior
```

### Phase 3: Component Migration (Week 3-4)
```python
# DSE Configuration Integration
from brainsmith.core.config import get_global_config_manager

class DSEConfiguration:
    def __init__(self, config_manager: Optional[ConfigurationManager] = None, **overrides):
        self.config_manager = config_manager or get_global_config_manager()
        
        # Get defaults from configuration profile
        self.max_evaluations = overrides.get(
            'max_evaluations', 
            self.config_manager.get_config('max_evaluations_default', 50, 'dse')
        )
        
        self.enable_parallel = overrides.get(
            'enable_parallel',
            self.config_manager.get_config('enable_parallel', True, 'dse')
        )
        
        self.continue_on_failure = overrides.get(
            'continue_on_failure',
            self.config_manager.get_config('continue_on_failure', True, 'dse')
        )

# FINN Interface Integration  
class FINNInterface(ComponentConfigMixin):
    def __init__(self, legacy_config: Dict[str, Any] = None, **kwargs):
        super().__init__(kwargs.get('config_manager'))
        
        # Merge explicit config with profile config
        profile_finn_config = self.get_component_config("finn")
        self.legacy_config = {**profile_finn_config, **(legacy_config or {})}
        
        # Environment-aware behavior
        self.use_mock_mode = self.get_feature_flag("mock_external_dependencies", False)
        self.validate_configs = self.get_config("validate_configurations", True)
```

## 5. Migration Benefits

### Before (Current State):
```python
# Scattered configuration across components
kernel_registry = KernelRegistry(kernel_dirs=["/custom/path"])
dse_config = DSEConfiguration(max_evaluations=50, continue_on_failure=True)
finn_interface = FINNInterface({'fpga_part': 'xcvu9p-flga2104-2-i'})

# No environment awareness
if "pytest" in sys.modules:
    # Manual test configuration
    kernel_registry._enable_caching = False
    dse_config.max_evaluations = 5
```

### After (With Configuration Management):
```python
# Environment-aware initialization
config_manager = ConfigurationManager()  # Auto-detects environment

# Components automatically use appropriate settings
kernel_registry = KernelRegistry(config_manager=config_manager)
dse_config = DSEConfiguration(config_manager=config_manager)
finn_interface = FINNInterface(config_manager=config_manager)

# Environment profiles handle the complexity
# - Development: caching disabled, debug mode on, mock dependencies
# - Testing: strict validation, minimal resource usage  
# - Production: full caching, performance optimized, strict error handling
```

## 6. Implementation Timeline

### Week 1: Foundation
- [ ] Core configuration classes and data structures
- [ ] Environment detection logic
- [ ] Basic profile loading and validation
- [ ] Unit tests for configuration system

### Week 2: Registry Integration
- [ ] Update BaseRegistry to use configuration
- [ ] Integrate with existing registries (kernels, transforms, analysis, blueprints, hooks)
- [ ] Create ComponentConfigMixin for easy adoption
- [ ] Integration tests for registry configuration

### Week 3: Component Migration
- [ ] Update DSEConfiguration to use profiles
- [ ] Integrate FINN interface with configuration
- [ ] Update core API to use configuration management
- [ ] Backward compatibility testing

### Week 4: Profile Creation and Testing
- [ ] Create comprehensive default profiles
- [ ] Environment-specific testing
- [ ] Performance benchmarking
- [ ] Documentation and migration guide

## 7. Backward Compatibility Strategy

### Compatibility Layer
```python
# Old way still works
kernel_registry = KernelRegistry(kernel_dirs=["/custom/path"])

# But now internally uses configuration system
class KernelRegistry(BaseRegistry[KernelPackage]):
    def __init__(self, kernel_dirs: Optional[List[str]] = None, config_manager: Optional[ConfigurationManager] = None):
        super().__init__(config_manager)
        
        # Explicit parameters override configuration
        if kernel_dirs is not None:
            self.search_dirs = kernel_dirs
        else:
            self.search_dirs = self.get_config("search_dirs", self._get_default_dirs())
```

### Migration Path
1. **Phase 1**: New system available alongside old system
2. **Phase 2**: Deprecation warnings for old patterns
3. **Phase 3**: Old patterns removed (major version bump)

## 8. Success Metrics

### Quantitative Metrics
- **Configuration Consistency**: 100% of components use unified configuration
- **Environment Support**: 3 complete profiles (dev/test/prod) working
- **Dependency Handling**: 90% of dependency issues handled gracefully
- **Test Coverage**: 95% coverage for configuration system

### Qualitative Metrics
- **Developer Experience**: Single environment variable switches behavior
- **Debugging Improvement**: Clear configuration source tracing
- **Deployment Simplification**: Environment-specific behavior without code changes
- **Dependency Resilience**: Graceful handling of missing optional dependencies

## 9. Risk Mitigation

### Configuration Conflicts
- **Risk**: Multiple configuration sources conflicting
- **Mitigation**: Clear precedence rules and validation

### Performance Impact  
- **Risk**: Configuration lookups adding overhead
- **Mitigation**: Caching and lazy loading strategies

### Migration Complexity
- **Risk**: Breaking existing integrations
- **Mitigation**: Comprehensive backward compatibility layer

### Profile Management
- **Risk**: Profile drift between environments
- **Mitigation**: Schema validation and automated testing

---

*This configuration management system will provide a solid foundation for the other architectural improvements while solving immediate pain points around dependency handling and environment-specific behavior.*