# FINN-Brainsmith API V2 - Plugin System for Open Source Contributors

**Version**: 2.1  
**Date**: December 2024  
**Purpose**: Enhanced design for seamless community contributions

## Executive Summary

To make Brainsmith a premium open source platform that attracts contributors, we need a plugin system that makes it trivially easy to:
1. Create new kernels and transforms
2. Package and distribute them
3. Discover and use community contributions
4. Maintain quality and compatibility

## Plugin Architecture

### 1. Declarative Component Registration

Instead of programmatic registration, use decorators and metadata:

```python
from brainsmith.api import kernel, transform, backend

@kernel(
    name="FlashAttention",
    operation_type="attention",
    description="Efficient attention mechanism for transformers",
    author="community-member",
    version="1.0.0",
    tags=["transformer", "attention", "efficient"]
)
class FlashAttentionKernel:
    """Community-contributed Flash Attention kernel."""
    
    constraints = {
        "min_sequence_length": 64,
        "supported_dtypes": ["float16", "bfloat16"]
    }
    
    @backend("cuda", default=True)
    def cuda_implementation(self, config):
        """CUDA implementation of Flash Attention."""
        return CUDAFlashAttentionOp(config)
    
    @backend("hls")
    def hls_implementation(self, config):
        """HLS implementation for FPGA."""
        return HLSFlashAttentionOp(config)
```

### 2. Transform Plugins with Auto-Registration

```python
from brainsmith.api import transform, requires

@transform(
    name="QuantizeAttention",
    stage="topology_optimization",
    description="Quantize attention layers to INT8",
    author="optimization-expert",
    version="2.1.0",
    tags=["quantization", "attention", "int8"]
)
@requires("numpy>=1.20", "onnx>=1.10")
class QuantizeAttentionTransform:
    """Intelligently quantize attention mechanisms."""
    
    def __init__(self):
        self.statistics = {}
    
    def apply(self, model, config):
        """Apply quantization to attention layers."""
        # Transform implementation
        return quantized_model
    
    @property
    def compatible_kernels(self):
        """Kernels this transform works well with."""
        return ["FlashAttention", "StandardAttention"]
```

### 3. Plugin Discovery System

```python
# brainsmith/plugin_system.py

class PluginRegistry:
    """Central registry for all plugins with discovery capabilities."""
    
    def __init__(self):
        self.search_paths = [
            "~/.brainsmith/plugins",      # User plugins
            "/usr/share/brainsmith",       # System plugins
            "./brainsmith_plugins"         # Project plugins
        ]
        self._plugins = {}
        self._metadata_cache = {}
    
    def discover_plugins(self):
        """Auto-discover plugins from all search paths."""
        for path in self.search_paths:
            self._scan_directory(path)
        
        # Also discover from pip-installed packages
        self._scan_pip_packages("brainsmith-plugin-*")
    
    def search(self, query: str, filters: Dict[str, Any] = None):
        """Search for plugins with advanced filtering."""
        results = []
        
        for plugin in self._plugins.values():
            if self._matches_query(plugin, query, filters):
                results.append(plugin)
        
        return sorted(results, key=lambda p: p.downloads, reverse=True)
    
    def install_from_hub(self, plugin_name: str, version: str = "latest"):
        """Install plugin from BrainSmith Hub."""
        # Download from community hub
        plugin_data = self.hub_client.download(plugin_name, version)
        
        # Validate plugin
        if self.validate_plugin(plugin_data):
            self._install_plugin(plugin_data)
        else:
            raise PluginValidationError(f"Plugin {plugin_name} failed validation")
```

### 4. Plugin Manifest Format

Each plugin includes a `brainsmith.yaml` manifest:

```yaml
# brainsmith.yaml
name: flash-attention-plugin
version: 1.0.0
author: community-member
license: Apache-2.0
description: Efficient Flash Attention implementation for transformers

kernels:
  - name: FlashAttention
    module: flash_attention.kernel
    class: FlashAttentionKernel
    
transforms:
  - name: OptimizeFlashAttention  
    module: flash_attention.transform
    class: OptimizeFlashAttentionTransform

dependencies:
  - numpy>=1.20
  - torch>=2.0
  
compatibility:
  brainsmith: ">=2.0"
  finn: ">=0.9"
  
tags:
  - transformer
  - attention
  - performance
  
examples:
  - examples/bert_flash_attention.py
  - examples/gpt_optimization.py
```

### 5. Plugin Development Kit (PDK)

Make it super easy to create plugins:

```bash
# Create new plugin from template
$ brainsmith create-plugin my-awesome-kernel --type kernel

# This generates:
my-awesome-kernel/
├── brainsmith.yaml
├── src/
│   └── my_awesome_kernel/
│       ├── __init__.py
│       ├── kernel.py
│       └── backends/
│           ├── hls.py
│           └── rtl.py
├── tests/
│   └── test_kernel.py
├── examples/
│   └── example_usage.py
├── docs/
│   └── README.md
└── setup.py
```

### 6. Quality Assurance System

Ensure plugin quality without being restrictive:

```python
class PluginValidator:
    """Validate plugins for quality and compatibility."""
    
    def validate(self, plugin_path: str) -> ValidationResult:
        """Run comprehensive validation checks."""
        checks = [
            self.check_manifest(),
            self.check_code_quality(),
            self.check_tests(),
            self.check_documentation(),
            self.check_examples(),
            self.check_performance(),
            self.check_compatibility()
        ]
        
        return ValidationResult(checks)
    
    def certify(self, plugin: Plugin) -> CertificationLevel:
        """Certify plugin quality level."""
        score = self.calculate_quality_score(plugin)
        
        if score >= 90:
            return CertificationLevel.GOLD  # Fully tested, documented
        elif score >= 70:
            return CertificationLevel.SILVER  # Good quality
        elif score >= 50:
            return CertificationLevel.BRONZE  # Basic quality
        else:
            return CertificationLevel.COMMUNITY  # Experimental
```

### 7. BrainSmith Hub - Community Platform

Web platform for sharing and discovering plugins:

```python
# API for BrainSmith Hub
class BrainSmithHub:
    """Community hub for sharing plugins."""
    
    def publish(self, plugin: Plugin, api_key: str):
        """Publish plugin to community hub."""
        # Validate plugin
        validation = self.validator.validate(plugin)
        if not validation.passed:
            raise ValidationError(validation.errors)
        
        # Run automated tests
        test_results = self.test_runner.run(plugin)
        
        # Calculate metrics
        metrics = {
            "performance": self.benchmark(plugin),
            "compatibility": self.check_compatibility(plugin),
            "quality_score": self.calculate_quality(plugin)
        }
        
        # Publish to hub
        return self.api.publish(plugin, metrics, api_key)
    
    def search(self, query: str, filters: Dict = None):
        """Search for plugins with advanced filtering."""
        return self.api.search(
            query=query,
            filters={
                "certification": filters.get("certification"),
                "min_downloads": filters.get("min_downloads", 0),
                "tags": filters.get("tags", []),
                "author": filters.get("author"),
                "compatible_with": filters.get("compatible_with")
            }
        )
```

### 8. Seamless Integration in User Code

Make using plugins as easy as possible:

```python
from brainsmith import Compiler, PluginRegistry

# Auto-discover all available plugins
registry = PluginRegistry()
registry.discover_plugins()

# Search for attention kernels
attention_kernels = registry.search("attention", filters={
    "certification": "gold",
    "tags": ["transformer"]
})

# Install new plugin from hub
registry.install_from_hub("flash-attention", version="2.0")

# Use in compilation strategy - plugins work just like built-ins
strategy = CompilationStrategy(
    name="bert_with_flash_attention",
    kernels=[
        # Built-in kernel
        KernelSpec("MatMul", backend="hls"),
        # Community plugin kernel - seamlessly integrated
        KernelSpec("FlashAttention", backend="cuda"),
    ],
    transforms=[
        # Mix built-in and plugin transforms
        TransformSpec("Streamline"),  # Built-in
        TransformSpec("QuantizeAttention"),  # Plugin
    ]
)

# Compile normally - plugins are transparent to user
compiler = Compiler()
result = compiler.compile("model.onnx", strategy)
```

### 9. Plugin Development Workflow

Make contribution as smooth as possible:

```bash
# 1. Create plugin from template
$ brainsmith create-plugin custom-conv2d --type kernel

# 2. Develop with hot reload
$ brainsmith develop --watch
✓ Plugin loaded: CustomConv2D
✓ Running tests... passed
✓ Checking quality... score: 85/100

# 3. Test with example models
$ brainsmith test-plugin custom-conv2d --model resnet50.onnx
✓ Performance: 1.2x faster than standard Conv2D
✓ Resource usage: 15% less DSPs
✓ Compatibility: Works with all tested models

# 4. Package for distribution
$ brainsmith package custom-conv2d
✓ Created: custom-conv2d-1.0.0.bspkg
✓ Size: 125KB
✓ Ready to publish

# 5. Publish to hub
$ brainsmith publish custom-conv2d-1.0.0.bspkg
✓ Validation passed
✓ Automated tests passed
✓ Published to BrainSmith Hub
✓ Available at: https://hub.brainsmith.ai/plugins/custom-conv2d
```

### 10. Incentive System

Encourage quality contributions:

```python
class ContributorRewards:
    """Track and reward contributors."""
    
    def calculate_impact(self, plugin: Plugin) -> ImpactScore:
        """Calculate plugin impact score."""
        return ImpactScore(
            downloads=plugin.download_count,
            usage_in_strategies=plugin.usage_count,
            performance_improvement=plugin.benchmark_scores,
            community_rating=plugin.average_rating,
            compatibility_range=plugin.compatible_versions
        )
    
    def assign_badges(self, author: Author) -> List[Badge]:
        """Assign badges based on contributions."""
        badges = []
        
        if author.total_downloads > 10000:
            badges.append(Badge.POPULAR_CONTRIBUTOR)
        
        if author.plugin_count > 5:
            badges.append(Badge.PROLIFIC_CREATOR)
        
        if author.average_rating > 4.5:
            badges.append(Badge.QUALITY_CHAMPION)
        
        return badges
```

## Benefits for Contributors

1. **Low Barrier to Entry**: Templates and examples make starting easy
2. **Clear Guidelines**: Quality metrics show what makes a good plugin  
3. **Immediate Feedback**: Automated testing and validation
4. **Wide Reach**: Hub makes plugins discoverable
5. **Recognition**: Badges and metrics highlight top contributors
6. **Seamless Integration**: Plugins work exactly like built-in components

## Benefits for Users

1. **Easy Discovery**: Search and filter plugins by need
2. **Quality Assurance**: Certification levels indicate plugin quality
3. **Simple Installation**: One command to add new capabilities
4. **Transparent Usage**: Plugins integrate seamlessly with core
5. **Community Support**: Ratings and reviews guide choices

## Technical Implementation

### Plugin Loading System

```python
class PluginLoader:
    """Dynamic plugin loading system."""
    
    def load_plugin(self, manifest_path: str) -> Plugin:
        """Load plugin from manifest."""
        manifest = self._parse_manifest(manifest_path)
        
        # Validate dependencies
        self._check_dependencies(manifest.dependencies)
        
        # Load Python modules
        modules = {}
        for component in manifest.components:
            module = self._import_module(component.module)
            modules[component.name] = getattr(module, component.class)
        
        # Register with appropriate registries
        for kernel in manifest.kernels:
            self.kernel_registry.register(modules[kernel.name])
        
        for transform in manifest.transforms:
            self.transform_registry.register(modules[transform.name])
        
        return Plugin(manifest, modules)
```

### Sandboxed Execution

Ensure plugins can't break the system:

```python
class PluginSandbox:
    """Sandbox for safe plugin execution."""
    
    def execute_transform(self, transform: Transform, model: Model, config: Dict):
        """Execute transform in sandboxed environment."""
        with self.create_sandbox() as sandbox:
            # Limit resources
            sandbox.set_memory_limit(self.memory_limit)
            sandbox.set_time_limit(self.time_limit)
            
            # Restrict file system access
            sandbox.restrict_filesystem(read_only_paths=["/tmp"])
            
            # Execute transform
            try:
                result = sandbox.run(transform.apply, model, config)
                return result
            except SandboxViolation as e:
                logger.error(f"Plugin violated sandbox: {e}")
                raise PluginSecurityError(f"Plugin {transform.name} violated security constraints")
```

## Conclusion

This enhanced plugin system makes BrainSmith truly open and extensible:

1. **Declarative Registration**: Simple decorators instead of complex APIs
2. **Auto-Discovery**: Plugins found automatically from multiple sources
3. **Quality Assurance**: Automated validation and certification
4. **Community Hub**: Central place for sharing and discovery
5. **Seamless Integration**: Plugins indistinguishable from built-in components
6. **Developer Friendly**: Templates, tools, and documentation
7. **Safe Execution**: Sandboxing protects system integrity

This design positions BrainSmith as a premium open source platform where contributors can easily create, share, and monetize high-quality kernels and transforms, while users can confidently discover and use community contributions.