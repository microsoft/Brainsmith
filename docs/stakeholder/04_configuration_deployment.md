# Brainsmith-2: Configuration & Deployment

## Configuration Management

Brainsmith-2 implements a **hierarchical configuration system** that supports environment-specific settings, template customization, and deployment optimization. The system balances ease-of-use with comprehensive customization capabilities.

### Multi-Level Configuration Architecture

#### Configuration Hierarchy
```
Global Defaults
    ↓
Environment Configuration (development/production/testing)
    ↓
Project-Specific Configuration 
    ↓
Command-Line Overrides
```

#### Core Configuration Components

**Pipeline Configuration** (`PipelineConfig`)
```python
from brainsmith.tools.hw_kernel_gen.enhanced_config import PipelineConfig

# Complete pipeline configuration
config = PipelineConfig(
    template=TemplateConfig(
        template_dirs=['templates/', 'custom_templates/'],
        selection_strategy='auto',  # auto, dataflow, legacy, custom
        cache_templates=True,
        auto_reload=True
    ),
    
    generation=GenerationConfig(
        enabled_generators={'hwcustomop', 'rtlbackend', 'test_suite', 'documentation'},
        output_organization='hierarchical',  # flat, hierarchical, by_type
        overwrite_existing=True,
        preserve_intermediates=False
    ),
    
    analysis=AnalysisConfig(
        interface_detection='enhanced',  # basic, enhanced, strict
        pragma_processing=True,
        validation_level='comprehensive',  # basic, standard, comprehensive
        caching_enabled=True,
        cache_size_limit=1000  # MB
    ),
    
    dataflow=DataflowConfig(
        mode='DATAFLOW_ONLY',  # DISABLED, HYBRID, DATAFLOW_ONLY
        optimization_level='balanced',  # conservative, balanced, aggressive
        parallelism_analysis=True,
        resource_constraints={
            'max_luts': 100000,
            'max_dsps': 500,
            'max_bram': 1000
        }
    ),
    
    validation=ValidationConfig(
        strict_validation=True,
        error_on_warnings=False,
        detailed_reports=True,
        performance_validation=True
    )
)
```

### Environment-Specific Configuration

**Development Environment**
```python
# Development optimized for fast iteration
development_config = PipelineConfig(
    template=TemplateConfig(
        cache_templates=True,
        auto_reload=True,  # Reload templates on change
        selection_strategy='base_class_preferred'  # Faster generation
    ),
    
    generation=GenerationConfig(
        preserve_intermediates=True,  # Keep debugging artifacts
        debug_output=True,
        verbose_logging=True
    ),
    
    analysis=AnalysisConfig(
        caching_enabled=True,
        cache_persistence=True,  # Cache across sessions
        validation_level='standard'  # Faster validation
    ),
    
    dataflow=DataflowConfig(
        optimization_level='conservative',  # Predictable results
        resource_constraints={'max_luts': 50000}  # Development FPGA
    )
)
```

**Production Environment**  
```python
# Production optimized for performance and reliability
production_config = PipelineConfig(
    template=TemplateConfig(
        cache_templates=True,
        auto_reload=False,  # Stability over flexibility
        selection_strategy='performance_optimized'
    ),
    
    generation=GenerationConfig(
        preserve_intermediates=False,  # Clean output
        debug_output=False,
        code_optimization=True
    ),
    
    analysis=AnalysisConfig(
        validation_level='comprehensive',  # Thorough validation
        caching_enabled=True,
        cache_persistence=False  # Fresh analysis each run
    ),
    
    dataflow=DataflowConfig(
        optimization_level='aggressive',  # Maximum performance
        parallelism_analysis=True,
        resource_constraints={
            'max_luts': 500000,      # Production FPGA
            'max_dsps': 2000,
            'target_frequency': 300   # MHz
        }
    ),
    
    validation=ValidationConfig(
        strict_validation=True,
        error_on_warnings=True,  # Strict quality control
        performance_validation=True
    )
)
```

**Testing Environment**
```python
# Testing optimized for validation and coverage
testing_config = PipelineConfig(
    generation=GenerationConfig(
        enabled_generators={'hwcustomop', 'rtlbackend', 'test_suite'},
        preserve_intermediates=True,  # For test analysis
        generate_coverage_reports=True
    ),
    
    analysis=AnalysisConfig(
        validation_level='comprehensive',
        detailed_reports=True,
        caching_enabled=False  # Fresh analysis for each test
    ),
    
    validation=ValidationConfig(
        strict_validation=True,
        error_on_warnings=True,
        detailed_reports=True,
        generate_test_vectors=True
    )
)
```

### Configuration File Management

**YAML Configuration Format**
```yaml
# brainsmith_config.yaml
pipeline:
  template:
    template_dirs:
      - "custom_templates/"
      - "dataflow_templates/"
    selection_strategy: "auto"
    cache_templates: true
    
  generation:
    enabled_generators:
      - "hwcustomop"
      - "rtlbackend" 
      - "test_suite"
    output_organization: "hierarchical"
    overwrite_existing: true
    
  analysis:
    interface_detection: "enhanced"
    pragma_processing: true
    validation_level: "comprehensive"
    caching_enabled: true
    
  dataflow:
    mode: "DATAFLOW_ONLY"
    optimization_level: "balanced"
    resource_constraints:
      max_luts: 100000
      max_dsps: 500
      target_frequency: 250
      
  validation:
    strict_validation: true
    detailed_reports: true
    performance_validation: true
```

**Loading Configuration**
```python
# Load from file
config = PipelineConfig.from_file('brainsmith_config.yaml')

# Load from environment variables
config = PipelineConfig.from_environment('BRAINSMITH_')

# Merge configurations
base_config = PipelineConfig.development_defaults()
custom_config = PipelineConfig.from_file('custom.yaml')
final_config = base_config.merge(custom_config)
```

### Template System Configuration

**Template Directory Structure**
```
templates/
├── dataflow/                    # Dataflow-aware templates
│   ├── hw_custom_op_slim.py.j2 # Minimal HWCustomOp using base classes
│   ├── rtl_backend.py.j2        # RTL backend with dataflow integration
│   └── test_suite_enhanced.py.j2 # Comprehensive test generation
├── legacy/                      # Full generation templates
│   ├── hw_custom_op_full.py.j2  # Complete HWCustomOp implementation
│   └── rtl_backend_full.py.j2   # Full RTL backend generation
└── custom/                      # Project-specific templates
    ├── specialized_op.py.j2     # Custom operation templates
    └── integration_test.py.j2   # Custom testing templates
```

**Template Selection Strategies**
```python
# Automatic selection based on complexity analysis
TemplateConfig(selection_strategy='auto')

# Prefer dataflow-aware templates with base class inheritance
TemplateConfig(selection_strategy='dataflow_preferred')

# Use legacy full-generation templates
TemplateConfig(selection_strategy='legacy')

# Custom selection logic
def custom_template_selector(context):
    if context['operation_complexity'] > 0.8:
        return 'legacy'
    return 'dataflow'

TemplateConfig(selection_strategy=custom_template_selector)
```

## Deployment Patterns

### Development Environment Setup

**Local Development Environment**
```bash
# Standard setup for individual developers
git clone <repository-url> brainsmith-2
cd brainsmith-2

# Create isolated environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install in development mode
pip install -e .

# Verify setup
python -c "from brainsmith.core.hw_compiler import forge; print('Setup complete')"

# Optional: Install development tools
pip install pytest black isort pre-commit
pre-commit install
```

**Docker Development Environment**
```dockerfile
# Dockerfile.development
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY . .
RUN pip install -e .

# Setup development tools
RUN pip install pytest black isort ipython jupyter

# Expose Jupyter port
EXPOSE 8888

# Default command for development
CMD ["bash"]
```

**Development Workflow**
```bash
# Build development container
docker build -f Dockerfile.development -t brainsmith-dev .

# Run with volume mounting for live code editing
docker run -it --rm \
    -v $(pwd):/workspace \
    -p 8888:8888 \
    brainsmith-dev

# Inside container - run development tasks
pytest tests/ -v                    # Run tests
python demos/bert/end2end_bert.py   # Test BERT pipeline
jupyter notebook --allow-root      # Start Jupyter for exploration
```

### CI/CD Integration

**GitHub Actions Workflow**
```yaml
# .github/workflows/ci.yml
name: Brainsmith-2 CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: 3.9

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']
        
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        pip install pytest pytest-cov black isort
        
    - name: Code quality checks
      run: |
        black --check .
        isort --check-only .
        
    - name: Run tests
      run: |
        pytest tests/ -v --cov=brainsmith --cov-report=xml
        
    - name: Test BERT pipeline
      run: |
        cd demos/bert
        python end2end_bert.py --quick_test
        
    - name: Test hardware kernel generation
      run: |
        python -m brainsmith.tools.hw_kernel_gen.hkg \
          examples/thresholding/thresholding_axi.sv \
          examples/thresholding/dummy_compiler_data.py \
          --output_dir test_output
          
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build-and-publish:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t brainsmith-2:latest .
        docker build -f Dockerfile.production -t brainsmith-2:production .
        
    - name: Run integration tests
      run: |
        docker run --rm brainsmith-2:latest python -m pytest tests/integration/ -v
        
    - name: Push to registry (if needed)
      run: |
        # docker push brainsmith-2:latest
        echo "Would push to container registry"
```

**GitLab CI/CD Pipeline**
```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  PYTHON_VERSION: "3.9"
  
test:
  stage: test
  image: python:${PYTHON_VERSION}
  before_script:
    - pip install -e .
    - pip install pytest pytest-cov
  script:
    - pytest tests/ -v --cov=brainsmith
    - python demos/bert/end2end_bert.py --validate_only
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main

deploy:
  stage: deploy
  script:
    - echo "Deploy to production environment"
    - docker run $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main
  when: manual
```

### Production Deployment

**Production Docker Configuration**
```dockerfile
# Dockerfile.production
FROM python:3.9-slim as builder

# Build dependencies
WORKDIR /build
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production stage
FROM python:3.9-slim

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local

# Add to PATH
ENV PATH=/root/.local/bin:$PATH

# Create app user
RUN useradd --create-home --shell /bin/bash brainsmith
USER brainsmith
WORKDIR /app

# Copy application
COPY --chown=brainsmith:brainsmith . .
RUN pip install --user -e .

# Production configuration
ENV BRAINSMITH_CONFIG_FILE=/app/config/production.yaml
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "from brainsmith.core.hw_compiler import forge" || exit 1

# Default command
CMD ["python", "-m", "brainsmith.core.hw_compiler", "--server-mode"]
```

**Production Configuration**
```yaml
# config/production.yaml
pipeline:
  template:
    cache_templates: true
    auto_reload: false
    selection_strategy: "performance_optimized"
    
  generation:
    preserve_intermediates: false
    debug_output: false
    output_organization: "hierarchical"
    
  analysis:
    validation_level: "comprehensive"
    caching_enabled: true
    cache_persistence: false
    
  dataflow:
    mode: "DATAFLOW_ONLY"
    optimization_level: "aggressive"
    resource_constraints:
      max_luts: 1000000
      max_dsps: 5000
      target_frequency: 300
      
  validation:
    strict_validation: true
    error_on_warnings: true
    performance_validation: true

logging:
  level: "INFO"
  format: "json"
  file: "/var/log/brainsmith/application.log"
  
monitoring:
  metrics_enabled: true
  prometheus_port: 9090
  health_check_interval: 30
```

**Kubernetes Deployment**
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: brainsmith-2
spec:
  replicas: 3
  selector:
    matchLabels:
      app: brainsmith-2
  template:
    metadata:
      labels:
        app: brainsmith-2
    spec:
      containers:
      - name: brainsmith-2
        image: brainsmith-2:production
        ports:
        - containerPort: 8080
        - containerPort: 9090  # metrics
        env:
        - name: BRAINSMITH_CONFIG_FILE
          value: "/config/production.yaml"
        volumeMounts:
        - name: config
          mountPath: /config
        - name: output
          mountPath: /output
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 30
      volumes:
      - name: config
        configMap:
          name: brainsmith-config
      - name: output
        persistentVolumeClaim:
          claimName: brainsmith-output

---
apiVersion: v1
kind: Service
metadata:
  name: brainsmith-2-service
spec:
  selector:
    app: brainsmith-2
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
```

## Monitoring & Diagnostics

### Performance Monitoring

**Built-in Metrics Collection**
```python
from brainsmith.tools.profiling import SystemProfiler

# Enable performance monitoring
profiler = SystemProfiler(
    metrics=['compilation_time', 'memory_usage', 'resource_utilization'],
    export_format='prometheus'
)

# Monitor compilation pipeline
with profiler.monitor('bert_compilation'):
    result = forge('bert', model_path, args)

# Export metrics
profiler.export_metrics('http://prometheus:9090')
```

**Resource Usage Tracking**
```python
from brainsmith.tools.profiling import ResourceMonitor

# Track FPGA resource utilization
monitor = ResourceMonitor()

# During compilation
resource_usage = monitor.track_compilation(model_path)
print(f"LUT usage: {resource_usage.luts}/{resource_usage.max_luts}")
print(f"DSP usage: {resource_usage.dsps}/{resource_usage.max_dsps}")
print(f"BRAM usage: {resource_usage.bram}/{resource_usage.max_bram}")

# Historical tracking
monitor.save_usage_history('resource_usage.json')
```

### Diagnostic Tools

**System Health Check**
```bash
# Built-in diagnostic tool
python -m brainsmith.tools.diagnostics --full-check

# Output:
# ✅ Python environment: OK (3.9.7)
# ✅ Dependencies: OK (FINN 0.8.1, QONNX 0.3.0)
# ✅ Template system: OK (15 templates loaded)
# ✅ RTL parser: OK (tree-sitter grammar loaded)
# ✅ Configuration: OK (production.yaml)
# ⚠️  Cache directory: 95% full (cleanup recommended)
# ✅ Example models: OK (thresholding test passed)
```

**Configuration Validation**
```python
from brainsmith.tools.diagnostics import ConfigValidator

# Validate configuration
validator = ConfigValidator()
result = validator.validate_config('production.yaml')

if not result.is_valid:
    print("Configuration issues found:")
    for issue in result.issues:
        print(f"  {issue.severity}: {issue.message}")
        print(f"    Suggested fix: {issue.suggestion}")
```

**Performance Diagnostics**
```python
from brainsmith.tools.profiling import PerformanceDiagnostics

# Analyze compilation performance
diagnostics = PerformanceDiagnostics()
report = diagnostics.analyze_compilation('compilation_log.json')

print(f"Total compilation time: {report.total_time}s")
print(f"Bottleneck phase: {report.slowest_phase}")
print(f"Memory peak usage: {report.peak_memory}MB")

# Optimization suggestions
for suggestion in report.optimization_suggestions:
    print(f"💡 {suggestion}")
```

### Common Troubleshooting Scenarios

**Memory Issues**
```bash
# Check memory usage patterns
python -m brainsmith.tools.diagnostics --memory-analysis

# Optimize for low memory
export BRAINSMITH_MEMORY_LIMIT=4G
export BRAINSMITH_CACHE_SIZE=512M
python demos/bert/end2end_bert.py
```

**Template Issues**
```bash
# Validate templates
python -m brainsmith.tools.diagnostics --template-check

# Clear template cache
python -m brainsmith.tools.cache --clear-templates

# Debug template rendering
python -m brainsmith.tools.hw_kernel_gen.hkg \
  --debug-templates \
  --preserve-intermediates \
  example.sv metadata.py
```

**Performance Issues**
```bash
# Profile compilation
python -m brainsmith.tools.profiling --profile-compilation model.onnx

# Optimize configuration
python -m brainsmith.tools.optimization --suggest-config model.onnx

# Benchmark different configurations
python -m brainsmith.tools.benchmarking --compare-configs \
  config1.yaml config2.yaml model.onnx
```

This configuration and deployment guide provides comprehensive coverage of Brainsmith-2's configuration management, deployment patterns, and operational monitoring capabilities, enabling teams to effectively deploy and maintain the platform in various environments.