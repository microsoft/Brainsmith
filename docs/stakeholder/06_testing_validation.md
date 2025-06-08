# Brainsmith-2: Testing & Validation

## Testing Framework Overview

Brainsmith-2 implements a **comprehensive testing architecture** with 575+ automated tests organized across multiple validation layers. The testing framework emphasizes both functional correctness and performance validation, ensuring reliable FPGA acceleration capabilities.

### Test Organization Structure

#### Hierarchical Test Architecture
```
tests/
├── dataflow/              # Dataflow framework validation (127 tests)
│   ├── core/             # Core component tests (89 tests)
│   ├── integration/      # Cross-component tests (23 tests) 
│   └── unit/            # Individual unit tests (15 tests)
├── tools/               # Tool-specific validation (312 tests)
│   └── hw_kernel_gen/   # HKG pipeline tests (312 tests)
│       ├── analysis/    # Analysis component tests (78 tests)
│       ├── generators/  # Generator tests (89 tests)
│       ├── orchestration/ # Pipeline tests (45 tests)
│       ├── rtl_parser/  # RTL parser tests (67 tests)
│       └── compatibility/ # Legacy compatibility (33 tests)
├── integration/         # End-to-end tests (89 tests)
│   ├── bert_pipeline/   # BERT workflow tests (34 tests)
│   ├── custom_ops/     # Custom operation tests (28 tests)
│   └── performance/    # Performance validation (27 tests)
└── validation/         # Compatibility tests (47 tests)
    ├── phase_compatibility/ # Version compatibility (23 tests)
    └── regression/     # Regression prevention (24 tests)
```

#### Test Categories by Purpose

**Unit Tests** (267 tests)
- Individual component testing in isolation
- Fast execution (< 5 seconds per test)
- High code coverage (>90% for core components)
- Mock external dependencies

**Integration Tests** (198 tests) 
- Cross-component interaction validation
- Moderate execution time (5-30 seconds per test)
- Real dependency integration
- Data flow validation

**End-to-End Tests** (89 tests)
- Complete workflow validation
- Longer execution time (30 seconds - 5 minutes)
- Real FPGA compilation testing
- Performance validation

**Validation Tests** (47 tests)
- Backward compatibility verification
- Regression prevention
- Quality assurance validation
- Migration testing

### Core Testing Components

#### Dataflow Framework Testing

**Interface Validation Testing**
```python
# tests/dataflow/core/test_dataflow_interface.py
import pytest
from brainsmith.dataflow.core.dataflow_interface import DataflowInterface

class TestDataflowInterface:
    """Comprehensive testing of DataflowInterface functionality."""
    
    def test_dimensional_constraint_validation(self):
        """Test qDim/tDim/sDim constraint enforcement."""
        
        # Valid configuration
        valid_interface = DataflowInterface(
            name="test_input",
            interface_type="INPUT", 
            qDim=768, tDim=64, sDim=8,
            dtype="INT8"
        )
        assert valid_interface.validate_constraints().success
        
        # Invalid: sDim > tDim
        with pytest.raises(ValueError, match="sDim must be <= tDim"):
            DataflowInterface(
                name="invalid", interface_type="INPUT",
                qDim=768, tDim=32, sDim=64, dtype="INT8"
            )
        
        # Invalid: tDim > qDim  
        with pytest.raises(ValueError, match="tDim must be <= qDim"):
            DataflowInterface(
                name="invalid", interface_type="INPUT", 
                qDim=256, tDim=512, sDim=8, dtype="INT8"
            )
    
    @pytest.mark.parametrize("qDim,tDim,sDim,expected_parallelism", [
        (768, 64, 8, 8),      # Standard BERT configuration
        (1024, 128, 16, 16),  # Higher parallelism
        (512, 32, 4, 4),      # Conservative configuration
    ])
    def test_parallelism_calculation(self, qDim, tDim, sDim, expected_parallelism):
        """Test parallelism calculation accuracy."""
        
        interface = DataflowInterface(
            name="test", interface_type="INPUT",
            qDim=qDim, tDim=tDim, sDim=sDim, dtype="INT8"
        )
        
        calculated_parallelism = interface.calculate_stream_parallelism()
        assert calculated_parallelism == expected_parallelism
```

**Dataflow Model Testing**
```python
# tests/dataflow/core/test_dataflow_model.py
class TestDataflowModelPerformance:
    """Test dataflow model performance analysis capabilities."""
    
    @pytest.fixture
    def bert_attention_model(self):
        """Create BERT attention dataflow model for testing."""
        interfaces = [
            DataflowInterface("query", "INPUT", qDim=512, tDim=64, sDim=8),
            DataflowInterface("key", "INPUT", qDim=512, tDim=64, sDim=8),
            DataflowInterface("value", "INPUT", qDim=512, tDim=64, sDim=8),
            DataflowInterface("output", "OUTPUT", qDim=512, tDim=64, sDim=8)
        ]
        
        return DataflowModel(
            interfaces=interfaces,
            operation_type="multi_head_attention"
        )
    
    def test_initiation_interval_calculation(self, bert_attention_model):
        """Test II calculation for different parallelism configurations."""
        
        # Test various parallelism configurations
        test_configs = [
            {"iPar": 1, "wPar": 1, "expected_ii": 64},
            {"iPar": 8, "wPar": 1, "expected_ii": 8}, 
            {"iPar": 8, "wPar": 8, "expected_ii": 1}
        ]
        
        for config in test_configs:
            ii_result = bert_attention_model.calculate_initiation_intervals(
                config["iPar"], config["wPar"]
            )
            
            assert ii_result["compute_ii"] <= config["expected_ii"]
            assert ii_result["memory_ii"] >= 1
            assert ii_result["overall_ii"] >= max(ii_result["compute_ii"], ii_result["memory_ii"])
    
    def test_parallelism_optimization(self, bert_attention_model):
        """Test automatic parallelism optimization."""
        
        constraints = {
            'max_luts': 50000,
            'max_dsps': 200, 
            'target_frequency': 250  # MHz
        }
        
        optimal_config = bert_attention_model.optimize_parallelism(constraints)
        
        # Validate optimization results
        assert 'input_parallelism' in optimal_config
        assert 'compute_parallelism' in optimal_config
        assert optimal_config['input_parallelism'] >= 1
        assert optimal_config['compute_parallelism'] >= 1
        
        # Ensure resource constraints are respected
        estimated_resources = bert_attention_model.estimate_resources(optimal_config)
        assert estimated_resources['luts'] <= constraints['max_luts']
        assert estimated_resources['dsps'] <= constraints['max_dsps']
```

#### Hardware Kernel Generator Testing

**RTL Parser Testing**
```python
# tests/tools/hw_kernel_gen/rtl_parser/test_rtl_parser.py
class TestRTLParser:
    """Test RTL parsing and interface detection."""
    
    def test_systemverilog_parsing_accuracy(self):
        """Test parsing of complex SystemVerilog modules."""
        
        test_rtl = '''
        module complex_processor #(
            parameter DATA_WIDTH = 32,
            parameter FIFO_DEPTH = 512
        )(
            input clk,
            input rst_n,
            
            // AXI-Stream input
            input [DATA_WIDTH-1:0] s_axis_tdata,
            input s_axis_tvalid,
            output s_axis_tready,
            
            // AXI-Stream output  
            output [DATA_WIDTH-1:0] m_axis_tdata,
            output m_axis_tvalid,
            input m_axis_tready,
            
            // Control interface
            input [31:0] config_data,
            input config_valid,
            output config_ready
        );
        '''
        
        parser = RTLParser()
        ast = parser.parse(test_rtl)
        
        # Validate parsing results
        assert ast is not None
        assert parser.get_module_name(ast) == "complex_processor"
        
        # Test interface detection
        interfaces = parser.extract_interfaces(ast)
        
        # Should detect 3 interfaces: input stream, output stream, control
        assert len(interfaces) == 3
        
        # Validate AXI-Stream detection
        axis_interfaces = [iface for iface in interfaces if iface.protocol == "AXI_STREAM"]
        assert len(axis_interfaces) == 2  # input and output
        
        # Validate control interface detection
        control_interfaces = [iface for iface in interfaces if iface.protocol == "CONTROL"]
        assert len(control_interfaces) == 1
    
    def test_pragma_extraction(self):
        """Test extraction of dataflow pragmas from RTL."""
        
        rtl_with_pragmas = '''
        module test_module (
            (* dataflow interface_type="INPUT" qDim=768 tDim=64 sDim=8 *)
            input [255:0] data_input,
            
            (* dataflow interface_type="OUTPUT" qDim=768 tDim=64 sDim=8 *)
            output [255:0] data_output
        );
        '''
        
        parser = RTLParser()
        ast = parser.parse(rtl_with_pragmas)
        pragmas = parser.extract_pragmas(ast)
        
        # Validate pragma extraction
        assert len(pragmas) == 2
        assert pragmas['data_input']['interface_type'] == 'INPUT'
        assert pragmas['data_input']['qDim'] == 768
        assert pragmas['data_output']['interface_type'] == 'OUTPUT'
```

**Generator Testing**
```python
# tests/tools/hw_kernel_gen/generators/test_enhanced_generators.py
class TestHWCustomOpGenerator:
    """Test HWCustomOp generation capabilities."""
    
    @pytest.fixture
    def sample_dataflow_model(self):
        """Create sample dataflow model for testing."""
        return DataflowModel(
            interfaces=[
                DataflowInterface("input", "INPUT", qDim=256, tDim=32, sDim=4),
                DataflowInterface("output", "OUTPUT", qDim=256, tDim=32, sDim=4)
            ],
            operation_type="test_operation"
        )
    
    def test_hwcustomop_generation_completeness(self, sample_dataflow_model):
        """Test that generated HWCustomOp contains all required methods."""
        
        generator = HWCustomOpGenerator(template_manager, config)
        
        context = {
            'class_name': 'TestOperation',
            'dataflow_model': sample_dataflow_model,
            'kernel_name': 'test_op'
        }
        
        generated_artifact = generator.generate(context)
        
        # Validate generated content
        assert generated_artifact.artifact_type == "hwcustomop"
        
        generated_code = generated_artifact.content
        
        # Check for required base class
        assert "from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp" in generated_code
        assert "class TestOperation(AutoHWCustomOp):" in generated_code
        
        # Check for required method signatures
        required_methods = [
            "def __init__(self, onnx_node, **kwargs):",
            "def get_input_datatype(self, ind: int = 0):",
            "def get_output_datatype(self, ind: int = 0):",
            "def bram_estimation(self):",
            "def lut_estimation(self):",
            "def dsp_estimation(self, fpgapart: str):",
            "def get_exp_cycles(self):"
        ]
        
        for method in required_methods:
            assert method in generated_code, f"Missing required method: {method}"
    
    def test_generated_code_syntax_validation(self, sample_dataflow_model):
        """Test that generated code has valid Python syntax."""
        import ast
        
        generator = HWCustomOpGenerator(template_manager, config)
        
        context = {
            'class_name': 'SyntaxTestOperation',
            'dataflow_model': sample_dataflow_model,
            'kernel_name': 'syntax_test'
        }
        
        generated_artifact = generator.generate(context)
        
        # Validate Python syntax
        try:
            ast.parse(generated_artifact.content)
        except SyntaxError as e:
            pytest.fail(f"Generated code has invalid syntax: {e}")
```

### Integration Testing Strategy

#### End-to-End Pipeline Testing

**BERT Compilation Testing**
```python
# tests/integration/test_bert_compilation.py
class TestBERTCompilationPipeline:
    """End-to-end testing of BERT compilation workflow."""
    
    @pytest.mark.slow
    def test_complete_bert_compilation(self, tmp_path):
        """Test complete BERT model compilation pipeline."""
        
        # Setup test environment
        model_path = "test_models/bert_tiny.onnx"
        output_dir = tmp_path / "bert_output"
        
        compilation_args = {
            'target_fps': 100,
            'resource_budget': 'moderate',
            'precision': 'INT8'
        }
        
        # Run compilation
        result = forge('bert', model_path, compilation_args)
        
        # Validate compilation success
        assert result.success, f"Compilation failed: {result.error_message}"
        
        # Validate output artifacts
        expected_artifacts = [
            'model_wrapper.py',
            'driver_files/',
            'performance_report.json',
            'resource_utilization.json'
        ]
        
        for artifact in expected_artifacts:
            artifact_path = output_dir / artifact
            assert artifact_path.exists(), f"Missing expected artifact: {artifact}"
        
        # Validate performance metrics
        perf_report = result.performance_metrics
        assert perf_report['estimated_fps'] >= compilation_args['target_fps'] * 0.8
        assert perf_report['resource_utilization'] <= 0.9  # Within resource budget
    
    def test_bert_configuration_variants(self):
        """Test BERT compilation with different configurations."""
        
        test_configurations = [
            {'layers': 3, 'heads': 4, 'hidden': 256, 'intermediate': 1024},
            {'layers': 6, 'heads': 8, 'hidden': 512, 'intermediate': 2048},
            {'layers': 12, 'heads': 12, 'hidden': 768, 'intermediate': 3072}
        ]
        
        for config in test_configurations:
            # Generate test model for configuration
            model_path = self._generate_bert_model(config)
            
            # Compile with configuration
            result = forge('bert', model_path, {
                'validate_only': True,  # Fast validation mode
                'configuration': config
            })
            
            assert result.success, f"Failed for config {config}: {result.error_message}"
            
            # Validate configuration-specific results
            assert result.model_info['num_layers'] == config['layers']
            assert result.model_info['num_attention_heads'] == config['heads']
```

**Hardware Kernel Generation Testing**
```python
# tests/integration/test_kernel_generation_pipeline.py
class TestKernelGenerationPipeline:
    """Test complete kernel generation workflow."""
    
    def test_thresholding_kernel_generation(self, tmp_path):
        """Test generation of thresholding kernel components."""
        
        # Input files
        rtl_file = "examples/thresholding/thresholding_axi.sv"
        metadata_file = "examples/thresholding/dummy_compiler_data.py"
        
        # Create HKG instance
        hkg = HardwareKernelGenerator(
            rtl_file_path=rtl_file,
            compiler_data_path=metadata_file,
            output_dir=str(tmp_path)
        )
        
        # Run complete generation pipeline
        generated_artifacts = hkg.run()
        
        # Validate generated artifacts
        expected_artifacts = [
            'hwcustomop', 'rtlbackend', 'test_suite', 
            'documentation', 'rtl_template'
        ]
        
        for artifact_type in expected_artifacts:
            assert artifact_type in generated_artifacts
            
            artifact_path = generated_artifacts[artifact_type]
            assert artifact_path.exists()
            assert artifact_path.stat().st_size > 0
        
        # Validate generated code quality
        hwcustomop_file = generated_artifacts['hwcustomop']
        with open(hwcustomop_file, 'r') as f:
            hwcustomop_content = f.read()
        
        # Check for proper imports and class structure
        assert "from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp" in hwcustomop_content
        assert "class AutoThresholdingAxi(AutoHWCustomOp):" in hwcustomop_content
        
        # Validate Python syntax
        import ast
        try:
            ast.parse(hwcustomop_content)
        except SyntaxError as e:
            pytest.fail(f"Generated HWCustomOp has invalid syntax: {e}")
    
    def test_custom_operation_generation(self, tmp_path):
        """Test generation for custom operation types."""
        
        custom_rtl = '''
        module custom_conv2d #(
            parameter INPUT_WIDTH = 8,
            parameter OUTPUT_WIDTH = 8,
            parameter KERNEL_SIZE = 3
        )(
            input clk, rst_n,
            
            (* dataflow interface_type="INPUT" qDim=224 tDim=28 sDim=4 *)
            input [INPUT_WIDTH-1:0] feature_input,
            
            (* dataflow interface_type="WEIGHT" qDim=9 tDim=9 sDim=1 *)
            input [INPUT_WIDTH-1:0] weight_input,
            
            (* dataflow interface_type="OUTPUT" qDim=224 tDim=28 sDim=4 *)
            output [OUTPUT_WIDTH-1:0] feature_output
        );
        // Module implementation...
        endmodule
        '''
        
        # Write custom RTL to temporary file
        rtl_file = tmp_path / "custom_conv2d.sv"
        rtl_file.write_text(custom_rtl)
        
        # Create minimal metadata file
        metadata_content = '''
        operation_type = "convolution"
        input_precision = 8
        output_precision = 8
        kernel_size = [3, 3]
        '''
        metadata_file = tmp_path / "conv2d_metadata.py"
        metadata_file.write_text(metadata_content)
        
        # Generate components
        hkg = HardwareKernelGenerator(
            rtl_file_path=str(rtl_file),
            compiler_data_path=str(metadata_file),
            output_dir=str(tmp_path / "generated")
        )
        
        artifacts = hkg.run()
        
        # Validate convolution-specific generation
        hwcustomop_file = artifacts['hwcustomop']
        with open(hwcustomop_file, 'r') as f:
            content = f.read()
        
        # Check for convolution-specific adaptations
        assert "convolution" in content.lower()
        assert "weight_input" in content
        assert "feature_input" in content
```

## Validation Procedures

### Functional Validation

#### Automated Functional Testing

**Dataflow Interface Validation**
```python
# Comprehensive validation suite for dataflow interfaces
class DataflowValidationSuite:
    """Comprehensive validation of dataflow interface behavior."""
    
    def validate_interface_mathematical_relationships(self, interface):
        """Validate mathematical consistency of interface dimensions."""
        
        validation_results = []
        
        # Test dimensional constraints
        if interface.sDim > interface.tDim:
            validation_results.append(ValidationError(
                "sDim exceeds tDim",
                severity="ERROR",
                suggestion="Reduce sDim or increase tDim"
            ))
        
        if interface.tDim > interface.qDim:
            validation_results.append(ValidationError(
                "tDim exceeds qDim", 
                severity="ERROR",
                suggestion="Reduce tDim or increase qDim"
            ))
        
        # Test performance characteristics
        parallelism = interface.calculate_stream_parallelism()
        if parallelism > 64:  # Hardware limitation
            validation_results.append(ValidationWarning(
                f"High parallelism ({parallelism}) may exceed hardware capabilities",
                suggestion="Consider reducing sDim for better resource utilization"
            ))
        
        # Test memory bandwidth requirements
        bandwidth = interface.calculate_memory_bandwidth()
        if bandwidth > 100:  # GB/s threshold
            validation_results.append(ValidationWarning(
                f"High memory bandwidth requirement ({bandwidth} GB/s)",
                suggestion="Consider reducing data width or parallelism"
            ))
        
        return ValidationResult(
            success=not any(r.severity == "ERROR" for r in validation_results),
            results=validation_results
        )
```

#### Performance Validation

**Resource Estimation Validation**
```python
class ResourceEstimationValidator:
    """Validate accuracy of resource estimation algorithms."""
    
    def __init__(self):
        self.known_benchmarks = self._load_benchmark_data()
    
    def validate_estimation_accuracy(self, dataflow_model, actual_results=None):
        """Validate resource estimation against known benchmarks or actual results."""
        
        estimated_resources = dataflow_model.estimate_resources()
        
        if actual_results:
            # Compare against actual implementation results
            accuracy_metrics = self._calculate_accuracy_metrics(
                estimated_resources, actual_results
            )
            
            validation_results = []
            
            for resource_type in ['luts', 'dsps', 'bram']:
                accuracy = accuracy_metrics[resource_type]
                
                if accuracy < 0.8:  # Less than 80% accuracy
                    validation_results.append(ValidationWarning(
                        f"Low estimation accuracy for {resource_type}: {accuracy:.2%}",
                        suggestion=f"Review {resource_type} estimation model"
                    ))
            
            return ValidationResult(
                success=all(accuracy_metrics[rt] >= 0.7 for rt in ['luts', 'dsps', 'bram']),
                accuracy_metrics=accuracy_metrics,
                results=validation_results
            )
        
        else:
            # Compare against benchmark patterns
            return self._validate_against_benchmarks(estimated_resources)
```

### Quality Assurance

#### Code Generation Validation

**Template Testing Framework**
```python
class TemplateValidationFramework:
    """Comprehensive validation of template generation quality."""
    
    def validate_generated_code_quality(self, generated_artifact):
        """Validate quality of generated code artifacts."""
        
        validation_results = []
        
        # Syntax validation
        syntax_result = self._validate_python_syntax(generated_artifact.content)
        if not syntax_result.success:
            validation_results.extend(syntax_result.errors)
        
        # Import validation
        import_result = self._validate_imports(generated_artifact.content)
        if not import_result.success:
            validation_results.extend(import_result.errors)
        
        # Code structure validation
        structure_result = self._validate_code_structure(generated_artifact.content)
        if not structure_result.success:
            validation_results.extend(structure_result.errors)
        
        # Performance characteristics validation
        performance_result = self._validate_performance_characteristics(generated_artifact)
        if not performance_result.success:
            validation_results.extend(performance_result.warnings)
        
        return ValidationResult(
            success=not any(r.severity == "ERROR" for r in validation_results),
            results=validation_results
        )
    
    def _validate_python_syntax(self, code_content):
        """Validate Python syntax of generated code."""
        import ast
        
        try:
            ast.parse(code_content)
            return ValidationResult(success=True)
        except SyntaxError as e:
            return ValidationResult(
                success=False,
                errors=[ValidationError(
                    f"Syntax error: {e.msg} at line {e.lineno}",
                    severity="ERROR"
                )]
            )
    
    def _validate_imports(self, code_content):
        """Validate that all imports are available and correct."""
        
        import_errors = []
        
        # Extract import statements
        import_lines = [line.strip() for line in code_content.split('\n') 
                       if line.strip().startswith(('import ', 'from '))]
        
        for import_line in import_lines:
            try:
                # Test import validity (simplified)
                module_name = self._extract_module_name(import_line)
                if not self._is_module_available(module_name):
                    import_errors.append(ValidationError(
                        f"Module not available: {module_name}",
                        severity="ERROR",
                        suggestion=f"Ensure {module_name} is installed or available"
                    ))
            except Exception as e:
                import_errors.append(ValidationError(
                    f"Invalid import statement: {import_line}",
                    severity="ERROR"
                ))
        
        return ValidationResult(
            success=len(import_errors) == 0,
            errors=import_errors
        )
```

#### End-to-End Pipeline Verification

**Complete System Validation**
```python
class SystemValidationPipeline:
    """End-to-end validation of complete system functionality."""
    
    def run_complete_validation(self, test_scenarios):
        """Run comprehensive system validation across multiple scenarios."""
        
        validation_report = {
            'scenarios_tested': len(test_scenarios),
            'scenarios_passed': 0,
            'scenarios_failed': 0,
            'performance_metrics': {},
            'detailed_results': []
        }
        
        for scenario in test_scenarios:
            scenario_result = self._validate_scenario(scenario)
            
            validation_report['detailed_results'].append({
                'scenario': scenario.name,
                'result': scenario_result,
                'execution_time': scenario_result.execution_time,
                'memory_usage': scenario_result.memory_usage
            })
            
            if scenario_result.success:
                validation_report['scenarios_passed'] += 1
            else:
                validation_report['scenarios_failed'] += 1
        
        # Calculate overall system health
        success_rate = validation_report['scenarios_passed'] / validation_report['scenarios_tested']
        validation_report['overall_health'] = success_rate
        
        return validation_report
    
    def _validate_scenario(self, scenario):
        """Validate individual test scenario."""
        
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            # Execute scenario
            if scenario.type == 'bert_compilation':
                result = self._validate_bert_compilation(scenario)
            elif scenario.type == 'kernel_generation':
                result = self._validate_kernel_generation(scenario)
            elif scenario.type == 'custom_operation':
                result = self._validate_custom_operation(scenario)
            else:
                raise ValueError(f"Unknown scenario type: {scenario.type}")
            
            execution_time = time.time() - start_time
            memory_peak = self._get_memory_usage() - memory_before
            
            return ScenarioResult(
                success=result.success,
                execution_time=execution_time,
                memory_usage=memory_peak,
                performance_metrics=result.performance_metrics,
                artifacts=result.artifacts
            )
            
        except Exception as e:
            return ScenarioResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
                memory_usage=self._get_memory_usage() - memory_before
            )
```

## Test Execution Strategies

### Continuous Integration Testing

**Automated Test Pipeline**
```yaml
# CI test execution strategy
test_pipeline:
  stages:
    - fast_tests:       # < 2 minutes
        - unit_tests
        - syntax_validation
        - import_validation
        
    - integration_tests: # 2-10 minutes  
        - component_integration
        - api_integration
        - template_generation
        
    - system_tests:     # 10-30 minutes
        - end_to_end_pipelines  
        - performance_validation
        - resource_estimation
        
    - acceptance_tests: # 30+ minutes
        - full_bert_compilation
        - custom_operation_workflows
        - regression_testing

  parallel_execution:
    unit_tests: 4_workers
    integration_tests: 2_workers  
    system_tests: 1_worker
```

**Performance Regression Testing**
```python
class PerformanceRegressionTester:
    """Monitor performance regressions across system updates."""
    
    def __init__(self):
        self.baseline_metrics = self._load_baseline_metrics()
        self.performance_thresholds = {
            'compilation_time': 1.2,      # 20% slowdown threshold
            'memory_usage': 1.3,          # 30% increase threshold  
            'resource_efficiency': 0.9    # 10% efficiency decrease threshold
        }
    
    def test_performance_regression(self, current_metrics):
        """Test for performance regressions against baseline."""
        
        regression_results = []
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric_name]
                threshold = self.performance_thresholds.get(metric_name, 1.1)
                
                ratio = current_value / baseline_value
                
                if ratio > threshold:
                    regression_results.append(PerformanceRegression(
                        metric=metric_name,
                        baseline=baseline_value,
                        current=current_value,
                        ratio=ratio,
                        threshold=threshold,
                        severity="WARNING" if ratio < threshold * 1.2 else "ERROR"
                    ))
        
        return RegressionTestResult(
            regressions_detected=len(regression_results),
            regressions=regression_results,
            overall_performance_delta=self._calculate_overall_delta(current_metrics)
        )
```

This comprehensive testing and validation framework ensures that Brainsmith-2 maintains high quality, performance, and reliability across all system components and workflows, providing stakeholders with confidence in the platform's production readiness.