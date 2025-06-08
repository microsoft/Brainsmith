"""
Enhanced hardware compiler for Brainsmith platform.

This module provides the main compilation interface with comprehensive metrics
collection and design space exploration capabilities.
"""

import os
import time
import tempfile
from typing import Dict, Any, List, Optional, Union
import onnx

from .config import CompilerConfig, ParameterSweepConfig, DSEConfig
from .metrics import BrainsmithMetrics, MetricsCollector
from .design_space import DesignPoint, DesignSpace, create_parameter_sweep_points
from .finn_interface import FINNInterfaceLayer
from .result import BrainsmithResult, ParameterSweepResult, DSEResult
from ..blueprints import Blueprint


class HardwareCompiler:
    """Main compiler with metric collection and extensibility."""
    
    def __init__(self, config: Optional[CompilerConfig] = None):
        """Initialize compiler with configuration."""
        self.config = config or CompilerConfig()
        self.finn_interface = FINNInterfaceLayer()
        self.blueprint: Optional[Blueprint] = None
        
    def load_blueprint(self, blueprint_name: str) -> Blueprint:
        """Load blueprint and extract design space if available."""
        from ..blueprints import get_blueprint
        
        self.blueprint = get_blueprint(blueprint_name)
        return self.blueprint
    
    def compile(self, model: Union[str, onnx.ModelProto], 
                blueprint_name: str = "") -> BrainsmithResult:
        """
        Enhanced compilation with comprehensive metrics.
        
        Args:
            model: ONNX model file path or ModelProto object
            blueprint_name: Blueprint to use (overrides config)
        
        Returns:
            BrainsmithResult with comprehensive metrics
        """
        # Load model if path provided
        if isinstance(model, str):
            model = onnx.load(model)
        
        # Use blueprint from parameter or config
        blueprint_name = blueprint_name or self.config.blueprint
        if not blueprint_name:
            raise ValueError("Blueprint name must be specified")
        
        # Load blueprint
        blueprint = self.load_blueprint(blueprint_name)
        
        # Create design point from configuration
        design_point = self.config.to_design_point()
        
        # Set up output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Execute build with FINN interface
        result = self.finn_interface.execute_build(
            model=model,
            design_point=design_point,
            blueprint=blueprint,
            output_dir=self.config.output_dir
        )
        
        # Set additional result metadata
        result.blueprint_name = blueprint_name
        result.config_data = self.config.to_dict()
        
        # Save result if requested
        if self.config.export_research_data:
            result.save_result()
        
        return result
    
    def parameter_sweep(self, model: Union[str, onnx.ModelProto],
                       parameters: Dict[str, List[Any]],
                       blueprint_name: str = "",
                       sweep_config: Optional[ParameterSweepConfig] = None) -> ParameterSweepResult:
        """
        Execute parameter sweep for optimization.
        
        Args:
            model: ONNX model to compile
            parameters: Dictionary of parameter names to value lists
            blueprint_name: Blueprint to use
            sweep_config: Parameter sweep configuration
        
        Returns:
            ParameterSweepResult with all build results
        """
        # Load model if path provided
        if isinstance(model, str):
            model = onnx.load(model)
        
        # Use blueprint from parameter or config
        blueprint_name = blueprint_name or self.config.blueprint
        if not blueprint_name:
            raise ValueError("Blueprint name must be specified")
        
        # Load blueprint
        blueprint = self.load_blueprint(blueprint_name)
        
        # Set up sweep configuration
        if sweep_config is None:
            sweep_config = ParameterSweepConfig(
                parameters=parameters,
                base_output_dir=self.config.output_dir
            )
        
        # Generate design points
        design_points = create_parameter_sweep_points(parameters)
        
        # Initialize result
        sweep_result = ParameterSweepResult(
            sweep_parameters=parameters
        )
        
        start_time = time.time()
        
        print(f"Starting parameter sweep with {len(design_points)} configurations...")
        
        # Execute builds for each design point
        for i, design_point in enumerate(design_points):
            print(f"Configuration {i+1}/{len(design_points)}: {design_point.parameters}")
            
            try:
                # Set up output directory for this configuration
                config_output_dir = sweep_config.get_output_dir(i)
                os.makedirs(config_output_dir, exist_ok=True)
                
                # Execute build
                result = self.finn_interface.execute_build(
                    model=model,
                    design_point=design_point,
                    blueprint=blueprint,
                    output_dir=config_output_dir
                )
                
                # Set result metadata
                result.blueprint_name = blueprint_name
                result.config_data = self.config.to_dict()
                
                # Save individual result
                result.save_result()
                
                sweep_result.results.append(result)
                
                if result.success:
                    print(f"  ✓ Success: {result.get_summary()}")
                else:
                    print(f"  ✗ Failed: {'; '.join(result.errors)}")
                    
                    if not sweep_config.continue_on_error:
                        break
                        
            except Exception as e:
                error_result = BrainsmithResult(
                    success=False,
                    output_dir=sweep_config.get_output_dir(i),
                    design_point=design_point,
                    blueprint_name=blueprint_name
                )
                error_result.add_error(f"Build exception: {str(e)}")
                sweep_result.results.append(error_result)
                
                print(f"  ✗ Exception: {str(e)}")
                
                if not sweep_config.continue_on_error:
                    break
        
        # Finalize sweep result
        sweep_result.total_time = time.time() - start_time
        sweep_result.success_count = sum(1 for r in sweep_result.results if r.success)
        
        # Generate analysis if requested
        if sweep_config.generate_comparison_report:
            self._generate_sweep_analysis(sweep_result, sweep_config)
        
        print(f"\nParameter sweep completed:")
        print(f"  Total configurations: {len(sweep_result.results)}")
        print(f"  Successful builds: {sweep_result.success_count}")
        print(f"  Total time: {sweep_result.total_time:.1f}s")
        
        return sweep_result
    
    def design_space_exploration(self, model: Union[str, onnx.ModelProto],
                                blueprint_name: str = "",
                                dse_config: Optional[DSEConfig] = None) -> DSEResult:
        """
        Execute design space exploration.
        
        This is a placeholder for future DSE implementation. Currently provides
        basic random sampling functionality.
        
        Args:
            model: ONNX model to explore
            blueprint_name: Blueprint to use
            dse_config: DSE configuration
        
        Returns:
            DSEResult with exploration results
        """
        # Load model if path provided
        if isinstance(model, str):
            model = onnx.load(model)
        
        # Use blueprint from parameter or config
        blueprint_name = blueprint_name or self.config.blueprint
        if not blueprint_name:
            raise ValueError("Blueprint name must be specified")
        
        # Load blueprint and extract design space
        blueprint = self.load_blueprint(blueprint_name)
        design_space = self._extract_design_space_from_blueprint(blueprint)
        
        # Set up DSE configuration
        if dse_config is None:
            dse_config = DSEConfig(max_evaluations=20)
        
        # Initialize result
        dse_result = DSEResult(
            design_space_info=design_space.to_dict(),
            strategy_used=dse_config.strategy
        )
        
        start_time = time.time()
        
        print(f"Starting design space exploration with {dse_config.max_evaluations} evaluations...")
        print(f"Strategy: {dse_config.strategy}")
        
        # Simple random sampling (placeholder for future DSE algorithms)
        from .design_space import sample_design_space
        
        design_points = sample_design_space(
            design_space,
            n_samples=dse_config.max_evaluations,
            strategy="random",
            seed=dse_config.random_seed
        )
        
        # Execute builds for each design point
        for i, design_point in enumerate(design_points):
            print(f"Evaluation {i+1}/{len(design_points)}: {design_point.parameters}")
            
            try:
                # Set up output directory
                eval_output_dir = os.path.join(dse_config.base_output_dir, f"eval_{i+1:03d}")
                os.makedirs(eval_output_dir, exist_ok=True)
                
                # Execute build
                result = self.finn_interface.execute_build(
                    model=model,
                    design_point=design_point,
                    blueprint=blueprint,
                    output_dir=eval_output_dir
                )
                
                # Set result metadata
                result.blueprint_name = blueprint_name
                result.config_data = self.config.to_dict()
                
                # Save individual result
                if dse_config.save_all_results:
                    result.save_result()
                
                dse_result.results.append(result)
                
                if result.success:
                    summary = result.get_summary()
                    print(f"  ✓ Success: {summary}")
                else:
                    print(f"  ✗ Failed: {'; '.join(result.errors)}")
                
            except Exception as e:
                error_result = BrainsmithResult(
                    success=False,
                    output_dir=eval_output_dir,
                    design_point=design_point,
                    blueprint_name=blueprint_name
                )
                error_result.add_error(f"Build exception: {str(e)}")
                dse_result.results.append(error_result)
                
                print(f"  ✗ Exception: {str(e)}")
        
        # Finalize DSE result
        dse_result.exploration_time = time.time() - start_time
        
        # Basic analysis (placeholder for future sophisticated analysis)
        successful_results = dse_result.get_successful_results()
        if successful_results:
            # Find best configurations for different metrics
            if dse_config.objectives:
                for objective in dse_config.objectives:
                    best = self._find_best_result(successful_results, objective)
                    if best:
                        dse_result.best_configurations[objective] = best
        
        # Generate analysis report if requested
        if dse_config.generate_analysis_report:
            self._generate_dse_analysis(dse_result, dse_config)
        
        print(f"\nDesign space exploration completed:")
        print(f"  Total evaluations: {len(dse_result.results)}")
        print(f"  Successful evaluations: {len(successful_results)}")
        print(f"  Exploration time: {dse_result.exploration_time:.1f}s")
        
        return dse_result
    
    def _extract_design_space_from_blueprint(self, blueprint: Blueprint) -> DesignSpace:
        """Extract design space from blueprint definition."""
        # Try to get design space from blueprint YAML data
        if hasattr(blueprint, 'yaml_data') and blueprint.yaml_data:
            return DesignSpace.from_blueprint_data(blueprint.yaml_data)
        
        # Fallback: create basic design space from common parameters
        design_space = DesignSpace(name=f"{blueprint.name}_default")
        
        # Add basic parameters that most blueprints support
        from .design_space import ParameterDefinition, ParameterType
        
        design_space.add_parameter(ParameterDefinition(
            name="target_fps",
            type=ParameterType.INTEGER,
            range=(1000, 10000),
            default=3000
        ))
        
        design_space.add_parameter(ParameterDefinition(
            name="clk_period_ns", 
            type=ParameterType.CONTINUOUS,
            range=(2.0, 10.0),
            default=3.33
        ))
        
        design_space.add_parameter(ParameterDefinition(
            name="board",
            type=ParameterType.CATEGORICAL,
            values=["V80", "ZCU104", "U250"],
            default="V80"
        ))
        
        return design_space
    
    def _find_best_result(self, results: List[BrainsmithResult], 
                         metric: str) -> Optional[BrainsmithResult]:
        """Find best result according to specified metric."""
        if not results:
            return None
        
        def get_metric_value(result: BrainsmithResult) -> float:
            if not result.metrics:
                return 0.0
            
            if metric == "throughput_ops_sec":
                return result.metrics.performance.throughput_ops_sec or 0.0
            elif metric == "efficiency":
                return result.metrics.custom_metrics.get("throughput_per_lut", 0.0)
            elif metric == "power_efficiency":
                return result.metrics.custom_metrics.get("throughput_per_watt", 0.0)
            else:
                return result.metrics.custom_metrics.get(metric, 0.0)
        
        return max(results, key=get_metric_value)
    
    def _generate_sweep_analysis(self, sweep_result: ParameterSweepResult,
                                sweep_config: ParameterSweepConfig):
        """Generate analysis report for parameter sweep."""
        # Export to CSV if requested
        if sweep_config.export_csv:
            csv_path = os.path.join(sweep_config.base_output_dir, "parameter_sweep_results.csv")
            sweep_result.to_csv(csv_path)
            print(f"  Results exported to: {csv_path}")
        
        # Generate comparison report
        if sweep_config.generate_comparison_report:
            comparison_path = os.path.join(sweep_config.base_output_dir, "comparison_report.json")
            comparison_data = sweep_result.export_comparison_data()
            
            import json
            with open(comparison_path, 'w') as f:
                json.dump(comparison_data, f, indent=2, default=str)
            
            print(f"  Comparison report saved to: {comparison_path}")
    
    def _generate_dse_analysis(self, dse_result: DSEResult, dse_config: DSEConfig):
        """Generate analysis report for design space exploration."""
        analysis_dir = os.path.join(dse_config.base_output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Export comprehensive dataset
        dataset_path = os.path.join(analysis_dir, "dse_dataset.json")
        dse_result.export_research_dataset(dataset_path)
        print(f"  Research dataset exported to: {dataset_path}")
        
        # Generate coverage report
        coverage_path = os.path.join(analysis_dir, "coverage_report.json")
        coverage_data = dse_result.get_coverage_report()
        
        import json
        with open(coverage_path, 'w') as f:
            json.dump(coverage_data, f, indent=2, default=str)
        
        print(f"  Coverage report saved to: {coverage_path}")


# Legacy compatibility function
def forge(blueprint: Blueprint, model_path: str, config) -> BrainsmithResult:
    """
    Legacy forge function for backward compatibility.
    
    This function maintains compatibility with existing code while
    using the new enhanced compiler infrastructure.
    """
    # Convert legacy config to new format
    if hasattr(config, 'to_compiler_config'):
        compiler_config = config.to_compiler_config(
            blueprint=blueprint.name,
            model_path=model_path
        )
    else:
        # Handle raw config objects
        compiler_config = CompilerConfig(
            blueprint=blueprint.name,
            model_path=model_path,
            output_dir=getattr(config, 'output_dir', './build'),
            target_fps=getattr(config, 'target_fps', 3000),
            synth_clk_period_ns=getattr(config, 'synth_clk_period_ns', 3.33),
            board=getattr(config, 'board', 'V80')
        )
    
    # Create compiler and execute
    compiler = HardwareCompiler(compiler_config)
    result = compiler.compile(model_path, blueprint.name)
    
    # For legacy compatibility, raise exception on failure
    if not result.success:
        raise RuntimeError(f"Build failed: {'; '.join(result.errors)}")
    
    return result