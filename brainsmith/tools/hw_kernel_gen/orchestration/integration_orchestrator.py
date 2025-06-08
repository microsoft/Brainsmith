"""
Integration Orchestrator for Hardware Kernel Generator.

This module provides seamless integration coordination between all Week 1, Week 2,
and Week 3 components, ensuring proper data flow, validation, and result aggregation.
"""

import time
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable, Set
from enum import Enum
import logging

from ..enhanced_config import PipelineConfig, GeneratorType
from ..enhanced_data_structures import RTLModule, ParsedRTLData
from ..enhanced_generator_base import GenerationResult, GeneratedArtifact
from ..enhanced_template_context import EnhancedTemplateContextBuilder
from ..enhanced_template_manager import EnhancedTemplateManager
from ..errors import BrainsmithError, IntegrationError

# Week 2 imports
from ..analysis.analysis_integration import AnalysisOrchestrator, AnalysisResults
from ..analysis.enhanced_interface_analyzer import InterfaceAnalysisResult
from ..analysis.enhanced_pragma_processor import PragmaProcessingResult

# Week 3 imports
from .generator_factory import GeneratorFactory, GeneratorConfiguration, GeneratorCapability
from .pipeline_orchestrator import PipelineOrchestrator, PipelineContext, StageResult
from .generation_workflow import WorkflowEngine, WorkflowDefinition, WorkflowContext, WorkflowResult
from .generator_management import GeneratorManager

# Check for dataflow availability
try:
    from ...dataflow.core.dataflow_model import DataflowModel
    from ...dataflow.core.auto_hw_custom_op import AutoHWCustomOp
    from ...dataflow.core.auto_rtl_backend import AutoRTLBackend
    DATAFLOW_AVAILABLE = True
except ImportError:
    DATAFLOW_AVAILABLE = False


class IntegrationMode(Enum):
    """Integration modes for orchestrator."""
    ANALYSIS_ONLY = "analysis_only"
    GENERATION_ONLY = "generation_only"
    FULL_PIPELINE = "full_pipeline"
    WORKFLOW_BASED = "workflow_based"
    CUSTOM = "custom"


class ValidationLevel(Enum):
    """Validation levels for integration."""
    NONE = "none"
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    STRICT = "strict"


@dataclass
class IntegrationConfiguration:
    """Configuration for integration orchestrator."""
    mode: IntegrationMode = IntegrationMode.FULL_PIPELINE
    validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE
    enable_caching: bool = True
    enable_parallel_processing: bool = True
    enable_error_recovery: bool = True
    enable_performance_monitoring: bool = True
    enable_dataflow_integration: bool = True
    enable_legacy_compatibility: bool = False
    timeout: Optional[float] = None
    max_retries: int = 2
    
    # Component-specific configurations
    analysis_config: Dict[str, Any] = field(default_factory=dict)
    generation_config: Dict[str, Any] = field(default_factory=dict)
    workflow_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationResult:
    """Result of integrated generation process."""
    rtl_module: RTLModule
    success: bool = False
    execution_time: float = 0.0
    
    # Analysis results
    analysis_results: Optional[AnalysisResults] = None
    
    # Generation results
    generation_results: Dict[str, GenerationResult] = field(default_factory=dict)
    
    # Workflow results
    workflow_results: Optional[WorkflowResult] = None
    
    # Pipeline results
    pipeline_results: Dict[str, StageResult] = field(default_factory=dict)
    
    # Aggregated outputs
    all_artifacts: List[GeneratedArtifact] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Performance metrics
    analysis_time: float = 0.0
    generation_time: float = 0.0
    validation_time: float = 0.0
    integration_time: float = 0.0
    
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    def finalize(self) -> None:
        """Finalize integration result."""
        self.end_time = time.time()
        if self.start_time:
            self.execution_time = self.end_time - self.start_time
        
        # Aggregate all artifacts
        self.all_artifacts.clear()
        
        if self.analysis_results:
            # Add analysis artifacts if any
            pass
        
        for gen_result in self.generation_results.values():
            self.all_artifacts.extend(gen_result.artifacts)
        
        if self.workflow_results:
            self.all_artifacts.extend(self.workflow_results.global_artifacts)
        
        # Determine overall success
        self.success = (
            len(self.errors) == 0 and
            (self.analysis_results is None or self.analysis_results.success) and
            all(result.success for result in self.generation_results.values())
        )


class ComponentIntegrator:
    """Integrates individual components into cohesive workflows."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._integration_stats = {
            "integrations": 0,
            "successful_integrations": 0,
            "failed_integrations": 0,
            "total_integration_time": 0.0
        }
    
    def integrate_analysis_and_generation(
        self,
        analysis_results: AnalysisResults,
        generator_factory: GeneratorFactory,
        template_manager: EnhancedTemplateManager
    ) -> Dict[str, GenerationResult]:
        """Integrate analysis results with generation process."""
        start_time = time.time()
        generation_results = {}
        
        try:
            # Create template context from analysis
            context_builder = EnhancedTemplateContextBuilder(self.config)
            
            # Generate for each required generator type
            if self.config.generator_type in [GeneratorType.AUTO_HW_CUSTOM_OP, GeneratorType.HW_CUSTOM_OP]:
                hw_result = self._generate_hw_custom_op(
                    analysis_results, generator_factory, template_manager, context_builder
                )
                if hw_result:
                    generation_results["hw_custom_op"] = hw_result
            
            if self.config.generator_type in [GeneratorType.AUTO_RTL_BACKEND, GeneratorType.RTL_BACKEND]:
                rtl_result = self._generate_rtl_backend(
                    analysis_results, generator_factory, template_manager, context_builder
                )
                if rtl_result:
                    generation_results["rtl_backend"] = rtl_result
            
            self._integration_stats["successful_integrations"] += 1
            
        except Exception as e:
            self._integration_stats["failed_integrations"] += 1
            logging.error(f"Analysis-generation integration failed: {e}")
            raise IntegrationError(f"Failed to integrate analysis and generation: {e}")
        
        finally:
            self._integration_stats["integrations"] += 1
            self._integration_stats["total_integration_time"] += time.time() - start_time
        
        return generation_results
    
    def _generate_hw_custom_op(
        self,
        analysis_results: AnalysisResults,
        generator_factory: GeneratorFactory,
        template_manager: EnhancedTemplateManager,
        context_builder: EnhancedTemplateContextBuilder
    ) -> Optional[GenerationResult]:
        """Generate HW Custom Op using analysis results."""
        try:
            # Create generator configuration
            generator_config = GeneratorConfiguration(
                generator_type=GeneratorType.AUTO_HW_CUSTOM_OP,
                config=self.config,
                capabilities_required={
                    GeneratorCapability.HW_CUSTOM_OP,
                    GeneratorCapability.DATAFLOW_INTEGRATION
                }
            )
            
            # Get generator
            generator = generator_factory.create_generator(generator_config, analysis_results.rtl_module)
            
            # Build context
            context = context_builder.build_hwcustomop_context(
                analysis_results.rtl_module,
                self.config,
                analysis_results
            )
            
            # Generate
            inputs = {
                "rtl_module": analysis_results.rtl_module,
                "analysis_results": analysis_results,
                "template_context": context,
                "config": self.config
            }
            
            return generator.generate(inputs)
            
        except Exception as e:
            logging.error(f"HW Custom Op generation failed: {e}")
            return None
    
    def _generate_rtl_backend(
        self,
        analysis_results: AnalysisResults,
        generator_factory: GeneratorFactory,
        template_manager: EnhancedTemplateManager,
        context_builder: EnhancedTemplateContextBuilder
    ) -> Optional[GenerationResult]:
        """Generate RTL Backend using analysis results."""
        try:
            # Create generator configuration
            generator_config = GeneratorConfiguration(
                generator_type=GeneratorType.AUTO_RTL_BACKEND,
                config=self.config,
                capabilities_required={
                    GeneratorCapability.RTL_BACKEND,
                    GeneratorCapability.DATAFLOW_INTEGRATION
                }
            )
            
            # Get generator
            generator = generator_factory.create_generator(generator_config, analysis_results.rtl_module)
            
            # Build context
            context = context_builder.build_rtlbackend_context(
                analysis_results.rtl_module,
                self.config,
                analysis_results
            )
            
            # Generate
            inputs = {
                "rtl_module": analysis_results.rtl_module,
                "analysis_results": analysis_results,
                "template_context": context,
                "config": self.config
            }
            
            return generator.generate(inputs)
            
        except Exception as e:
            logging.error(f"RTL Backend generation failed: {e}")
            return None
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        stats = self._integration_stats.copy()
        
        if stats["integrations"] > 0:
            stats["success_rate"] = (
                stats["successful_integrations"] / stats["integrations"]
            )
            stats["average_integration_time"] = (
                stats["total_integration_time"] / stats["integrations"]
            )
        else:
            stats["success_rate"] = 0.0
            stats["average_integration_time"] = 0.0
        
        return stats


class ResultsAggregator:
    """Aggregates results from different components."""
    
    def __init__(self):
        self._aggregation_stats = {
            "aggregations": 0,
            "total_artifacts_processed": 0,
            "total_aggregation_time": 0.0
        }
    
    def aggregate_results(
        self,
        analysis_results: Optional[AnalysisResults] = None,
        generation_results: Dict[str, GenerationResult] = None,
        workflow_results: Optional[WorkflowResult] = None,
        pipeline_results: Dict[str, StageResult] = None
    ) -> IntegrationResult:
        """Aggregate results from different components."""
        start_time = time.time()
        
        try:
            # Create integration result
            rtl_module = None
            if analysis_results:
                rtl_module = analysis_results.rtl_module
            elif generation_results:
                for result in generation_results.values():
                    if hasattr(result, 'rtl_module'):
                        rtl_module = result.rtl_module
                        break
            
            integration_result = IntegrationResult(
                rtl_module=rtl_module,
                analysis_results=analysis_results,
                generation_results=generation_results or {},
                workflow_results=workflow_results,
                pipeline_results=pipeline_results or {}
            )
            
            # Aggregate metadata
            self._aggregate_metadata(integration_result)
            
            # Aggregate artifacts
            self._aggregate_artifacts(integration_result)
            
            # Aggregate errors and warnings
            self._aggregate_errors_and_warnings(integration_result)
            
            # Calculate performance metrics
            self._calculate_performance_metrics(integration_result)
            
            # Finalize
            integration_result.finalize()
            
            self._aggregation_stats["aggregations"] += 1
            self._aggregation_stats["total_artifacts_processed"] += len(integration_result.all_artifacts)
            
            return integration_result
            
        finally:
            self._aggregation_stats["total_aggregation_time"] += time.time() - start_time
    
    def _aggregate_metadata(self, result: IntegrationResult) -> None:
        """Aggregate metadata from all sources."""
        if result.analysis_results:
            result.metadata["analysis"] = {
                "interface_count": len(result.analysis_results.interface_results),
                "pragma_count": (
                    result.analysis_results.pragma_results.pragma_count
                    if result.analysis_results.pragma_results else 0
                ),
                "analysis_time": result.analysis_results.total_analysis_time
            }
        
        if result.generation_results:
            result.metadata["generation"] = {
                "generator_count": len(result.generation_results),
                "generators": list(result.generation_results.keys())
            }
        
        if result.workflow_results:
            result.metadata["workflow"] = {
                "workflow_name": result.workflow_results.workflow_name,
                "step_count": len(result.workflow_results.step_results),
                "workflow_time": result.workflow_results.execution_time
            }
    
    def _aggregate_artifacts(self, result: IntegrationResult) -> None:
        """Aggregate artifacts from all sources."""
        # This is handled in IntegrationResult.finalize()
        pass
    
    def _aggregate_errors_and_warnings(self, result: IntegrationResult) -> None:
        """Aggregate errors and warnings from all sources."""
        if result.analysis_results:
            result.errors.extend(result.analysis_results.errors)
            result.warnings.extend(result.analysis_results.warnings)
        
        for gen_result in result.generation_results.values():
            if hasattr(gen_result, 'errors'):
                result.errors.extend(gen_result.errors)
            if hasattr(gen_result, 'warnings'):
                result.warnings.extend(gen_result.warnings)
        
        if result.workflow_results:
            result.errors.extend(result.workflow_results.errors)
            result.warnings.extend(result.workflow_results.warnings)
    
    def _calculate_performance_metrics(self, result: IntegrationResult) -> None:
        """Calculate performance metrics."""
        if result.analysis_results:
            result.analysis_time = result.analysis_results.total_analysis_time
        
        if result.generation_results:
            result.generation_time = sum(
                getattr(gen_result, 'execution_time', 0.0)
                for gen_result in result.generation_results.values()
            )
        
        if result.workflow_results:
            result.integration_time = result.workflow_results.execution_time
    
    def get_aggregation_statistics(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        stats = self._aggregation_stats.copy()
        
        if stats["aggregations"] > 0:
            stats["average_artifacts_per_aggregation"] = (
                stats["total_artifacts_processed"] / stats["aggregations"]
            )
            stats["average_aggregation_time"] = (
                stats["total_aggregation_time"] / stats["aggregations"]
            )
        else:
            stats["average_artifacts_per_aggregation"] = 0.0
            stats["average_aggregation_time"] = 0.0
        
        return stats


class ValidationCoordinator:
    """Coordinates validation across all components."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE):
        self.validation_level = validation_level
        self._validation_stats = {
            "validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "total_validation_time": 0.0
        }
    
    def validate_integration_result(self, result: IntegrationResult) -> Dict[str, Any]:
        """Validate complete integration result."""
        start_time = time.time()
        
        try:
            validation_result = {
                "success": True,
                "errors": [],
                "warnings": [],
                "checks_performed": [],
                "validation_level": self.validation_level.value
            }
            
            # Perform validation based on level
            if self.validation_level in [ValidationLevel.BASIC, ValidationLevel.COMPREHENSIVE, ValidationLevel.STRICT]:
                self._validate_basic_requirements(result, validation_result)
            
            if self.validation_level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.STRICT]:
                self._validate_comprehensive_requirements(result, validation_result)
            
            if self.validation_level == ValidationLevel.STRICT:
                self._validate_strict_requirements(result, validation_result)
            
            # Determine overall success
            validation_result["success"] = len(validation_result["errors"]) == 0
            
            if validation_result["success"]:
                self._validation_stats["successful_validations"] += 1
            else:
                self._validation_stats["failed_validations"] += 1
            
            return validation_result
            
        finally:
            self._validation_stats["validations"] += 1
            self._validation_stats["total_validation_time"] += time.time() - start_time
    
    def _validate_basic_requirements(self, result: IntegrationResult, validation_result: Dict[str, Any]) -> None:
        """Validate basic requirements."""
        validation_result["checks_performed"].append("basic_requirements")
        
        # Check RTL module exists
        if not result.rtl_module:
            validation_result["errors"].append("No RTL module provided")
        
        # Check for any results
        if (not result.analysis_results and 
            not result.generation_results and 
            not result.workflow_results):
            validation_result["errors"].append("No results generated")
        
        # Check for critical errors
        critical_errors = [error for error in result.errors if "critical" in error.lower()]
        if critical_errors:
            validation_result["errors"].extend(critical_errors)
    
    def _validate_comprehensive_requirements(self, result: IntegrationResult, validation_result: Dict[str, Any]) -> None:
        """Validate comprehensive requirements."""
        validation_result["checks_performed"].append("comprehensive_requirements")
        
        # Validate analysis results if present
        if result.analysis_results:
            if not result.analysis_results.success:
                validation_result["warnings"].append("Analysis was not fully successful")
            
            if len(result.analysis_results.interface_results) == 0:
                validation_result["warnings"].append("No interfaces detected in analysis")
        
        # Validate generation results
        if result.generation_results:
            for gen_name, gen_result in result.generation_results.items():
                if not gen_result.success:
                    validation_result["errors"].append(f"Generation failed for {gen_name}")
                
                if len(gen_result.artifacts) == 0:
                    validation_result["warnings"].append(f"No artifacts generated for {gen_name}")
        
        # Check artifact consistency
        if len(result.all_artifacts) == 0:
            validation_result["warnings"].append("No artifacts generated")
    
    def _validate_strict_requirements(self, result: IntegrationResult, validation_result: Dict[str, Any]) -> None:
        """Validate strict requirements."""
        validation_result["checks_performed"].append("strict_requirements")
        
        # Strict validation: no warnings allowed
        if result.warnings:
            validation_result["errors"].append(f"Warnings not allowed in strict mode: {result.warnings}")
        
        # Require specific artifacts
        if result.generation_results:
            required_artifacts = ["hw_custom_op.py", "rtl_backend.py"]
            generated_files = [artifact.file_path for artifact in result.all_artifacts]
            
            for required in required_artifacts:
                if not any(required in file_path for file_path in generated_files):
                    validation_result["errors"].append(f"Required artifact missing: {required}")
        
        # Performance requirements
        if result.execution_time > 60.0:  # 1 minute max
            validation_result["errors"].append(f"Execution time too long: {result.execution_time}s")
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        stats = self._validation_stats.copy()
        
        if stats["validations"] > 0:
            stats["success_rate"] = (
                stats["successful_validations"] / stats["validations"]
            )
            stats["average_validation_time"] = (
                stats["total_validation_time"] / stats["validations"]
            )
        else:
            stats["success_rate"] = 0.0
            stats["average_validation_time"] = 0.0
        
        return stats


class IntegrationOrchestrator:
    """Main orchestrator for integrating all components."""
    
    def __init__(
        self,
        config: PipelineConfig,
        integration_config: Optional[IntegrationConfiguration] = None,
        generator_factory: Optional[GeneratorFactory] = None
    ):
        self.config = config
        self.integration_config = integration_config or IntegrationConfiguration()
        
        # Initialize components
        self.generator_factory = generator_factory or GeneratorFactory(config)
        self.template_manager = EnhancedTemplateManager(config.template)
        self.generator_manager = GeneratorManager(config, self.generator_factory)
        
        # Week 2 components
        self.analysis_orchestrator = AnalysisOrchestrator(config)
        
        # Week 3 integration components
        self.component_integrator = ComponentIntegrator(config)
        self.results_aggregator = ResultsAggregator()
        self.validation_coordinator = ValidationCoordinator(
            self.integration_config.validation_level
        )
        
        # Optional components
        self.pipeline_orchestrator: Optional[PipelineOrchestrator] = None
        self.workflow_engine: Optional[WorkflowEngine] = None
        
        # Statistics
        self._orchestration_stats = {
            "orchestrations": 0,
            "successful_orchestrations": 0,
            "failed_orchestrations": 0,
            "total_orchestration_time": 0.0
        }
    
    def initialize_pipeline_orchestrator(self) -> PipelineOrchestrator:
        """Initialize pipeline orchestrator if needed."""
        if not self.pipeline_orchestrator:
            from .pipeline_orchestrator import create_pipeline_orchestrator
            self.pipeline_orchestrator = create_pipeline_orchestrator(
                self.config,
                self.generator_factory
            )
        return self.pipeline_orchestrator
    
    def initialize_workflow_engine(self) -> WorkflowEngine:
        """Initialize workflow engine if needed."""
        if not self.workflow_engine:
            from .generation_workflow import create_workflow_engine
            self.workflow_engine = create_workflow_engine(self.config)
        return self.workflow_engine
    
    async def orchestrate_complete_generation(
        self,
        rtl_module: RTLModule,
        pragma_sources: Optional[Union[List[str], str]] = None,
        workflow_definition: Optional[WorkflowDefinition] = None
    ) -> IntegrationResult:
        """Orchestrate complete generation process."""
        start_time = time.time()
        
        try:
            self._orchestration_stats["orchestrations"] += 1
            
            if self.integration_config.mode == IntegrationMode.ANALYSIS_ONLY:
                return await self._orchestrate_analysis_only(rtl_module, pragma_sources)
            elif self.integration_config.mode == IntegrationMode.GENERATION_ONLY:
                return await self._orchestrate_generation_only(rtl_module)
            elif self.integration_config.mode == IntegrationMode.WORKFLOW_BASED:
                return await self._orchestrate_workflow_based(rtl_module, workflow_definition)
            else:  # FULL_PIPELINE
                return await self._orchestrate_full_pipeline(rtl_module, pragma_sources)
            
        except Exception as e:
            self._orchestration_stats["failed_orchestrations"] += 1
            raise IntegrationError(f"Orchestration failed: {e}")
        
        finally:
            self._orchestration_stats["total_orchestration_time"] += time.time() - start_time
    
    async def _orchestrate_analysis_only(
        self,
        rtl_module: RTLModule,
        pragma_sources: Optional[Union[List[str], str]]
    ) -> IntegrationResult:
        """Orchestrate analysis-only workflow."""
        # Perform analysis
        analysis_results = self.analysis_orchestrator.analyze_rtl_module(
            rtl_module,
            pragma_sources,
            enable_caching=self.integration_config.enable_caching
        )
        
        # Create integration result
        result = self.results_aggregator.aggregate_results(
            analysis_results=analysis_results
        )
        
        # Validate
        if self.integration_config.validation_level != ValidationLevel.NONE:
            validation_result = self.validation_coordinator.validate_integration_result(result)
            result.metadata["validation"] = validation_result
        
        self._orchestration_stats["successful_orchestrations"] += 1
        return result
    
    async def _orchestrate_generation_only(self, rtl_module: RTLModule) -> IntegrationResult:
        """Orchestrate generation-only workflow."""
        # Create minimal analysis for generation
        analysis_results = AnalysisResults(rtl_module=rtl_module)
        analysis_results.success = True
        analysis_results.finalize()
        
        # Perform generation
        generation_results = self.component_integrator.integrate_analysis_and_generation(
            analysis_results,
            self.generator_factory,
            self.template_manager
        )
        
        # Create integration result
        result = self.results_aggregator.aggregate_results(
            analysis_results=analysis_results,
            generation_results=generation_results
        )
        
        # Validate
        if self.integration_config.validation_level != ValidationLevel.NONE:
            validation_result = self.validation_coordinator.validate_integration_result(result)
            result.metadata["validation"] = validation_result
        
        self._orchestration_stats["successful_orchestrations"] += 1
        return result
    
    async def _orchestrate_full_pipeline(
        self,
        rtl_module: RTLModule,
        pragma_sources: Optional[Union[List[str], str]]
    ) -> IntegrationResult:
        """Orchestrate full pipeline workflow."""
        # Step 1: Analysis
        analysis_results = self.analysis_orchestrator.analyze_rtl_module(
            rtl_module,
            pragma_sources,
            enable_caching=self.integration_config.enable_caching
        )
        
        # Step 2: Generation
        generation_results = self.component_integrator.integrate_analysis_and_generation(
            analysis_results,
            self.generator_factory,
            self.template_manager
        )
        
        # Step 3: Aggregation
        result = self.results_aggregator.aggregate_results(
            analysis_results=analysis_results,
            generation_results=generation_results
        )
        
        # Step 4: Validation
        if self.integration_config.validation_level != ValidationLevel.NONE:
            validation_result = self.validation_coordinator.validate_integration_result(result)
            result.metadata["validation"] = validation_result
            
            # Check if validation failed and error recovery is enabled
            if not validation_result["success"] and self.integration_config.enable_error_recovery:
                # Attempt error recovery (simplified)
                logging.warning(f"Validation failed, attempting recovery: {validation_result['errors']}")
        
        self._orchestration_stats["successful_orchestrations"] += 1
        return result
    
    async def _orchestrate_workflow_based(
        self,
        rtl_module: RTLModule,
        workflow_definition: Optional[WorkflowDefinition]
    ) -> IntegrationResult:
        """Orchestrate workflow-based generation."""
        if not workflow_definition:
            raise IntegrationError("Workflow definition required for workflow-based orchestration")
        
        # Initialize workflow engine
        workflow_engine = self.initialize_workflow_engine()
        
        # Create workflow context
        workflow_context = WorkflowContext(
            rtl_module=rtl_module,
            config=self.config,
            generator_factory=self.generator_factory,
            pipeline_orchestrator=self.pipeline_orchestrator
        )
        
        # Execute workflow
        workflow_results = workflow_engine.execute_workflow(workflow_definition, workflow_context)
        
        # Create integration result
        result = self.results_aggregator.aggregate_results(
            workflow_results=workflow_results
        )
        
        # Validate
        if self.integration_config.validation_level != ValidationLevel.NONE:
            validation_result = self.validation_coordinator.validate_integration_result(result)
            result.metadata["validation"] = validation_result
        
        self._orchestration_stats["successful_orchestrations"] += 1
        return result
    
    def get_orchestration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive orchestration statistics."""
        stats = self._orchestration_stats.copy()
        
        # Add component statistics
        stats["component_integration"] = self.component_integrator.get_integration_statistics()
        stats["results_aggregation"] = self.results_aggregator.get_aggregation_statistics()
        stats["validation_coordination"] = self.validation_coordinator.get_validation_statistics()
        
        # Add factory and manager statistics
        stats["generator_factory"] = self.generator_factory.get_factory_statistics()
        stats["generator_manager"] = self.generator_manager.get_manager_statistics()
        
        # Add analysis statistics
        stats["analysis_orchestrator"] = self.analysis_orchestrator.get_orchestration_statistics()
        
        # Calculate derived statistics
        if stats["orchestrations"] > 0:
            stats["success_rate"] = (
                stats["successful_orchestrations"] / stats["orchestrations"]
            )
            stats["average_orchestration_time"] = (
                stats["total_orchestration_time"] / stats["orchestrations"]
            )
        else:
            stats["success_rate"] = 0.0
            stats["average_orchestration_time"] = 0.0
        
        return stats


# Factory functions and convenience methods

def create_integration_orchestrator(
    config: PipelineConfig,
    integration_config: Optional[IntegrationConfiguration] = None
) -> IntegrationOrchestrator:
    """Create integration orchestrator with configuration."""
    return IntegrationOrchestrator(config, integration_config)


async def run_integrated_generation(
    orchestrator: IntegrationOrchestrator,
    rtl_module: RTLModule,
    pragma_sources: Optional[Union[List[str], str]] = None,
    workflow_definition: Optional[WorkflowDefinition] = None
) -> IntegrationResult:
    """Run integrated generation using orchestrator."""
    return await orchestrator.orchestrate_complete_generation(
        rtl_module,
        pragma_sources,
        workflow_definition
    )


def create_integration_configuration(
    mode: IntegrationMode = IntegrationMode.FULL_PIPELINE,
    validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE,
    **kwargs
) -> IntegrationConfiguration:
    """Create integration configuration with defaults."""
    return IntegrationConfiguration(
        mode=mode,
        validation_level=validation_level,
        **kwargs
    )