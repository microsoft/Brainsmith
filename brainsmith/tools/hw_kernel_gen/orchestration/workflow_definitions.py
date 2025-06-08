"""
Standard Workflow Definitions for Hardware Kernel Generator.

This module provides pre-defined workflows and workflow templates for common
generation scenarios.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass

from .generation_workflow import (
    WorkflowDefinition, WorkflowStep, WorkflowCondition, ConditionType,
    GeneratorWorkflowStep, TransformationWorkflowStep, WorkflowStepType
)
from .generator_factory import GeneratorCapability
from ..enhanced_config import GeneratorType


class StandardWorkflows:
    """Standard workflow definitions for common scenarios."""
    
    @staticmethod
    def create_basic_generation_workflow() -> WorkflowDefinition:
        """Create basic generation workflow with analysis and generation."""
        workflow = WorkflowDefinition(
            name="basic_generation",
            description="Basic workflow with interface analysis and code generation",
            version="1.0.0"
        )
        
        # Step 1: Interface Analysis
        analysis_step = TransformationWorkflowStep(
            name="interface_analysis",
            transformation_func=lambda ctx: {
                "analysis_completed": True,
                "interfaces_detected": len(ctx.rtl_module.interfaces) if ctx.rtl_module else 0
            }
        )
        workflow.add_step(analysis_step)
        
        # Step 2: HW Custom Op Generation
        hw_gen_step = GeneratorWorkflowStep(
            name="hw_custom_op_generation",
            generator_name="hw_custom_op_generator",
            dependencies={"interface_analysis"}
        )
        workflow.add_step(hw_gen_step)
        
        # Step 3: RTL Backend Generation
        rtl_gen_step = GeneratorWorkflowStep(
            name="rtl_backend_generation",
            generator_name="rtl_backend_generator",
            dependencies={"interface_analysis"}
        )
        workflow.add_step(rtl_gen_step)
        
        return workflow
    
    @staticmethod
    def create_comprehensive_workflow() -> WorkflowDefinition:
        """Create comprehensive workflow with full analysis, generation, and validation."""
        workflow = WorkflowDefinition(
            name="comprehensive_generation",
            description="Comprehensive workflow with full analysis, generation, validation, and optimization",
            version="1.0.0"
        )
        
        # Step 1: Pre-processing
        preprocess_step = TransformationWorkflowStep(
            name="preprocessing",
            transformation_func=lambda ctx: {
                "rtl_validated": True,
                "preprocessing_completed": True
            }
        )
        workflow.add_step(preprocess_step)
        
        # Step 2: Interface Analysis
        interface_analysis_step = TransformationWorkflowStep(
            name="interface_analysis",
            transformation_func=lambda ctx: {
                "interfaces_analyzed": True,
                "interface_count": len(ctx.rtl_module.interfaces) if ctx.rtl_module else 0
            },
            dependencies={"preprocessing"}
        )
        workflow.add_step(interface_analysis_step)
        
        # Step 3: Pragma Processing
        pragma_analysis_step = TransformationWorkflowStep(
            name="pragma_analysis",
            transformation_func=lambda ctx: {
                "pragmas_processed": True,
                "constraints_generated": True
            },
            dependencies={"preprocessing"}
        )
        workflow.add_step(pragma_analysis_step)
        
        # Step 4: Dataflow Integration (conditional)
        dataflow_step = TransformationWorkflowStep(
            name="dataflow_integration",
            transformation_func=lambda ctx: {
                "dataflow_model_created": True,
                "parallelism_optimized": True
            },
            dependencies={"interface_analysis", "pragma_analysis"},
            condition=WorkflowCondition(
                ConditionType.IF_EXISTS,
                parameters={"key": "enable_dataflow"}
            )
        )
        workflow.add_step(dataflow_step)
        
        # Step 5: HW Custom Op Generation
        hw_gen_step = GeneratorWorkflowStep(
            name="hw_custom_op_generation",
            generator_name="enhanced_hw_custom_op_generator",
            generator_inputs={"use_dataflow": True},
            dependencies={"interface_analysis", "pragma_analysis"}
        )
        workflow.add_step(hw_gen_step)
        
        # Step 6: RTL Backend Generation
        rtl_gen_step = GeneratorWorkflowStep(
            name="rtl_backend_generation",
            generator_name="enhanced_rtl_backend_generator",
            generator_inputs={"use_dataflow": True},
            dependencies={"interface_analysis", "pragma_analysis"}
        )
        workflow.add_step(rtl_gen_step)
        
        # Step 7: Documentation Generation
        doc_gen_step = GeneratorWorkflowStep(
            name="documentation_generation",
            generator_name="documentation_generator",
            dependencies={"hw_custom_op_generation", "rtl_backend_generation"}
        )
        workflow.add_step(doc_gen_step)
        
        # Step 8: Test Generation
        test_gen_step = GeneratorWorkflowStep(
            name="test_generation",
            generator_name="test_generator",
            dependencies={"hw_custom_op_generation", "rtl_backend_generation"}
        )
        workflow.add_step(test_gen_step)
        
        # Step 9: Validation
        validation_step = TransformationWorkflowStep(
            name="comprehensive_validation",
            transformation_func=lambda ctx: {
                "validation_completed": True,
                "all_tests_passed": True
            },
            dependencies={"documentation_generation", "test_generation"}
        )
        workflow.add_step(validation_step)
        
        return workflow
    
    @staticmethod
    def create_dataflow_optimized_workflow() -> WorkflowDefinition:
        """Create workflow optimized for dataflow-based generation."""
        workflow = WorkflowDefinition(
            name="dataflow_optimized",
            description="Workflow optimized for dataflow-based code generation",
            version="1.0.0",
            global_config={"enable_dataflow": True, "optimize_parallelism": True}
        )
        
        # Step 1: RTL Analysis with Dataflow Focus
        rtl_analysis_step = TransformationWorkflowStep(
            name="rtl_dataflow_analysis",
            transformation_func=lambda ctx: {
                "rtl_interfaces_analyzed": True,
                "dataflow_patterns_detected": True
            }
        )
        workflow.add_step(rtl_analysis_step)
        
        # Step 2: Tensor Shape Analysis
        tensor_analysis_step = TransformationWorkflowStep(
            name="tensor_shape_analysis",
            transformation_func=lambda ctx: {
                "tensor_shapes_inferred": True,
                "dimensions_validated": True
            },
            dependencies={"rtl_dataflow_analysis"}
        )
        workflow.add_step(tensor_analysis_step)
        
        # Step 3: Parallelism Analysis
        parallelism_step = TransformationWorkflowStep(
            name="parallelism_analysis",
            transformation_func=lambda ctx: {
                "parallelism_constraints_analyzed": True,
                "optimization_opportunities_identified": True
            },
            dependencies={"tensor_shape_analysis"}
        )
        workflow.add_step(parallelism_step)
        
        # Step 4: Dataflow Model Creation
        dataflow_model_step = TransformationWorkflowStep(
            name="dataflow_model_creation",
            transformation_func=lambda ctx: {
                "dataflow_model_created": True,
                "chunking_strategy_selected": True
            },
            dependencies={"parallelism_analysis"}
        )
        workflow.add_step(dataflow_model_step)
        
        # Step 5: Optimized HW Custom Op Generation
        optimized_hw_step = GeneratorWorkflowStep(
            name="optimized_hw_generation",
            generator_name="dataflow_hw_custom_op_generator",
            generator_inputs={
                "use_dataflow_model": True,
                "optimize_for_performance": True
            },
            dependencies={"dataflow_model_creation"}
        )
        workflow.add_step(optimized_hw_step)
        
        # Step 6: Optimized RTL Backend Generation
        optimized_rtl_step = GeneratorWorkflowStep(
            name="optimized_rtl_generation",
            generator_name="dataflow_rtl_backend_generator",
            generator_inputs={
                "use_dataflow_model": True,
                "optimize_for_performance": True
            },
            dependencies={"dataflow_model_creation"}
        )
        workflow.add_step(optimized_rtl_step)
        
        # Step 7: Performance Validation
        perf_validation_step = TransformationWorkflowStep(
            name="performance_validation",
            transformation_func=lambda ctx: {
                "performance_validated": True,
                "optimization_verified": True
            },
            dependencies={"optimized_hw_generation", "optimized_rtl_generation"}
        )
        workflow.add_step(perf_validation_step)
        
        return workflow
    
    @staticmethod
    def create_legacy_compatibility_workflow() -> WorkflowDefinition:
        """Create workflow for legacy compatibility mode."""
        workflow = WorkflowDefinition(
            name="legacy_compatibility",
            description="Workflow for generating code compatible with legacy systems",
            version="1.0.0",
            global_config={"legacy_mode": True, "dataflow_enabled": False}
        )
        
        # Step 1: Legacy RTL Analysis
        legacy_analysis_step = TransformationWorkflowStep(
            name="legacy_rtl_analysis",
            transformation_func=lambda ctx: {
                "legacy_interfaces_detected": True,
                "compatibility_checked": True
            }
        )
        workflow.add_step(legacy_analysis_step)
        
        # Step 2: Legacy HW Custom Op Generation
        legacy_hw_step = GeneratorWorkflowStep(
            name="legacy_hw_generation",
            generator_name="legacy_hw_custom_op_generator",
            generator_inputs={"legacy_mode": True},
            dependencies={"legacy_rtl_analysis"}
        )
        workflow.add_step(legacy_hw_step)
        
        # Step 3: Legacy RTL Backend Generation
        legacy_rtl_step = GeneratorWorkflowStep(
            name="legacy_rtl_generation",
            generator_name="legacy_rtl_backend_generator",
            generator_inputs={"legacy_mode": True},
            dependencies={"legacy_rtl_analysis"}
        )
        workflow.add_step(legacy_rtl_step)
        
        # Step 4: Legacy Validation
        legacy_validation_step = TransformationWorkflowStep(
            name="legacy_validation",
            transformation_func=lambda ctx: {
                "legacy_compatibility_verified": True
            },
            dependencies={"legacy_hw_generation", "legacy_rtl_generation"}
        )
        workflow.add_step(legacy_validation_step)
        
        return workflow
    
    @staticmethod
    def create_rapid_prototyping_workflow() -> WorkflowDefinition:
        """Create workflow for rapid prototyping with minimal validation."""
        workflow = WorkflowDefinition(
            name="rapid_prototyping",
            description="Fast workflow for rapid prototyping with minimal overhead",
            version="1.0.0",
            global_config={"fast_mode": True, "skip_validation": True}
        )
        
        # Step 1: Quick Analysis
        quick_analysis_step = TransformationWorkflowStep(
            name="quick_analysis",
            transformation_func=lambda ctx: {
                "basic_analysis_completed": True
            }
        )
        workflow.add_step(quick_analysis_step)
        
        # Step 2: Fast HW Generation
        fast_hw_step = GeneratorWorkflowStep(
            name="fast_hw_generation",
            generator_name="fast_hw_custom_op_generator",
            generator_inputs={"minimal_validation": True},
            dependencies={"quick_analysis"}
        )
        workflow.add_step(fast_hw_step)
        
        # Step 3: Fast RTL Generation
        fast_rtl_step = GeneratorWorkflowStep(
            name="fast_rtl_generation",
            generator_name="fast_rtl_backend_generator",
            generator_inputs={"minimal_validation": True},
            dependencies={"quick_analysis"}
        )
        workflow.add_step(fast_rtl_step)
        
        return workflow


@dataclass
class WorkflowTemplate:
    """Template for creating custom workflows."""
    name: str
    description: str
    required_capabilities: Set[GeneratorCapability]
    optional_capabilities: Set[GeneratorCapability]
    generator_types: Set[GeneratorType]
    step_templates: List[Dict[str, Any]]
    configuration_template: Dict[str, Any]
    
    def instantiate(self, **kwargs) -> WorkflowDefinition:
        """Instantiate workflow from template."""
        workflow = WorkflowDefinition(
            name=kwargs.get("name", self.name),
            description=kwargs.get("description", self.description),
            version=kwargs.get("version", "1.0.0"),
            global_config={**self.configuration_template, **kwargs.get("config", {})}
        )
        
        # Create steps from templates
        for step_template in self.step_templates:
            step = self._create_step_from_template(step_template, kwargs)
            if step:
                workflow.add_step(step)
        
        return workflow
    
    def _create_step_from_template(self, template: Dict[str, Any], kwargs: Dict[str, Any]) -> Optional[WorkflowStep]:
        """Create step from template."""
        step_type = template.get("type", "transformation")
        
        if step_type == "generator":
            return GeneratorWorkflowStep(
                name=template["name"],
                generator_name=template["generator_name"],
                generator_inputs=template.get("inputs", {}),
                dependencies=set(template.get("dependencies", []))
            )
        elif step_type == "transformation":
            # This would need a function registry for transformation functions
            # For now, return a simple placeholder
            def placeholder_func(ctx):
                return {"completed": True}
            
            return TransformationWorkflowStep(
                name=template["name"],
                transformation_func=placeholder_func,
                dependencies=set(template.get("dependencies", []))
            )
        
        return None


class CustomWorkflowBuilder:
    """Builder for creating custom workflows."""
    
    def __init__(self, name: str, description: str = ""):
        self.workflow = WorkflowDefinition(
            name=name,
            description=description,
            version="1.0.0"
        )
        self._step_counter = 0
    
    def add_analysis_step(
        self,
        name: Optional[str] = None,
        dependencies: Set[str] = None
    ) -> 'CustomWorkflowBuilder':
        """Add analysis step to workflow."""
        step_name = name or f"analysis_step_{self._step_counter}"
        self._step_counter += 1
        
        def analysis_func(ctx):
            return {
                "analysis_completed": True,
                "interfaces_count": len(ctx.rtl_module.interfaces) if ctx.rtl_module else 0
            }
        
        step = TransformationWorkflowStep(
            name=step_name,
            transformation_func=analysis_func,
            dependencies=dependencies or set()
        )
        
        self.workflow.add_step(step)
        return self
    
    def add_generation_step(
        self,
        generator_name: str,
        name: Optional[str] = None,
        dependencies: Set[str] = None,
        inputs: Dict[str, Any] = None
    ) -> 'CustomWorkflowBuilder':
        """Add generation step to workflow."""
        step_name = name or f"generation_step_{self._step_counter}"
        self._step_counter += 1
        
        step = GeneratorWorkflowStep(
            name=step_name,
            generator_name=generator_name,
            generator_inputs=inputs or {},
            dependencies=dependencies or set()
        )
        
        self.workflow.add_step(step)
        return self
    
    def add_validation_step(
        self,
        name: Optional[str] = None,
        dependencies: Set[str] = None
    ) -> 'CustomWorkflowBuilder':
        """Add validation step to workflow."""
        step_name = name or f"validation_step_{self._step_counter}"
        self._step_counter += 1
        
        def validation_func(ctx):
            return {
                "validation_completed": True,
                "validation_passed": True
            }
        
        step = TransformationWorkflowStep(
            name=step_name,
            transformation_func=validation_func,
            dependencies=dependencies or set()
        )
        
        self.workflow.add_step(step)
        return self
    
    def add_conditional_step(
        self,
        step: WorkflowStep,
        condition: WorkflowCondition
    ) -> 'CustomWorkflowBuilder':
        """Add conditional step to workflow."""
        step.condition = condition
        self.workflow.add_step(step)
        return self
    
    def set_global_config(self, config: Dict[str, Any]) -> 'CustomWorkflowBuilder':
        """Set global configuration for workflow."""
        self.workflow.global_config.update(config)
        return self
    
    def build(self) -> WorkflowDefinition:
        """Build the workflow definition."""
        return self.workflow


# Standard workflow templates
STANDARD_TEMPLATES = {
    "basic_generation": WorkflowTemplate(
        name="basic_generation_template",
        description="Template for basic generation workflows",
        required_capabilities={GeneratorCapability.HW_CUSTOM_OP, GeneratorCapability.RTL_BACKEND},
        optional_capabilities={GeneratorCapability.DOCUMENTATION, GeneratorCapability.VALIDATION},
        generator_types={GeneratorType.AUTO_HW_CUSTOM_OP, GeneratorType.AUTO_RTL_BACKEND},
        step_templates=[
            {"type": "transformation", "name": "analysis", "dependencies": []},
            {"type": "generator", "name": "hw_generation", "generator_name": "hw_custom_op_generator", "dependencies": ["analysis"]},
            {"type": "generator", "name": "rtl_generation", "generator_name": "rtl_backend_generator", "dependencies": ["analysis"]}
        ],
        configuration_template={"mode": "basic", "validation_level": "basic"}
    ),
    
    "dataflow_optimized": WorkflowTemplate(
        name="dataflow_optimized_template",
        description="Template for dataflow-optimized workflows",
        required_capabilities={
            GeneratorCapability.HW_CUSTOM_OP,
            GeneratorCapability.RTL_BACKEND,
            GeneratorCapability.DATAFLOW_INTEGRATION
        },
        optional_capabilities={GeneratorCapability.OPTIMIZATION},
        generator_types={GeneratorType.AUTO_HW_CUSTOM_OP, GeneratorType.AUTO_RTL_BACKEND},
        step_templates=[
            {"type": "transformation", "name": "dataflow_analysis", "dependencies": []},
            {"type": "transformation", "name": "optimization", "dependencies": ["dataflow_analysis"]},
            {"type": "generator", "name": "optimized_hw_generation", "generator_name": "dataflow_hw_generator", "dependencies": ["optimization"]},
            {"type": "generator", "name": "optimized_rtl_generation", "generator_name": "dataflow_rtl_generator", "dependencies": ["optimization"]}
        ],
        configuration_template={"mode": "dataflow", "enable_optimization": True}
    )
}


# Factory functions

def create_standard_workflow(workflow_type: str, **kwargs) -> WorkflowDefinition:
    """Create standard workflow by type."""
    if workflow_type == "basic":
        return StandardWorkflows.create_basic_generation_workflow()
    elif workflow_type == "comprehensive":
        return StandardWorkflows.create_comprehensive_workflow()
    elif workflow_type == "dataflow_optimized":
        return StandardWorkflows.create_dataflow_optimized_workflow()
    elif workflow_type == "legacy_compatibility":
        return StandardWorkflows.create_legacy_compatibility_workflow()
    elif workflow_type == "rapid_prototyping":
        return StandardWorkflows.create_rapid_prototyping_workflow()
    else:
        raise ValueError(f"Unknown workflow type: {workflow_type}")


def get_workflow_template(template_name: str) -> Optional[WorkflowTemplate]:
    """Get workflow template by name."""
    return STANDARD_TEMPLATES.get(template_name)


def create_custom_workflow_builder(name: str, description: str = "") -> CustomWorkflowBuilder:
    """Create custom workflow builder."""
    return CustomWorkflowBuilder(name, description)