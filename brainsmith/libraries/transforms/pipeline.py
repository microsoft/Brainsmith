"""
Transform pipeline management.

Manages sequences of transformations and their execution order,
providing pipeline-based transform application.
"""

import uuid
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TransformStep:
    """Represents a single transform step in a pipeline."""
    name: str
    transform_type: str
    parameters: Dict[str, Any]
    enabled: bool = True
    order: int = 0


class TransformPipeline:
    """
    Manages a pipeline of transform steps.
    
    Provides ordered execution of transforms with dependency management
    and configuration from blueprint specifications.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize transform pipeline.
        
        Args:
            name: Pipeline name (generated if None)
        """
        self.id = str(uuid.uuid4())[:8]
        self.name = name or f"pipeline_{self.id}"
        self.steps = []
        self.configuration = {}
        self.logger = logging.getLogger("brainsmith.libraries.transforms.pipeline")
    
    def configure(self, config: Dict[str, Any]):
        """
        Configure pipeline from blueprint configuration.
        
        Args:
            config: Pipeline configuration
        """
        self.configuration = config
        self.steps.clear()
        
        # Build pipeline steps based on configuration
        self._build_pipeline_from_config(config)
        
        self.logger.info(f"Configured pipeline {self.name} with {len(self.steps)} steps")
    
    def _build_pipeline_from_config(self, config: Dict[str, Any]):
        """Build pipeline steps from configuration."""
        order = 0
        
        # Add folding transforms if configured
        if 'folding_factors' in config:
            folding_factors = config['folding_factors']
            if folding_factors:
                step = TransformStep(
                    name='folding_transform',
                    transform_type='folding',
                    parameters={'folding_factors': folding_factors},
                    order=order
                )
                self.steps.append(step)
                order += 1
        
        # Add memory optimization if configured
        memory_opt = config.get('memory_optimization', 'balanced')
        if memory_opt != 'none':
            step = TransformStep(
                name='memory_optimization',
                transform_type='memory',
                parameters={'optimization_level': memory_opt},
                order=order
            )
            self.steps.append(step)
            order += 1
        
        # Add pipeline insertion if depth specified
        if 'pipeline_depth' in config:
            depth = config['pipeline_depth']
            if depth > 1:
                step = TransformStep(
                    name='pipeline_insertion',
                    transform_type='pipelining',
                    parameters={'pipeline_depth': depth},
                    order=order
                )
                self.steps.append(step)
                order += 1
        
        # Add streaming if enabled
        if config.get('streaming_enabled', False):
            step = TransformStep(
                name='streaming_dataflow',
                transform_type='streaming',
                parameters={'enable_streaming': True},
                order=order
            )
            self.steps.append(step)
            order += 1
        
        # Sort steps by order
        self.steps.sort(key=lambda s: s.order)
    
    def add_step(self, name: str, transform_type: str, 
                 parameters: Dict[str, Any], order: int = None):
        """
        Add a transform step to the pipeline.
        
        Args:
            name: Step name
            transform_type: Type of transform
            parameters: Transform parameters
            order: Execution order (appended if None)
        """
        if order is None:
            order = len(self.steps)
        
        step = TransformStep(
            name=name,
            transform_type=transform_type,
            parameters=parameters,
            order=order
        )
        
        self.steps.append(step)
        self.steps.sort(key=lambda s: s.order)
        
        self.logger.debug(f"Added step {name} to pipeline {self.name}")
    
    def remove_step(self, name: str) -> bool:
        """
        Remove a step from the pipeline.
        
        Args:
            name: Step name to remove
            
        Returns:
            True if step was removed
        """
        for i, step in enumerate(self.steps):
            if step.name == name:
                del self.steps[i]
                self.logger.debug(f"Removed step {name} from pipeline {self.name}")
                return True
        
        return False
    
    def enable_step(self, name: str, enabled: bool = True):
        """Enable or disable a pipeline step."""
        for step in self.steps:
            if step.name == name:
                step.enabled = enabled
                self.logger.debug(f"{'Enabled' if enabled else 'Disabled'} step {name}")
                break
    
    def get_transform_sequence(self) -> List[str]:
        """Get the sequence of transform names."""
        return [step.name for step in self.steps if step.enabled]
    
    def get_enabled_steps(self) -> List[TransformStep]:
        """Get list of enabled steps."""
        return [step for step in self.steps if step.enabled]
    
    def get_estimated_depth(self) -> int:
        """Estimate pipeline depth based on transforms."""
        base_depth = 1
        
        for step in self.get_enabled_steps():
            if step.transform_type == 'pipelining':
                depth = step.parameters.get('pipeline_depth', 1)
                base_depth += depth - 1
            elif step.transform_type == 'folding':
                # Folding may add some pipeline stages
                base_depth += 1
        
        return base_depth
    
    def get_resource_estimate(self) -> Dict[str, Any]:
        """Estimate resource impact of pipeline."""
        estimates = {
            'lut_overhead': 1.0,
            'ff_overhead': 1.0,
            'bram_overhead': 1.0,
            'dsp_overhead': 1.0
        }
        
        for step in self.get_enabled_steps():
            if step.transform_type == 'folding':
                # Folding reduces resource usage
                folding_factors = step.parameters.get('folding_factors', [1])
                max_folding = max(folding_factors) if folding_factors else 1
                estimates['lut_overhead'] *= (1.0 / max_folding)
                estimates['dsp_overhead'] *= (1.0 / max_folding)
            
            elif step.transform_type == 'pipelining':
                # Pipelining increases FF usage
                depth = step.parameters.get('pipeline_depth', 1)
                estimates['ff_overhead'] *= (1.0 + depth * 0.1)
            
            elif step.transform_type == 'streaming':
                # Streaming increases BRAM usage
                estimates['bram_overhead'] *= 1.5
            
            elif step.transform_type == 'memory':
                # Memory optimization affects BRAM usage
                opt_level = step.parameters.get('optimization_level', 'balanced')
                multipliers = {'conservative': 1.2, 'balanced': 1.0, 'aggressive': 0.8}
                estimates['bram_overhead'] *= multipliers.get(opt_level, 1.0)
        
        return estimates
    
    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate pipeline configuration.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for conflicting transforms
        step_types = [step.transform_type for step in self.get_enabled_steps()]
        
        # Check for reasonable pipeline depth
        estimated_depth = self.get_estimated_depth()
        if estimated_depth > 20:
            errors.append(f"Pipeline depth too high: {estimated_depth}")
        
        # Check for step dependencies
        folding_steps = [s for s in self.steps if s.transform_type == 'folding']
        streaming_steps = [s for s in self.steps if s.transform_type == 'streaming']
        
        # Folding and aggressive streaming might conflict
        if folding_steps and streaming_steps:
            for folding_step in folding_steps:
                factors = folding_step.parameters.get('folding_factors', [])
                if factors and max(factors) > 8:
                    errors.append("High folding factors may conflict with streaming")
        
        return len(errors) == 0, errors
    
    def execute(self, model: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute pipeline on a model (simplified for Week 4).
        
        Args:
            model: Model to transform
            context: Execution context
            
        Returns:
            Execution results
        """
        context = context or {}
        results = {
            'input_model': model,
            'transformed_model': model,  # In real implementation, would apply transforms
            'applied_steps': [],
            'resource_estimates': self.get_resource_estimate(),
            'pipeline_depth': self.get_estimated_depth()
        }
        
        # Simulate step execution
        for step in self.get_enabled_steps():
            step_result = {
                'step_name': step.name,
                'transform_type': step.transform_type,
                'parameters': step.parameters,
                'execution_time': 0.1  # Mock execution time
            }
            
            results['applied_steps'].append(step_result)
            self.logger.debug(f"Executed step: {step.name}")
        
        self.logger.info(f"Pipeline {self.name} executed {len(results['applied_steps'])} steps")
        return results
    
    def from_config(self, config: Dict[str, Any]) -> 'TransformPipeline':
        """
        Create pipeline from configuration.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Configured pipeline (self)
        """
        self.configure(config)
        return self
    
    def to_config(self) -> Dict[str, Any]:
        """Convert pipeline to configuration dictionary."""
        config = self.configuration.copy()
        
        # Add step information
        config['steps'] = []
        for step in self.steps:
            step_config = {
                'name': step.name,
                'type': step.transform_type,
                'parameters': step.parameters,
                'enabled': step.enabled,
                'order': step.order
            }
            config['steps'].append(step_config)
        
        return config
    
    def get_id(self) -> str:
        """Get pipeline ID."""
        return self.id
    
    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline summary."""
        return {
            'id': self.id,
            'name': self.name,
            'total_steps': len(self.steps),
            'enabled_steps': len(self.get_enabled_steps()),
            'estimated_depth': self.get_estimated_depth(),
            'transform_sequence': self.get_transform_sequence(),
            'resource_estimates': self.get_resource_estimate()
        }
    
    def cleanup(self):
        """Cleanup pipeline resources."""
        self.steps.clear()
        self.configuration.clear()
        self.logger.debug(f"Cleaned up pipeline {self.name}")