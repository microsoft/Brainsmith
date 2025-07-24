"""
Postprocessing pipeline for Phase 3.

This module implements the shared postprocessing pipeline used by all backends
to analyze and enhance build results using the plugin registry system.
"""

import json
import os
from typing import Dict

from brainsmith.legacy.phase2.data_structures import BuildConfig
from brainsmith.core.plugins.registry import get_registry
from .data_structures import BuildResult


class PostprocessingPipeline:
    """
    Shared postprocessing pipeline for all backends using plugin registry.
    """
    
    def analyze(self, config: BuildConfig, result: BuildResult):
        """
        Execute postprocessing transforms from 'post_proc' stage.

        Args:
            config: Build configuration with transform stages
            result: Build result to analyze and enhance
        """
        # Create postprocessing output directory
        postprocess_dir = os.path.join(config.output_dir, "postprocessing")
        os.makedirs(postprocess_dir, exist_ok=True)
        
        # Get transforms from 'post_proc' stage
        registry = get_registry()
        post_proc_transforms = config.transforms_by_stage.get('post_proc', [])
        
        if post_proc_transforms:
            print(f"Applying {len(post_proc_transforms)} post_proc transforms")
            
            # Apply transforms using QONNX ModelWrapper
            try:
                from qonnx.core.modelwrapper import ModelWrapper
                
                # Load model from build result or final model path
                model_path = result.artifacts.get('final_model') or config.model_path
                if model_path and os.path.exists(model_path):
                    model = ModelWrapper(model_path)
                    
                    for i, transform_name in enumerate(post_proc_transforms):
                        print(f"Applying post_proc transform {i+1}/{len(post_proc_transforms)}: {transform_name}")
                        
                        # Get transform from registry
                        transform_class = registry.get_transform(transform_name)
                        if transform_class:
                            # Apply transform
                            model = transform_class().apply(model)
                            
                            # Save analysis result
                            analysis_path = os.path.join(postprocess_dir, f"{transform_name}_analysis.onnx")
                            model.save(analysis_path)
                            result.artifacts[f"{transform_name}_analysis"] = analysis_path
                        else:
                            print(f"Warning: Transform '{transform_name}' not found in registry")
                            self._create_placeholder_analysis(transform_name, postprocess_dir, result)
                else:
                    print("Warning: No model available for postprocessing, creating placeholder analyses")
                    for transform_name in post_proc_transforms:
                        self._create_placeholder_analysis(transform_name, postprocess_dir, result)
                        
            except ImportError:
                print("Warning: QONNX not available, creating placeholder analyses")
                for transform_name in post_proc_transforms:
                    self._create_placeholder_analysis(transform_name, postprocess_dir, result)
        else:
            print("No post_proc transforms specified")
    
    def _create_placeholder_analysis(self, transform_name: str, output_dir: str, result: BuildResult):
        """
        Create placeholder analysis when transform cannot be applied.
        """
        analysis_file = os.path.join(output_dir, f"{transform_name}_analysis.json")
        analysis_data = {
            "transform_name": transform_name,
            "config_id": result.config_id,
            "placeholder": True,
            "message": f"Placeholder analysis for {transform_name}"
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        # Add to artifacts
        result.artifacts[f"{transform_name}_analysis"] = analysis_file