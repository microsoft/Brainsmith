"""
Combination generator for systematic design space exploration.

This module generates all valid configurations from a design space by taking
the cartesian product of all options and filtering based on constraints.
"""

import hashlib
import os
from datetime import datetime
from typing import List, Tuple
import itertools
import logging

from .data_structures import BuildConfig
from ..phase1.data_structures import DesignSpace, SearchConstraint


logger = logging.getLogger(__name__)


class CombinationGenerator:
    """
    Generate all valid combinations from a design space.
    
    This class handles the systematic generation of BuildConfig objects
    from the design space, including cartesian product generation and
    constraint filtering.
    """
    
    def generate_all(self, design_space: DesignSpace) -> List[BuildConfig]:
        """
        Generate all valid combinations from the design space.
        
        Args:
            design_space: The design space to generate combinations from
            
        Returns:
            List of BuildConfig objects representing all valid combinations
        """
        logger.info("Generating combinations from design space")
        
        # Get all combinations from each component
        kernel_combos = design_space.hw_compiler_space.get_kernel_combinations()
        transform_combos = design_space.hw_compiler_space.get_transform_combinations_by_stage()
        preproc_combos = design_space.processing_space.get_preprocessing_combinations()
        postproc_combos = design_space.processing_space.get_postprocessing_combinations()
        
        logger.debug(f"Kernel combinations: {len(kernel_combos)}")
        logger.debug(f"Transform combinations: {len(transform_combos)}")
        logger.debug(f"Preprocessing combinations: {len(preproc_combos)}")
        logger.debug(f"Postprocessing combinations: {len(postproc_combos)}")
        
        # Calculate total combinations
        total = (
            len(kernel_combos) * 
            len(transform_combos) * 
            len(preproc_combos) * 
            len(postproc_combos)
        )
        logger.info(f"Total possible combinations: {total}")
        
        # Generate design space ID
        design_space_id = self._generate_design_space_id(design_space)
        
        # Calculate number of digits needed for padding
        num_digits = len(str(total - 1)) if total > 0 else 1
        
        # Generate all combinations
        configs = []
        combo_index = 0
        
        for kernels, transforms, preprocessing, postprocessing in itertools.product(
            kernel_combos, transform_combos, preproc_combos, postproc_combos
        ):
            # Filter out empty/skipped elements
            active_kernels = [k for k in kernels if k[0]]  # Non-empty names
            
            # transforms is now a dict mapping stage -> list of transforms
            # No need to filter since that's already done in get_transform_combinations_by_stage
            
            # Filter processing steps by enabled flag
            active_preprocessing = [step for step in preprocessing if step.enabled]
            active_postprocessing = [step for step in postprocessing if step.enabled]
            
            # Generate output directory path with dynamic padding
            config_dirname = f"config_{combo_index:0{num_digits}d}"
            output_dir = os.path.join(
                design_space.global_config.working_directory,
                design_space_id,
                "builds",
                config_dirname
            )
            
            # Create BuildConfig
            config = BuildConfig(
                id=f"{design_space_id}_{config_dirname}",
                design_space_id=design_space_id,
                model_path=design_space.model_path,
                kernels=list(active_kernels),
                transforms=transforms,  # Now a dict
                preprocessing=list(active_preprocessing),
                postprocessing=list(active_postprocessing),
                build_steps=design_space.hw_compiler_space.build_steps,
                config_flags=design_space.hw_compiler_space.config_flags.copy(),
                global_config=design_space.global_config,
                output_dir=output_dir,
                timestamp=datetime.now(),
                combination_index=combo_index,
                total_combinations=total
            )
            
            # Apply constraints (pre-filtering if possible)
            if self._satisfies_constraints(config, design_space.search_config.constraints):
                configs.append(config)
            else:
                logger.debug(f"Config {config.id} filtered by constraints")
            
            combo_index += 1
        
        logger.info(f"Generated {len(configs)} valid configurations after constraint filtering")
        return configs
    
    def _generate_design_space_id(self, design_space: DesignSpace) -> str:
        """
        Generate a unique ID for the design space.
        
        Uses a hash of the model path and key configuration elements.
        """
        # Create a string representation of key elements
        id_components = [
            design_space.model_path,
            str(len(design_space.hw_compiler_space.kernels)),
            str(len(design_space.hw_compiler_space.transforms)),
            design_space.search_config.strategy.value,
            str(len(design_space.search_config.constraints)),
        ]
        
        # Generate hash
        id_string = "|".join(id_components)
        hash_digest = hashlib.sha256(id_string.encode()).hexdigest()[:8]
        
        return f"dse_{hash_digest}"
    
    def _satisfies_constraints(
        self, 
        config: BuildConfig, 
        constraints: List[SearchConstraint]
    ) -> bool:
        """
        Check if a configuration satisfies all constraints.
        
        Currently this is a placeholder that accepts all configurations.
        In a real implementation, some constraints could be checked before
        build execution (e.g., configuration-based constraints).
        
        Args:
            config: The configuration to check
            constraints: List of constraints to apply
            
        Returns:
            True if the configuration satisfies all pre-checkable constraints
        """
        # For now, we can't check metric-based constraints without building
        # In the future, we could check configuration-based constraints here
        # For example:
        # - Maximum number of kernels
        # - Required/forbidden kernel combinations
        # - Transform ordering constraints
        
        # Example: Check if we have too many kernels (hypothetical constraint)
        # if len(config.kernels) > 10:
        #     return False
        
        # For now, accept all configurations
        return True
    
    def filter_by_indices(
        self, 
        configs: List[BuildConfig], 
        indices: List[int]
    ) -> List[BuildConfig]:
        """
        Filter configurations to only include specific indices.
        
        Useful for distributed execution or sampling.
        
        Args:
            configs: List of all configurations
            indices: Indices of configurations to keep
            
        Returns:
            Filtered list of configurations
        """
        return [configs[i] for i in indices if 0 <= i < len(configs)]
    
    def filter_by_resume(
        self,
        configs: List[BuildConfig],
        last_completed_id: str
    ) -> List[BuildConfig]:
        """
        Filter configurations to resume from a specific point.
        
        Args:
            configs: List of all configurations
            last_completed_id: ID of the last successfully completed configuration
            
        Returns:
            List of configurations after the resume point
        """
        # Find the index of the last completed configuration
        resume_index = -1
        for i, config in enumerate(configs):
            if config.id == last_completed_id:
                resume_index = i
                break
        
        if resume_index == -1:
            logger.warning(f"Resume ID {last_completed_id} not found, starting from beginning")
            return configs
        
        # Return configurations after the resume point
        remaining = configs[resume_index + 1:]
        logger.info(f"Resuming from configuration {resume_index + 1}, {len(remaining)} configs remaining")
        return remaining