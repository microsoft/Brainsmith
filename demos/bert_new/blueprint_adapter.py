"""
Blueprint Adapter for BERT Demo

Dynamically updates blueprint configuration based on runtime parameters.
Allows a single unified blueprint to adapt to different model configurations.
"""

import yaml
import copy
from typing import Dict, Any, Optional
from pathlib import Path


class BlueprintAdapter:
    """Adapts blueprint configuration based on runtime parameters."""
    
    def __init__(self, blueprint_path: str):
        """Initialize with base blueprint."""
        self.blueprint_path = Path(blueprint_path)
        self.base_blueprint = self._load_blueprint()
    
    def _load_blueprint(self) -> Dict[str, Any]:
        """Load the base blueprint from YAML."""
        with open(self.blueprint_path, 'r') as f:
            return yaml.safe_load(f)
    
    def adapt_for_model_config(self, 
                              hidden_size: int,
                              num_hidden_layers: int, 
                              num_attention_heads: int,
                              intermediate_size: int,
                              sequence_length: int,
                              bitwidth: int,
                              ultra_small: bool = False,
                              target_device: str = "V80",
                              folding_config_file: Optional[str] = None,
                              target_fps: Optional[int] = None) -> Dict[str, Any]:
        """
        Adapt blueprint for specific model configuration.
        
        Args:
            hidden_size: BERT hidden dimension
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            intermediate_size: Feed-forward intermediate size
            sequence_length: Maximum sequence length
            bitwidth: Quantization bit width
            target_device: Target FPGA device
            
        Returns:
            Adapted blueprint configuration
        """
        # Create deep copy to avoid modifying original
        adapted = copy.deepcopy(self.base_blueprint)
        
        # Update model configuration
        model_config = adapted['model_configuration']
        model_config['hidden_size'] = hidden_size
        model_config['num_hidden_layers'] = num_hidden_layers
        model_config['num_attention_heads'] = num_attention_heads
        model_config['intermediate_size'] = intermediate_size
        model_config['max_position_embeddings'] = sequence_length
        model_config['bitwidth'] = bitwidth
        
        # Update expected model parameters
        expected = adapted['expected_model']
        expected['input_shape'] = [1, sequence_length]
        expected['parameters']['hidden_size'] = hidden_size
        expected['parameters']['num_attention_heads'] = num_attention_heads
        expected['parameters']['num_hidden_layers'] = num_hidden_layers
        expected['parameters']['intermediate_size'] = intermediate_size
        expected['parameters']['max_position_embeddings'] = sequence_length
        
        # Apply optimizations
        self._apply_standard_optimizations(adapted)
        
        # Update platform settings
        adapted['platform']['board'] = target_device
        
        # Ensure finn_config section exists
        if 'finn_config' not in adapted:
            adapted['finn_config'] = {}
            
        # Add folding config file to finn_config if provided
        if folding_config_file:
            # Add folding config file path
            adapted['finn_config']['folding_config_file'] = folding_config_file
            
            # When using a folding config, we should disable auto folding
            adapted['finn_config']['target_fps'] = None
        else:
            # When not using folding config, use a conservative target_fps
            # to avoid aggressive parallelization that causes shape mismatches
            if target_fps is not None:
                adapted['finn_config']['target_fps'] = target_fps
            else:
                # Use a very conservative default to avoid shape mismatches
                adapted['finn_config']['target_fps'] = 10
        
        # Update blueprint metadata
        self._update_metadata(adapted, ultra_small, hidden_size, num_hidden_layers)
        
        return adapted
    
    def _apply_standard_optimizations(self, blueprint: Dict[str, Any]) -> None:
        """Apply optimizations for standard models."""
        # Use standard configuration (defaults in blueprint)
        blueprint['expected_model']['architecture'] = "bert_standard"
        
        # Estimate size based on model dimensions
        hidden_size = blueprint['model_configuration']['hidden_size']
        num_layers = blueprint['model_configuration']['num_hidden_layers']
        estimated_params = self._estimate_parameters(hidden_size, num_layers)
        estimated_size_mb = estimated_params * 4 / (1024 * 1024)  # 4 bytes per float32 param
        
        blueprint['expected_model']['estimated_size'] = f"~{estimated_size_mb:.0f}MB"
        
        # Adjust build time estimate based on model size
        if estimated_size_mb < 20:
            blueprint['expected_model']['estimated_build_time'] = "15-30 minutes"
        elif estimated_size_mb < 100:
            blueprint['expected_model']['estimated_build_time'] = "30-60 minutes" 
        else:
            blueprint['expected_model']['estimated_build_time'] = "60-120 minutes"
    
    def _estimate_parameters(self, hidden_size: int, num_layers: int) -> int:
        """Estimate number of parameters in BERT model."""
        # Rough estimation for BERT parameter count
        vocab_size = 30522  # Standard BERT vocab
        
        # Embedding parameters
        embedding_params = vocab_size * hidden_size + 512 * hidden_size + 2 * hidden_size
        
        # Transformer layer parameters (per layer)
        attention_params = 4 * hidden_size * hidden_size + 4 * hidden_size  # Q, K, V, O projections
        ffn_params = 2 * hidden_size * hidden_size * 4 + hidden_size * 4 + hidden_size  # Two linear layers
        layer_norm_params = 2 * hidden_size * 2  # Two layer norms per layer
        layer_params = attention_params + ffn_params + layer_norm_params
        
        # Total parameters
        total_params = embedding_params + num_layers * layer_params
        
        return total_params
    
    def _update_metadata(self, blueprint: Dict[str, Any], ultra_small: bool, 
                        hidden_size: int, num_layers: int) -> None:
        """Update blueprint metadata."""
        mode = "ultra_small" if ultra_small else "standard"
        blueprint['name'] = f"bert_demo_{mode}"
        blueprint['description'] = f"BERT accelerator blueprint - {mode} mode ({hidden_size}D, {num_layers}L)"
    
    def save_adapted_blueprint(self, adapted_blueprint: Dict[str, Any], 
                              output_path: str) -> str:
        """Save adapted blueprint to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(adapted_blueprint, f, default_flow_style=False, indent=2)
        
        return str(output_path)


def create_runtime_blueprint(base_blueprint_path: str,
                           hidden_size: int,
                           num_hidden_layers: int,
                           num_attention_heads: int,
                           intermediate_size: int, 
                           sequence_length: int,
                           bitwidth: int,
                           ultra_small: bool = False,
                           target_device: str = "V80",
                           output_dir: str = "./",
                           folding_config_file: Optional[str] = None,
                           target_fps: Optional[int] = None) -> str:
    """
    Convenience function to create a runtime-adapted blueprint.
    
    Returns:
        Path to the generated blueprint file
    """
    adapter = BlueprintAdapter(base_blueprint_path)
    
    adapted = adapter.adapt_for_model_config(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        sequence_length=sequence_length,
        bitwidth=bitwidth,
        ultra_small=ultra_small,
        target_device=target_device,
        folding_config_file=folding_config_file,
        target_fps=target_fps
    )
    
    # Generate output filename based on configuration
    mode = "ultra_small" if ultra_small else "standard"
    output_filename = f"bert_demo_{mode}_{hidden_size}D_{num_hidden_layers}L.yaml"
    output_path = Path(output_dir) / output_filename
    
    return adapter.save_adapted_blueprint(adapted, output_path)