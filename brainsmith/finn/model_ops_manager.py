"""
Model Operations Manager for FINN Integration Engine.

Handles configuration of FINN model operations including:
- Supported operations validation
- Custom operations processing
- Frontend cleanup configuration
- Preprocessing steps determination
"""

import logging
from typing import Dict, Any, List

from .types import ModelOpsConfig

logger = logging.getLogger(__name__)

class ModelOpsManager:
    """Manager for FINN model operations configuration"""
    
    def __init__(self):
        self.supported_ops_registry = self._load_supported_ops()
        self.custom_ops_registry = self._load_custom_ops()
        self.cleanup_steps_registry = self._load_cleanup_steps()
    
    def configure(self, 
                 supported_ops: List[str],
                 custom_ops: Dict[str, Any] = None,
                 frontend_cleanup: List[str] = None) -> ModelOpsConfig:
        """Configure model operations for FINN"""
        
        if custom_ops is None:
            custom_ops = {}
        if frontend_cleanup is None:
            frontend_cleanup = []
        
        # Validate supported operations
        validated_ops = self._validate_supported_ops(supported_ops)
        
        # Process custom operations
        processed_custom_ops = self._process_custom_ops(custom_ops)
        
        # Configure frontend cleanup
        cleanup_steps = self._configure_cleanup_steps(frontend_cleanup)
        
        # Add preprocessing steps
        preprocessing_steps = self._determine_preprocessing_steps(validated_ops)
        
        # Define validation rules
        validation_rules = self._create_validation_rules(validated_ops, processed_custom_ops)
        
        return ModelOpsConfig(
            supported_ops=validated_ops,
            custom_ops=processed_custom_ops,
            frontend_cleanup=cleanup_steps,
            preprocessing_steps=preprocessing_steps,
            validation_rules=validation_rules
        )
    
    def _load_supported_ops(self) -> Dict[str, Any]:
        """Load registry of supported FINN operations"""
        return {
            'Conv': {
                'input_types': ['float32', 'int8'], 
                'parameters': ['kernel_size', 'stride', 'padding'],
                'constraints': {'kernel_size': [1, 2, 3, 4, 5, 7]}
            },
            'MatMul': {
                'input_types': ['float32', 'int8'], 
                'parameters': ['transpose_a', 'transpose_b'],
                'constraints': {}
            },
            'Add': {
                'input_types': ['float32', 'int8'], 
                'parameters': [],
                'constraints': {}
            },
            'Relu': {
                'input_types': ['float32', 'int8'], 
                'parameters': [],
                'constraints': {}
            },
            'MaxPool': {
                'input_types': ['float32', 'int8'], 
                'parameters': ['kernel_size', 'stride', 'padding'],
                'constraints': {'kernel_size': [2, 3, 4, 5]}
            },
            'Reshape': {
                'input_types': ['float32', 'int8'], 
                'parameters': ['shape'],
                'constraints': {}
            },
            'Transpose': {
                'input_types': ['float32', 'int8'], 
                'parameters': ['perm'],
                'constraints': {}
            },
            'Concat': {
                'input_types': ['float32', 'int8'], 
                'parameters': ['axis'],
                'constraints': {}
            }
        }
    
    def _load_custom_ops(self) -> Dict[str, Any]:
        """Load registry of custom FINN operations"""
        return {
            'Thresholding': {
                'backend': 'rtl', 
                'parameters': ['threshold', 'num_bits'],
                'implementation_path': 'finn.custom_op.fpgadataflow.thresholding'
            },
            'LookupTable': {
                'backend': 'hls', 
                'parameters': ['table_size', 'input_bits'],
                'implementation_path': 'finn.custom_op.fpgadataflow.lookup'
            },
            'StreamingFCLayer': {
                'backend': 'hls', 
                'parameters': ['pe', 'simd', 'weight_mem'],
                'implementation_path': 'finn.custom_op.fpgadataflow.fc'
            },
            'VectorVectorActivation': {
                'backend': 'rtl', 
                'parameters': ['pe', 'activation_type'],
                'implementation_path': 'finn.custom_op.fpgadataflow.vvau'
            }
        }
    
    def _load_cleanup_steps(self) -> Dict[str, Any]:
        """Load available frontend cleanup steps"""
        return {
            'RemoveUnusedNodes': {
                'description': 'Remove nodes that are not used in computation',
                'safe': True,
                'impact': 'low'
            },
            'FoldConstants': {
                'description': 'Fold constant expressions',
                'safe': True,
                'impact': 'medium'
            },
            'InlineSubgraphs': {
                'description': 'Inline small subgraphs',
                'safe': False,
                'impact': 'high'
            },
            'RemoveIdentity': {
                'description': 'Remove identity operations',
                'safe': True,
                'impact': 'low'
            },
            'SimplifyReshapes': {
                'description': 'Simplify unnecessary reshape operations',
                'safe': True,
                'impact': 'medium'
            }
        }
    
    def _validate_supported_ops(self, supported_ops: List[str]) -> List[str]:
        """Validate and filter supported operations"""
        validated = []
        for op in supported_ops:
            if op in self.supported_ops_registry:
                validated.append(op)
                logger.debug(f"Validated operation: {op}")
            else:
                logger.warning(f"Unsupported operation: {op}")
        
        if not validated:
            logger.warning("No valid operations found, using basic set")
            validated = ['Conv', 'MatMul', 'Add', 'Relu']
        
        return validated
    
    def _process_custom_ops(self, custom_ops: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate custom operations"""
        processed = {}
        
        for op_name, op_config in custom_ops.items():
            if self._validate_custom_op(op_name, op_config):
                # Merge with registry defaults if available
                if op_name in self.custom_ops_registry:
                    merged_config = self.custom_ops_registry[op_name].copy()
                    merged_config.update(op_config)
                    processed[op_name] = merged_config
                else:
                    processed[op_name] = op_config
                logger.debug(f"Processed custom operation: {op_name}")
            else:
                logger.warning(f"Invalid custom operation: {op_name}")
        
        return processed
    
    def _validate_custom_op(self, op_name: str, op_config: Dict[str, Any]) -> bool:
        """Validate custom operation configuration"""
        required_fields = ['backend']
        optional_fields = ['implementation_path', 'parameters']
        
        # Check required fields
        if not all(field in op_config for field in required_fields):
            return False
        
        # Validate backend
        valid_backends = ['hls', 'rtl', 'python']
        if op_config.get('backend') not in valid_backends:
            return False
        
        return True
    
    def _configure_cleanup_steps(self, frontend_cleanup: List[str]) -> List[str]:
        """Configure frontend cleanup steps"""
        configured = []
        
        for step in frontend_cleanup:
            if step in self.cleanup_steps_registry:
                configured.append(step)
                logger.debug(f"Added cleanup step: {step}")
            else:
                logger.warning(f"Unknown cleanup step: {step}")
        
        # Add safe default steps if none specified
        if not configured:
            safe_defaults = [
                step for step, config in self.cleanup_steps_registry.items()
                if config.get('safe', False)
            ]
            configured.extend(safe_defaults)
            logger.info(f"Using default safe cleanup steps: {configured}")
        
        return configured
    
    def _determine_preprocessing_steps(self, supported_ops: List[str]) -> List[str]:
        """Determine required preprocessing steps based on operations"""
        steps = ['ValidateModel', 'InferShapes']
        
        # Add operation-specific preprocessing
        if 'Conv' in supported_ops:
            steps.append('ConvertConv2d')
        if 'MatMul' in supported_ops:
            steps.append('ConvertMatMul')
        if 'MaxPool' in supported_ops:
            steps.append('ConvertMaxPool')
        
        # Add data type conversion if needed
        if any(op in supported_ops for op in ['Conv', 'MatMul']):
            steps.append('ConvertDataTypes')
        
        logger.debug(f"Determined preprocessing steps: {steps}")
        return steps
    
    def _create_validation_rules(self, 
                               supported_ops: List[str],
                               custom_ops: Dict[str, Any]) -> Dict[str, Any]:
        """Create validation rules for model operations"""
        rules = {
            'required_ops': supported_ops,
            'custom_ops': list(custom_ops.keys()),
            'input_constraints': {
                'data_types': ['float32', 'int8', 'int16'],
                'tensor_rank': [2, 3, 4, 5]
            },
            'model_constraints': {
                'max_nodes': 1000,
                'max_parameters': 100_000_000,
                'max_depth': 100
            },
            'operation_constraints': {}
        }
        
        # Add operation-specific constraints
        for op in supported_ops:
            if op in self.supported_ops_registry:
                op_info = self.supported_ops_registry[op]
                if 'constraints' in op_info and op_info['constraints']:
                    rules['operation_constraints'][op] = op_info['constraints']
        
        logger.debug(f"Created validation rules for {len(supported_ops)} operations")
        return rules
    
    def get_supported_operations(self) -> List[str]:
        """Get list of all supported operations"""
        return list(self.supported_ops_registry.keys())
    
    def get_custom_operations(self) -> List[str]:
        """Get list of all available custom operations"""
        return list(self.custom_ops_registry.keys())
    
    def validate_operation_config(self, op_name: str, op_config: Dict[str, Any]) -> bool:
        """Validate a specific operation configuration"""
        if op_name in self.supported_ops_registry:
            return True  # Standard ops are always valid
        elif op_name in self.custom_ops_registry:
            return self._validate_custom_op(op_name, op_config)
        else:
            return False