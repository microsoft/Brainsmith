"""
Unified HWCustomOp generator with optional BDIM sophistication.

Based on hw_kernel_gen_simple generator pattern with enhanced template
context for advanced BDIM pragma processing when enabled.
"""

from pathlib import Path
from .base import GeneratorBase
from ..data import UnifiedHWKernel


class UnifiedHWCustomOpGenerator(GeneratorBase):
    """
    Unified HWCustomOp generator with optional BDIM sophistication.
    
    Generates HWCustomOp Python classes using the enhanced template context
    that supports both simple mode (identical to hw_kernel_gen_simple) and
    advanced mode with BDIM pragma integration.
    """
    
    def __init__(self, template_dir: Path = None):
        super().__init__('hw_custom_op_slim.py.j2', template_dir)
    
    def _get_output_filename(self, hw_kernel: UnifiedHWKernel) -> str:
        """Get output filename for HWCustomOp class."""
        return f"{hw_kernel.name.lower()}_hwcustomop.py"
    
    def _get_template_context(self, hw_kernel: UnifiedHWKernel) -> dict:
        """
        Get enhanced template context for HWCustomOp generation.
        
        Builds on base context with HWCustomOp-specific enhancements
        for both simple and advanced modes.
        """
        context = super()._get_template_context(hw_kernel)
        
        # HWCustomOp-specific context enhancements
        context.update({
            'class_name': f"{hw_kernel.class_name}HWCustomOp",
            'verification_required': hw_kernel.verification_required,
            'kernel_verifications': self._get_kernel_verifications(hw_kernel),
            'resource_estimation': self._get_resource_estimation_context(hw_kernel)
        })
        
        # Advanced BDIM context if available
        if hw_kernel.has_enhanced_bdim:
            context.update({
                'interface_metadata': self._build_enhanced_interface_metadata(hw_kernel),
                'chunking_context': self._build_chunking_context(hw_kernel),
                'dataflow_model_integration': True
            })
        else:
            context.update({
                'dataflow_model_integration': False
            })
        
        return context
    
    def _get_kernel_verifications(self, hw_kernel: UnifiedHWKernel) -> list:
        """
        Get kernel-specific verification rules.
        
        Enhanced from simple system with comprehensive BDIM-aware verifications
        following Interface-Wise Dataflow Modeling axioms.
        """
        verifications = []
        
        # Basic verifications based on kernel type (from simple system)
        if hw_kernel.kernel_type == 'threshold':
            verifications.extend(self._get_threshold_verifications())
        elif hw_kernel.kernel_type in ['matmul', 'conv']:
            verifications.extend(self._get_computation_verifications())
        elif hw_kernel.kernel_type == 'norm':
            verifications.extend(self._get_normalization_verifications())
        
        # Enhanced verifications for BDIM-aware kernels
        if hw_kernel.has_enhanced_bdim:
            verifications.extend(self._get_bdim_verifications(hw_kernel))
        
        # Interface-specific verifications
        verifications.extend(self._get_interface_verifications(hw_kernel))
        
        # Parallelism and performance verifications
        if hw_kernel.has_enhanced_bdim:
            verifications.extend(self._get_parallelism_verifications(hw_kernel))
        
        return verifications
    
    def _get_threshold_verifications(self) -> list:
        """Get threshold-specific verification rules."""
        return [
            {
                'description': 'Verify threshold parameters are within valid range',
                'code': 'assert self.get_nodeattr("threshold") > 0, "Threshold must be positive"',
                'category': 'parameter_validation'
            },
            {
                'description': 'Verify input/output bit widths are compatible',
                'code': 'assert self.get_nodeattr("input_bits") >= self.get_nodeattr("output_bits"), "Output precision cannot exceed input precision"',
                'category': 'datatype_validation'
            }
        ]
    
    def _get_computation_verifications(self) -> list:
        """Get computation-specific verification rules for matmul/conv kernels."""
        return [
            {
                'description': 'Verify input/output dimension compatibility',
                'code': 'assert len(self.get_input_dataflow_interface().block_dims) >= 2, "Need at least 2D tensors for computation kernels"',
                'category': 'dimension_validation'
            },
            {
                'description': 'Verify weight dimensions match computation requirements',
                'code': 'assert self.get_weight_interface() is not None, "Computation kernels require weight interface"',
                'category': 'interface_validation'
            },
            {
                'description': 'Verify parallelism parameters are consistent',
                'code': 'assert self.get_nodeattr("simd", 1) * self.get_nodeattr("pe", 1) <= 1024, "Total parallelism should not exceed hardware limits"',
                'category': 'parallelism_validation'
            }
        ]
    
    def _get_normalization_verifications(self) -> list:
        """Get normalization-specific verification rules."""
        return [
            {
                'description': 'Verify normalization axis is valid',
                'code': 'assert 0 <= self.get_nodeattr("axis", 1) < len(self.get_input_dataflow_interface().tensor_dims), "Normalization axis must be within tensor dimensions"',
                'category': 'parameter_validation'
            },
            {
                'description': 'Verify epsilon parameter for numerical stability',
                'code': 'assert self.get_nodeattr("epsilon", 1e-5) > 0, "Epsilon must be positive for numerical stability"',
                'category': 'numerical_validation'
            }
        ]
    
    def _get_bdim_verifications(self, hw_kernel: UnifiedHWKernel) -> list:
        """
        Get BDIM-specific verification rules following Interface-Wise Dataflow axioms.
        
        Comprehensive validations for advanced BDIM pragma processing.
        """
        verifications = []
        
        # Axiom 1: Data Hierarchy validation (Tensor → Block → Stream → Element)
        verifications.append({
            'description': 'Verify data hierarchy consistency (Tensor → Block → Stream)',
            'code': '''
for iface in self.get_dataflow_interfaces():
    if hasattr(iface, 'tensor_dims') and hasattr(iface, 'block_dims') and hasattr(iface, 'stream_dims'):
        assert all(t >= b for t, b in zip(iface.tensor_dims, iface.block_dims) if isinstance(t, int) and isinstance(b, int)), f"Tensor dimensions must be >= block dimensions for {iface.name}"
        assert all(b >= s for b, s in zip(iface.block_dims, iface.stream_dims) if isinstance(b, int) and isinstance(s, int)), f"Block dimensions must be >= stream dimensions for {iface.name}"
            '''.strip(),
            'category': 'hierarchy_validation'
        })
        
        # Axiom 2: Core Relationship validation (tensor_dims → chunked into → block_dims)
        verifications.append({
            'description': 'Verify chunking relationship consistency',
            'code': '''
for iface in self.get_dataflow_interfaces():
    if hasattr(iface, 'tensor_dims') and hasattr(iface, 'block_dims'):
        for i, (t_dim, b_dim) in enumerate(zip(iface.tensor_dims, iface.block_dims)):
            if isinstance(t_dim, int) and isinstance(b_dim, int) and b_dim > 0:
                assert t_dim % b_dim == 0, f"Tensor dimension {i} ({t_dim}) must be divisible by block dimension ({b_dim}) for {iface.name}"
            '''.strip(),
            'category': 'chunking_validation'
        })
        
        # Axiom 3: Interface Types validation
        verifications.append({
            'description': 'Verify interface type classification consistency',
            'code': '''
input_interfaces = [iface for iface in self.get_dataflow_interfaces() if iface.dataflow_type == "INPUT"]
output_interfaces = [iface for iface in self.get_dataflow_interfaces() if iface.dataflow_type == "OUTPUT"]
assert len(input_interfaces) >= 1, "At least one INPUT interface required"
assert len(output_interfaces) >= 1, "At least one OUTPUT interface required"
            '''.strip(),
            'category': 'interface_type_validation'
        })
        
        # Interface-specific BDIM validations
        if hw_kernel.weight_interfaces_count > 0:
            verifications.append({
                'description': 'Verify weight interface BDIM alignment with inputs',
                'code': '''
weight_iface = self.get_weight_interface()
input_iface = self.get_input_interface()
if hasattr(weight_iface, 'block_dims') and hasattr(input_iface, 'block_dims'):
    if len(weight_iface.block_dims) >= 2 and len(input_iface.block_dims) >= 1:
        assert weight_iface.block_dims[-1] == input_iface.block_dims[-1], "Weight input dimension must match input feature dimension"
                '''.strip(),
                'category': 'weight_alignment_validation'
            })
        
        # Chunking strategy consistency
        verifications.append({
            'description': 'Verify BDIM chunking strategies are feasible',
            'code': '''
for iface in self.get_dataflow_interfaces():
    if hasattr(iface, 'chunking_strategy') and hasattr(iface, 'block_dims'):
        if iface.chunking_strategy == 'block_based':
            assert all(isinstance(dim, int) and dim > 0 for dim in iface.block_dims), f"Block-based chunking requires positive integer block dimensions for {iface.name}"
            '''.strip(),
            'category': 'strategy_validation'
        })
        
        return verifications
    
    def _get_interface_verifications(self, hw_kernel: UnifiedHWKernel) -> list:
        """Get interface-specific verification rules."""
        verifications = []
        
        # AXI-Stream interface validations
        axi_stream_interfaces = [iface for iface in hw_kernel.interfaces if self._get_interface_type_name(iface) == 'AXI_STREAM']
        
        if axi_stream_interfaces:
            verifications.append({
                'description': 'Verify AXI-Stream interfaces have required ports',
                'code': '''
for iface in self.get_dataflow_interfaces():
    if iface.interface_type == "AXI_STREAM":
        required_signals = ["TDATA", "TVALID", "TREADY"]
        iface_ports = [port.name.upper() for port in iface.ports] if hasattr(iface, 'ports') else []
        for signal in required_signals:
            assert any(signal in port for port in iface_ports), f"AXI-Stream interface {iface.name} missing required signal {signal}"
                '''.strip(),
                'category': 'interface_protocol_validation'
            })
        
        # Dataflow consistency across interfaces
        verifications.append({
            'description': 'Verify dataflow consistency between interfaces',
            'code': '''
input_interfaces = [iface for iface in self.get_dataflow_interfaces() if iface.dataflow_type == "INPUT"]
output_interfaces = [iface for iface in self.get_dataflow_interfaces() if iface.dataflow_type == "OUTPUT"]

# Verify basic flow: inputs produce outputs
assert len(input_interfaces) > 0 and len(output_interfaces) > 0, "Must have both input and output interfaces for dataflow"

# For kernels with weights, verify weight-input-output relationship
if self.get_weight_interface() is not None:
    assert len(input_interfaces) >= 1, "Kernels with weights must have activation inputs"
            '''.strip(),
            'category': 'dataflow_validation'
        })
        
        return verifications
    
    def _get_parallelism_verifications(self, hw_kernel: UnifiedHWKernel) -> list:
        """
        Get parallelism verification rules following Axiom 5: Parallelism Parameters.
        
        Validates iPar, wPar, and effective parallelism relationships.
        """
        verifications = []
        
        # Basic parallelism validation
        verifications.append({
            'description': 'Verify parallelism parameters are within reasonable bounds',
            'code': '''
if hasattr(self, 'parallelism_analysis'):
    iPar = getattr(self.parallelism_analysis, 'iPar', 1)
    wPar = getattr(self.parallelism_analysis, 'wPar', 1)
    
    assert 1 <= iPar <= 1024, f"Input parallelism (iPar={iPar}) must be between 1 and 1024"
    assert 1 <= wPar <= 1024, f"Weight parallelism (wPar={wPar}) must be between 1 and 1024"
    assert iPar * wPar <= 4096, f"Total parallelism ({iPar * wPar}) should not exceed 4096 for resource efficiency"
            '''.strip(),
            'category': 'parallelism_bounds_validation'
        })
        
        # Stream parallelism validation following Axiom 6: Stream Relationships
        verifications.append({
            'description': 'Verify stream parallelism relationships (Axiom 6)',
            'code': '''
for iface in self.get_dataflow_interfaces():
    if hasattr(iface, 'stream_dims') and hasattr(iface, 'block_dims'):
        if iface.dataflow_type == "INPUT" and len(iface.stream_dims) > 0 and len(iface.block_dims) > 0:
            # stream_dims_I = iPar (from Axiom 6)
            iPar = getattr(self.parallelism_analysis, 'iPar', 1) if hasattr(self, 'parallelism_analysis') else 1
            expected_stream_dim = min(iPar, iface.block_dims[0]) if isinstance(iface.block_dims[0], int) else iPar
            # Allow some flexibility in stream dimensions
            assert iface.stream_dims[0] <= expected_stream_dim, f"Input stream dimension should not exceed iPar for {iface.name}"
            '''.strip(),
            'category': 'stream_parallelism_validation'
        })
        
        # Performance-oriented parallelism validation
        if hw_kernel.bdim_metadata and 'parallelism' in hw_kernel.bdim_metadata:
            verifications.append({
                'description': 'Verify parallelism utilization efficiency',
                'code': '''
if hasattr(self, 'parallelism_analysis') and hasattr(self.parallelism_analysis, 'utilization_percentage'):
    utilization = self.parallelism_analysis.utilization_percentage
    assert utilization >= 25.0, f"Parallelism utilization ({utilization:.1f}%) is too low - consider optimizing chunking strategy"
    
    if utilization < 50.0:
        import warnings
        warnings.warn(f"Parallelism utilization ({utilization:.1f}%) could be improved", UserWarning)
                '''.strip(),
                'category': 'parallelism_efficiency_validation'
            })
        
        return verifications
    
    def _get_resource_estimation_context(self, hw_kernel: UnifiedHWKernel) -> dict:
        """Get resource estimation context for template."""
        return {
            'complexity': hw_kernel.kernel_complexity,
            'interface_count': len(hw_kernel.interfaces),
            'parameter_count': len(hw_kernel.rtl_parameters),
            'weight_interfaces': hw_kernel.weight_interfaces_count,
            'requires_estimation': hw_kernel.resource_estimation_required
        }
    
    def _build_enhanced_interface_metadata(self, hw_kernel: UnifiedHWKernel) -> list:
        """
        Build enhanced interface metadata for BDIM-aware templates.
        
        Following Interface-Wise Dataflow Axioms for complete metadata structure:
        - Axiom 1: Data Hierarchy (Tensor → Block → Stream → Element)
        - Axiom 2: Core Relationship (tensor_dims → block_dims → stream_dims)
        - Axiom 3: Interface Types (Input, Output, Weight, Config/Control)
        """
        enhanced_interfaces = []
        
        for iface in hw_kernel.interfaces:
            # Base interface metadata
            interface_metadata = {
                'name': iface.get('name', ''),
                'dataflow_type': self._classify_interface_dataflow_type(iface),
                'interface_type': self._get_interface_type_name(iface),
                'has_chunking_strategy': False,
                'tensor_dims': [],
                'block_dims': [],
                'stream_dims': [],
                'port_count': len(iface.get('ports', [])),
                'is_dataflow_interface': self._is_dataflow_interface(iface)
            }
            
            # Enhanced BDIM chunking information if available
            if hw_kernel.has_enhanced_bdim and self._has_bdim_metadata(iface):
                chunking_info = self._extract_bdim_chunking_info(iface, hw_kernel)
                interface_metadata.update({
                    'has_chunking_strategy': True,
                    'tensor_dims': chunking_info.get('tensor_dims', []),
                    'block_dims': chunking_info.get('block_dims', []),
                    'stream_dims': chunking_info.get('stream_dims', []),
                    'chunking_strategy': chunking_info.get('strategy_type', 'block_based'),
                    'chunk_index': chunking_info.get('chunk_index', 0),
                    'parallelism_factors': chunking_info.get('parallelism_factors', {}),
                    'layout_analysis': chunking_info.get('layout_analysis', {})
                })
            else:
                # Infer basic dimensions for simple mode
                basic_dims = self._infer_basic_dimensions(iface)
                interface_metadata.update(basic_dims)
            
            # Add datatype constraints and validation rules
            interface_metadata.update({
                'datatype_constraints': self._extract_datatype_constraints(iface),
                'validation_rules': self._build_interface_validation_rules(interface_metadata),
                'resource_impact': self._estimate_interface_resource_impact(interface_metadata)
            })
            
            enhanced_interfaces.append(interface_metadata)
        
        return enhanced_interfaces
    
    def _classify_interface_dataflow_type(self, iface: dict) -> str:
        """
        Classify interface dataflow type following Interface-Wise Dataflow Axiom 3.
        
        Enhanced classification with better pattern recognition.
        """
        # Use existing dataflow_type if available
        if 'dataflow_type' in iface:
            return iface['dataflow_type']
        
        # Enhanced pattern-based classification
        name = iface.get('name', '').lower()
        interface_type = self._get_interface_type_name(iface)
        
        if interface_type == 'GLOBAL_CONTROL':
            return 'CONTROL'
        elif interface_type == 'AXI_LITE':
            return 'CONFIG'
        elif interface_type == 'AXI_STREAM':
            # Enhanced AXI-Stream classification
            if any(pattern in name for pattern in ['weight', 'w_axis', 'param', 'kernel', 'filter']):
                return 'WEIGHT'
            elif any(pattern in name for pattern in ['s_axis', 'input', 'in_', 'slave']) or name.startswith('s_'):
                return 'INPUT'
            elif any(pattern in name for pattern in ['m_axis', 'output', 'out_', 'master']) or name.startswith('m_'):
                return 'OUTPUT'
            else:
                return 'INPUT'  # Default for unknown AXI-Stream
        
        return 'UNKNOWN'
    
    def _get_interface_type_name(self, iface: dict) -> str:
        """Get interface type name safely handling both dict and object types."""
        interface_type = iface.get('type', {})
        if hasattr(interface_type, 'name'):
            return interface_type.name
        elif isinstance(interface_type, dict):
            return interface_type.get('name', 'UNKNOWN')
        else:
            return str(interface_type)
    
    def _is_dataflow_interface(self, iface: dict) -> bool:
        """Check if interface participates in dataflow (AXI-Stream)."""
        return self._get_interface_type_name(iface) == 'AXI_STREAM'
    
    def _has_bdim_metadata(self, iface: dict) -> bool:
        """Check if interface has BDIM metadata available."""
        return any(key in iface for key in ['tensor_dims', 'block_dims', 'stream_dims', 'chunking_strategy'])
    
    def _extract_bdim_chunking_info(self, iface: dict, hw_kernel: UnifiedHWKernel) -> dict:
        """
        Extract BDIM chunking information from interface and kernel metadata.
        
        Following Interface-Wise Dataflow Axiom 2: Core Relationship.
        """
        chunking_info = {
            'tensor_dims': iface.get('tensor_dims', []),
            'block_dims': iface.get('block_dims', []),
            'stream_dims': iface.get('stream_dims', []),
            'strategy_type': iface.get('chunking_strategy', 'block_based'),
            'chunk_index': iface.get('chunk_index', 0)
        }
        
        # Extract parallelism factors from kernel metadata
        if hw_kernel.bdim_metadata and 'parallelism' in hw_kernel.bdim_metadata:
            parallelism = hw_kernel.bdim_metadata['parallelism']
            chunking_info['parallelism_factors'] = {
                'iPar': parallelism.get('iPar', 1),
                'wPar': parallelism.get('wPar', 1),
                'effective_parallelism': parallelism.get('iPar', 1) * parallelism.get('wPar', 1)
            }
        
        # Add layout analysis if available
        interface_name = iface.get('name', '')
        if hw_kernel.chunking_strategies and interface_name in hw_kernel.chunking_strategies:
            strategy = hw_kernel.chunking_strategies[interface_name]
            chunking_info['layout_analysis'] = {
                'inferred_layout': self._infer_tensor_layout(chunking_info['tensor_dims']),
                'chunking_dimension': strategy.get('chunk_index', 0),
                'chunk_sizes': strategy.get('chunk_sizes', chunking_info['block_dims'])
            }
        
        return chunking_info
    
    def _infer_basic_dimensions(self, iface: dict) -> dict:
        """
        Infer basic tensor dimensions for simple mode interfaces.
        
        Provides reasonable defaults following Interface-Wise Dataflow Axiom 1.
        """
        dataflow_type = self._classify_interface_dataflow_type(iface)
        
        # Default dimensions based on interface type
        if dataflow_type == 'WEIGHT':
            return {
                'tensor_dims': [256, 128],  # [OutChannels, InChannels]
                'block_dims': [16, 16],     # Conservative block size
                'stream_dims': [4]          # Conservative stream parallelism
            }
        elif dataflow_type == 'INPUT':
            return {
                'tensor_dims': [128, 32, 32],  # [Channels, Height, Width]
                'block_dims': [16, 8, 8],      # Conservative block size
                'stream_dims': [4]             # Conservative stream parallelism
            }
        elif dataflow_type == 'OUTPUT':
            return {
                'tensor_dims': [256, 32, 32],  # [Channels, Height, Width]
                'block_dims': [16, 8, 8],      # Conservative block size
                'stream_dims': [4]             # Conservative stream parallelism
            }
        else:
            return {
                'tensor_dims': [128],  # Generic 1D tensor
                'block_dims': [16],    # Conservative block size
                'stream_dims': [1]     # Single element per cycle
            }
    
    def _infer_tensor_layout(self, tensor_dims: list) -> str:
        """
        Infer tensor layout from dimensions following Interface-Wise Dataflow Axiom 9.
        
        Layout-Driven Chunking: ONNX tensor layout determines chunking dimension.
        """
        if not tensor_dims:
            return 'unknown'
        
        dim_count = len(tensor_dims)
        if dim_count == 1:
            return '[N]'
        elif dim_count == 2:
            return '[N, C]'
        elif dim_count == 3:
            return '[N, L, C]'  # Sequence data
        elif dim_count == 4:
            return '[N, C, H, W]'  # Image data
        else:
            return f'[{dim_count}D]'
    
    def _extract_datatype_constraints(self, iface: dict) -> list:
        """Extract datatype constraints for interface validation."""
        dataflow_type = self._classify_interface_dataflow_type(iface)
        
        # Default constraints based on interface type
        if dataflow_type == 'WEIGHT':
            return [
                {'finn_type': 'INT8', 'bit_width': 8, 'signed': True},
                {'finn_type': 'INT16', 'bit_width': 16, 'signed': True}
            ]
        elif dataflow_type in ['INPUT', 'OUTPUT']:
            return [
                {'finn_type': 'UINT8', 'bit_width': 8, 'signed': False},
                {'finn_type': 'UINT16', 'bit_width': 16, 'signed': False}
            ]
        else:
            return [
                {'finn_type': 'UINT32', 'bit_width': 32, 'signed': False}
            ]
    
    def _build_interface_validation_rules(self, interface_metadata: dict) -> list:
        """Build validation rules for interface based on metadata."""
        rules = []
        
        # Basic validation rules
        rules.append({
            'rule_type': 'dimension_consistency',
            'description': f"Validate {interface_metadata['name']} dimension consistency",
            'check': 'tensor_dims_compatible_with_block_dims'
        })
        
        # BDIM-specific validation rules
        if interface_metadata['has_chunking_strategy']:
            rules.append({
                'rule_type': 'chunking_alignment',
                'description': f"Validate {interface_metadata['name']} chunking alignment",
                'check': 'block_dims_align_with_tensor_dims'
            })
            
            if interface_metadata['dataflow_type'] == 'WEIGHT':
                rules.append({
                    'rule_type': 'weight_dimension_compatibility',
                    'description': f"Validate {interface_metadata['name']} weight dimensions",
                    'check': 'weight_dims_compatible_with_input_dims'
                })
        
        return rules
    
    def _estimate_interface_resource_impact(self, interface_metadata: dict) -> dict:
        """Estimate resource impact of interface for template optimization."""
        impact = {
            'memory_requirement': 'low',
            'bandwidth_requirement': 'low',
            'complexity_contribution': 'low'
        }
        
        # Calculate based on dimensions and parallelism
        if interface_metadata['has_chunking_strategy']:
            tensor_size = 1
            for dim in interface_metadata['tensor_dims']:
                if isinstance(dim, int):
                    tensor_size *= dim
            
            block_size = 1
            for dim in interface_metadata['block_dims']:
                if isinstance(dim, int):
                    block_size *= dim
            
            # Classify resource requirements
            if tensor_size > 100000:  # Large tensors
                impact['memory_requirement'] = 'high'
            elif tensor_size > 10000:
                impact['memory_requirement'] = 'medium'
            
            if block_size > 1000:  # Large blocks
                impact['bandwidth_requirement'] = 'high'
            elif block_size > 100:
                impact['bandwidth_requirement'] = 'medium'
            
            if interface_metadata['dataflow_type'] == 'WEIGHT':
                impact['complexity_contribution'] = 'high'  # Weights add complexity
        
        return impact
    
    def _build_chunking_context(self, hw_kernel: UnifiedHWKernel) -> dict:
        """
        Build chunking context for BDIM-aware template generation.
        
        Following Interface-Wise Dataflow Axiom 2: Core Relationship and
        Axiom 5: Parallelism Parameters (iPar, wPar).
        """
        if not hw_kernel.has_enhanced_bdim:
            return {
                'strategies_available': False,
                'uses_basic_chunking': True,
                'default_chunking_applied': True
            }
        
        chunking_context = {
            'strategies_available': len(hw_kernel.chunking_strategies) > 0,
            'strategy_count': len(hw_kernel.chunking_strategies),
            'interfaces_with_chunking': [],
            'parallelism_analysis': {},
            'dimension_relationships': {},
            'layout_information': {},
            'performance_estimates': {}
        }
        
        # Build interface-specific chunking information
        total_parallelism = 1
        for interface_name, strategy in hw_kernel.chunking_strategies.items():
            chunking_info = {
                'interface_name': interface_name,
                'chunking_type': strategy.get('chunking_type', 'block_based'),
                'tensor_dims': strategy.get('tensor_dims', []),
                'block_dims': strategy.get('block_dims', []),
                'stream_dims': strategy.get('stream_dims', []),
                'chunk_index': strategy.get('chunk_index', 0),
                'layout_compatibility': self._analyze_layout_compatibility(strategy)
            }
            
            # Calculate chunking efficiency
            chunking_info['efficiency_metrics'] = self._calculate_chunking_efficiency(chunking_info)
            
            chunking_context['interfaces_with_chunking'].append(chunking_info)
            
            # Accumulate parallelism
            stream_dims = chunking_info['stream_dims']
            if stream_dims and len(stream_dims) > 0:
                total_parallelism *= stream_dims[0]
        
        # Build parallelism analysis following Axiom 5
        parallelism_info = hw_kernel.bdim_metadata.get('parallelism', {}) if hw_kernel.bdim_metadata else {}
        chunking_context['parallelism_analysis'] = {
            'iPar': parallelism_info.get('iPar', 1),
            'wPar': parallelism_info.get('wPar', 1),
            'effective_parallelism': parallelism_info.get('iPar', 1) * parallelism_info.get('wPar', 1),
            'total_stream_parallelism': total_parallelism,
            'parallelism_utilization': self._calculate_parallelism_utilization(parallelism_info, total_parallelism)
        }
        
        # Build dimension relationships following Axiom 2: tensor_dims → block_dims → stream_dims
        chunking_context['dimension_relationships'] = self._analyze_dimension_relationships(hw_kernel)
        
        # Build layout information following Axiom 9: Layout-Driven Chunking
        chunking_context['layout_information'] = self._analyze_layout_patterns(hw_kernel)
        
        # Build performance estimates following Axiom 7: Timing Relationships
        chunking_context['performance_estimates'] = self._estimate_performance_characteristics(hw_kernel)
        
        return chunking_context
    
    def _analyze_layout_compatibility(self, strategy: dict) -> dict:
        """Analyze layout compatibility for chunking strategy."""
        tensor_dims = strategy.get('tensor_dims', [])
        block_dims = strategy.get('block_dims', [])
        
        compatibility = {
            'is_compatible': len(tensor_dims) >= len(block_dims),
            'layout_type': self._infer_tensor_layout(tensor_dims),
            'chunking_feasible': True,
            'alignment_issues': []
        }
        
        # Check dimension alignment
        for i, (t_dim, b_dim) in enumerate(zip(tensor_dims, block_dims)):
            if isinstance(t_dim, int) and isinstance(b_dim, int):
                if t_dim % b_dim != 0:
                    compatibility['alignment_issues'].append(f"Dimension {i}: {t_dim} not divisible by {b_dim}")
                    compatibility['chunking_feasible'] = False
        
        return compatibility
    
    def _calculate_chunking_efficiency(self, chunking_info: dict) -> dict:
        """Calculate efficiency metrics for chunking strategy."""
        tensor_dims = chunking_info['tensor_dims']
        block_dims = chunking_info['block_dims']
        stream_dims = chunking_info['stream_dims']
        
        efficiency = {
            'utilization_ratio': 1.0,
            'memory_efficiency': 1.0,
            'bandwidth_efficiency': 1.0,
            'parallelism_efficiency': 1.0
        }
        
        if tensor_dims and block_dims:
            # Calculate utilization ratio
            tensor_size = 1
            block_size = 1
            for t_dim, b_dim in zip(tensor_dims, block_dims):
                if isinstance(t_dim, int) and isinstance(b_dim, int):
                    tensor_size *= t_dim
                    block_size *= b_dim
            
            if tensor_size > 0:
                efficiency['utilization_ratio'] = block_size / tensor_size
            
            # Calculate memory efficiency (how well blocks fit into memory hierarchy)
            if block_size <= 1024:  # Fits in L1 cache
                efficiency['memory_efficiency'] = 1.0
            elif block_size <= 32768:  # Fits in L2 cache
                efficiency['memory_efficiency'] = 0.8
            else:
                efficiency['memory_efficiency'] = 0.6
            
            # Calculate bandwidth efficiency based on stream dimensions
            if stream_dims and len(stream_dims) > 0:
                stream_size = stream_dims[0]
                if stream_size >= block_size:
                    efficiency['bandwidth_efficiency'] = 1.0
                else:
                    efficiency['bandwidth_efficiency'] = stream_size / block_size
        
        return efficiency
    
    def _calculate_parallelism_utilization(self, parallelism_info: dict, total_stream_parallelism: int) -> dict:
        """Calculate parallelism utilization metrics."""
        iPar = parallelism_info.get('iPar', 1)
        wPar = parallelism_info.get('wPar', 1)
        theoretical_parallelism = iPar * wPar
        
        return {
            'theoretical_parallelism': theoretical_parallelism,
            'actual_parallelism': total_stream_parallelism,
            'utilization_percentage': (total_stream_parallelism / max(theoretical_parallelism, 1)) * 100,
            'bottleneck_analysis': self._identify_parallelism_bottlenecks(iPar, wPar, total_stream_parallelism)
        }
    
    def _identify_parallelism_bottlenecks(self, iPar: int, wPar: int, actual_parallelism: int) -> list:
        """Identify potential parallelism bottlenecks."""
        bottlenecks = []
        theoretical = iPar * wPar
        
        if actual_parallelism < theoretical:
            if actual_parallelism < iPar:
                bottlenecks.append("Input parallelism underutilized")
            if actual_parallelism < wPar:
                bottlenecks.append("Weight parallelism underutilized")
            if actual_parallelism < theoretical * 0.5:
                bottlenecks.append("Significant parallelism gap detected")
        
        return bottlenecks
    
    def _analyze_dimension_relationships(self, hw_kernel: UnifiedHWKernel) -> dict:
        """
        Analyze dimension relationships following Axiom 2: Core Relationship.
        
        tensor_dims → chunked into → num_blocks pieces of shape block_dims → streamed as stream_dims per cycle
        """
        relationships = {
            'tensor_to_block_ratios': {},
            'block_to_stream_ratios': {},
            'dimension_consistency': True,
            'relationship_violations': []
        }
        
        for interface_name, strategy in hw_kernel.chunking_strategies.items():
            tensor_dims = strategy.get('tensor_dims', [])
            block_dims = strategy.get('block_dims', [])
            stream_dims = strategy.get('stream_dims', [])
            
            # Calculate tensor → block ratios
            t_to_b_ratios = []
            for t_dim, b_dim in zip(tensor_dims, block_dims):
                if isinstance(t_dim, int) and isinstance(b_dim, int) and b_dim > 0:
                    t_to_b_ratios.append(t_dim / b_dim)
                else:
                    relationships['relationship_violations'].append(f"{interface_name}: Invalid tensor→block ratio")
            
            relationships['tensor_to_block_ratios'][interface_name] = t_to_b_ratios
            
            # Calculate block → stream ratios
            b_to_s_ratios = []
            for b_dim, s_dim in zip(block_dims, stream_dims):
                if isinstance(b_dim, int) and isinstance(s_dim, int) and s_dim > 0:
                    b_to_s_ratios.append(b_dim / s_dim)
                else:
                    relationships['relationship_violations'].append(f"{interface_name}: Invalid block→stream ratio")
            
            relationships['block_to_stream_ratios'][interface_name] = b_to_s_ratios
        
        relationships['dimension_consistency'] = len(relationships['relationship_violations']) == 0
        
        return relationships
    
    def _analyze_layout_patterns(self, hw_kernel: UnifiedHWKernel) -> dict:
        """
        Analyze layout patterns following Axiom 9: Layout-Driven Chunking.
        """
        layout_info = {
            'detected_layouts': {},
            'chunking_dimensions': {},
            'layout_compatibility': {},
            'optimization_suggestions': []
        }
        
        for interface_name, strategy in hw_kernel.chunking_strategies.items():
            tensor_dims = strategy.get('tensor_dims', [])
            chunk_index = strategy.get('chunk_index', 0)
            
            # Detect layout pattern
            layout_pattern = self._infer_tensor_layout(tensor_dims)
            layout_info['detected_layouts'][interface_name] = layout_pattern
            
            # Analyze chunking dimension
            if chunk_index < len(tensor_dims):
                layout_info['chunking_dimensions'][interface_name] = {
                    'chunking_index': chunk_index,
                    'chunking_dimension_size': tensor_dims[chunk_index] if isinstance(tensor_dims[chunk_index], int) else 'dynamic',
                    'is_optimal_for_layout': self._is_optimal_chunking_dimension(layout_pattern, chunk_index)
                }
            
            # Analyze compatibility
            layout_info['layout_compatibility'][interface_name] = self._analyze_layout_chunking_compatibility(layout_pattern, chunk_index)
        
        # Generate optimization suggestions
        layout_info['optimization_suggestions'] = self._generate_layout_optimization_suggestions(layout_info)
        
        return layout_info
    
    def _is_optimal_chunking_dimension(self, layout_pattern: str, chunk_index: int) -> bool:
        """Check if chunking dimension is optimal for the detected layout."""
        # Layout-specific optimization rules
        if layout_pattern == '[N, C, H, W]':
            return chunk_index == 1  # Chunk along C dimension is typically optimal
        elif layout_pattern == '[N, L, C]':
            return chunk_index == 1  # Chunk along L dimension for sequence data
        elif layout_pattern == '[N, H, W, C]':
            return chunk_index in [1, 2]  # Chunk along H×W dimensions
        
        return True  # Default to optimal for unknown layouts
    
    def _analyze_layout_chunking_compatibility(self, layout_pattern: str, chunk_index: int) -> dict:
        """Analyze compatibility between layout and chunking strategy."""
        compatibility = {
            'is_compatible': True,
            'efficiency_rating': 'high',
            'potential_issues': [],
            'recommendations': []
        }
        
        if layout_pattern == '[N, C, H, W]' and chunk_index not in [1, 2, 3]:
            compatibility['is_compatible'] = False
            compatibility['potential_issues'].append("Chunking batch dimension not recommended for image data")
            compatibility['recommendations'].append("Consider chunking along channel or spatial dimensions")
        
        return compatibility
    
    def _generate_layout_optimization_suggestions(self, layout_info: dict) -> list:
        """Generate optimization suggestions based on layout analysis."""
        suggestions = []
        
        for interface_name, layout_data in layout_info['detected_layouts'].items():
            if interface_name in layout_info['chunking_dimensions']:
                chunk_info = layout_info['chunking_dimensions'][interface_name]
                if not chunk_info['is_optimal_for_layout']:
                    suggestions.append({
                        'interface': interface_name,
                        'issue': f"Suboptimal chunking dimension for {layout_data} layout",
                        'suggestion': f"Consider chunking along more cache-friendly dimensions",
                        'priority': 'medium'
                    })
        
        return suggestions
    
    def _estimate_performance_characteristics(self, hw_kernel: UnifiedHWKernel) -> dict:
        """
        Estimate performance characteristics following Axiom 7: Timing Relationships.
        
        - cII: Cycles per calculation (Input block × Weight block)
        - eII: Cycles per execution (Input block × Weight tensor)  
        - L: Cycles per inference (Input tensor)
        """
        performance = {
            'estimated_cII': {},
            'estimated_eII': {},
            'estimated_L': {},
            'bottleneck_analysis': {},
            'optimization_potential': {}
        }
        
        input_interfaces = [iface for iface in hw_kernel.interfaces if self._classify_interface_dataflow_type(iface) == 'INPUT']
        weight_interfaces = [iface for iface in hw_kernel.interfaces if self._classify_interface_dataflow_type(iface) == 'WEIGHT']
        
        for input_iface in input_interfaces:
            interface_name = input_iface.get('name', '')
            
            if interface_name in hw_kernel.chunking_strategies:
                strategy = hw_kernel.chunking_strategies[interface_name]
                block_dims = strategy.get('block_dims', [])
                stream_dims = strategy.get('stream_dims', [])
                tensor_dims = strategy.get('tensor_dims', [])
                
                # Calculate cII: cycles per input block processing
                if block_dims and stream_dims:
                    block_size = 1
                    stream_size = 1
                    for dim in block_dims:
                        if isinstance(dim, int):
                            block_size *= dim
                    for dim in stream_dims:
                        if isinstance(dim, int):
                            stream_size *= dim
                    
                    cII = max(1, block_size // stream_size) if stream_size > 0 else block_size
                    performance['estimated_cII'][interface_name] = cII
                
                # Calculate eII and L based on tensor dimensions
                if tensor_dims and block_dims:
                    tensor_size = 1
                    for dim in tensor_dims:
                        if isinstance(dim, int):
                            tensor_size *= dim
                    
                    num_blocks = max(1, tensor_size // block_size) if block_size > 0 else 1
                    performance['estimated_eII'][interface_name] = cII * num_blocks
                    performance['estimated_L'][interface_name] = performance['estimated_eII'][interface_name]
        
        # Analyze bottlenecks
        performance['bottleneck_analysis'] = self._analyze_performance_bottlenecks(performance)
        
        # Identify optimization potential
        performance['optimization_potential'] = self._identify_optimization_opportunities(performance, hw_kernel)
        
        return performance
    
    def _analyze_performance_bottlenecks(self, performance: dict) -> dict:
        """Analyze potential performance bottlenecks."""
        bottlenecks = {
            'high_cII_interfaces': [],
            'memory_bound_interfaces': [],
            'bandwidth_bound_interfaces': []
        }
        
        for interface_name, cII in performance.get('estimated_cII', {}).items():
            if cII > 100:  # High cycle count per input block
                bottlenecks['high_cII_interfaces'].append({
                    'interface': interface_name,
                    'cII': cII,
                    'severity': 'high' if cII > 1000 else 'medium'
                })
        
        return bottlenecks
    
    def _identify_optimization_opportunities(self, performance: dict, hw_kernel: UnifiedHWKernel) -> list:
        """Identify optimization opportunities."""
        opportunities = []
        
        # Check for parallelism underutilization
        if hw_kernel.bdim_metadata and 'parallelism' in hw_kernel.bdim_metadata:
            parallelism = hw_kernel.bdim_metadata['parallelism']
            theoretical_parallelism = parallelism.get('iPar', 1) * parallelism.get('wPar', 1)
            
            for interface_name, cII in performance.get('estimated_cII', {}).items():
                if cII > theoretical_parallelism:
                    opportunities.append({
                        'type': 'parallelism_increase',
                        'interface': interface_name,
                        'current_cII': cII,
                        'potential_improvement': f"Could reduce to {cII // theoretical_parallelism} cycles",
                        'recommendation': "Increase stream parallelism or reduce block size"
                    })
        
        return opportunities