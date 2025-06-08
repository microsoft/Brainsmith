"""
Enhanced Template Context Builder with Dataflow Integration.

This module provides a comprehensive template context building system that
integrates with the Interface-Wise Dataflow Modeling framework to generate
rich, validated template contexts for code generation.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import hashlib
import json
from datetime import datetime
from abc import ABC, abstractmethod

# Import dataflow components
try:
    from ...dataflow.core.dataflow_interface import DataflowInterface, DataflowInterfaceType
    from ...dataflow.core.dataflow_model import DataflowModel, InitiationIntervals
    from ...dataflow.integration.rtl_conversion import RTLInterfaceConverter
    from ...dataflow.core.class_naming import generate_class_name, generate_backend_class_name
    from ...dataflow.core.validation import ValidationResult, create_validation_result
    DATAFLOW_AVAILABLE = True
except ImportError:
    DATAFLOW_AVAILABLE = False
    # Create placeholder types for when dataflow is not available
    class DataflowInterface: pass
    class DataflowInterfaceType: pass
    class DataflowModel: pass
    class InitiationIntervals: pass
    class RTLInterfaceConverter: pass
    ValidationResult = Dict[str, Any]

from .enhanced_config import PipelineConfig, DataflowMode
from .errors import TemplateError, ConfigurationError


@dataclass
class BaseContext:
    """Base template context used by all generators."""
    
    # Basic information
    kernel_name: str
    class_name: str
    file_name: str
    source_file: str
    generation_timestamp: str
    generator_version: str = "2.0.0"
    
    # RTL metadata
    rtl_parameters: Dict[str, Any] = field(default_factory=dict)
    rtl_interfaces: Dict[str, Any] = field(default_factory=dict)
    rtl_pragmas: List[Dict[str, Any]] = field(default_factory=list)
    
    # Configuration metadata
    config_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> ValidationResult:
        """Validate base context requirements."""
        if DATAFLOW_AVAILABLE:
            result = create_validation_result()
        else:
            result = {"errors": [], "warnings": [], "success": True}
        
        # Validate required fields
        required_fields = ["kernel_name", "class_name", "file_name", "source_file"]
        for field_name in required_fields:
            if not getattr(self, field_name, None):
                error = {
                    "component": "BaseContext",
                    "error_type": "missing_field",
                    "message": f"Required field '{field_name}' is missing or empty",
                    "context": {"field": field_name}
                }
                if DATAFLOW_AVAILABLE:
                    result.add_error(error)
                else:
                    result["errors"].append(error)
                    result["success"] = False
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for template rendering."""
        return asdict(self)


@dataclass
class DataflowContext(BaseContext):
    """Enhanced context with dataflow modeling information."""
    
    # Dataflow model information
    dataflow_model: Optional[DataflowModel] = None
    dataflow_interfaces: List[Dict[str, Any]] = field(default_factory=list)
    
    # Interface organization
    input_interfaces: List[Dict[str, Any]] = field(default_factory=list)
    output_interfaces: List[Dict[str, Any]] = field(default_factory=list)
    weight_interfaces: List[Dict[str, Any]] = field(default_factory=list)
    config_interfaces: List[Dict[str, Any]] = field(default_factory=list)
    control_interfaces: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    initiation_intervals: Optional[Dict[str, Any]] = None
    parallelism_bounds: Dict[str, Any] = field(default_factory=dict)
    resource_estimates: Dict[str, Any] = field(default_factory=dict)
    
    # Tensor information
    tensor_shapes: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    chunking_information: Dict[str, Any] = field(default_factory=dict)
    
    # AXI signal specifications
    axi_signals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Validation results
    validation_summary: Dict[str, Any] = field(default_factory=dict)
    
    def has_axi_interfaces(self) -> bool:
        """Check if kernel has AXI-Stream interfaces."""
        return len(self.input_interfaces) > 0 or len(self.output_interfaces) > 0 or len(self.weight_interfaces) > 0
    
    def get_axi_interfaces(self) -> List[Dict[str, Any]]:
        """Get all AXI-Stream interfaces."""
        return self.input_interfaces + self.output_interfaces + self.weight_interfaces
    
    def get_control_interfaces(self) -> List[Dict[str, Any]]:
        """Get all control interfaces."""
        return self.control_interfaces + self.config_interfaces
    
    def get_interface_count_by_type(self) -> Dict[str, int]:
        """Get count of interfaces by type."""
        return {
            "input": len(self.input_interfaces),
            "output": len(self.output_interfaces),
            "weight": len(self.weight_interfaces),
            "config": len(self.config_interfaces),
            "control": len(self.control_interfaces)
        }


@dataclass
class HWCustomOpContext(DataflowContext):
    """HWCustomOp-specific template context."""
    
    # HWCustomOp specific fields
    onnx_op_type: str = "HWCustomOp"
    supports_batching: bool = True
    requires_weight_loading: bool = False
    
    # FINN integration
    finn_datatype_support: List[str] = field(default_factory=list)
    parallelism_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Code generation flags
    generate_get_exp_cycles: bool = True
    generate_resource_estimation: bool = True
    generate_characteristic_fxns: bool = True
    
    # Enhanced functionality flags
    use_autohwcustomop_base: bool = True
    minimize_generated_code: bool = True


@dataclass
class RTLBackendContext(DataflowContext):
    """RTLBackend-specific template context."""
    
    # RTL Backend specific fields
    backend_type: str = "RTLBackend"
    synthesis_config: Dict[str, Any] = field(default_factory=dict)
    
    # RTL generation
    generate_wrapper: bool = True
    wrapper_module_name: str = ""
    clock_domain_info: Dict[str, Any] = field(default_factory=dict)
    
    # Code generation flags
    generate_code_dict: bool = True
    generate_params_method: bool = True
    
    # Enhanced functionality flags
    use_autortlbackend_base: bool = True


class ContextBuilder(ABC):
    """Abstract base for context builders."""
    
    @abstractmethod
    def build_context(self, *args, **kwargs) -> BaseContext:
        """Build template context."""
        pass


class EnhancedTemplateContextBuilder:
    """
    Enhanced template context builder with comprehensive dataflow integration.
    
    This builder creates rich template contexts by combining RTL parsing results
    with dataflow modeling information, providing all the data needed for
    high-quality code generation.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize context builder with configuration."""
        self.config = config or PipelineConfig()
        self._context_cache: Dict[str, BaseContext] = {}
        self._converter: Optional[RTLInterfaceConverter] = None
        
        # Initialize RTL to dataflow converter if dataflow is available
        if DATAFLOW_AVAILABLE and self.config.is_dataflow_enabled():
            self._converter = RTLInterfaceConverter(
                onnx_metadata=self.config.dataflow.onnx_metadata
            )
    
    def clear_cache(self) -> None:
        """Clear the template context cache."""
        self._context_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._context_cache),
            "cache_limit": self.config.dataflow.cache_size_limit if self.config.is_dataflow_enabled() else 0,
            "hits": getattr(self, "_cache_hits", 0),
            "misses": getattr(self, "_cache_misses", 0)
        }
    
    def build_base_context(
        self,
        hw_kernel: Any,
        config: PipelineConfig,
        source_file: Optional[str] = None
    ) -> BaseContext:
        """Build common base context used by all generators."""
        
        # Create cache key
        cache_key = self._create_cache_key("base", hw_kernel, config, source_file)
        
        # Check cache if enabled
        if config.dataflow.enable_interface_caching and cache_key in self._context_cache:
            self._cache_hits = getattr(self, "_cache_hits", 0) + 1
            return self._context_cache[cache_key]
        
        self._cache_misses = getattr(self, "_cache_misses", 0) + 1
        
        # Generate class name
        if DATAFLOW_AVAILABLE:
            class_name = generate_class_name(hw_kernel.name, prefix="Auto")
        else:
            class_name = self._legacy_class_name_generation(hw_kernel.name)
        
        # Build base context
        context = BaseContext(
            kernel_name=hw_kernel.name,
            class_name=class_name,
            file_name=f"{hw_kernel.name.lower()}.py",
            source_file=source_file or str(config.rtl_file_path or "unknown.sv"),
            generation_timestamp=datetime.now().isoformat(),
            generator_version="2.0.0",
            rtl_parameters=self._process_parameters(hw_kernel.parameters),
            rtl_interfaces=self._process_interfaces(hw_kernel.interfaces),
            rtl_pragmas=self._process_pragmas(getattr(hw_kernel, 'pragmas', [])),
            config_metadata={
                "generator_type": config.generator_type.value,
                "dataflow_enabled": config.is_dataflow_enabled(),
                "dataflow_mode": config.dataflow.mode.value,
                "template_selection": config.template.template_selection_strategy
            }
        )
        
        # Cache if enabled
        if config.dataflow.enable_interface_caching:
            self._context_cache[cache_key] = context
        
        return context
    
    def build_dataflow_context(
        self,
        hw_kernel: Any,
        config: PipelineConfig,
        onnx_metadata: Optional[Dict] = None,
        source_file: Optional[str] = None
    ) -> DataflowContext:
        """Build enhanced dataflow context with comprehensive dataflow information."""
        
        if not DATAFLOW_AVAILABLE or not config.is_dataflow_enabled():
            raise TemplateError(
                "Dataflow context requested but dataflow modeling is not available or disabled",
                template_name="dataflow_context",
                suggestion="Enable dataflow mode or install dataflow dependencies"
            )
        
        # Create cache key
        cache_key = self._create_cache_key("dataflow", hw_kernel, config, source_file, onnx_metadata)
        
        # Check cache
        if config.dataflow.enable_interface_caching and cache_key in self._context_cache:
            self._cache_hits = getattr(self, "_cache_hits", 0) + 1
            return self._context_cache[cache_key]
        
        self._cache_misses = getattr(self, "_cache_misses", 0) + 1
        
        # Build base context
        base_context = self.build_base_context(hw_kernel, config, source_file)
        
        # Convert RTL interfaces to dataflow interfaces
        dataflow_interfaces = []
        if self._converter:
            try:
                dataflow_interfaces = self._converter.convert_interfaces(
                    hw_kernel.interfaces,
                    parameters=hw_kernel.parameters
                )
            except Exception as e:
                raise TemplateError(
                    f"Failed to convert RTL interfaces to dataflow interfaces: {e}",
                    template_name="dataflow_context",
                    suggestion="Check RTL interface compatibility with dataflow modeling"
                )
        
        # Create dataflow model
        dataflow_model = None
        if dataflow_interfaces:
            try:
                dataflow_model = DataflowModel(dataflow_interfaces, hw_kernel.parameters)
            except Exception as e:
                raise TemplateError(
                    f"Failed to create dataflow model: {e}",
                    template_name="dataflow_context",
                    suggestion="Verify interface definitions and parameters"
                )
        
        # Organize interfaces by type
        organized_interfaces = self._organize_dataflow_interfaces(dataflow_interfaces)
        
        # Calculate performance metrics if model available
        performance_metrics = {}
        if dataflow_model and config.dataflow.enable_parallelism_optimization:
            try:
                performance_metrics = self._calculate_performance_metrics(dataflow_model)
            except Exception as e:
                # Don't fail context building for performance metric errors
                performance_metrics = {"error": str(e)}
        
        # Generate AXI signal specifications
        axi_signals = self._generate_axi_signals(dataflow_interfaces)
        
        # Build enhanced context
        context = DataflowContext(
            # Base context fields
            kernel_name=base_context.kernel_name,
            class_name=base_context.class_name,
            file_name=base_context.file_name,
            source_file=base_context.source_file,
            generation_timestamp=base_context.generation_timestamp,
            generator_version=base_context.generator_version,
            rtl_parameters=base_context.rtl_parameters,
            rtl_interfaces=base_context.rtl_interfaces,
            rtl_pragmas=base_context.rtl_pragmas,
            config_metadata=base_context.config_metadata,
            
            # Dataflow-specific fields
            dataflow_model=dataflow_model,
            dataflow_interfaces=[self._interface_to_dict(iface) for iface in dataflow_interfaces],
            input_interfaces=organized_interfaces["input"],
            output_interfaces=organized_interfaces["output"],
            weight_interfaces=organized_interfaces["weight"],
            config_interfaces=organized_interfaces["config"],
            control_interfaces=organized_interfaces["control"],
            initiation_intervals=performance_metrics.get("intervals"),
            parallelism_bounds=performance_metrics.get("bounds", {}),
            resource_estimates=performance_metrics.get("resources", {}),
            tensor_shapes=self._extract_tensor_shapes(dataflow_interfaces),
            chunking_information=self._extract_chunking_info(dataflow_interfaces),
            axi_signals=axi_signals,
            validation_summary=self._create_validation_summary(dataflow_interfaces, dataflow_model)
        )
        
        # Cache if enabled
        if config.dataflow.enable_interface_caching:
            self._context_cache[cache_key] = context
        
        return context
    
    def build_hwcustomop_context(
        self,
        hw_kernel: Any,
        config: PipelineConfig,
        finn_config: Optional[Dict] = None,
        onnx_metadata: Optional[Dict] = None,
        source_file: Optional[str] = None
    ) -> HWCustomOpContext:
        """Build HWCustomOp-specific context."""
        
        # Build dataflow context first
        if config.is_dataflow_enabled():
            dataflow_context = self.build_dataflow_context(hw_kernel, config, onnx_metadata, source_file)
        else:
            base_context = self.build_base_context(hw_kernel, config, source_file)
            # Create minimal dataflow context for compatibility
            dataflow_context = DataflowContext(**base_context.to_dict())
        
        # Extract FINN-specific information
        finn_info = self._extract_finn_information(finn_config or {})
        
        # Create HWCustomOp context
        context = HWCustomOpContext(
            # Copy all fields from dataflow context
            **dataflow_context.to_dict(),
            
            # HWCustomOp-specific fields
            onnx_op_type=finn_info.get("op_type", "HWCustomOp"),
            supports_batching=finn_info.get("supports_batching", True),
            requires_weight_loading=len(dataflow_context.weight_interfaces) > 0,
            finn_datatype_support=finn_info.get("supported_datatypes", ["UINT8", "INT8", "UINT16", "INT16"]),
            parallelism_parameters=finn_info.get("parallelism_params", {}),
            generate_get_exp_cycles=config.dataflow.resource_estimation_enabled,
            generate_resource_estimation=config.dataflow.resource_estimation_enabled,
            generate_characteristic_fxns=config.generation.include_debug_info,
            use_autohwcustomop_base=config.generation.use_autogenerated_base_classes,
            minimize_generated_code=config.generation.minimize_template_code
        )
        
        return context
    
    def build_rtlbackend_context(
        self,
        hw_kernel: Any,
        config: PipelineConfig,
        backend_config: Optional[Dict] = None,
        onnx_metadata: Optional[Dict] = None,
        source_file: Optional[str] = None
    ) -> RTLBackendContext:
        """Build RTLBackend-specific context."""
        
        # Build dataflow context first
        if config.is_dataflow_enabled():
            dataflow_context = self.build_dataflow_context(hw_kernel, config, onnx_metadata, source_file)
        else:
            base_context = self.build_base_context(hw_kernel, config, source_file)
            # Create minimal dataflow context for compatibility
            dataflow_context = DataflowContext(**base_context.to_dict())
        
        # Extract backend-specific information
        backend_info = self._extract_backend_information(backend_config or {})
        
        # Generate backend class name
        if DATAFLOW_AVAILABLE:
            backend_class_name = generate_backend_class_name(hw_kernel.name)
        else:
            backend_class_name = f"{dataflow_context.class_name}RTLBackend"
        
        # Create RTLBackend context data
        context_data = dataflow_context.to_dict()
        
        # Override specific fields for backend
        context_data.update({
            "class_name": backend_class_name,
            "file_name": f"{hw_kernel.name.lower()}_rtlbackend.py",
            
            # RTLBackend-specific fields
            "backend_type": backend_info.get("backend_type", "RTLBackend"),
            "synthesis_config": backend_info.get("synthesis_config", {}),
            "generate_wrapper": backend_info.get("generate_wrapper", True),
            "wrapper_module_name": backend_info.get("wrapper_name", f"{hw_kernel.name}_wrapper"),
            "clock_domain_info": backend_info.get("clock_domains", {}),
            "generate_code_dict": True,
            "generate_params_method": True,
            "use_autortlbackend_base": config.generation.use_autogenerated_base_classes
        })
        
        # Create and return RTLBackend context
        return RTLBackendContext(**context_data)
    
    def _create_cache_key(self, context_type: str, *args) -> str:
        """Create a cache key for context caching."""
        # Create a hash of the arguments
        content = json.dumps(str(args), sort_keys=True)
        hash_obj = hashlib.md5(content.encode())
        return f"{context_type}_{hash_obj.hexdigest()[:16]}"
    
    def _legacy_class_name_generation(self, kernel_name: str) -> str:
        """Legacy class name generation when dataflow is not available."""
        # Convert snake_case to CamelCase
        words = kernel_name.replace('-', '_').split('_')
        return ''.join(word.capitalize() for word in words)
    
    def _process_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process RTL parameters for template context."""
        processed = {}
        for name, param in parameters.items():
            if hasattr(param, 'to_dict'):
                processed[name] = param.to_dict()
            else:
                processed[name] = {
                    "name": name,
                    "value": param,
                    "type": type(param).__name__
                }
        return processed
    
    def _process_interfaces(self, interfaces: Dict[str, Any]) -> Dict[str, Any]:
        """Process RTL interfaces for template context."""
        processed = {}
        for name, interface in interfaces.items():
            if hasattr(interface, 'to_dict'):
                processed[name] = interface.to_dict()
            else:
                processed[name] = {
                    "name": name,
                    "type": getattr(interface, 'type', 'unknown'),
                    "signals": getattr(interface, 'signals', [])
                }
        return processed
    
    def _process_pragmas(self, pragmas: List[Any]) -> List[Dict[str, Any]]:
        """Process RTL pragmas for template context."""
        processed = []
        for pragma in pragmas:
            if hasattr(pragma, 'to_dict'):
                processed.append(pragma.to_dict())
            else:
                processed.append({
                    "type": getattr(pragma, 'type', 'unknown'),
                    "content": str(pragma)
                })
        return processed
    
    def _organize_dataflow_interfaces(self, interfaces: List[DataflowInterface]) -> Dict[str, List[Dict[str, Any]]]:
        """Organize dataflow interfaces by type."""
        organized = {
            "input": [],
            "output": [],
            "weight": [],
            "config": [],
            "control": []
        }
        
        for interface in interfaces:
            interface_dict = self._interface_to_dict(interface)
            
            if interface.interface_type == DataflowInterfaceType.INPUT:
                organized["input"].append(interface_dict)
            elif interface.interface_type == DataflowInterfaceType.OUTPUT:
                organized["output"].append(interface_dict)
            elif interface.interface_type == DataflowInterfaceType.WEIGHT:
                organized["weight"].append(interface_dict)
            elif interface.interface_type == DataflowInterfaceType.CONFIG:
                organized["config"].append(interface_dict)
            elif interface.interface_type == DataflowInterfaceType.CONTROL:
                organized["control"].append(interface_dict)
        
        return organized
    
    def _interface_to_dict(self, interface: DataflowInterface) -> Dict[str, Any]:
        """Convert DataflowInterface to dictionary representation."""
        return {
            "name": interface.name,
            "type": interface.interface_type.value,
            "qDim": interface.qDim,
            "tDim": interface.tDim,
            "sDim": interface.sDim,
            "dtype": {
                "base_type": interface.dtype.base_type,
                "bitwidth": interface.dtype.bitwidth,
                "signed": interface.dtype.signed,
                "finn_type": interface.dtype.finn_type
            },
            "stream_width": interface.calculate_stream_width(),
            "memory_footprint": interface.get_memory_footprint(),
            "transfer_cycles": interface.get_transfer_cycles(),
            "axi_metadata": interface.axi_metadata,
            "pragma_metadata": interface.pragma_metadata
        }
    
    def _calculate_performance_metrics(self, model: DataflowModel) -> Dict[str, Any]:
        """Calculate performance metrics from dataflow model."""
        try:
            # Get parallelism bounds
            bounds = model.get_parallelism_bounds()
            
            # Calculate with default parallelism
            default_ipar = {name: bound.min_value for name, bound in bounds.items() if "input" in name}
            default_wpar = {name: bound.min_value for name, bound in bounds.items() if "weight" in name}
            
            intervals = model.calculate_initiation_intervals(default_ipar, default_wpar)
            
            # Get resource estimates
            from ...dataflow.core.dataflow_model import ParallelismConfiguration
            config = ParallelismConfiguration(default_ipar, default_wpar, {})
            resources = model.get_resource_requirements(config)
            
            return {
                "intervals": {
                    "cII": intervals.cII,
                    "eII": intervals.eII,
                    "L": intervals.L,
                    "bottleneck_analysis": intervals.bottleneck_analysis
                },
                "bounds": {name: {"min": bound.min_value, "max": bound.max_value} 
                          for name, bound in bounds.items()},
                "resources": resources
            }
        except Exception as e:
            return {"error": f"Performance calculation failed: {e}"}
    
    def _generate_axi_signals(self, interfaces: List[DataflowInterface]) -> Dict[str, Dict[str, Any]]:
        """Generate AXI signal specifications for interfaces."""
        axi_signals = {}
        
        for interface in interfaces:
            if interface.interface_type in [DataflowInterfaceType.INPUT, 
                                          DataflowInterfaceType.OUTPUT, 
                                          DataflowInterfaceType.WEIGHT]:
                signals = interface.get_axi_signals()
                axi_signals[interface.name] = signals
        
        return axi_signals
    
    def _extract_tensor_shapes(self, interfaces: List[DataflowInterface]) -> Dict[str, Dict[str, List[int]]]:
        """Extract tensor shape information from interfaces."""
        shapes = {}
        
        for interface in interfaces:
            shapes[interface.name] = {
                "qDim": interface.qDim,
                "tDim": interface.tDim,
                "sDim": interface.sDim,
                "original_shape": interface.reconstruct_tensor_shape()
            }
        
        return shapes
    
    def _extract_chunking_info(self, interfaces: List[DataflowInterface]) -> Dict[str, Any]:
        """Extract tensor chunking information."""
        chunking_info = {}
        
        for interface in interfaces:
            if hasattr(interface, 'chunking_metadata'):
                chunking_info[interface.name] = interface.chunking_metadata
        
        return chunking_info
    
    def _create_validation_summary(self, interfaces: List[DataflowInterface], model: Optional[DataflowModel]) -> Dict[str, Any]:
        """Create validation summary for the dataflow components."""
        summary = {
            "interface_validation": {},
            "model_validation": {},
            "overall_status": "success"
        }
        
        # Validate interfaces
        for interface in interfaces:
            result = interface.validate_constraints()
            summary["interface_validation"][interface.name] = {
                "success": result.success if DATAFLOW_AVAILABLE else result.get("success", True),
                "error_count": len(result.errors) if DATAFLOW_AVAILABLE else len(result.get("errors", [])),
                "warning_count": len(result.warnings) if DATAFLOW_AVAILABLE else len(result.get("warnings", []))
            }
            
            if not (result.success if DATAFLOW_AVAILABLE else result.get("success", True)):
                summary["overall_status"] = "validation_errors"
        
        # Validate model
        if model:
            try:
                model_result = model.validate_mathematical_constraints()
                summary["model_validation"] = {
                    "success": model_result.success if DATAFLOW_AVAILABLE else model_result.get("success", True),
                    "error_count": len(model_result.errors) if DATAFLOW_AVAILABLE else len(model_result.get("errors", [])),
                    "warning_count": len(model_result.warnings) if DATAFLOW_AVAILABLE else len(model_result.get("warnings", []))
                }
                
                if not (model_result.success if DATAFLOW_AVAILABLE else model_result.get("success", True)):
                    summary["overall_status"] = "validation_errors"
            except Exception as e:
                summary["model_validation"] = {"error": str(e)}
                summary["overall_status"] = "validation_failed"
        
        return summary
    
    def _extract_finn_information(self, finn_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract FINN-specific information from configuration."""
        return {
            "op_type": finn_config.get("function", "HWCustomOp"),
            "supports_batching": finn_config.get("supports_batching", True),
            "supported_datatypes": finn_config.get("datatypes", ["UINT8", "INT8", "UINT16", "INT16"]),
            "parallelism_params": finn_config.get("parallelism", {})
        }
    
    def _extract_backend_information(self, backend_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract RTL backend-specific information."""
        return {
            "backend_type": backend_config.get("type", "RTLBackend"),
            "synthesis_config": backend_config.get("synthesis", {}),
            "generate_wrapper": backend_config.get("wrapper", True),
            "wrapper_name": backend_config.get("wrapper_name"),
            "clock_domains": backend_config.get("clocks", {})
        }


# Factory functions for convenience
def create_template_context_builder(config: Optional[PipelineConfig] = None) -> EnhancedTemplateContextBuilder:
    """Create a template context builder with the given configuration."""
    return EnhancedTemplateContextBuilder(config)


def build_hwcustomop_context(
    hw_kernel: Any,
    config: Optional[PipelineConfig] = None,
    finn_config: Optional[Dict] = None,
    onnx_metadata: Optional[Dict] = None
) -> HWCustomOpContext:
    """Convenience function to build HWCustomOp context."""
    builder = create_template_context_builder(config)
    return builder.build_hwcustomop_context(hw_kernel, config or PipelineConfig(), finn_config, onnx_metadata)


def build_rtlbackend_context(
    hw_kernel: Any,
    config: Optional[PipelineConfig] = None,
    backend_config: Optional[Dict] = None,
    onnx_metadata: Optional[Dict] = None
) -> RTLBackendContext:
    """Convenience function to build RTLBackend context."""
    builder = create_template_context_builder(config)
    return builder.build_rtlbackend_context(hw_kernel, config or PipelineConfig(), backend_config, onnx_metadata)