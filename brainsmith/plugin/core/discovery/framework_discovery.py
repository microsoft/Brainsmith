"""
Framework-Native Plugin Discovery

Discovers transforms from QONNX and FINN frameworks by directly
scanning their modules, without requiring registration.
"""

import logging
import inspect
import importlib
from typing import List, Dict, Any, Optional, Set

from .base import DiscoveryInterface
from ..data_models import PluginInfo

logger = logging.getLogger(__name__)


class FrameworkDiscoveryBase(DiscoveryInterface):
    """Base class for framework-specific discovery."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._discovered_classes: Set[type] = set()
    
    def _scan_module_for_transforms(self, module_name: str, 
                                   framework: str) -> List[PluginInfo]:
        """Scan a module for transformation classes."""
        plugins = []
        
        try:
            module = importlib.import_module(module_name)
            
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    not name.startswith('_') and
                    obj not in self._discovered_classes and
                    self._is_transform_class(obj, name)):
                    
                    self._discovered_classes.add(obj)
                    
                    plugin_info = PluginInfo(
                        name=name,
                        plugin_class=obj,
                        framework=framework,
                        plugin_type="transform",
                        metadata={
                            'module': module_name,
                            'docstring': (obj.__doc__ or "").strip()
                        },
                        discovery_method="framework_native"
                    )
                    
                    plugins.append(plugin_info)
                    logger.debug(f"Discovered {framework} transform: {name}")
                    
        except ImportError as e:
            logger.debug(f"Could not import {framework} module {module_name}: {e}")
        
        return plugins
    
    def _is_transform_class(self, obj, name: str) -> bool:
        """Check if object is a valid transform class."""
        # Skip base classes and non-transforms
        if name in ['Transformation', 'NodeLocalTransformation', 
                    'AnalysisTransformation', 'BatchedTransformation']:
            return False
        
        # Must be callable (have __call__ or apply method)
        if not (hasattr(obj, '__call__') or hasattr(obj, 'apply')):
            return False
        
        # Check if it's a transformation subclass
        try:
            # Try to check inheritance without importing base class
            mro_names = [c.__name__ for c in inspect.getmro(obj)]
            return 'Transformation' in mro_names
        except:
            # Fallback: check for apply method
            return hasattr(obj, 'apply') and callable(getattr(obj, 'apply', None))


class QONNXDiscovery(FrameworkDiscoveryBase):
    """Discovers QONNX transforms from their native modules."""
    
    # Default QONNX modules to scan
    DEFAULT_MODULES = [
        'qonnx.transformation.general',
        'qonnx.transformation.remove',
        'qonnx.transformation.fold_constants',
        'qonnx.transformation.infer_data_layouts',
        'qonnx.transformation.infer_datatypes', 
        'qonnx.transformation.infer_shapes',
        'qonnx.transformation.change_datalayout',
        'qonnx.transformation.double_to_single_float',
        'qonnx.transformation.extract_conv_bias',
        'qonnx.transformation.gemm_to_matmul',
        'qonnx.transformation.lower_convs_to_matmul',
        'qonnx.transformation.merge_onnx_models',
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.modules = self.config.get('modules', self.DEFAULT_MODULES)
    
    @property
    def name(self) -> str:
        return "QONNXDiscovery"
    
    def discover(self) -> List[PluginInfo]:
        """Discover QONNX transforms."""
        plugins = []
        
        for module_name in self.modules:
            module_plugins = self._scan_module_for_transforms(module_name, "qonnx")
            plugins.extend(module_plugins)
        
        self.log_discovery_summary(plugins)
        return plugins


class FINNDiscovery(FrameworkDiscoveryBase):
    """Discovers FINN transforms from their native modules."""
    
    # Default FINN modules to scan
    DEFAULT_MODULES = [
        'finn.transformation.streamline',
        'finn.transformation.streamline.absorb',
        'finn.transformation.streamline.collapse_repeated',
        'finn.transformation.streamline.reorder',
        'finn.transformation.streamline.round_thresholds',
        'finn.transformation.streamline.sign_to_thres',
        'finn.transformation.move_reshape',
        'finn.transformation.fpgadataflow.convert_to_hw_layers',
        'finn.transformation.fpgadataflow.create_dataflow_partition',
        'finn.transformation.fpgadataflow.create_stitched_ip',
        'finn.transformation.fpgadataflow.derive_characteristic',
        'finn.transformation.fpgadataflow.floorplan',
        'finn.transformation.fpgadataflow.hlssynth_ip',
        'finn.transformation.fpgadataflow.insert_dwc',
        'finn.transformation.fpgadataflow.insert_fifo',
        'finn.transformation.fpgadataflow.insert_iodma',
        'finn.transformation.fpgadataflow.insert_tlastmarker',
        'finn.transformation.fpgadataflow.minimize_accumulator_width',
        'finn.transformation.fpgadataflow.minimize_weight_bit_width',
        'finn.transformation.fpgadataflow.prepare_cppsim',
        'finn.transformation.fpgadataflow.prepare_ip',
        'finn.transformation.fpgadataflow.prepare_rtlsim',
        'finn.transformation.fpgadataflow.set_exec_mode',
        'finn.transformation.fpgadataflow.set_fifo_depths',
        'finn.transformation.fpgadataflow.set_folding',
        'finn.transformation.fpgadataflow.specialize_layers',
        'finn.transformation.fpgadataflow.synth_ooc',
        'finn.transformation.fpgadataflow.vitis_build',
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.modules = self.config.get('modules', self.DEFAULT_MODULES)
    
    @property 
    def name(self) -> str:
        return "FINNDiscovery"
    
    def discover(self) -> List[PluginInfo]:
        """Discover FINN transforms."""
        plugins = []
        
        for module_name in self.modules:
            module_plugins = self._scan_module_for_transforms(module_name, "finn")
            plugins.extend(module_plugins)
        
        self.log_discovery_summary(plugins)
        return plugins