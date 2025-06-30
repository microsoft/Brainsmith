"""
Unified Plugin System

The single, canonical plugin registration system for BrainSmith.
No compatibility layers, no fallbacks - just one clean approach.
"""

import logging
from typing import Dict, Type, Optional, List, Any
from threading import Lock
from qonnx.transformation.registry import register_transformation
from qonnx.custom_op.registry import register_op

logger = logging.getLogger(__name__)


class UnifiedRegistry:
    """The only plugin registry. No fallbacks, no compatibility."""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """Thread-safe singleton."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize if not already done."""
        if self._initialized:
            return
            
        self._plugins = {}  # type:name -> {class, metadata}
        self._initialized = True
        logger.info("UnifiedRegistry initialized")
    
    def register(self, plugin_type: str, name: str, cls: Type, **metadata):
        """
        Register a plugin. Overwrites any existing registration.
        
        Args:
            plugin_type: Type of plugin (transform, kernel, backend, inference)
            name: Unique name within type
            cls: The plugin class
            **metadata: Additional metadata (stage, kernel, backend_type, etc.)
        """
        key = f"{plugin_type}:{name}"
        
        self._plugins[key] = {
            "class": cls,
            "type": plugin_type,
            "name": name,
            **metadata
        }
        
        logger.debug(f"Registered {key}")
    
    def get(self, plugin_type: str, name: str) -> Optional[Type]:
        """Get plugin by type and name."""
        key = f"{plugin_type}:{name}"
        entry = self._plugins.get(key)
        return entry["class"] if entry else None
    
    def query(self, **filters) -> List[Dict[str, Any]]:
        """
        Query plugins by any metadata field.
        
        Args:
            **filters: Field=value pairs to match
            
        Returns:
            List of matching plugin entries
        """
        results = []
        
        for key, entry in self._plugins.items():
            # Check if all filters match
            match = True
            for field, value in filters.items():
                if field in ["type", "name"]:
                    # These are in the main entry
                    if entry.get(field) != value:
                        match = False
                        break
                else:
                    # Check in metadata
                    if entry.get(field) != value:
                        match = False
                        break
            
            if match:
                results.append(entry.copy())
        
        return results
    
    def get_with_conflict_detection(self, plugin_type: str, name: str) -> Optional[Type]:
        """
        Get plugin by name with intelligent conflict detection.
        
        For unprefixed names, returns the plugin if unique, or raises AmbiguousTransformError
        if multiple frameworks have plugins with the same name.
        
        Args:
            plugin_type: Type of plugin (e.g., "transform")
            name: Plugin name (with or without framework prefix)
            
        Returns:
            Plugin class if found and unique
            
        Raises:
            AmbiguousTransformError: If unprefixed name matches multiple frameworks
        """
        # If name has prefix, use direct lookup
        if ":" in name:
            return self.get(plugin_type, name)
        
        # Find all matches for unprefixed name
        matches = self._find_unprefixed_matches(plugin_type, name)
        
        if len(matches) == 0:
            return None
        elif len(matches) == 1:
            return matches[0]["class"]
        else:
            # Multiple matches - require disambiguation
            frameworks = sorted(set(m.get("framework", "unknown") for m in matches))
            from brainsmith.steps.transform_resolver import AmbiguousTransformError
            raise AmbiguousTransformError(
                f"Transform '{name}' found in multiple frameworks: {frameworks}. "
                f"Use explicit prefix like {', '.join(f'{fw}:{name}' for fw in frameworks)}"
            )
    
    def _find_unprefixed_matches(self, plugin_type: str, name: str) -> List[Dict[str, Any]]:
        """
        Find all plugins matching an unprefixed name.
        
        Args:
            plugin_type: Type of plugin to search for
            name: Unprefixed name to match
            
        Returns:
            List of matching plugin entries
        """
        matches = []
        
        for key, entry in self._plugins.items():
            if entry.get("type") == plugin_type:
                # Extract name without prefix
                stored_name = entry.get("name", "")
                if ":" in stored_name:
                    unprefixed = stored_name.split(":", 1)[1]
                else:
                    unprefixed = stored_name
                
                if unprefixed == name:
                    matches.append(entry)
        
        return matches
    
    def analyze_conflicts(self, plugin_type: str = "transform") -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze naming conflicts for a plugin type.
        
        Args:
            plugin_type: Type of plugin to analyze
            
        Returns:
            Dict mapping conflicted names to lists of conflicting entries
        """
        # Group by unprefixed name
        by_name = {}
        
        for key, entry in self._plugins.items():
            if entry.get("type") == plugin_type:
                stored_name = entry.get("name", "")
                if ":" in stored_name:
                    unprefixed = stored_name.split(":", 1)[1]
                else:
                    unprefixed = stored_name
                
                if unprefixed not in by_name:
                    by_name[unprefixed] = []
                by_name[unprefixed].append(entry)
        
        # Return only conflicts (multiple entries per name)
        return {name: entries for name, entries in by_name.items() if len(entries) > 1}
    
    def clear(self):
        """Clear registry (mainly for testing)."""
        self._plugins.clear()
        logger.info("Registry cleared")


# Global instance
_registry = None


def get_registry() -> UnifiedRegistry:
    """Get the global registry instance."""
    global _registry
    if _registry is None:
        _registry = UnifiedRegistry()
    return _registry


# Plugin types and their requirements
PLUGIN_TYPES = {
    "transform": {
        "required": ["stage"],
        "stages": ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt"]
    },
    "kernel": {
        "required": [],
        "optional": ["op_type", "domain"]
    },
    "backend": {
        "required": ["kernel", "backend_type"],
        "backend_types": ["hls", "rtl"]
    },
    "kernel_inference": {
        "required": ["kernel"],
        # stage is always None for kernel inference
    }
}


def transform(name: str, stage: Optional[str] = None, kernel: Optional[str] = None, 
              description: Optional[str] = None, author: Optional[str] = None,
              version: Optional[str] = None, **kwargs):
    """
    Decorator for transforms.
    
    Args:
        name: Transform name (required)
        stage: Compilation stage (mutually exclusive with kernel)
        kernel: Target kernel name (mutually exclusive with stage)
        description: Human-readable description
        author: Author name or organization
        version: Version string (e.g., "1.0.0")
        **kwargs: Additional metadata
        
    Note:
        - If 'kernel' is specified, registers as a kernel inference transform
        - If 'stage' is specified, registers as a regular transform
        - Exactly one of 'kernel' or 'stage' must be provided (mutually exclusive)
        
    Example:
        # Regular transform
        @transform(name="ExpandNorms", stage="topology_opt")
        class ExpandNorms(Transformation):
            pass
            
        # Kernel inference transform
        @transform(name="InferLayerNorm", kernel="LayerNorm")
        class InferLayerNorm(Transformation):
            pass
    """
    def decorator(cls):
        # Validate mutual exclusion
        if kernel and stage:
            raise ValueError(
                f"Transform '{name}' cannot specify both 'kernel' and 'stage'. "
                f"Use 'kernel' for kernel inference transforms or 'stage' for regular transforms."
            )
        
        if not kernel and not stage:
            raise ValueError(
                f"Transform '{name}' must specify either 'kernel' or 'stage'. "
                f"Use 'kernel' for kernel inference transforms or 'stage' for regular transforms."
            )
        
        # Determine plugin type based on presence of kernel
        if kernel:
            plugin_type = "kernel_inference"
            metadata = {
                "kernel": kernel,
                "stage": None,
                "framework": kwargs.get("framework", "brainsmith"),
                "description": description,
                "author": author,
                "version": version,
                **kwargs
            }
        else:
            plugin_type = "transform"
            
            # Validate stage
            valid_stages = PLUGIN_TYPES["transform"]["stages"]
            if stage not in valid_stages:
                logger.warning(
                    f"Transform '{name}' uses non-standard stage '{stage}'. "
                    f"Standard stages are: {valid_stages}"
                )
            
            metadata = {
                "stage": stage,
                "framework": kwargs.get("framework", "brainsmith"),
                "description": description,
                "author": author,
                "version": version,
                **kwargs
            }
        
        # Store metadata on class
        cls._plugin_metadata = {
            "type": plugin_type,
            "name": name,
            **metadata
        }
        
        # Register with unified registry
        registry = get_registry()
        registry.register(plugin_type, name, cls, **metadata)
        
        # Also register with QONNX transformation registry
        # Filter out BrainSmith-specific metadata and those we pass explicitly
        qonnx_kwargs = {k: v for k, v in kwargs.items() 
                        if k not in ['stage', 'kernel', 'framework', 'tags']}
        
        qonnx_decorator = register_transformation(
            name=name,
            description=description,
            tags=kwargs.get('tags', []),
            author=author,
            version=version,
            **qonnx_kwargs
        )
        cls = qonnx_decorator(cls)
        logger.debug(f"Registered transform with QONNX: {name}")
        
        return cls
    
    return decorator


def kernel(name: str, op_type: Optional[str] = None, domain: Optional[str] = None,
           description: Optional[str] = None, author: Optional[str] = None,
           version: Optional[str] = None, **kwargs):
    """
    Decorator for kernels.
    
    Args:
        name: Kernel name (required)
        op_type: ONNX operation type
        domain: ONNX domain for custom ops
        description: Human-readable description
        author: Author name or organization
        version: Version string (e.g., "1.0.0")
        **kwargs: Additional metadata
        
    Example:
        @kernel(name="LayerNorm", op_type="LayerNorm")
        class LayerNorm(HWCustomOp):
            pass
    """
    def decorator(cls):
        metadata = {
            "framework": kwargs.get("framework", "brainsmith"),
            "description": description,
            "author": author,
            "version": version,
            **kwargs
        }
        if op_type:
            metadata["op_type"] = op_type
        if domain:
            metadata["domain"] = domain
        
        # Store metadata on class
        cls._plugin_metadata = {
            "type": "kernel",
            "name": name,
            **metadata
        }
        
        # Register with unified registry
        registry = get_registry()
        try:
            registry.register("kernel", name, cls, **metadata)
        except ValueError as e:
            logger.debug(f"Plugin registration skipped: {e}")
        
        # Also register with QONNX custom op registry
        op_domain = domain or f"brainsmith.kernels.{name.lower()}"
        op_type_name = op_type or name
        
        cls = register_op(op_domain, op_type_name)(cls)
        logger.debug(f"Registered kernel with QONNX: {op_type_name} in domain {op_domain}")
        
        return cls
    
    return decorator


def backend(name: str, kernel: str, backend_type: str,
            description: Optional[str] = None, author: Optional[str] = None,
            version: Optional[str] = None, **kwargs):
    """
    Decorator for backends.
    
    Args:
        name: Backend name (required)
        kernel: Kernel this implements (required)
        backend_type: Either "hls" or "rtl" (required)
        description: Human-readable description
        author: Author name or organization
        version: Version string (e.g., "1.0.0")
        **kwargs: Additional metadata
        
    Example:
        @backend(name="LayerNormHLS", kernel="LayerNorm", backend_type="hls")
        class LayerNormHLS(LayerNorm, HLSBackend):
            pass
    """
    def decorator(cls):
        # Validate backend type
        valid_types = PLUGIN_TYPES["backend"]["backend_types"]
        if backend_type not in valid_types:
            raise ValueError(f"Invalid backend_type '{backend_type}'. Must be one of: {valid_types}")
        
        metadata = {
            "kernel": kernel,
            "backend_type": backend_type,
            "framework": kwargs.get("framework", "brainsmith"),
            "description": description,
            "author": author,
            "version": version,
            **kwargs
        }
        
        # Store metadata on class
        cls._plugin_metadata = {
            "type": "backend",
            "name": name,
            **metadata
        }
        
        # Register with unified registry
        registry = get_registry()
        try:
            registry.register("backend", name, cls, **metadata)
        except ValueError as e:
            logger.debug(f"Plugin registration skipped: {e}")
        
        # Also register with QONNX custom op registry
        # Backends use same domain/op_type as their parent kernel
        kernel_cls = registry.get("kernel", kernel)
        if kernel_cls and hasattr(kernel_cls, '_plugin_metadata'):
            kernel_meta = kernel_cls._plugin_metadata
            op_domain = kernel_meta.get('domain') or f"brainsmith.kernels.{kernel.lower()}"
            op_type_name = kernel_meta.get('op_type') or kernel
        else:
            # Fallback if kernel not found
            op_domain = f"brainsmith.kernels.{kernel.lower()}"
            op_type_name = kernel
        
        cls = register_op(op_domain, op_type_name)(cls)
        logger.debug(f"Registered backend with QONNX: {op_type_name} in domain {op_domain}")
        
        return cls
    
    return decorator