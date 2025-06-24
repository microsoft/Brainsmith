"""
Plugin Discovery

Automatic discovery and loading of plugins from various sources.
"""

import importlib
import importlib.util
import logging
import pkgutil
import sys
from pathlib import Path
from typing import List, Set, Optional

from .exceptions import PluginError

logger = logging.getLogger(__name__)


class PluginDiscovery:
    """
    Discover and load plugins from various sources.
    
    Plugins can be discovered from:
    - Built-in BrainSmith transforms/kernels
    - User plugin directory (~/.brainsmith/plugins)
    - Project plugin directory (./brainsmith_plugins)
    - Installed Python packages (brainsmith-plugin-*)
    """
    
    def __init__(self):
        """Initialize discovery with default search paths."""
        self.search_paths = self._get_default_search_paths()
        self._discovered_modules: Set[str] = set()
        
    def _get_default_search_paths(self) -> List[Path]:
        """Get default plugin search paths."""
        paths = []
        
        # User plugins
        user_plugin_dir = Path.home() / ".brainsmith" / "plugins"
        if user_plugin_dir.exists():
            paths.append(user_plugin_dir)
            
        # Project plugins
        project_plugin_dir = Path("./brainsmith_plugins")
        if project_plugin_dir.exists():
            paths.append(project_plugin_dir)
            
        return paths
    
    def add_search_path(self, path: Path) -> None:
        """
        Add a custom search path for plugins.
        
        Args:
            path: Directory path to search for plugins
        """
        path = Path(path)
        if path.exists() and path.is_dir():
            if path not in self.search_paths:
                self.search_paths.append(path)
                logger.info(f"Added plugin search path: {path}")
        else:
            logger.warning(f"Plugin search path does not exist: {path}")
    
    def discover_all(self) -> None:
        """Discover all plugins from all sources."""
        logger.info("Starting plugin discovery...")
        
        # Discover built-in plugins
        self._discover_builtin_transforms()
        self._discover_builtin_kernels()
        
        # Discover from search paths
        for search_path in self.search_paths:
            self._discover_directory(search_path)
        
        # Discover from installed packages
        self._discover_installed_packages()
        
        logger.info(f"Plugin discovery complete. Discovered {len(self._discovered_modules)} modules.")
    
    def _discover_builtin_transforms(self) -> None:
        """Discover built-in transforms from the new structure."""
        try:
            # Check if brainsmith.transforms exists
            import brainsmith.transforms
            base_package = brainsmith.transforms
            
            # Discover transforms in each stage subdirectory
            stages = [
                "graph_cleanup",
                "topology_optimization",
                "kernel_mapping", 
                "kernel_optimization",
                "graph_optimization"
            ]
            
            for stage in stages:
                try:
                    stage_module = importlib.import_module(f"brainsmith.transforms.{stage}")
                    self._discover_package_modules(stage_module, f"brainsmith.transforms.{stage}")
                except ImportError as e:
                    logger.debug(f"No transforms found for stage {stage}: {e}")
                    
        except ImportError:
            logger.debug("brainsmith.transforms not found, skipping built-in transform discovery")
    
    def _discover_builtin_kernels(self) -> None:
        """Discover built-in kernels from the new structure."""
        try:
            # Check if brainsmith.kernels exists
            import brainsmith.kernels
            base_path = Path(brainsmith.kernels.__file__).parent
            
            # Discover each kernel package
            for kernel_dir in base_path.iterdir():
                if kernel_dir.is_dir() and not kernel_dir.name.startswith('_'):
                    kernel_name = kernel_dir.name
                    
                    # Import the main kernel module
                    try:
                        kernel_module = importlib.import_module(f"brainsmith.kernels.{kernel_name}.{kernel_name}")
                        self._discovered_modules.add(f"brainsmith.kernels.{kernel_name}.{kernel_name}")
                        logger.debug(f"Discovered kernel module: {kernel_name}")
                    except ImportError as e:
                        logger.debug(f"No main kernel module for {kernel_name}: {e}")
                    
                    # Import backend modules
                    for py_file in kernel_dir.glob("*_hls.py"):
                        module_name = py_file.stem
                        try:
                            importlib.import_module(f"brainsmith.kernels.{kernel_name}.{module_name}")
                            self._discovered_modules.add(f"brainsmith.kernels.{kernel_name}.{module_name}")
                            logger.debug(f"Discovered backend module: {kernel_name}.{module_name}")
                        except ImportError as e:
                            logger.debug(f"Failed to import backend {module_name}: {e}")
                    
                    # Import RTL backends
                    for py_file in kernel_dir.glob("*_rtl.py"):
                        module_name = py_file.stem
                        try:
                            importlib.import_module(f"brainsmith.kernels.{kernel_name}.{module_name}")
                            self._discovered_modules.add(f"brainsmith.kernels.{kernel_name}.{module_name}")
                            logger.debug(f"Discovered backend module: {kernel_name}.{module_name}")
                        except ImportError as e:
                            logger.debug(f"Failed to import backend {module_name}: {e}")
                    
                    # Import optimization transforms
                    for py_file in kernel_dir.glob("optimize_*.py"):
                        module_name = py_file.stem
                        try:
                            importlib.import_module(f"brainsmith.kernels.{kernel_name}.{module_name}")
                            self._discovered_modules.add(f"brainsmith.kernels.{kernel_name}.{module_name}")
                            logger.debug(f"Discovered hw_transform module: {kernel_name}.{module_name}")
                        except ImportError as e:
                            logger.debug(f"Failed to import hw_transform {module_name}: {e}")
                            
        except ImportError:
            logger.debug("brainsmith.kernels not found, skipping built-in kernel discovery")
    
    def _discover_package_modules(self, package, package_name: str) -> None:
        """
        Discover all modules in a package.
        
        Args:
            package: Package object to scan
            package_name: Full package name for imports
        """
        if not hasattr(package, '__path__'):
            return
            
        for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
            if modname in self._discovered_modules:
                continue
                
            if not ispkg and not modname.endswith('__'):  # Skip __init__ and similar
                try:
                    module = importlib.import_module(modname)
                    self._discovered_modules.add(modname)
                    logger.debug(f"Discovered module: {modname}")
                except Exception as e:
                    logger.warning(f"Failed to import {modname}: {e}")
    
    def _discover_directory(self, directory: Path) -> None:
        """
        Discover plugins in a directory.
        
        Args:
            directory: Directory path to search
        """
        logger.debug(f"Discovering plugins in: {directory}")
        
        # Add directory to Python path temporarily
        str_path = str(directory.absolute())
        if str_path not in sys.path:
            sys.path.insert(0, str_path)
            
        try:
            # Find all Python files
            for py_file in directory.rglob("*.py"):
                if py_file.name.startswith('_'):
                    continue
                    
                # Calculate module name relative to directory
                relative_path = py_file.relative_to(directory)
                module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
                module_name = '.'.join(module_parts)
                
                if module_name in self._discovered_modules:
                    continue
                
                try:
                    # Load module using spec
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        self._discovered_modules.add(module_name)
                        logger.debug(f"Loaded plugin module: {module_name} from {py_file}")
                except Exception as e:
                    logger.warning(f"Failed to load plugin {py_file}: {e}")
                    
        finally:
            # Remove from path if we added it
            if str_path in sys.path:
                sys.path.remove(str_path)
    
    def _discover_installed_packages(self) -> None:
        """Discover plugins from installed Python packages."""
        # Look for packages with specific naming pattern
        for finder, name, ispkg in pkgutil.iter_modules():
            if name.startswith('brainsmith_plugin_') or name.startswith('brainsmith-plugin-'):
                if name in self._discovered_modules:
                    continue
                    
                try:
                    module = importlib.import_module(name)
                    self._discovered_modules.add(name)
                    logger.info(f"Discovered installed plugin package: {name}")
                    
                    # If it's a package, discover its submodules
                    if ispkg and hasattr(module, '__path__'):
                        self._discover_package_modules(module, name)
                        
                except Exception as e:
                    logger.warning(f"Failed to import plugin package {name}: {e}")
    
    def get_discovered_modules(self) -> List[str]:
        """
        Get list of all discovered module names.
        
        Returns:
            List of module names
        """
        return sorted(list(self._discovered_modules))
    
    @staticmethod
    def discover_and_register_all() -> None:
        """
        Convenience method to discover and register all plugins.
        
        This is the main entry point for plugin discovery.
        """
        discovery = PluginDiscovery()
        discovery.discover_all()
        
        # Log registry stats
        from .registry import PluginRegistry
        registry = PluginRegistry()
        stats = registry.get_stats()
        
        logger.info(
            f"Plugin registration complete: "
            f"{stats['transform']} transforms, "
            f"{stats['kernel']} kernels, "
            f"{stats['backend']} backends, "
            f"{stats['hw_transform']} hw_transforms"
        )