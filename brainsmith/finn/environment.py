"""
FINN Environment Management
Automatic FINN installation discovery, version management, and environment validation.
"""

import os
import sys
import json
import logging
import subprocess
import tempfile
import shutil
import requests
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import re
import pkg_resources
from packaging import version

logger = logging.getLogger(__name__)


@dataclass
class FINNInstallation:
    """Represents a FINN installation."""
    path: str
    version: str
    type: str  # 'git', 'pip', 'docker', 'conda'
    is_valid: bool = False
    python_path: Optional[str] = None
    dependencies: Dict[str, str] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    last_validated: Optional[float] = None
    validation_issues: List[str] = field(default_factory=list)


@dataclass
class FINNEnvironmentInfo:
    """Complete FINN environment information."""
    installations: List[FINNInstallation] = field(default_factory=list)
    active_installation: Optional[FINNInstallation] = None
    system_info: Dict[str, Any] = field(default_factory=dict)
    docker_available: bool = False
    conda_available: bool = False


class FINNInstallationRegistry:
    """Registry of discovered FINN installations."""
    
    def __init__(self, registry_file: Optional[str] = None):
        self.registry_file = registry_file or os.path.expanduser("~/.brainsmith/finn_registry.json")
        self.installations = {}
        self.load_registry()
    
    def add_installation(self, installation: FINNInstallation):
        """Add FINN installation to registry."""
        self.installations[installation.path] = installation
        self.save_registry()
        logger.info(f"Added FINN installation: {installation.path} (v{installation.version})")
    
    def remove_installation(self, path: str):
        """Remove installation from registry."""
        if path in self.installations:
            del self.installations[path]
            self.save_registry()
            logger.info(f"Removed FINN installation: {path}")
    
    def get_installation(self, path: str) -> Optional[FINNInstallation]:
        """Get installation by path."""
        return self.installations.get(path)
    
    def list_installations(self) -> List[FINNInstallation]:
        """List all registered installations."""
        return list(self.installations.values())
    
    def get_valid_installations(self) -> List[FINNInstallation]:
        """Get only valid installations."""
        return [inst for inst in self.installations.values() if inst.is_valid]
    
    def get_latest_installation(self) -> Optional[FINNInstallation]:
        """Get installation with latest version."""
        valid_installations = self.get_valid_installations()
        if not valid_installations:
            return None
        
        return max(valid_installations, key=lambda x: version.parse(x.version))
    
    def save_registry(self):
        """Save registry to file."""
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        
        registry_data = {
            'installations': {
                path: {
                    'path': inst.path,
                    'version': inst.version,
                    'type': inst.type,
                    'is_valid': inst.is_valid,
                    'python_path': inst.python_path,
                    'dependencies': inst.dependencies,
                    'capabilities': inst.capabilities,
                    'last_validated': inst.last_validated,
                    'validation_issues': inst.validation_issues
                }
                for path, inst in self.installations.items()
            }
        }
        
        with open(self.registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def load_registry(self):
        """Load registry from file."""
        if not os.path.exists(self.registry_file):
            return
        
        try:
            with open(self.registry_file, 'r') as f:
                registry_data = json.load(f)
            
            for path, data in registry_data.get('installations', {}).items():
                installation = FINNInstallation(**data)
                self.installations[path] = installation
                
        except Exception as e:
            logger.warning(f"Failed to load FINN registry: {e}")


class FINNVersionManager:
    """Manage FINN versions and installations."""
    
    def __init__(self):
        self.finn_repo_url = "https://github.com/Xilinx/finn.git"
        self.finn_releases_api = "https://api.github.com/repos/Xilinx/finn/releases"
        self.supported_versions = ["0.8.1", "0.9.0", "0.10.0", "main"]
    
    def get_available_versions(self) -> List[str]:
        """Get available FINN versions from GitHub."""
        try:
            response = requests.get(self.finn_releases_api, timeout=10)
            response.raise_for_status()
            
            releases = response.json()
            versions = [release['tag_name'] for release in releases]
            
            # Add main branch
            versions.insert(0, "main")
            
            return versions
            
        except Exception as e:
            logger.warning(f"Failed to fetch FINN versions: {e}")
            return self.supported_versions
    
    def install_finn_version(self, version: str, install_path: str, method: str = "git") -> bool:
        """Install specific FINN version."""
        logger.info(f"Installing FINN {version} to {install_path} using {method}")
        
        os.makedirs(install_path, exist_ok=True)
        
        try:
            if method == "git":
                return self._install_via_git(version, install_path)
            elif method == "pip":
                return self._install_via_pip(version, install_path)
            elif method == "conda":
                return self._install_via_conda(version, install_path)
            else:
                logger.error(f"Unsupported installation method: {method}")
                return False
        
        except Exception as e:
            logger.error(f"FINN installation failed: {e}")
            return False
    
    def _install_via_git(self, version: str, install_path: str) -> bool:
        """Install FINN via git clone."""
        try:
            # Clone repository
            clone_cmd = ["git", "clone", self.finn_repo_url, install_path]
            result = subprocess.run(clone_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Git clone failed: {result.stderr}")
                return False
            
            # Checkout specific version if not main
            if version != "main":
                checkout_cmd = ["git", "checkout", version]
                result = subprocess.run(checkout_cmd, cwd=install_path, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Git checkout failed: {result.stderr}")
                    return False
            
            # Install dependencies
            return self._install_dependencies(install_path)
            
        except Exception as e:
            logger.error(f"Git installation failed: {e}")
            return False
    
    def _install_via_pip(self, version: str, install_path: str) -> bool:
        """Install FINN via pip."""
        try:
            # Create virtual environment
            venv_path = os.path.join(install_path, "venv")
            subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
            
            # Get pip executable
            if os.name == 'nt':  # Windows
                pip_exe = os.path.join(venv_path, "Scripts", "pip")
            else:  # Unix-like
                pip_exe = os.path.join(venv_path, "bin", "pip")
            
            # Install FINN
            if version == "main":
                install_cmd = [pip_exe, "install", f"git+{self.finn_repo_url}"]
            else:
                install_cmd = [pip_exe, "install", f"git+{self.finn_repo_url}@{version}"]
            
            result = subprocess.run(install_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Pip install failed: {result.stderr}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Pip installation failed: {e}")
            return False
    
    def _install_via_conda(self, version: str, install_path: str) -> bool:
        """Install FINN via conda."""
        try:
            # Create conda environment
            env_name = f"finn_{version.replace('.', '_')}"
            create_cmd = ["conda", "create", "-n", env_name, "python=3.8", "-y"]
            result = subprocess.run(create_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Conda environment creation failed: {result.stderr}")
                return False
            
            # Install FINN in conda environment
            if version == "main":
                install_cmd = ["conda", "run", "-n", env_name, "pip", "install", f"git+{self.finn_repo_url}"]
            else:
                install_cmd = ["conda", "run", "-n", env_name, "pip", "install", f"git+{self.finn_repo_url}@{version}"]
            
            result = subprocess.run(install_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Conda FINN install failed: {result.stderr}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Conda installation failed: {e}")
            return False
    
    def _install_dependencies(self, install_path: str) -> bool:
        """Install FINN dependencies."""
        try:
            requirements_file = os.path.join(install_path, "requirements.txt")
            if os.path.exists(requirements_file):
                install_cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_file]
                result = subprocess.run(install_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.warning(f"Some dependencies failed to install: {result.stderr}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Dependency installation failed: {e}")
            return False


class FINNDependencyResolver:
    """Resolve and validate FINN dependencies."""
    
    def __init__(self):
        self.required_packages = {
            'onnx': '>=1.10.0',
            'numpy': '>=1.19.0',
            'torch': '>=1.7.0',
            'qonnx': '>=0.0.1',
            'brevitas': '>=0.7.0',
            'matplotlib': '>=3.3.0',
            'jupyter': '>=1.0.0'
        }
        
        self.optional_packages = {
            'vivado': None,  # External tool
            'vitis_hls': None,  # External tool
            'docker': '>=20.0.0',
            'verilator': '>=4.0.0'
        }
    
    def check_dependencies(self, installation_path: str) -> Tuple[Dict[str, str], List[str]]:
        """Check dependencies for FINN installation."""
        found_packages = {}
        missing_packages = []
        
        # Check Python packages
        for package, version_req in self.required_packages.items():
            try:
                installed_version = pkg_resources.get_distribution(package).version
                found_packages[package] = installed_version
                
                if version_req and not self._version_satisfies(installed_version, version_req):
                    missing_packages.append(f"{package} {version_req} (found {installed_version})")
                    
            except pkg_resources.DistributionNotFound:
                missing_packages.append(f"{package} {version_req or ''}")
        
        # Check optional packages
        for package, version_req in self.optional_packages.items():
            try:
                if package in ['vivado', 'vitis_hls']:
                    # Check for Xilinx tools in PATH
                    if shutil.which(package):
                        found_packages[package] = "detected"
                elif package == 'docker':
                    # Check docker
                    result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
                    if result.returncode == 0:
                        version_match = re.search(r'Docker version (\d+\.\d+\.\d+)', result.stdout)
                        if version_match:
                            found_packages[package] = version_match.group(1)
                elif package == 'verilator':
                    # Check verilator
                    result = subprocess.run(['verilator', '--version'], capture_output=True, text=True)
                    if result.returncode == 0:
                        version_match = re.search(r'Verilator (\d+\.\d+)', result.stdout)
                        if version_match:
                            found_packages[package] = version_match.group(1)
                else:
                    installed_version = pkg_resources.get_distribution(package).version
                    found_packages[package] = installed_version
                    
            except (pkg_resources.DistributionNotFound, subprocess.CalledProcessError, FileNotFoundError):
                # Optional packages - don't add to missing
                pass
        
        return found_packages, missing_packages
    
    def _version_satisfies(self, installed: str, requirement: str) -> bool:
        """Check if installed version satisfies requirement."""
        try:
            from packaging.specifiers import SpecifierSet
            spec = SpecifierSet(requirement)
            return version.parse(installed) in spec
        except Exception:
            return True  # If we can't parse, assume it's okay


class FINNEnvironmentManager:
    """Main FINN environment management class."""
    
    def __init__(self):
        self.installation_registry = FINNInstallationRegistry()
        self.version_manager = FINNVersionManager()
        self.dependency_resolver = FINNDependencyResolver()
        self.environment_info = FINNEnvironmentInfo()
        
        # Discover existing installations on init
        self.discover_finn_installations()
    
    def discover_finn_installations(self) -> List[FINNInstallation]:
        """Discover available FINN installations."""
        logger.info("Discovering FINN installations...")
        
        discovered = []
        
        # Check common installation locations
        search_paths = [
            os.path.expanduser("~/finn"),
            os.path.expanduser("~/FINN"),
            "/opt/finn",
            "/usr/local/finn",
            "./finn",
            "../finn"
        ]
        
        # Add paths from environment variables
        if 'FINN_ROOT' in os.environ:
            search_paths.insert(0, os.environ['FINN_ROOT'])
        
        if 'PYTHONPATH' in os.environ:
            for path in os.environ['PYTHONPATH'].split(os.pathsep):
                if 'finn' in path.lower():
                    search_paths.append(path)
        
        # Search for installations
        for search_path in search_paths:
            if os.path.exists(search_path):
                installation = self._analyze_installation(search_path)
                if installation:
                    discovered.append(installation)
                    self.installation_registry.add_installation(installation)
        
        # Check pip installations
        try:
            import finn
            pip_installation = self._analyze_pip_installation()
            if pip_installation:
                discovered.append(pip_installation)
                self.installation_registry.add_installation(pip_installation)
        except ImportError:
            pass
        
        # Update environment info
        self.environment_info.installations = discovered
        if discovered:
            self.environment_info.active_installation = self.installation_registry.get_latest_installation()
        
        logger.info(f"Discovered {len(discovered)} FINN installations")
        return discovered
    
    def install_finn_version(self, version: str, install_path: str, method: str = "git") -> bool:
        """Install specific FINN version."""
        success = self.version_manager.install_finn_version(version, install_path, method)
        
        if success:
            # Re-discover installations to include the new one
            self.discover_finn_installations()
        
        return success
    
    def validate_finn_environment(self, finn_path: str) -> Tuple[bool, List[str]]:
        """Validate FINN installation completeness."""
        issues = []
        
        # Check if path exists
        if not os.path.exists(finn_path):
            return False, ["FINN path does not exist"]
        
        # Check for required directories and files
        required_items = [
            'src/finn',
            'notebooks',
            'requirements.txt'
        ]
        
        for item in required_items:
            item_path = os.path.join(finn_path, item)
            if not os.path.exists(item_path):
                issues.append(f"Missing required item: {item}")
        
        # Check dependencies
        dependencies, missing = self.dependency_resolver.check_dependencies(finn_path)
        issues.extend([f"Missing dependency: {dep}" for dep in missing])
        
        # Check if FINN can be imported
        try:
            old_path = sys.path.copy()
            sys.path.insert(0, os.path.join(finn_path, 'src'))
            import finn
            sys.path = old_path
        except ImportError as e:
            issues.append(f"Cannot import FINN: {e}")
        
        return len(issues) == 0, issues
    
    def get_environment_info(self) -> FINNEnvironmentInfo:
        """Get complete FINN environment information."""
        # Update system info
        self.environment_info.system_info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'python_executable': sys.executable,
            'pythonpath': os.environ.get('PYTHONPATH', ''),
        }
        
        # Check Docker availability
        try:
            subprocess.run(['docker', '--version'], capture_output=True, check=True)
            self.environment_info.docker_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.environment_info.docker_available = False
        
        # Check Conda availability
        try:
            subprocess.run(['conda', '--version'], capture_output=True, check=True)
            self.environment_info.conda_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.environment_info.conda_available = False
        
        return self.environment_info
    
    def set_active_installation(self, finn_path: str) -> bool:
        """Set active FINN installation."""
        installation = self.installation_registry.get_installation(finn_path)
        if installation and installation.is_valid:
            self.environment_info.active_installation = installation
            os.environ['FINN_ROOT'] = finn_path
            return True
        
        return False
    
    def _analyze_installation(self, path: str) -> Optional[FINNInstallation]:
        """Analyze potential FINN installation."""
        try:
            # Determine installation type
            if os.path.exists(os.path.join(path, '.git')):
                install_type = 'git'
            elif os.path.exists(os.path.join(path, 'setup.py')):
                install_type = 'source'
            else:
                install_type = 'unknown'
            
            # Extract version
            version = self._extract_version(path)
            if not version:
                return None
            
            # Create installation object
            installation = FINNInstallation(
                path=path,
                version=version,
                type=install_type
            )
            
            # Validate installation
            is_valid, issues = self.validate_finn_environment(path)
            installation.is_valid = is_valid
            installation.validation_issues = issues
            
            # Get dependencies
            dependencies, _ = self.dependency_resolver.check_dependencies(path)
            installation.dependencies = dependencies
            
            # Determine capabilities
            installation.capabilities = self._determine_capabilities(path)
            
            return installation
            
        except Exception as e:
            logger.warning(f"Failed to analyze installation at {path}: {e}")
            return None
    
    def _analyze_pip_installation(self) -> Optional[FINNInstallation]:
        """Analyze pip-installed FINN."""
        try:
            import finn
            
            installation = FINNInstallation(
                path=os.path.dirname(finn.__file__),
                version=getattr(finn, '__version__', 'unknown'),
                type='pip',
                is_valid=True
            )
            
            # Get dependencies
            dependencies, _ = self.dependency_resolver.check_dependencies('')
            installation.dependencies = dependencies
            
            return installation
            
        except Exception as e:
            logger.warning(f"Failed to analyze pip installation: {e}")
            return None
    
    def _extract_version(self, path: str) -> Optional[str]:
        """Extract FINN version from installation."""
        # Try setup.py
        setup_py = os.path.join(path, 'setup.py')
        if os.path.exists(setup_py):
            try:
                with open(setup_py, 'r') as f:
                    content = f.read()
                    version_match = re.search(r'version\s*=\s*[\'"]([^\'"]+)[\'"]', content)
                    if version_match:
                        return version_match.group(1)
            except Exception:
                pass
        
        # Try version.py
        version_py = os.path.join(path, 'src', 'finn', 'version.py')
        if os.path.exists(version_py):
            try:
                with open(version_py, 'r') as f:
                    content = f.read()
                    version_match = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]', content)
                    if version_match:
                        return version_match.group(1)
            except Exception:
                pass
        
        # Try git tag if it's a git installation
        if os.path.exists(os.path.join(path, '.git')):
            try:
                result = subprocess.run(
                    ['git', 'describe', '--tags', '--abbrev=0'],
                    cwd=path,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            except Exception:
                pass
        
        return 'unknown'
    
    def _determine_capabilities(self, path: str) -> List[str]:
        """Determine capabilities of FINN installation."""
        capabilities = []
        
        # Check for specific modules/capabilities
        src_path = os.path.join(path, 'src', 'finn')
        if os.path.exists(src_path):
            capabilities.append('core')
            
            # Check for specific transformation capabilities
            transform_path = os.path.join(src_path, 'transformation')
            if os.path.exists(transform_path):
                capabilities.append('transformations')
                
                # Check for dataflow transformations
                dataflow_path = os.path.join(transform_path, 'fpgadataflow')
                if os.path.exists(dataflow_path):
                    capabilities.append('fpgadataflow')
            
            # Check for custom ops
            custom_op_path = os.path.join(src_path, 'custom_op')
            if os.path.exists(custom_op_path):
                capabilities.append('custom_ops')
        
        # Check for notebooks
        notebooks_path = os.path.join(path, 'notebooks')
        if os.path.exists(notebooks_path):
            capabilities.append('notebooks')
        
        # Check for build tools
        if shutil.which('vivado'):
            capabilities.append('vivado')
        
        if shutil.which('vitis_hls'):
            capabilities.append('vitis_hls')
        
        return capabilities