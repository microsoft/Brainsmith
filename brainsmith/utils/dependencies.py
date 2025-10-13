# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Simple dependency management for non-Python dependencies."""

import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

# All non-Python dependencies
DEPENDENCIES = {
    # C++ Simulation Dependencies
    'cnpy': {
        'type': 'git',
        'url': 'https://github.com/maltanar/cnpy.git',
        'ref': '8c82362372ce600bbd1cf11d64661ab69d38d7de',
        'group': 'cppsim',
        'description': 'NumPy file I/O in C++',
        'requires': {
            'g++': 'C++ compiler - install with: sudo apt install g++'
        }
    },
    'finn-hlslib': {
        'type': 'git',
        'url': 'https://github.com/Xilinx/finn-hlslib.git',
        'ref': '5c5ad631e3602a8dd5bd3399a016477a407d6ee7',
        'group': 'cppsim',
        'description': 'Header-only HLS library'
    },
    
    # RTL Simulation Dependencies
    'oh-my-xilinx': {
        'type': 'git',
        'url': 'https://github.com/ddanag/oh-my-xilinx.git',
        'ref': '0b59762f9e4c4f7e5aa535ee9bc29f292434ca7a',
        'group': 'xsim',
        'description': 'Xilinx Vivado utility scripts for RTL compilation',
        'requires': {
            'vivado': 'Xilinx Vivado - set xilinx_path in config or BSMITH_XILINX_PATH env var'
        }
    },
    'finn-xsim': {
        'type': 'build',
        'source': 'finn',  # This refers to the poetry-installed finn package
        'build_cmd': ['python3', '-m', 'finn.xsi.setup'],
        'group': 'xsim',
        'description': 'FINN XSI module',
        'requires': {
            'vivado': 'Xilinx Vivado - set xilinx_path in config or BSMITH_XILINX_PATH env var'
        }
    },
    
    # Board Files
    'avnet-boards': {
        'type': 'git',
        'url': 'https://github.com/Avnet/bdf.git',
        'ref': '2d49cfc25766f07792c0b314489f21fe916b639b',
        'group': 'boards',
        'description': 'Avnet board files'
    },
    'xilinx-boards': {
        'type': 'git',
        'url': 'https://github.com/Xilinx/XilinxBoardStore.git',
        'ref': '8cf4bb674a919ac34e3d99d8d71a9e60af93d14e',
        'sparse_dirs': ['boards/Xilinx/rfsoc2x2'],
        'group': 'boards',
        'description': 'Xilinx board files (RFSoC2x2)'
    },
    'rfsoc4x2-boards': {
        'type': 'git',
        'url': 'https://github.com/RealDigitalOrg/RFSoC4x2-BSP.git',
        'ref': '13fb6f6c02c7dfd7e4b336b18b959ad5115db696',
        'sparse_dirs': ['board_files/rfsoc4x2'],
        'group': 'boards',
        'description': 'RFSoC 4x2 board files'
    },
    'kv260-som-boards': {
        'type': 'git',
        'url': 'https://github.com/Xilinx/XilinxBoardStore.git',
        'ref': '98e0d3efc901f0b974006bc4370c2a7ad8856c79',
        'sparse_dirs': ['boards/Xilinx/kv260_som'],
        'group': 'boards',
        'description': 'KV260 SOM board files'
    },
    'aupzu3-boards': {
        'type': 'git',
        'url': 'https://github.com/RealDigitalOrg/aup-zu3-bsp.git',
        'ref': 'b595ecdf37c7204129517de1773b0895bcdcc2ed',
        'sparse_dirs': ['board-files/aup-zu3-8gb'],
        'group': 'boards',
        'description': 'AUP ZU3 8GB board files'
    },
    'pynq-z1': {
        'type': 'zip',
        'url': 'https://github.com/cathalmccabe/pynq-z1_board_files/raw/master/pynq-z1.zip',
        'group': 'boards',
        'description': 'PYNQ-Z1 board files'
    },
    'pynq-z2': {
        'type': 'zip',
        'url': 'https://dpoauwgwqsy2x.cloudfront.net/Download/pynq-z2.zip',
        'group': 'boards',
        'description': 'PYNQ-Z2 board files'
    }
}


class DependencyManager:
    """Manages non-Python dependencies for Brainsmith."""
    
    def __init__(self, deps_dir: Optional[Path] = None):
        """Initialize with deps_dir from config if not provided."""
        if deps_dir is None:
            from brainsmith.config import get_config
            deps_dir = get_config().deps_dir
        self.deps_dir = Path(deps_dir)
        self.deps_dir.mkdir(parents=True, exist_ok=True)
        
    def install(self, name: str, force: bool = False, quiet: bool = False) -> bool:
        """Install a single dependency.
        
        Args:
            name: Dependency name from DEPENDENCIES
            force: Force reinstall even if exists
            quiet: Suppress output
            
        Returns:
            True if installation succeeded
        """
        if name not in DEPENDENCIES:
            raise ValueError(f"Unknown dependency: {name}")
            
        dep = DEPENDENCIES[name]
        
        # Check requirements first
        missing = self._check_requirements(dep)
        if missing:
            error_msg = f"Error: {name} requires the following tools to be installed:"
            print(error_msg, file=sys.stderr)
            logger.error(error_msg)
            for tool, message in missing:
                tool_error = f"  ✗ {tool} - {message}"
                print(tool_error, file=sys.stderr)
                logger.error(tool_error)
            return False
        
        # Handle build dependencies differently
        if dep['type'] == 'build':
            return self._install_build(name, dep, force, quiet)
            
        # For git/zip dependencies, check if already exists
        # Special handling for board files - they go in board-files subdirectory
        if dep.get('group') == 'boards':
            dest = self.deps_dir / 'board-files' / name
        else:
            dest = self.deps_dir / name
        
        if dest.exists() and not force:
            if not quiet:
                logger.info(f"{name} already installed at {dest}")
            return True
            
        # Remove existing if force
        if dest.exists():
            shutil.rmtree(dest)
            
        # Install based on type
        if dep['type'] == 'git':
            return self._install_git(name, dep, dest, quiet)
        elif dep['type'] == 'zip':
            return self._install_zip(name, dep, dest, quiet)
        else:
            raise ValueError(f"Unknown dependency type: {dep['type']}")
            
    def install_group(self, group: str, force: bool = False, quiet: bool = False) -> Dict[str, bool]:
        """Install all dependencies in a group.
        
        Args:
            group: Group name (cppsim, xsim, boards)
            force: Force reinstall
            quiet: Suppress output
            
        Returns:
            Dict mapping dependency name to success status
        """
        deps = [k for k, v in DEPENDENCIES.items() if v.get('group') == group]
        return {dep: self.install(dep, force, quiet) for dep in deps}
        
    def _check_requirements(self, dep: dict) -> List[Tuple[str, str]]:
        """Check if required tools are available.
        
        Args:
            dep: Dependency dictionary
            
        Returns:
            List of (tool, message) tuples for missing requirements
        """
        missing = []
        if 'requires' in dep:
            for tool, message in dep['requires'].items():
                # Special handling for Vivado - use config's robust detection
                if tool == 'vivado':
                    try:
                        from brainsmith.config import get_config
                        config = get_config()
                        if not config.effective_vivado_path:
                            missing.append((tool, message))
                    except ImportError:
                        # Fallback to simple PATH check if config not available
                        if not shutil.which(tool):
                            missing.append((tool, message))
                else:
                    # For other tools, check if they're on PATH
                    if not shutil.which(tool):
                        missing.append((tool, message))
        return missing
        
    def _install_git(self, name: str, dep: dict, dest: Path, quiet: bool) -> bool:
        """Clone a git repository."""
        cmd = ['git', 'clone']
        
        # Handle sparse checkout
        if 'sparse_dirs' in dep:
            cmd.extend(['--filter=blob:none', '--sparse'])
            
        cmd.extend([dep['url'], str(dest)])
        
        # Clone
        if not quiet:
            logger.info(f"Cloning {name} from {dep['url']}")
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = f"Failed to clone {name}: {result.stderr}"
            print(error_msg, file=sys.stderr)
            logger.error(error_msg)
            return False
            
        # Checkout specific ref
        if 'ref' in dep:
            # First fetch the ref (in case it's not a branch/tag)
            fetch_result = subprocess.run(
                ['git', '-C', str(dest), 'fetch', 'origin', dep['ref']],
                capture_output=True
            )
            # Now checkout
            result = subprocess.run(
                ['git', '-C', str(dest), 'checkout', dep['ref']], 
                capture_output=True
            )
            if result.returncode != 0:
                logger.error(f"Failed to checkout {dep['ref']} for {name}: {result.stderr.decode() if result.stderr else 'No error details'}")
                return False
                
        # Setup sparse checkout if needed
        if 'sparse_dirs' in dep:
            subprocess.run(['git', '-C', str(dest), 'sparse-checkout', 'init'])
            subprocess.run(['git', '-C', str(dest), 'sparse-checkout', 'set'] + dep['sparse_dirs'])
            
        return True
        
    def _install_zip(self, name: str, dep: dict, dest: Path, quiet: bool) -> bool:
        """Download and extract a zip file."""
        zip_path = self.deps_dir / f"{name}.zip"
        
        try:
            # Download
            if not quiet:
                logger.info(f"Downloading {name} from {dep['url']}")
                
            urlretrieve(dep['url'], zip_path)
            
            # Create destination directory
            dest.mkdir(exist_ok=True)
            
            # Extract
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
                
        except Exception as e:
            error_msg = f"Failed to download/extract {name}: {e}"
            print(error_msg, file=sys.stderr)
            logger.error(error_msg)
            return False
        finally:
            # Cleanup zip file
            if zip_path.exists():
                zip_path.unlink()
                
        return True
        
    def _install_build(self, name: str, dep: dict, force: bool, quiet: bool) -> bool:
        """Build a local dependency."""
        # Special handling for finn-xsim which uses poetry-installed finn
        if name == 'finn-xsim':
            # Check if FINN package is available
            try:
                import finn
                # Check if already built - finn-xsim builds to finn_xsi/xsi.so
                finn_root = Path(finn.__file__).parent.parent.parent  # Go up to deps/finn
                output_file = finn_root / 'finn_xsi' / 'xsi.so'
                
                if output_file.exists() and not force:
                    if not quiet:
                        logger.info(f"{name} already built at {output_file}")
                    return True
            except ImportError:
                logger.error("FINN package not found. Please run 'poetry install' first.")
                return False
                
            # Run build command - Python modules can be run from any directory
            if not quiet:
                logger.info(f"Building {name}...")

            # Get Vivado settings path from config
            from brainsmith.config import get_config
            config = get_config()
            vivado_path = config.effective_vivado_path

            if not vivado_path:
                error_msg = f"Failed to build {name}: Vivado path not configured"
                print(error_msg, file=sys.stderr)
                logger.error(error_msg)
                return False

            settings_script = Path(vivado_path) / "settings64.sh"
            if not settings_script.exists():
                error_msg = f"Failed to build {name}: settings64.sh not found at {settings_script}"
                print(error_msg, file=sys.stderr)
                logger.error(error_msg)
                return False

            # Construct build command
            build_cmd = dep['build_cmd'].copy()
            if force:
                build_cmd.append('--force')

            # Build bash command that sources Vivado settings before running build
            python_cmd = ' '.join(build_cmd)
            bash_cmd = f"source {settings_script} && {python_cmd}"

            # Run build with Vivado environment
            if quiet:
                # In quiet mode, capture output
                result = subprocess.run(
                    ['bash', '-c', bash_cmd],
                    capture_output=True,
                    text=True
                )
            else:
                # In non-quiet mode, stream output to terminal
                result = subprocess.run(
                    ['bash', '-c', bash_cmd],
                    text=True
                )

            # Always print errors to stderr for visibility
            if result.returncode != 0:
                error_msg = f"Failed to build {name}"
                if hasattr(result, 'stderr') and result.stderr:
                    error_msg += f": {result.stderr}"
                print(error_msg, file=sys.stderr)
                logger.error(error_msg)
                return False

            # Verify output file exists after build
            if not output_file.exists():
                error_msg = f"Build command completed but output file not found: {output_file}"
                print(error_msg, file=sys.stderr)
                logger.error(error_msg)
                return False

            return True
        else:
            # Generic build dependency handling (for future use)
            source_dir = self.deps_dir / dep['source']
            if not source_dir.exists():
                # Need to install source first
                if not self.install(dep['source'], force, quiet):
                    return False
                    
            # Run build command in source directory
            env = os.environ.copy()
            result = subprocess.run(
                dep['build_cmd'],
                cwd=source_dir,
                capture_output=True,
                text=True,
                env=env
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to build {name}: {result.stderr}")
                return False
                
            return True
    
    def remove(self, name: str, quiet: bool = False) -> bool:
        """Remove a single dependency.
        
        Args:
            name: Dependency name from DEPENDENCIES
            quiet: Suppress output
            
        Returns:
            True if removal succeeded or dependency wasn't installed
        """
        if name not in DEPENDENCIES:
            raise ValueError(f"Unknown dependency: {name}")
            
        dep = DEPENDENCIES[name]
        
        # Handle build dependencies differently
        if dep['type'] == 'build':
            return self._remove_build(name, dep, quiet)
            
        # For git/zip dependencies, check if exists
        if dep.get('group') == 'boards':
            dest = self.deps_dir / 'board-files' / name
        else:
            dest = self.deps_dir / name
        
        if not dest.exists():
            if not quiet:
                logger.info(f"{name} is not installed")
            return True
            
        # Remove directory
        try:
            if not quiet:
                logger.info(f"Removing {name} from {dest}")
            shutil.rmtree(dest)
            return True
        except Exception as e:
            logger.error(f"Failed to remove {name}: {e}")
            return False
            
    def remove_group(self, group: str, quiet: bool = False) -> Dict[str, bool]:
        """Remove all dependencies in a group.
        
        Args:
            group: Group name (cppsim, xsim, boards)
            quiet: Suppress output
            
        Returns:
            Dict mapping dependency name to success status
        """
        deps = [k for k, v in DEPENDENCIES.items() if v.get('group') == group]
        return {dep: self.remove(dep, quiet) for dep in deps}
        
    def _remove_build(self, name: str, dep: dict, quiet: bool) -> bool:
        """Remove a build dependency."""
        # Special handling for finn-xsim
        if name == 'finn-xsim':
            try:
                import finn
                finn_root = Path(finn.__file__).parent.parent.parent
                output_file = finn_root / 'finn_xsi' / 'xsi.so'
                
                if not output_file.exists():
                    if not quiet:
                        logger.info(f"{name} is not built")
                    return True
                    
                # Run clean command if available
                if not quiet:
                    logger.info(f"Cleaning {name}...")
                    
                clean_cmd = ['python3', '-m', 'finn.xsi.setup', '--clean']
                result = subprocess.run(
                    clean_cmd,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    # Try manual removal as fallback
                    try:
                        output_file.unlink()
                        return True
                    except Exception as e:
                        logger.error(f"Failed to remove {name}: {e}")
                        return False
                        
                return True
                
            except ImportError:
                if not quiet:
                    logger.info(f"{name} is not installed (FINN not found)")
                return True
        else:
            # Generic build dependency removal (for future use)
            if not quiet:
                logger.info(f"Build dependency {name} removal not implemented")
            return True


class BoardManager:
    """Manages FPGA board definition files."""
    
    def __init__(self, board_dir: Path):
        self.board_dir = Path(board_dir)
        
    def list_downloaded_repositories(self) -> List[str]:
        """List all downloaded board repositories."""
        if not self.board_dir.exists():
            return []
            
        # Board repos are direct subdirectories
        repos = []
        for item in self.board_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                repos.append(item.name)
                
        return sorted(repos)
        
    def get_board_summary(self) -> Dict[str, List[str]]:
        """Get summary of all boards organized by repository.
        
        Returns:
            Dict mapping repository name to list of board names
        """
        summary = {}
        
        for repo in self.list_downloaded_repositories():
            repo_path = self.board_dir / repo
            board_files = self.find_board_files(repo_path)
            boards = self.extract_board_names(board_files, repo)
            if boards:
                summary[repo] = boards
                
        return summary
        
    def find_board_files(self, repo_path: Path) -> List[Path]:
        """Find all board.xml files in a repository."""
        return list(repo_path.glob("**/board.xml"))
        
    def extract_board_names(self, board_files: List[Path], repo_name: str) -> List[str]:
        """Extract board names from board.xml files."""
        boards = []
        
        for board_file in board_files:
            # Try to parse board name from XML
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(board_file)
                root = tree.getroot()
                
                # Look for board name in attributes or elements
                board_name = root.get('name')
                if not board_name:
                    # Try to find name element
                    name_elem = root.find('.//name')
                    if name_elem is not None and name_elem.text:
                        board_name = name_elem.text
                        
                if board_name:
                    boards.append(board_name)
                    
            except Exception:
                # Fall back to directory name
                board_dir = board_file.parent
                if board_dir.name not in ['boards', repo_name]:
                    boards.append(board_dir.name)
                    
        return sorted(set(boards))
        
    def find_board_path(self, board_name: str) -> Optional[Path]:
        """Find the path to a specific board's files.
        
        Args:
            board_name: Name of the board to find
            
        Returns:
            Path to board directory, or None if not found
        """
        for board_file in self.board_dir.glob("**/board.xml"):
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(board_file)
                root = tree.getroot()
                
                # Check if this is the board we're looking for
                name = root.get('name')
                if not name:
                    name_elem = root.find('.//name')
                    if name_elem is not None:
                        name = name_elem.text
                        
                if name == board_name:
                    return board_file.parent
                    
            except Exception:
                # Check directory name as fallback
                if board_file.parent.name == board_name:
                    return board_file.parent
                    
        return None
        
    def validate_repository_names(self, names: List[str]) -> Tuple[List[str], List[str]]:
        """Validate repository names against known repositories.
        
        Args:
            names: List of repository names to validate
            
        Returns:
            Tuple of (valid_names, invalid_names)
        """
        # Known board repositories from DEPENDENCIES
        known_repos = {
            'avnet': 'avnet-boards',
            'xilinx': 'xilinx-boards',
            'xilinx-rfsoc2x2': 'xilinx-boards',
            'rfsoc4x2': 'rfsoc4x2-boards',
            'kv260': 'kv260-som-boards',
            'kv260-som': 'kv260-som-boards',
            'aupzu3': 'aupzu3-boards',
            'pynq-z1': 'pynq-z1',
            'pynq-z2': 'pynq-z2',
            'pynq': ['pynq-z1', 'pynq-z2'],  # Both PYNQ boards
        }
        
        valid = []
        invalid = []
        
        for name in names:
            if name in known_repos:
                mapped = known_repos[name]
                if isinstance(mapped, list):
                    valid.extend(mapped)
                else:
                    valid.append(mapped)
            elif name in DEPENDENCIES and DEPENDENCIES[name].get('group') == 'boards':
                valid.append(name)
            else:
                invalid.append(name)
                
        return list(set(valid)), invalid  # Remove duplicates from valid