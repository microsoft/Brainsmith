# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Simple dependency management for non-Python dependencies."""

import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .dependency_installers import (
    BuildDependencyInstaller,
    DependencyError,
    GitDependencyInstaller,
    InstallationError,
    RemovalError,
    RequirementError,
    UnknownDependencyError,
    ZipDependencyInstaller,
)

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
    """Manages non-Python dependencies for Brainsmith.

    This class orchestrates dependency installation by delegating to
    specialized installer classes for each dependency type (git, zip, build).
    """

    def __init__(self, deps_dir: Optional[Path] = None):
        """Initialize with deps_dir from config if not provided."""
        if deps_dir is None:
            from brainsmith.settings import get_config
            deps_dir = get_config().deps_dir
        self.deps_dir = Path(deps_dir)
        self.deps_dir.mkdir(parents=True, exist_ok=True)

        # Create installer instances
        self.installers = {
            'git': GitDependencyInstaller(),
            'zip': ZipDependencyInstaller(temp_dir=self.deps_dir),
            'build': BuildDependencyInstaller()
        }
        
    def install(self, name: str, force: bool = False, quiet: bool = False) -> None:
        """Install a single dependency.

        Args:
            name: Dependency name from DEPENDENCIES
            force: Force reinstall even if exists
            quiet: Suppress output

        Raises:
            UnknownDependencyError: If dependency name is unknown
            RequirementError: If required tools are missing
            InstallationError: If installation fails
        """
        if name not in DEPENDENCIES:
            raise UnknownDependencyError(
                f"Unknown dependency: {name}. "
                f"Available: {', '.join(sorted(DEPENDENCIES.keys()))}"
            )

        dep = DEPENDENCIES[name]

        # Check requirements first
        missing = self._check_requirements(dep)
        if missing:
            error_msg = f"Error: {name} requires the following tools to be installed:"
            logger.error("%s", error_msg)
            for tool, message in missing:
                logger.error("  ✗ %s - %s", tool, message)
            # Build error message for exception
            requirements_list = '\n'.join([f"  ✗ {tool} - {msg}" for tool, msg in missing])
            raise RequirementError(
                f"{error_msg}\n{requirements_list}"
            )

        # Determine destination path
        # Special handling for board files - they go in board-files subdirectory
        if dep.get('group') == 'boards':
            dest = self.deps_dir / 'board-files' / name
        else:
            dest = self.deps_dir / name

        # Get appropriate installer and delegate
        dep_type = dep['type']
        if dep_type not in self.installers:
            raise ValueError(f"Unknown dependency type: {dep_type}")

        installer = self.installers[dep_type]

        # Install the dependency (installer handles exists checks and force)
        installer.install(name, dep, dest, force, quiet)
            
    def install_group(self, group: str, force: bool = False, quiet: bool = False) -> Dict[str, Optional[Exception]]:
        """Install all dependencies in a group.

        Args:
            group: Group name (cppsim, xsim, boards)
            force: Force reinstall
            quiet: Suppress output

        Returns:
            Dict mapping dependency name to error (None if successful)
        """
        deps = [k for k, v in DEPENDENCIES.items() if v.get('group') == group]
        results = {}

        for dep in deps:
            try:
                self.install(dep, force, quiet)
                results[dep] = None  # Success
            except DependencyError as e:
                results[dep] = e  # Store error for reporting
                if not quiet:
                    logger.error("Failed to install %s: %s", dep, e)

        return results
        
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
                        from brainsmith.settings import get_config
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

    def remove(self, name: str, quiet: bool = False) -> None:
        """Remove a single dependency.

        Args:
            name: Dependency name from DEPENDENCIES
            quiet: Suppress output

        Raises:
            UnknownDependencyError: If dependency name is unknown
            RemovalError: If removal fails
        """
        if name not in DEPENDENCIES:
            raise UnknownDependencyError(f"Unknown dependency: {name}")

        dep = DEPENDENCIES[name]

        # Determine destination path
        if dep.get('group') == 'boards':
            dest = self.deps_dir / 'board-files' / name
        else:
            dest = self.deps_dir / name

        # Get appropriate installer and delegate removal
        dep_type = dep['type']
        if dep_type not in self.installers:
            raise ValueError(f"Unknown dependency type: {dep_type}")

        installer = self.installers[dep_type]
        installer.remove(name, dest, quiet)
            
    def remove_group(self, group: str, quiet: bool = False) -> Dict[str, Optional[Exception]]:
        """Remove all dependencies in a group.

        Args:
            group: Group name (cppsim, xsim, boards)
            quiet: Suppress output

        Returns:
            Dict mapping dependency name to error (None if successful)
        """
        deps = [k for k, v in DEPENDENCIES.items() if v.get('group') == group]
        results = {}

        for dep in deps:
            try:
                self.remove(dep, quiet)
                results[dep] = None  # Success
            except DependencyError as e:
                results[dep] = e  # Store error for reporting
                if not quiet:
                    logger.error("Failed to remove %s: %s", dep, e)

        return results


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

    def _parse_board_name(self, board_file: Path) -> Optional[str]:
        """Extract board name from board.xml file.

        Args:
            board_file: Path to board.xml file

        Returns:
            Board name from XML, directory name as fallback, or None
        """
        try:
            tree = ET.parse(board_file)
            root = tree.getroot()

            # Try attribute first (use walrus operator for clarity)
            if name := root.get('name'):
                return name

            # Try name element
            if (name_elem := root.find('.//name')) is not None:
                if name := name_elem.text:
                    return name

            # XML parsed but no name found - use fallback
            return self._fallback_board_name(board_file)

        except (ET.ParseError, FileNotFoundError, PermissionError) as e:
            # Log parsing failures for debugging
            logger.debug("Failed to parse %s: %s", board_file, e)
            return self._fallback_board_name(board_file)

    def _fallback_board_name(self, board_file: Path) -> Optional[str]:
        """Get board name from directory structure.

        Args:
            board_file: Path to board.xml file

        Returns:
            Directory name or None if generic
        """
        parent = board_file.parent
        # Skip generic directory names
        if parent.name in ['boards', self.board_dir.name]:
            return None
        return parent.name

    def extract_board_names(self, board_files: List[Path], repo_name: str) -> List[str]:
        """Extract board names from board.xml files.

        Args:
            board_files: List of paths to board.xml files
            repo_name: Repository name (unused, kept for compatibility)

        Returns:
            Sorted list of unique board names
        """
        boards = []

        for board_file in board_files:
            board_name = self._parse_board_name(board_file)
            if board_name:
                boards.append(board_name)

        return sorted(set(boards))
        
    def find_board_path(self, board_name: str) -> Optional[Path]:
        """Find the path to a specific board's files.

        Args:
            board_name: Name of the board to find

        Returns:
            Path to board directory, or None if not found
        """
        for board_file in self.board_dir.glob("**/board.xml"):
            name = self._parse_board_name(board_file)
            if name == board_name:
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