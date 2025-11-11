# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Simple dependency management for non-Python dependencies."""

import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from .dependency_installers import (
    BuildDependencyInstaller,
    DependencyError,
    GitDependencyInstaller,
    RequirementError,
    UnknownDependencyError,
    ZipDependencyInstaller,
)

logger = logging.getLogger(__name__)

# All non-Python dependencies
DEPENDENCIES = {
    # C++ Simulation Dependencies
    "cnpy": {
        "type": "git",
        "url": "https://github.com/maltanar/cnpy.git",
        "ref": "8c82362372ce600bbd1cf11d64661ab69d38d7de",
        "group": "cppsim",
        "description": "NumPy file I/O in C++",
        "requires": {"g++": "C++ compiler - install with: sudo apt install g++"},
    },
    "finn-hlslib": {
        "type": "git",
        "url": "https://github.com/Xilinx/finn-hlslib.git",
        "ref": "a19482ba6886f6f26aff11b10126a82ce0dd7ab1",
        "group": "cppsim",
        "description": "Header-only HLS library",
    },
    # RTL Simulation Dependencies
    "oh-my-xilinx": {
        "type": "git",
        "url": "https://github.com/ddanag/oh-my-xilinx.git",
        "ref": "0b59762f9e4c4f7e5aa535ee9bc29f292434ca7a",
        "group": "xsim",
        "description": "Xilinx Vivado utility scripts for RTL compilation",
        "requires": {
            "vivado": "Xilinx Vivado - set xilinx_path in config or BSMITH_XILINX_PATH env var"
        },
    },
    "finn-xsim": {
        "type": "build",
        "source": "finn",  # This refers to the poetry-installed finn package
        "build_cmd": ["python3", "-m", "finn.xsi.setup"],
        "group": "xsim",
        "description": "FINN XSI module",
        "requires": {
            "vivado": "Xilinx Vivado - set xilinx_path in config or BSMITH_XILINX_PATH env var"
        },
    },
    # Board Files
    "avnet-boards": {
        "type": "git",
        "url": "https://github.com/Avnet/bdf.git",
        "ref": "2d49cfc25766f07792c0b314489f21fe916b639b",
        "group": "boards",
        "description": "Avnet board files",
        "aliases": ["avnet"],
    },
    "xilinx-boards": {
        "type": "git",
        "url": "https://github.com/Xilinx/XilinxBoardStore.git",
        "ref": "8cf4bb674a919ac34e3d99d8d71a9e60af93d14e",
        "sparse_dirs": ["boards/Xilinx/rfsoc2x2"],
        "group": "boards",
        "description": "Xilinx board files (RFSoC2x2)",
        "aliases": ["xilinx", "xilinx-rfsoc2x2"],
    },
    "rfsoc4x2-boards": {
        "type": "git",
        "url": "https://github.com/RealDigitalOrg/RFSoC4x2-BSP.git",
        "ref": "13fb6f6c02c7dfd7e4b336b18b959ad5115db696",
        "sparse_dirs": ["board_files/rfsoc4x2"],
        "group": "boards",
        "description": "RFSoC 4x2 board files",
        "aliases": ["rfsoc4x2"],
    },
    "kv260-som-boards": {
        "type": "git",
        "url": "https://github.com/Xilinx/XilinxBoardStore.git",
        "ref": "98e0d3efc901f0b974006bc4370c2a7ad8856c79",
        "sparse_dirs": ["boards/Xilinx/kv260_som"],
        "group": "boards",
        "description": "KV260 SOM board files",
        "aliases": ["kv260", "kv260-som"],
    },
    "aupzu3-boards": {
        "type": "git",
        "url": "https://github.com/RealDigitalOrg/aup-zu3-bsp.git",
        "ref": "b595ecdf37c7204129517de1773b0895bcdcc2ed",
        "sparse_dirs": ["board-files/aup-zu3-8gb"],
        "group": "boards",
        "description": "AUP ZU3 8GB board files",
        "aliases": ["aupzu3"],
    },
    "pynq-z1": {
        "type": "zip",
        "url": "https://github.com/cathalmccabe/pynq-z1_board_files/raw/master/pynq-z1.zip",
        "group": "boards",
        "description": "PYNQ-Z1 board files",
        "multi_alias": "pynq",
    },
    "pynq-z2": {
        "type": "zip",
        "url": "https://dpoauwgwqsy2x.cloudfront.net/Download/pynq-z2.zip",
        "group": "boards",
        "description": "PYNQ-Z2 board files",
        "multi_alias": "pynq",
    },
}


class DependencyManager:
    """Manages non-Python dependencies for Brainsmith.

    This class orchestrates dependency installation by delegating to
    specialized installer classes for each dependency type (git, zip, build).
    """

    def __init__(self, deps_dir: Path | None = None):
        if deps_dir is None:
            from brainsmith.settings import get_config

            deps_dir = get_config().deps_dir
        self.deps_dir = Path(deps_dir)
        self.deps_dir.mkdir(parents=True, exist_ok=True)

        # Create installer instances
        self.installers = {
            "git": GitDependencyInstaller(),
            "zip": ZipDependencyInstaller(temp_dir=self.deps_dir),
            "build": BuildDependencyInstaller(),
        }

    def _get_dest_path(self, dep: dict, name: str) -> Path:
        if dep.get("group") == "boards":
            return self.deps_dir / "board-files" / name
        return self.deps_dir / name

    def install(self, name: str, force: bool = False, quiet: bool = False) -> None:
        """Install a single dependency.

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
            requirements_list = "\n".join([f"  ✗ {tool} - {msg}" for tool, msg in missing])
            error_msg = (
                f"Error: {name} requires the following tools to be installed:\n{requirements_list}"
            )
            logger.error("%s", error_msg)
            raise RequirementError(error_msg)

        # Get appropriate installer and delegate
        dep_type = dep["type"]
        if dep_type not in self.installers:
            raise ValueError(f"Unknown dependency type: {dep_type}")

        dest = self._get_dest_path(dep, name)
        self.installers[dep_type].install(name, dep, dest, force, quiet)

    def install_group(
        self, group: str, force: bool = False, quiet: bool = False
    ) -> dict[str, Exception | None]:
        """Install all dependencies in a group.

        Returns dict mapping dependency name to error (None if successful).
        """
        deps = [k for k, v in DEPENDENCIES.items() if v.get("group") == group]
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

    def _check_requirements(self, dep: dict) -> list[tuple[str, str]]:
        """Returns list of (tool, message) tuples for missing requirements."""
        missing = []
        if "requires" not in dep:
            return missing

        for tool, message in dep["requires"].items():
            if not self._is_tool_available(tool):
                missing.append((tool, message))

        return missing

    def _is_tool_available(self, tool: str) -> bool:
        from brainsmith.settings import get_config

        config = get_config()

        # Config handles special tool detection
        if tool == "vivado":
            return bool(config.vivado_path)

        # Default: check PATH
        return bool(shutil.which(tool))

    def remove(self, name: str, quiet: bool = False) -> None:
        """Remove a single dependency.

        Raises:
            UnknownDependencyError: If dependency name is unknown
            RemovalError: If removal fails
        """
        if name not in DEPENDENCIES:
            raise UnknownDependencyError(f"Unknown dependency: {name}")

        dep = DEPENDENCIES[name]
        dep_type = dep["type"]
        if dep_type not in self.installers:
            raise ValueError(f"Unknown dependency type: {dep_type}")

        dest = self._get_dest_path(dep, name)
        self.installers[dep_type].remove(name, dest, quiet)

    def remove_group(self, group: str, quiet: bool = False) -> dict[str, Exception | None]:
        """Remove all dependencies in a group.

        Returns dict mapping dependency name to error (None if successful).
        """
        deps = [k for k, v in DEPENDENCIES.items() if v.get("group") == group]
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

    def list_downloaded_repositories(self) -> list[str]:
        if not self.board_dir.exists():
            return []

        # Board repos are direct subdirectories
        repos = []
        for item in self.board_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                repos.append(item.name)

        return sorted(repos)

    def get_board_summary(self) -> dict[str, list[str]]:
        """Get summary of all boards organized by repository."""
        summary = {}

        for repo in self.list_downloaded_repositories():
            repo_path = self.board_dir / repo
            boards = self.extract_board_names(self.find_board_files(repo_path))
            if boards:
                summary[repo] = boards

        return summary

    def find_board_files(self, repo_path: Path) -> list[Path]:
        return list(repo_path.glob("**/board.xml"))

    def _parse_board_name(self, board_file: Path) -> str | None:
        """Extract board name from board.xml (returns directory name as fallback, or None)."""
        try:
            tree = ET.parse(board_file)
            root = tree.getroot()

            # Prefer XML attribute over nested element
            if name := root.get("name"):
                return name

            # Try name element
            if (name_elem := root.find(".//name")) is not None:
                if name := name_elem.text:
                    return name

            # XML parsed but no name found - use fallback
            return self._fallback_board_name(board_file)

        except (ET.ParseError, FileNotFoundError, PermissionError) as e:
            # Log parsing failures for debugging
            logger.debug("Failed to parse %s: %s", board_file, e)
            return self._fallback_board_name(board_file)

    def _fallback_board_name(self, board_file: Path) -> str | None:
        """Get board name from directory structure (returns None if generic)."""
        parent = board_file.parent
        # Skip generic directory names
        if parent.name in ["boards", self.board_dir.name]:
            return None
        return parent.name

    def extract_board_names(self, board_files: list[Path]) -> list[str]:
        """Extract board names from board.xml files (returns sorted unique list)."""
        boards = []
        for board_file in board_files:
            if board_name := self._parse_board_name(board_file):
                boards.append(board_name)
        return sorted(set(boards))

    def find_board_path(self, board_name: str) -> Path | None:
        """Find path to board directory, or None if not found."""
        for board_file in self.board_dir.glob("**/board.xml"):
            name = self._parse_board_name(board_file)
            if name == board_name:
                return board_file.parent

        return None

    def validate_repository_names(self, names: list[str]) -> tuple[list[str], list[str]]:
        """Validate repository names against known repositories.

        Returns tuple of (valid_names, invalid_names).
        """
        # Build alias mapping from DEPENDENCIES
        alias_to_canonical = {}
        multi_aliases = {}  # Maps multi_alias -> [dependency names]

        board_deps = {k: v for k, v in DEPENDENCIES.items() if v.get("group") == "boards"}

        for dep_name, dep_info in board_deps.items():
            # Add canonical name
            alias_to_canonical[dep_name] = dep_name
            # Add any aliases
            for alias in dep_info.get("aliases", []):
                alias_to_canonical[alias] = dep_name
            # Track multi-aliases (e.g., 'pynq' -> ['pynq-z1', 'pynq-z2'])
            if multi := dep_info.get("multi_alias"):
                multi_aliases.setdefault(multi, []).append(dep_name)

        # Add multi-aliases to mapping
        alias_to_canonical.update(multi_aliases)

        valid = []
        invalid = []

        for name in names:
            if name in alias_to_canonical:
                mapped = alias_to_canonical[name]
                if isinstance(mapped, list):
                    valid.extend(mapped)
                else:
                    valid.append(mapped)
            else:
                invalid.append(name)

        return list(set(valid)), invalid
