"""Dependency management for Brainsmith.

Handles installation of external dependencies including simulation tools,
board files, and other required components.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console

console = Console()


# Dependency definitions
@dataclass
class GitDependency:
    """Git repository dependency."""
    name: str
    url: str
    rev: str
    dest_subdir: str  # Relative to deps_dir
    build_cmd: Optional[List[str]] = None  # None = no build needed
    build_dir: Optional[str] = None  # Subdirectory for building
    sparse_paths: Optional[List[str]] = None
    prerequisites: Optional[List[str]] = None
    env_requirements: Optional[Dict[str, str]] = None  # {VAR: description}
    
    def dest_path(self, deps_dir: Path) -> Path:
        return deps_dir / self.dest_subdir


@dataclass 
class LocalDependency:
    """Dependency that exists within another dependency."""
    name: str
    parent_dep: str  # Name of parent dependency
    subdir: str  # Path within parent
    build_cmd: List[str]
    build_dir: Optional[str] = None
    prerequisites: Optional[List[str]] = None
    env_requirements: Optional[Dict[str, str]] = None
    check_file: Optional[str] = None  # File to check if built
    
    def dest_path(self, deps_dir: Path) -> Path:
        return deps_dir / self.parent_dep / self.subdir


# Define all dependencies declaratively
DEPENDENCIES = {
    # C++ Simulation
    'cnpy': GitDependency(
        name='cnpy',
        url='https://github.com/maltanar/cnpy.git',
        rev='8c82362372ce600bbd1cf11d64661ab69d38d7de',
        dest_subdir='cnpy',
        # No build needed - FINN uses source directly
        prerequisites=['g++']  # Only g++ needed for compilation
    ),
    
    'finn-hlslib': GitDependency(
        name='finn-hlslib',
        url='https://github.com/Xilinx/finn-hlslib.git',
        rev='5c5ad631e3602a8dd5bd3399a016477a407d6ee7',
        dest_subdir='finn-hlslib'
        # No build_cmd = header-only
    ),
    
    # RTL Simulation
    'finn-xsim': LocalDependency(
        name='finn-xsim',
        parent_dep='finn',
        subdir='finn_xsi',
        build_cmd=['make'],
        prerequisites=['g++', 'python3-config'],
        env_requirements={'XILINX_VIVADO': 'Vivado installation'},
        check_file='xsi.so'
    ),
    
    # Board files
    'avnet-boards': GitDependency(
        name='avnet-boards',
        url='https://github.com/Avnet/bdf.git',
        rev='2d49cfc25766f07792c0b314489f21fe916b639b',
        dest_subdir='board-files/avnet'
    ),
    
    'xilinx-boards': GitDependency(
        name='xilinx-boards',
        url='https://github.com/Xilinx/XilinxBoardStore.git',
        rev='8cf4bb674a919ac34e3d99d8d71a9e60af93d14e',
        dest_subdir='board-files/xilinx',
        sparse_paths=['boards/Xilinx/rfsoc2x2', 'boards/Xilinx/kv260_som']
    ),
    
    'realdigital-boards': GitDependency(
        name='realdigital-boards',
        url='https://github.com/RealDigitalOrg/RFSoC4x2-BSP.git',
        rev='13fb6f6c02c7dfd7e4b336b18b959ad5115db696',
        dest_subdir='board-files/realdigital'
    ),
}

# Group dependencies by category
DEPENDENCY_GROUPS = {
    'cppsim': ['cnpy', 'finn-hlslib'],
    'xsim': ['finn-xsim'],
    'boards': ['avnet-boards', 'xilinx-boards', 'realdigital-boards'],
}


# Helper functions
def check_command(cmd: str, hint: str = "") -> bool:
    """Check if a command exists."""
    if shutil.which(cmd):
        console.print(f"[green]✓ {cmd} found[/green]")
        return True
    msg = f"[red]✗ {cmd} not found[/red]"
    if hint:
        msg += f" - {hint}"
    console.print(msg)
    return False


def check_env_var(var: str, description: str) -> bool:
    """Check if environment variable is set and path exists."""
    value = os.environ.get(var)
    if value and Path(value).exists():
        console.print(f"[green]✓ {description} found at {value}[/green]")
        return True
    console.print(f"[red]✗ {description} not found - set {var}[/red]")
    return False


def run_command(cmd: List[str], cwd: Path, description: str, env: Optional[Dict[str, str]] = None, quiet: bool = False) -> bool:
    """Run command with unified error handling."""
    try:
        # Use current environment as base
        cmd_env = os.environ.copy()
        if env:
            cmd_env.update(env)
        subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, env=cmd_env)
        if not quiet:
            console.print(f"[green]✓ {description}[/green]")
        return True
    except subprocess.CalledProcessError as e:
        if not quiet:
            console.print(f"[red]✗ {description} failed[/red]")
            if e.stderr:
                console.print(f"[red]{e.stderr.decode()}[/red]")
        return False


def git_clone(url: str, dest: Path, rev: str, sparse_paths: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Clone or update a git repository."""
    try:
        if dest.exists():
            # Update existing
            subprocess.run(["git", "fetch"], cwd=dest, check=True, capture_output=True)
            if rev:
                subprocess.run(["git", "checkout", rev], cwd=dest, check=True, capture_output=True)
            return True, "Updated"
        else:
            # Clone new
            cmd = ["git", "clone", "--quiet"]
            if sparse_paths:
                cmd.extend(["--filter=blob:none", "--sparse"])
            cmd.extend([url, str(dest)])
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            if sparse_paths and dest.exists():
                subprocess.run(
                    ["git", "sparse-checkout", "init", "--cone"],
                    cwd=dest,
                    check=True,
                    capture_output=True
                )
                subprocess.run(
                    ["git", "sparse-checkout", "set"] + sparse_paths,
                    cwd=dest,
                    check=True,
                    capture_output=True
                )
            
            if rev:
                subprocess.run(["git", "checkout", rev], cwd=dest, check=True, capture_output=True)
            
            return True, "Cloned"
            
    except subprocess.CalledProcessError as e:
        return False, str(e)


class DependencyManager:
    """Manages external dependencies for Brainsmith."""
    
    def __init__(self, deps_dir: Optional[Path] = None):
        """Initialize dependency manager.
        
        Args:
            deps_dir: Directory for dependencies. Defaults to deps/ in current directory.
        """
        self.deps_dir = Path(deps_dir) if deps_dir else Path.cwd() / "deps"
        self.deps_dir.mkdir(exist_ok=True)
    
    @property
    def deps_path(self) -> Path:
        """Get the dependencies directory path."""
        return self.deps_dir
    
    def install(self, dep_names: List[str], force: bool = False, quiet: bool = False) -> Dict[str, bool]:
        """Install specified dependencies.
        
        Args:
            dep_names: List of dependency names to install
            force: Force reinstall even if already installed
            quiet: Suppress output messages
            
        Returns:
            Dict mapping dependency names to success status
        """
        results = {}
        
        for name in dep_names:
            if name not in DEPENDENCIES:
                console.print(f"[red]Unknown dependency: {name}[/red]")
                results[name] = False
                continue
            
            dep = DEPENDENCIES[name]
            
            # Check if already installed
            if not force and self._is_installed(dep):
                if not quiet:
                    console.print(f"[yellow]{name} already installed[/yellow]")
                results[name] = True
                continue
            
            # Check prerequisites
            missing = self._check_prerequisites(dep)
            if missing:
                # Always show prerequisite errors, even in quiet mode
                console.print(f"[red]Missing prerequisites for {name}:[/red]")
                for item in missing:
                    console.print(f"  [red]✗ {item}[/red]")
                results[name] = False
                continue
            
            # Install
            if not quiet:
                console.print(f"\n[blue]Installing {name}...[/blue]")
            results[name] = self._install_dependency(dep, quiet=quiet)
            
        return results
    
    def _is_installed(self, dep) -> bool:
        """Check if a dependency is already installed."""
        dest = dep.dest_path(self.deps_dir)
        
        if isinstance(dep, GitDependency):
            # For git deps, check if directory exists
            return dest.exists() and any(dest.iterdir())
        elif isinstance(dep, LocalDependency):
            # For local deps, check if build output exists
            if dep.check_file:
                check_path = dest / dep.check_file
                return check_path.exists()
            # Otherwise just check if directory exists
            return dest.exists()
        
        return False
    
    def _check_prerequisites(self, dep) -> List[str]:
        """Check prerequisites for a dependency. Returns list of missing items."""
        missing = []
        
        # Check commands
        if hasattr(dep, 'prerequisites') and dep.prerequisites:
            for cmd in dep.prerequisites:
                if not shutil.which(cmd):
                    if cmd == 'g++':
                        missing.append(f"{cmd} - install with: sudo apt install g++")
                    elif cmd == 'cmake':
                        missing.append(f"{cmd} - install with: sudo apt install cmake")
                    elif cmd == 'make':
                        missing.append(f"{cmd} - install with: sudo apt install make")
                    elif cmd == 'python3-config':
                        missing.append(f"{cmd} - install with: sudo apt install python3-dev")
                    else:
                        missing.append(cmd)
        
        # Check environment variables
        if hasattr(dep, 'env_requirements') and dep.env_requirements:
            for var, desc in dep.env_requirements.items():
                value = os.environ.get(var)
                if not value or not Path(value).exists():
                    missing.append(f"{desc} - set {var}")
        
        return missing
    
    def _install_dependency(self, dep, quiet: bool = False) -> bool:
        """Install a single dependency."""
        if isinstance(dep, GitDependency):
            return self._install_git_dependency(dep, quiet)
        elif isinstance(dep, LocalDependency):
            return self._install_local_dependency(dep, quiet)
        return False
    
    def _install_git_dependency(self, dep: GitDependency, quiet: bool = False) -> bool:
        """Clone and optionally build a git dependency."""
        dest = dep.dest_path(self.deps_dir)
        
        # Clone/update repository
        success, msg = git_clone(dep.url, dest, dep.rev, dep.sparse_paths)
        if not success:
            console.print(f"[red]Failed to clone: {msg}[/red]")
            return False
        
        # Build if needed
        if dep.build_cmd:
            build_path = dest / dep.build_dir if dep.build_dir else dest
            build_path.mkdir(exist_ok=True)
            
            # Run build command (e.g., cmake)
            if not run_command(dep.build_cmd, build_path, f"Configuring {dep.name}", quiet=quiet):
                return False
            
            # Additional make command if cmake was used
            if dep.build_cmd and 'cmake' in dep.build_cmd:
                if not run_command(['make', '-j4'], build_path, f"Building {dep.name}", quiet=quiet):
                    return False
        
        if not quiet:
            console.print(f"[green]✓ {dep.name} installed successfully[/green]")
        return True
    
    def _install_local_dependency(self, dep: LocalDependency, quiet: bool = False) -> bool:
        """Build a dependency within another dependency."""
        build_path = dep.dest_path(self.deps_dir)
        
        # Check parent exists
        parent_path = self.deps_dir / dep.parent_dep
        if not parent_path.exists():
            console.print(f"[red]Parent dependency {dep.parent_dep} not found[/red]")
            console.print(f"[yellow]Make sure to run: poetry install[/yellow]")
            return False
        
        if not build_path.exists():
            console.print(f"[red]Source not found at {build_path}[/red]")
            return False
        
        if dep.build_dir:
            build_path = build_path / dep.build_dir
            build_path.mkdir(exist_ok=True)
        
        # Special handling for finn-xsim to include pybind11
        build_env = None
        if dep.name == 'finn-xsim':
            try:
                # Create a wrapper script for python3-config that includes pybind11
                wrapper_script = build_path / 'python3-config-wrapper'
                pybind11_result = subprocess.run(
                    ['poetry', 'run', 'python', '-m', 'pybind11', '--includes'],
                    capture_output=True, text=True, check=True
                )
                pybind11_includes = pybind11_result.stdout.strip()
                
                wrapper_content = f'''#!/bin/bash
# Wrapper script to include pybind11 headers
echo "{pybind11_includes}"
'''
                wrapper_script.write_text(wrapper_content)
                wrapper_script.chmod(0o755)
                
                # Add wrapper script directory to PATH
                new_path = f"{build_path}:{os.environ.get('PATH', '')}"
                build_env = {'PATH': new_path}
                if not quiet:
                    console.print(f"[dim]Created python3-config wrapper with pybind11 includes[/dim]")
                
                # Rename the wrapper to python3-config
                python3_config = build_path / 'python3-config'
                if python3_config.exists():
                    python3_config.unlink()
                wrapper_script.rename(python3_config)
                
            except subprocess.CalledProcessError as e:
                console.print(f"[yellow]Warning: Could not setup pybind11 includes: {e}[/yellow]")
        
        if not run_command(dep.build_cmd, build_path, f"Building {dep.name}", env=build_env, quiet=quiet):
            return False
        
        # Check if build succeeded
        if dep.check_file:
            check_path = build_path / dep.check_file
            if not check_path.exists():
                console.print(f"[red]Build output {dep.check_file} not found[/red]")
                return False
        
        if not quiet:
            console.print(f"[green]✓ {dep.name} built successfully[/green]")
        return True
    
    # Convenience methods for grouped installations
    def install_group(self, group_name: str, force: bool = False, quiet: bool = True) -> bool:
        """Install all dependencies in a group."""
        if group_name not in DEPENDENCY_GROUPS:
            if not quiet:
                console.print(f"[red]Unknown dependency group: {group_name}[/red]")
            return False
        
        deps = DEPENDENCY_GROUPS[group_name]
        results = self.install(deps, force, quiet=quiet)
        return all(results.values())
    
    # Public API for compatibility with existing code
    def setup_cppsim(self, force: bool = False, quiet: bool = True) -> bool:
        """Setup C++ simulation dependencies."""
        if not quiet:
            console.print("\n[bold]Setting up C++ Simulation Dependencies[/bold]")
        return self.install_group('cppsim', force=force)
    
    def setup_xsim(self) -> bool:
        """Setup Xilinx simulation dependencies."""
        console.print("\n[bold]Setting up Xilinx Simulation Dependencies[/bold]")
        return self.install_group('xsim')
    
    def download_board_files(self, boards: Optional[List[str]] = None, quiet: bool = True) -> bool:
        """Download board definition files."""
        if not quiet:
            console.print("\n[bold]Downloading Board Definition Files[/bold]")
        
        if boards:
            # Map board names to dependency names
            dep_names = []
            for board in boards:
                matching = [name for name in DEPENDENCIES if board in name and name.endswith('-boards')]
                dep_names.extend(matching)
            if not dep_names:
                if not quiet:
                    console.print(f"[red]No matching board repositories found for: {', '.join(boards)}[/red]")
                    console.print("\n[yellow]Available repositories:[/yellow]")
                    available_repos = [name.replace('-boards', '') for name in DEPENDENCIES 
                                     if name.endswith('-boards')]
                    for r in sorted(available_repos):
                        console.print(f"  • {r}")
                return False
        else:
            dep_names = DEPENDENCY_GROUPS['boards']
        
        results = self.install(dep_names, quiet=quiet)
        return all(results.values())
    
    # Individual method wrappers for CLI compatibility
    def install_cnpy(self) -> bool:
        """Install cnpy library."""
        results = self.install(['cnpy'])
        return results.get('cnpy', False)
    
    def install_hlslib_headers(self) -> bool:
        """Install finn-hlslib headers."""
        results = self.install(['finn-hlslib'])
        return results.get('finn-hlslib', False)
    
    def build_finnxsim(self, force: bool = False, quiet: bool = True) -> bool:
        """Build finn-xsim module."""
        results = self.install(['finn-xsim'], force=force, quiet=quiet)
        return results.get('finn-xsim', False)


# Legacy compatibility - maintain the old class name temporarily
SimulationSetup = DependencyManager