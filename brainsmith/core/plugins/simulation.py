"""Simplified simulation setup for Brainsmith."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from rich.console import Console

console = Console()


class SimulationSetup:
    """Manages simulation dependencies for Brainsmith."""
    
    def __init__(self, deps_dir: Optional[Path] = None):
        """Initialize simulation setup.
        
        Args:
            deps_dir: Directory for dependencies. Defaults to deps/ in current directory.
        """
        self.deps_dir = deps_dir or Path.cwd() / "deps"
        self.deps_dir.mkdir(exist_ok=True)
    
    def setup_cppsim(self) -> bool:
        """Download and setup dependencies for C++ simulation.
        
        Returns:
            True if setup successful, False otherwise.
        """
        console.print("\n[bold]Setting up C++ Simulation Dependencies[/bold]")
        
        # Check prerequisites
        if not self._check_cppsim_prerequisites():
            return False
        
        # Install cnpy (C++ NumPy library)
        console.print("\n[blue]Installing cnpy library...[/blue]")
        if not self._install_cnpy():
            console.print("[red]Failed to install cnpy[/red]")
            return False
        
        # Install finn-hlslib (header-only)
        console.print("\n[blue]Installing finn-hlslib...[/blue]")
        if not self._install_finn_hlslib():
            console.print("[red]Failed to install finn-hlslib[/red]")
            return False
        
        console.print("\n[green]✓ C++ simulation setup complete![/green]")
        return True
    
    def setup_rtlsim(self) -> bool:
        """Download and setup dependencies for RTL simulation.
        
        Returns:
            True if setup successful, False otherwise.
        """
        console.print("\n[bold]Setting up RTL Simulation Dependencies[/bold]")
        
        # Check prerequisites
        if not self._check_rtlsim_prerequisites():
            return False
        
        # Build finnxsi if FINN is available
        console.print("\n[blue]Building finnxsi module...[/blue]")
        if not self._build_finnxsi():
            console.print("[red]Failed to build finnxsi[/red]")
            return False
        
        console.print("\n[green]✓ RTL simulation setup complete![/green]")
        return True
    
    def download_board_files(self, boards: Optional[list] = None) -> bool:
        """Download optional board definition files.
        
        Args:
            boards: List of board names to download. None means all.
            
        Returns:
            True if download successful, False otherwise.
        """
        console.print("\n[bold]Downloading Board Definition Files[/bold]")
        
        board_repos = {
            "avnet": {
                "url": "https://github.com/Avnet/bdf.git",
                "rev": "2d49cfc25766f07792c0b314489f21fe916b639b",
            },
            "xilinx-rfsoc": {
                "url": "https://github.com/Xilinx/XilinxBoardStore.git",
                "rev": "8cf4bb674a919ac34e3d99d8d71a9e60af93d14e",
                "sparse": ["boards/Xilinx/rfsoc2x2", "boards/Xilinx/kv260_som"],
            },
            "realdigital": {
                "url": "https://github.com/RealDigitalOrg/RFSoC4x2-BSP.git",
                "rev": "13fb6f6c02c7dfd7e4b336b18b959ad5115db696",
            },
        }
        
        if boards is None:
            boards = list(board_repos.keys())
        
        board_dir = self.deps_dir / "board_files"
        board_dir.mkdir(exist_ok=True)
        
        for board_name in boards:
            if board_name not in board_repos:
                console.print(f"[yellow]Unknown board: {board_name}[/yellow]")
                continue
            
            console.print(f"\n[blue]Downloading {board_name} board files...[/blue]")
            board_info = board_repos[board_name]
            
            success, msg = self._git_clone(
                board_info["url"],
                board_dir / board_name,
                rev=board_info["rev"],
                sparse_paths=board_info.get("sparse")
            )
            
            if success:
                console.print(f"[green]✓ {board_name} downloaded[/green]")
            else:
                console.print(f"[red]✗ {board_name}: {msg}[/red]")
        
        console.print("\n[green]✓ Board files download complete![/green]")
        return True
    
    def _check_cppsim_prerequisites(self) -> bool:
        """Check prerequisites for C++ simulation."""
        console.print("\nChecking C++ simulation prerequisites...")
        
        # Check for g++
        if not shutil.which("g++"):
            console.print("[red]✗ g++ not found. Please install: sudo apt install g++[/red]")
            return False
        
        # Check g++ version
        try:
            result = subprocess.run(
                ["g++", "--version"], 
                capture_output=True, 
                text=True
            )
            console.print(f"[green]✓ g++ found[/green]")
        except:
            console.print("[red]✗ Failed to check g++ version[/red]")
            return False
        
        # Check for make
        if not shutil.which("make"):
            console.print("[red]✗ make not found. Please install: sudo apt install make[/red]")
            return False
        console.print("[green]✓ make found[/green]")
        
        # Check for cmake (needed for cnpy)
        if not shutil.which("cmake"):
            console.print("[red]✗ cmake not found. Please install: sudo apt install cmake[/red]")
            return False
        console.print("[green]✓ cmake found[/green]")
        
        # Optional: Check for Vitis HLS
        if os.environ.get("HLS_PATH"):
            console.print(f"[green]✓ Vitis HLS found at {os.environ['HLS_PATH']}[/green]")
        else:
            console.print("[yellow]⚠ Vitis HLS not found (optional for basic C++ sim)[/yellow]")
        
        return True
    
    def _check_rtlsim_prerequisites(self) -> bool:
        """Check prerequisites for RTL simulation."""
        console.print("\nChecking RTL simulation prerequisites...")
        
        # Check for Vivado
        vivado_path = os.environ.get("XILINX_VIVADO")
        if not vivado_path or not Path(vivado_path).exists():
            console.print("[red]✗ Vivado not found. Please set XILINX_VIVADO environment variable[/red]")
            return False
        console.print(f"[green]✓ Vivado found at {vivado_path}[/green]")
        
        # Check for g++
        if not shutil.which("g++"):
            console.print("[red]✗ g++ not found. Please install: sudo apt install g++[/red]")
            return False
        console.print("[green]✓ g++ found[/green]")
        
        # Check for Python headers
        try:
            subprocess.run(
                ["python3-config", "--includes"],
                capture_output=True,
                check=True
            )
            console.print("[green]✓ Python development headers found[/green]")
        except:
            console.print("[red]✗ Python headers not found. Please install: sudo apt install python3-dev[/red]")
            return False
        
        # Check for FINN
        finn_path = self.deps_dir / "finn"
        if not finn_path.exists():
            console.print("[red]✗ FINN not found. Please run: poetry install[/red]")
            return False
        console.print("[green]✓ FINN found[/green]")
        
        return True
    
    def _install_cnpy(self) -> bool:
        """Install cnpy library."""
        cnpy_dir = self.deps_dir / "cnpy"
        
        # Clone if needed
        if not cnpy_dir.exists():
            success, msg = self._git_clone(
                "https://github.com/maltanar/cnpy.git",
                cnpy_dir,
                rev="8c82362372ce600bbd1cf11d64661ab69d38d7de"
            )
            if not success:
                return False
        
        # Build cnpy
        build_dir = cnpy_dir / "build"
        build_dir.mkdir(exist_ok=True)
        
        try:
            # Configure with CMake
            subprocess.run(
                ["cmake", "-DBUILD_SHARED_LIBS=ON", ".."],
                cwd=build_dir,
                check=True
            )
            
            # Build
            subprocess.run(
                ["make", "-j4"],
                cwd=build_dir,
                check=True
            )
            
            console.print("[green]✓ cnpy built successfully[/green]")
            return True
            
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Build failed: {e}[/red]")
            return False
    
    def _install_finn_hlslib(self) -> bool:
        """Install finn-hlslib headers."""
        hlslib_dir = self.deps_dir / "finn-hlslib"
        
        success, msg = self._git_clone(
            "https://github.com/Xilinx/finn-hlslib.git",
            hlslib_dir,
            rev="5c5ad631e3602a8dd5bd3399a016477a407d6ee7"
        )
        
        if success:
            console.print("[green]✓ finn-hlslib installed[/green]")
        
        return success
    
    def _build_finnxsi(self) -> bool:
        """Build finnxsi module."""
        finn_xsi_dir = self.deps_dir / "finn" / "finn_xsi"
        
        if not finn_xsi_dir.exists():
            console.print("[yellow]finnxsi source not found[/yellow]")
            return False
        
        try:
            subprocess.run(
                ["make"],
                cwd=finn_xsi_dir,
                check=True
            )
            
            # Check if xsi.so was created
            if (self.deps_dir / "finn" / "xsi.so").exists():
                console.print("[green]✓ finnxsi built successfully[/green]")
                return True
            else:
                console.print("[red]xsi.so not found after build[/red]")
                return False
                
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Build failed: {e}[/red]")
            return False
    
    def _git_clone(self, url: str, dest: Path, rev: Optional[str] = None,
                   sparse_paths: Optional[list] = None) -> Tuple[bool, str]:
        """Clone or update a git repository."""
        try:
            if dest.exists():
                # Update existing
                subprocess.run(["git", "fetch"], cwd=dest, check=True)
                if rev:
                    subprocess.run(["git", "checkout", rev], cwd=dest, check=True)
                return True, "Updated"
            else:
                # Clone new
                cmd = ["git", "clone", "--quiet"]
                if sparse_paths:
                    cmd.extend(["--filter=blob:none", "--sparse"])
                cmd.extend([url, str(dest)])
                
                subprocess.run(cmd, check=True)
                
                if sparse_paths and dest.exists():
                    subprocess.run(
                        ["git", "sparse-checkout", "init", "--cone"],
                        cwd=dest,
                        check=True
                    )
                    subprocess.run(
                        ["git", "sparse-checkout", "set"] + sparse_paths,
                        cwd=dest,
                        check=True
                    )
                
                if rev:
                    subprocess.run(["git", "checkout", rev], cwd=dest, check=True)
                
                return True, "Cloned"
                
        except subprocess.CalledProcessError as e:
            return False, str(e)