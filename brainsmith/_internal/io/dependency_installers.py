# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Dependency installer implementations for non-Python dependencies.

This module provides focused installer classes for different dependency types,
following the Single Responsibility Principle. Each installer handles a specific
installation method (git, zip, build) and can be tested independently.
"""

import logging
import os
import shutil
import subprocess
import sys
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)


class DependencyInstallerBase(ABC):
    """Base class for dependency installers."""

    @abstractmethod
    def install(
        self,
        name: str,
        dep: Dict,
        dest: Path,
        force: bool,
        quiet: bool
    ) -> None:
        """Install a dependency.

        Args:
            name: Dependency name
            dep: Dependency metadata dictionary
            dest: Destination path for installation
            force: Force reinstall even if exists
            quiet: Suppress output

        Raises:
            InstallationError: If installation fails
        """
        pass

    @abstractmethod
    def remove(self, name: str, dest: Path, quiet: bool) -> None:
        """Remove a dependency.

        Args:
            name: Dependency name
            dest: Installation path
            quiet: Suppress output

        Raises:
            RemovalError: If removal fails
        """
        pass

    def _log_error(self, message: str, also_print: bool = True) -> None:
        """Log error message and optionally print to stderr.

        Args:
            message: Error message to log
            also_print: Whether to also print to stderr
        """
        logger.error(message)
        if also_print:
            print(message, file=sys.stderr)

    def remove_directory(self, name: str, dest: Path, quiet: bool) -> None:
        """Remove a dependency directory (common for git/zip types).

        Args:
            name: Dependency name (for logging)
            dest: Directory to remove
            quiet: Suppress output

        Raises:
            RemovalError: If removal fails
        """
        if not dest.exists():
            if not quiet:
                logger.info("%s is not installed", name)
            return

        try:
            if not quiet:
                logger.info("Removing %s from %s", name, dest)
            shutil.rmtree(dest)
        except OSError as e:
            error_msg = f"Failed to remove {name}: {e}"
            self._log_error(error_msg)
            raise RemovalError(error_msg) from e


class GitDependencyInstaller(DependencyInstallerBase):
    """Installer for git-based dependencies."""

    def install(
        self,
        name: str,
        dep: Dict,
        dest: Path,
        force: bool,
        quiet: bool
    ) -> None:
        """Clone a git repository.

        Args:
            name: Dependency name
            dep: Must contain 'url', optional 'ref' and 'sparse_dirs'
            dest: Destination directory for clone
            force: Force reclone even if exists
            quiet: Suppress output

        Raises:
            InstallationError: If git clone fails
        """
        # Check if already exists
        if dest.exists() and not force:
            if not quiet:
                logger.info("%s already installed at %s", name, dest)
            return

        # Remove existing if force
        if dest.exists():
            shutil.rmtree(dest)

        # Build git clone command
        cmd = ['git', 'clone', '--quiet']

        # Handle sparse checkout
        if 'sparse_dirs' in dep:
            cmd.extend(['--filter=blob:none', '--sparse'])

        cmd.extend([dep['url'], str(dest)])

        # Clone
        if not quiet:
            logger.info("Cloning %s from %s", name, dep['url'])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = f"Failed to clone {name}: {result.stderr}"
            self._log_error(error_msg)
            raise InstallationError(error_msg)

        # Checkout specific ref if specified
        if 'ref' in dep:
            # First fetch the ref (in case it's not a branch/tag)
            subprocess.run(
                ['git', '-C', str(dest), 'fetch', '--quiet', 'origin', dep['ref']],
                capture_output=True
            )

            # Now checkout
            result = subprocess.run(
                ['git', '-C', str(dest), 'checkout', '--quiet', dep['ref']],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                error_msg = f"Failed to checkout {dep['ref']} for {name}: {result.stderr}"
                self._log_error(error_msg)
                raise InstallationError(error_msg)

        # Setup sparse checkout if needed
        if 'sparse_dirs' in dep:
            subprocess.run(
                ['git', '-C', str(dest), 'sparse-checkout', 'init'],
                capture_output=True
            )
            subprocess.run(
                ['git', '-C', str(dest), 'sparse-checkout', 'set'] + dep['sparse_dirs'],
                capture_output=True
            )

    def remove(self, name: str, dest: Path, quiet: bool) -> None:
        """Remove a git repository.

        Args:
            name: Dependency name
            dest: Repository directory
            quiet: Suppress output

        Raises:
            RemovalError: If removal fails
        """
        self.remove_directory(name, dest, quiet)


class ZipDependencyInstaller(DependencyInstallerBase):
    """Installer for zip-based dependencies."""

    def __init__(self, temp_dir: Optional[Path] = None):
        """Initialize with optional temp directory.

        Args:
            temp_dir: Directory for temporary zip files (default: dest parent)
        """
        self.temp_dir = temp_dir

    def install(
        self,
        name: str,
        dep: Dict,
        dest: Path,
        force: bool,
        quiet: bool
    ) -> None:
        """Download and extract a zip file.

        Args:
            name: Dependency name
            dep: Must contain 'url'
            dest: Destination directory for extraction
            force: Force redownload even if exists
            quiet: Suppress output

        Raises:
            InstallationError: If download or extraction fails
        """
        # Check if already exists
        if dest.exists() and not force:
            if not quiet:
                logger.info("%s already installed at %s", name, dest)
            return

        # Remove existing if force
        if dest.exists():
            shutil.rmtree(dest)

        # Determine temp directory
        temp_dir = self.temp_dir or dest.parent
        zip_path = temp_dir / f"{name}.zip"

        try:
            # Download
            if not quiet:
                logger.info("Downloading %s from %s", name, dep['url'])

            urlretrieve(dep['url'], zip_path)

            # Create destination directory
            dest.mkdir(parents=True, exist_ok=True)

            # Extract
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)

        except Exception as e:
            error_msg = f"Failed to download/extract {name}: {e}"
            self._log_error(error_msg)
            raise InstallationError(error_msg)

        finally:
            # Cleanup zip file
            if zip_path.exists():
                zip_path.unlink()

    def remove(self, name: str, dest: Path, quiet: bool) -> None:
        """Remove a zip-extracted directory.

        Args:
            name: Dependency name
            dest: Extracted directory
            quiet: Suppress output

        Raises:
            RemovalError: If removal fails
        """
        self.remove_directory(name, dest, quiet)


class BuildDependencyInstaller(DependencyInstallerBase):
    """Installer for build-based dependencies."""

    def install(
        self,
        name: str,
        dep: Dict,
        dest: Path,
        force: bool,
        quiet: bool
    ) -> None:
        """Build a local dependency.

        Args:
            name: Dependency name
            dep: Must contain 'build_cmd', optional 'source'
            dest: Not used for build dependencies (builds in-place)
            force: Force rebuild
            quiet: Suppress output

        Raises:
            BuildError: If build fails
        """
        # Special handling for finn-xsim
        if name == 'finn-xsim':
            self._install_finn_xsim(force, quiet)
        else:
            self._install_generic_build(name, dep, force, quiet)

    def _install_finn_xsim(self, force: bool, quiet: bool) -> None:
        """Install FINN XSI module with Vivado.

        Args:
            force: Force rebuild
            quiet: Suppress output

        Raises:
            BuildError: If build fails or requirements missing
        """
        # Check if FINN package is available
        try:
            import finn
            # Check if already built
            finn_root = Path(finn.__file__).parent.parent.parent
            output_file = finn_root / 'finn_xsi' / 'xsi.so'

            if output_file.exists() and not force:
                if not quiet:
                    logger.info("finn-xsim already built at %s", output_file)
                return

        except ImportError:
            error_msg = "FINN package not found. Please run 'poetry install' first."
            self._log_error(error_msg)
            raise BuildError(error_msg)

        # Get Vivado settings path from config
        from brainsmith.settings import get_config
        config = get_config()
        vivado_path = config.effective_vivado_path

        if not vivado_path:
            error_msg = "Vivado path not configured. Set xilinx_path in config or BSMITH_XILINX_PATH env var."
            self._log_error(error_msg)
            raise RequirementError(error_msg)

        settings_script = Path(vivado_path) / "settings64.sh"
        if not settings_script.exists():
            error_msg = f"settings64.sh not found at {settings_script}"
            self._log_error(error_msg)
            raise RequirementError(error_msg)

        # Build with finn-xsim
        if not quiet:
            logger.info("Building finn-xsim...")

        # Construct build command
        build_cmd = ['python3', '-m', 'finn.xsi.setup']
        if force:
            build_cmd.append('--force')

        # Build bash command that sources Vivado settings
        python_cmd = ' '.join(build_cmd)
        bash_cmd = f"source {settings_script} && {python_cmd}"

        logger.info("Running: %s", bash_cmd)

        # Execute build
        result = subprocess.run(
            ['bash', '-c', bash_cmd],
            capture_output=True,
            text=True
        )

        # Log output at INFO level (visible with --logs info)
        if result.stdout:
            for line in result.stdout.splitlines():
                if line.strip():
                    logger.info(line)

        if result.stderr:
            for line in result.stderr.splitlines():
                if line.strip():
                    logger.warning(line)

        # Check for errors
        if result.returncode != 0:
            error_msg = f"Failed to build finn-xsim (exit code: {result.returncode})"
            self._log_error(error_msg)
            if result.stdout:
                print("\n--- Build Output ---", file=sys.stderr)
                print(result.stdout, file=sys.stderr)
            if result.stderr:
                print("\n--- Build Errors ---", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
            raise BuildError(error_msg)

        # Verify output file exists
        if not output_file.exists():
            error_msg = f"Build completed but output file not found: {output_file}"
            self._log_error(error_msg)
            raise BuildError(error_msg)

    def _install_generic_build(
        self,
        name: str,
        dep: Dict,
        force: bool,
        quiet: bool
    ) -> None:
        """Install generic build dependency.

        Args:
            name: Dependency name
            dep: Dependency metadata with 'source' and 'build_cmd'
            force: Force rebuild
            quiet: Suppress output

        Raises:
            BuildError: If build fails
        """
        # Get deps_dir from config
        from brainsmith.settings import get_config
        deps_dir = get_config().deps_dir

        source_dir = Path(deps_dir) / dep['source']
        if not source_dir.exists():
            error_msg = f"Source directory not found: {source_dir}. Install source dependency first."
            self._log_error(error_msg)
            raise BuildError(error_msg)

        if not quiet:
            logger.info("Building %s in %s", name, source_dir)

        # Run build command
        env = os.environ.copy()
        result = subprocess.run(
            dep['build_cmd'],
            cwd=source_dir,
            capture_output=True,
            text=True,
            env=env
        )

        # Log output
        if result.stdout:
            for line in result.stdout.splitlines():
                if line.strip():
                    logger.info(line)

        if result.stderr:
            for line in result.stderr.splitlines():
                if line.strip():
                    logger.warning(line)

        # Check for errors
        if result.returncode != 0:
            error_msg = f"Failed to build {name} (exit code: {result.returncode})"
            self._log_error(error_msg)
            if result.stdout:
                print("\n--- Build Output ---", file=sys.stderr)
                print(result.stdout, file=sys.stderr)
            if result.stderr:
                print("\n--- Build Errors ---", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
            raise BuildError(error_msg)

    def remove(self, name: str, dest: Path, quiet: bool) -> None:
        """Remove a build dependency.

        Args:
            name: Dependency name
            dest: Not used (builds are in-place)
            quiet: Suppress output
        """
        # Special handling for finn-xsim
        if name == 'finn-xsim':
            try:
                import finn
                finn_root = Path(finn.__file__).parent.parent.parent
                output_file = finn_root / 'finn_xsi' / 'xsi.so'

                if not output_file.exists():
                    if not quiet:
                        logger.info("finn-xsim is not built")
                    return

                # Run clean command if available
                if not quiet:
                    logger.info("Cleaning finn-xsim...")

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
                    except Exception as e:
                        error_msg = f"Failed to remove finn-xsim: {e}"
                        self._log_error(error_msg)
                        raise RemovalError(error_msg)

            except ImportError:
                if not quiet:
                    logger.info("finn-xsim is not installed (FINN not found)")

        else:
            # Generic build dependency removal not implemented
            if not quiet:
                logger.info("Build dependency %s removal not implemented", name)


# Custom exceptions for better error handling
class DependencyError(Exception):
    """Base exception for dependency operations."""
    pass


class UnknownDependencyError(DependencyError):
    """Requested dependency not found in registry."""
    pass


class InstallationError(DependencyError):
    """Dependency installation failed."""
    pass


class BuildError(DependencyError):
    """Dependency build failed."""
    pass


class RequirementError(DependencyError):
    """Required tool or dependency missing."""
    pass


class RemovalError(DependencyError):
    """Dependency removal failed."""
    pass
