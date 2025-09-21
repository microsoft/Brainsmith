# Adapted from FINN-plus (https://github.com/eki-project/finn-plus)
# Copyright (c) 2020-2025, AMD/Xilinx and Paderborn University
# Licensed under BSD License - see FINN-plus repository for full license text

"""Manage dependencies. Called by run_finn.py"""
from __future__ import annotations

import concurrent
import concurrent.futures
import os
import shlex
import shutil
import subprocess as sp
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from rich.console import Console
from rich.live import Live
from rich.table import Table
from threading import Lock

from brainsmith.interface import IS_POSIX
from brainsmith.config import get_config

FINN_DEPS = {
    "finn-experimental": (
        "https://github.com/Xilinx/finn-experimental.git",
        "0724be21111a21f0d81a072fccc1c446e053f851",
        False,
    ),
    "brevitas": (
        "https://github.com/iksnagreb/brevitas.git",
        "003f9f4070c20639790c7b406a28612a089fc502",
        True,
    ),
    "qonnx": (
        "https://github.com/fpjentzsch/qonnx.git",
        "fab7dca2e1734cf8767fa3f6161f674aaa26a402",
        True,
    ),
    "dataset_loading": (
        "https://github.com/fbcotter/dataset_loading.git",
        "5b9faa226e5f7c857579d31cdd9acde8cdfb816f",
        True,
    ),
    "cnpy": (
        "https://github.com/rogersce/cnpy.git",
        "4e8810b1a8637695171ed346ce68f6984e585ef4",
        False,
    ),
    "finn-hlslib": (
        "https://github.com/Xilinx/finn-hlslib.git",
        "5c5ad631e3602a8dd5bd3399a016477a407d6ee7",
        False,
    ),
    "attention-hlslib": (
        "https://github.com/iksnagreb/attention-hlslib.git",
        "afc9720f10e551e1f734e137b21bb6d0a8342177",
        False,
    ),
    "pyxsi": (
        "https://github.com/fpjentzsch/pyxsi.git",
        "bbef09f9520457186830505a99050256821d5079",
        False,
    ),
}

FINN_BOARDFILES = {
    "avnet-bdf": (
        "https://github.com/Avnet/bdf.git",
        "2d49cfc25766f07792c0b314489f21fe916b639b",
        Path(),
    ),
    "xil-bdf": (
        "https://github.com/Xilinx/XilinxBoardStore.git",
        "8cf4bb674a919ac34e3d99d8d71a9e60af93d14e",
        Path("boards/Xilinx/rfsoc2x2"),
    ),
    "rfsoc4x2-bdf": (
        "https://github.com/RealDigitalOrg/RFSoC4x2-BSP.git",
        "13fb6f6c02c7dfd7e4b336b18b959ad5115db696",
        Path("board_files/rfsoc4x2"),
    ),
    "kv260-som-bdf": (
        "https://github.com/Xilinx/XilinxBoardStore.git",
        "98e0d3efc901f0b974006bc4370c2a7ad8856c79",
        Path("boards/Xilinx/kv260_som"),
    ),
}

# URL, do_unzip, where to download to
DIRECT_DOWNLOAD_DEPS = {
    "pynq-z1": (
        "https://github.com/cathalmccabe/pynq-z1_board_files/raw/master/pynq-z1.zip",
        True,
        Path("board_files"),
    ),
    "pynq-z2": (
        "https://dpoauwgwqsy2x.cloudfront.net/Download/pynq-z2.zip",
        True,
        Path("board_files"),
    ),
}

# TODO: Change or make it configurable
GIT_CLONE_TIMEOUT = 120


# Tuple that defines a dep status
# Example: ("qonnx", False, "Wrong commit")
Status = tuple[str, bool, str]


def check_commit(repo: Path, commit: str) -> tuple[bool, str]:
    """Return if the given repo has the correct commit and what commit it read"""
    result = sp.run("git rev-parse HEAD", text=True, capture_output=True, shell=True, cwd=str(repo))
    return result.stdout.strip() == commit, result.stdout.strip()


def run_silent(s: str, loc: str | None | Path, timeout: int | None = None) -> None:
    """Run a command silently directly without shell"""
    sp.run(
        shlex.split(s, posix=IS_POSIX),
        cwd=loc,
        stdout=sp.DEVNULL,
        stderr=sp.DEVNULL,
        stdin=sp.DEVNULL,
        timeout=timeout,
    )


def make_status_table(data: dict[str, tuple[str, str]]) -> Table:
    """Get a dict of the form data[name] = (status, color) and convert it into a table"""
    t = Table()
    t.add_column("Name")
    t.add_column("Status")
    for name, (status, color) in data.items():
        t.add_row(name, f"[bold {color}]{status}[/bold {color}]")
    return t


def update_dependencies(location: Path = None) -> None:
    """Update dependencies at the given path. Display live status"""
    config = get_config()
    if location is None:
        location = config.bsmith_deps_dir
    if not location.exists():
        location.mkdir(parents=True)
    board_file_dir = location / "board_files"
    if not board_file_dir.exists():
        board_file_dir.mkdir(parents=True)
    current_state = {}
    state_lock = Lock()
    any_failed = False
    with Live(make_status_table(current_state)) as live:

        def update_status(key: str, msg: str, color: str) -> None:
            state_lock.acquire()
            current_state[key] = (msg, color)
            live.update(make_status_table(current_state))
            state_lock.release()

        def pull_data(args: tuple) -> bool:
            name, url, do_unzip, target = args
            purl = Path(url)
            target = location.absolute() / target
            update_status(name, "Downloading data...", "orange1")
            if shutil.which("wget") is None:
                update_status(name, "wget not found - could not download data!", "red")
                return False
            if not (target / purl.name).exists():
                result = sp.run(
                    shlex.split(f"wget {url}", posix=IS_POSIX),
                    capture_output=True,
                    text=True,
                    cwd=target,
                )
                if result.returncode != 0:
                    update_status(name, f"Download failed: {result.stderr}", "red")
                    return False
            if do_unzip and not (target / purl.name.replace(purl.suffix, "")).exists():
                update_status(name, "Unzipping data...", "orange1")
                run_silent(f"unzip {purl.name}", target)
            update_status(name, "Dependency ready!", "green")
            return True

        def pull_dep(args: tuple) -> bool:
            pkg_name, giturl, commit, install = args
            target = (location / pkg_name).absolute()
            update_status(pkg_name, "Pulling data...", "orange1")
            if target.exists():
                run_silent("git pull", target)
            else:
                run_silent(f"git clone {giturl} {target}", None, timeout=GIT_CLONE_TIMEOUT)
            if not target.exists():
                update_status(pkg_name, "Bad Git URL or missing network connection", "red")
                return False
            update_status(pkg_name, "Checking out commit...", "orange1")
            run_silent(f"git checkout {commit}", target)
            success, read_commit = check_commit(target, commit)
            if not success:
                update_status(pkg_name, "Failed pulling. Retrying...", "orange1")
                shutil.rmtree(target, ignore_errors=True)
                run_silent(f"git clone {giturl} {target}", None)
                run_silent(f"git checkout {commit}", target)
                success, read_commit = check_commit(target, commit)
            if success:
                if install:
                    update_status(pkg_name, "Installing dependency (pip)!", "orange1")
                    run_silent(f"{sys.executable} -m pip install {target}", None)
                    update_status(pkg_name, "Dependency ready & installed (pip)!", "green")
                else:
                    update_status(pkg_name, "Dependency ready!", "green")
            else:
                update_status(
                    pkg_name,
                    f"Installation failed! Expected commit {commit} but got {read_commit}",
                    "red",
                )
                return False
            return True

        def pull_board(args: tuple) -> bool:
            pkg_name, giturl, commit, copy_from_here = args
            clone_location = location / pkg_name
            copy_source = clone_location / copy_from_here
            copy_target = location / "board_files" / copy_source.name
            update_status(pkg_name, "Pulling data...", "orange1")
            if clone_location.exists():
                run_silent("git pull", clone_location)
            else:
                run_silent(f"git clone {giturl} {clone_location}", None, timeout=GIT_CLONE_TIMEOUT)
            if not clone_location.exists():
                update_status(pkg_name, "Bad Git URL or missing network connection", "red")
                return False
            update_status(pkg_name, "Checking out commit...", "orange1")
            run_silent(f"git checkout {commit}", clone_location)
            success, read_commit = check_commit(clone_location, commit)
            if not success:
                update_status(pkg_name, "Failed pulling. Retrying...", "orange1")
                shutil.rmtree(clone_location, ignore_errors=True)
                run_silent(f"git clone {giturl} {clone_location}", None)
                run_silent(f"git checkout {commit}", clone_location)
                success, read_commit = check_commit(clone_location, commit)
            if success:
                update_status(pkg_name, "Copying boardfiles over...", "orange1")
            else:
                update_status(
                    pkg_name,
                    f"Installation failed! Expected commit {commit} but got {read_commit}",
                    "red",
                )
                return False
            if copy_source != clone_location:
                shutil.copytree(copy_source, copy_target, dirs_exist_ok=True)
            else:
                run_silent(f"cp -r {copy_source}/* {copy_target}", None)
            update_status(pkg_name, "Dependency ready!", "green")
            return True

        with ThreadPoolExecutor(100) as tpe:
            futures = []
            for name, (giturl, commit, install) in FINN_DEPS.items():
                futures.append(tpe.submit(pull_dep, (name, giturl, commit, install)))
            for name, (giturl, commit, copy_from_here) in FINN_BOARDFILES.items():
                futures.append(tpe.submit(pull_board, (name, giturl, commit, copy_from_here)))
            for name, (url, do_unzip, target) in DIRECT_DOWNLOAD_DEPS.items():
                futures.append(tpe.submit(pull_data, (name, url, do_unzip, target)))
            for future in concurrent.futures.as_completed(futures):
                any_failed |= not future.result()

    if any_failed:
        Console().print(
            "[bold red]ERROR: [/bold red][red]"
            "Failed to retrieve all dependencies. Stopping...[/red]"
        )
        sys.exit(1)


def install_pyxsi() -> bool:
    # TODO: integrate properly into the rich.Live above?
    # Will soon be replaced by finnXSI
    config = get_config()
    pyxsi_path = config.bsmith_deps_dir / "pyxsi"
    pyxsi_so_path = pyxsi_path / "pyxsi.so"

    # Disable PyXSI makefile Docker wrapper
    os.environ["PYXSI_MAKE_USE_DOCKER"] = "0"

    # Run make
    res = sp.run(["make"], cwd=pyxsi_path, capture_output=True, text=True)
    if res.returncode != 0:
        Console().print(res.stderr)
        return False

    # Check if .so was created
    if not pyxsi_so_path.exists():
        return False

    # Set environment variables
    pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{pythonpath}:{pyxsi_path}:{pyxsi_path}/py"
    sys.path.append(str(pyxsi_path))
    sys.path.append(str(pyxsi_path / "py"))
    
    if config.xilinx.vivado_path:
        vivado_path = str(config.xilinx.vivado_path)
        ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
        new_paths = f"/lib/x86_64-linux-gnu/:{vivado_path}/lib/lnx64.o"
        if ld_lib_path:
            os.environ["LD_LIBRARY_PATH"] = f"{ld_lib_path}:{new_paths}"
        else:
            os.environ["LD_LIBRARY_PATH"] = new_paths
    return True
