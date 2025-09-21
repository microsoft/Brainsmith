# Adapted from FINN-plus (https://github.com/eki-project/finn-plus)
# Copyright (c) 2020-2025, AMD/Xilinx and Paderborn University
# Licensed under BSD License - see FINN-plus repository for full license text

import os
import shlex
import subprocess
import sys
from pathlib import Path

from brainsmith.interface import IS_POSIX
from brainsmith.interface.console import status
from brainsmith.config import get_config


def run_test(variant: str, num_workers: str) -> None:
    """Run a given test variant with the given number of workers"""
    config = get_config()
    original_dir = Path.cwd()

    # TODO: Make this optional
    if "CI_PROJECT_DIR" in os.environ.keys():
        ci_project_dir = os.environ["CI_PROJECT_DIR"]
    else:
        ci_project_dir = str(config.bsmith_build_dir)
    status(f"Putting test reports into {ci_project_dir}")

    test_dir = str(config.finn.finn_tests or config.bsmith_dir / "tests")
    os.chdir(test_dir)
    match variant:
        case "quick":
            subprocess.run(
                shlex.split(
                    f"{sys.executable} -m pytest -v -m 'not "
                    f"(vivado or slow or vitis or board or notebooks or bnn_pynq or end2end)' "
                    f"--dist=loadfile -n {num_workers}",
                    posix=IS_POSIX,
                )
            )
        case "quicktest_ci":
            subprocess.run(
                shlex.split(
                    f"{sys.executable} -m pytest -v -m 'not "
                    f"(vivado or slow or vitis or board or notebooks or bnn_pynq or end2end)' "
                    f"--junitxml={ci_project_dir}/reports/quick.xml "
                    f"--html={ci_project_dir}/reports/quick.html "
                    f"--reruns 1 --dist worksteal -n {num_workers}",
                    posix=IS_POSIX,
                )
            )
        case "full_ci":
            test_1_process = subprocess.Popen(
                shlex.split(
                    (
                        f"{sys.executable} -m pytest -v -m 'not "
                        f"(end2end or sanity_bnn or notebooks)' "
                        f"--junitxml={ci_project_dir}/reports/main.xml "
                        f"--html={ci_project_dir}/reports/main.html "
                        f"--reruns 1 --dist worksteal -n {num_workers}"
                    ),
                    posix=IS_POSIX,
                )
            )
            test_2_process = subprocess.Popen(
                shlex.split(
                    (
                        f"{sys.executable} -m pytest -v -m 'end2end or sanity_bnn or notebooks' "
                        f"--junitxml={ci_project_dir}/reports/end2end.xml "
                        f"--html={ci_project_dir}/reports/end2end.html "
                        f"--reruns 1 --dist loadgroup -n {num_workers}"
                    ),
                    posix=IS_POSIX,
                )
            )
            test_1_process.communicate()
            test_1_returncode = test_1_process.returncode
            test_2_process.communicate()
            test_2_returncode = test_2_process.returncode

            subprocess.run(
                shlex.split(
                    (
                        f"{sys.executable} -m pytest_html_merger -i {ci_project_dir}/reports/ "
                        f"-o {ci_project_dir}/reports/full_test_suite.html"
                    ),
                    posix=IS_POSIX,
                )
            )

            if test_1_returncode or test_2_returncode:
                sys.exit(1)

        case _:
            subprocess.run(
                shlex.split(f"{sys.executable} -m pytest -k '{variant}'", posix=IS_POSIX)
            )
    os.chdir(original_dir)
