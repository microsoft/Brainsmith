# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import setup, find_packages

setup(
    name="brainsmith",
    version="0.0.0",
    description="From PyTorch to RTL with no brakes",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Thomas Keller",
    author_email="thomaskeller@microsoft.com",
    url="https://github.com/microsoft/BrainSmith/",
    packages=find_packages(include=["brainsmith", "brainsmith.*"]),
    install_requires=[
        "docker",  # Required dependency for Docker interactions
        # TODO: Add other dependencies here
    ],
    classifiers =[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    license="MIT",
    # TODO: Setup HW compilers as entry_points
    python_requires=">=3.8",
)