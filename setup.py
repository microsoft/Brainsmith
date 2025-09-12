# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import setup, find_packages

setup(
    name="brainsmith",
    version="0.1.0",
    description="From PyTorch to RTL with no brakes",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Thomas Keller",
    author_email="thomaskeller@microsoft.com",
    url="https://github.com/microsoft/Brainsmith/",
    packages=find_packages(include=["brainsmith", "brainsmith.*"]),
    install_requires=[
        "bitstring==4.2.3",
        "clize==5.0.1",
        "dataclasses-json==0.5.7",
        "gspread==3.6.0",
        "importlib-resources==6.1.0",
        "ipython==8.12.2",
        "ml_dtypes>=0.5.1",
        "numpy==1.24.1",
        "onnx==1.17.0",
        "onnxoptimizer==0.3.13",
        "onnxruntime==1.18.1",
        "onnxscript==0.4.0",
        "onnxsim==0.4.36",
        "pre-commit==3.3.2",
        "packaging>=25.0",
        "protobuf==3.20.3",
        "psutil==5.9.4",
        "pyscaffold==4.4",
        "scipy==1.10.1",
        "setupext-janitor>=1.1.2",
        "sigtools==4.0.1",
        "toposort==1.7.0",
        "transformers==4.48.3",
        "tree-sitter==0.24.0",
        "typing_extensions>=4.10",
        "vcdvcd==1.0.5",
        "wget==3.2",
        "docker",  # Keep existing requirement
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    license="MIT",
    entry_points={
        'console_scripts': [
            'forge=brainsmith.cli.forge:main',
        ],
    },
    python_requires=">=3.8",
)
