# Getting Started

Welcome to Brainsmith! This guide will help you get up and running with FPGA accelerator development.

## Overview

Brainsmith automates the design and implementation of neural network accelerators on FPGAs. The typical workflow is:

1. **Quantize** your PyTorch model with Brevitas
2. **Export** to ONNX format
3. **Define** your design space in a Blueprint YAML file
4. **Explore** configurations with the `smith` CLI
5. **Generate** RTL and synthesize to bitstream

## Quick Navigation

### :material-download: [Installation](installation.md)
Set up your development environment with Poetry or Docker. Configure Vivado paths and verify your installation.

### :material-lightning-bolt: [Quick Start](quickstart.md)
Run your first design space exploration with the BERT example in under 5 minutes (build time 30-60 minutes).

### :material-cog: [Configuration](configuration.md)
Learn about the project-based configuration system, environment management, and how to work with multiple projects.

## Prerequisites

Before you begin, ensure you have:

- **Ubuntu 22.04+** (other Linux distributions may work but are untested)
- **Vivado Design Suite 2024.2** (migration to 2025.1 in progress)
- **Python 3.10+** installed
- **Poetry** package manager ([installation guide](https://python-poetry.org/docs/#installation))

## Next Steps

1. [Install Brainsmith](installation.md) - Set up your environment
2. [Run the Quickstart](quickstart.md) - Build your first accelerator
3. [Learn Blueprints](../developer-guide/3-reference/blueprints.md) - Understand the configuration format
4. [Explore the CLI](../developer-guide/3-reference/cli.md) - Master the command-line tools

## Getting Help

- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/microsoft/brainsmith/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/microsoft/brainsmith/discussions)
- **Roadmap**: Follow planned features on the [Brainsmith Feature Roadmap](https://github.com/orgs/microsoft/projects/2017)
