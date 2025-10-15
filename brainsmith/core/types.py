# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Type definitions for Brainsmith core.

This module provides Enums and type definitions used across
the core system for type safety and consistency.
"""

from enum import Enum


class SegmentStatus(Enum):
    """Status of a DSE segment execution.

    Tracks the execution state of a segment in the design space
    exploration tree.
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class OutputType(Enum):
    """Blueprint output product type.

    Defines what artifacts the build system should generate:
    - ESTIMATES: Resource estimates only (fast)
    - RTL: RTL simulation and IP generation
    - BITFILE: Full bitstream generation (slow)
    """
    ESTIMATES = "estimates"
    RTL = "rtl"
    BITFILE = "bitfile"
