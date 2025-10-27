# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""AddStreams kernel module.

Provides element-wise addition of two integer streams with hardware acceleration.
"""

from .addstreams import AddStreams, ADDSTREAMS_SCHEMA
from .addstreams_hls import AddStreams_hls

__all__ = ["AddStreams", "ADDSTREAMS_SCHEMA", "AddStreams_hls"]
