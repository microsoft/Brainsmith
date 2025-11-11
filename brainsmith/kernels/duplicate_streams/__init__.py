# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DuplicateStreams kernel package."""

from .duplicate_streams import DuplicateStreams
from .duplicate_streams_hls import DuplicateStreams_hls

__all__ = ["DuplicateStreams", "DuplicateStreams_hls"]
