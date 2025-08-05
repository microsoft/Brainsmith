############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Common utilities for Kernel Integrator demos."""

from .utils import (
    Timer,
    format_file_size,
    load_rtl_file,
    create_demo_header,
    create_progress_bar,
    highlight_code,
    save_demo_output
)

from .visualizer import (
    create_interface_diagram,
    create_comparison_chart,
    create_pragma_visualization,
    export_visualization
)

__all__ = [
    'Timer',
    'format_file_size',
    'load_rtl_file',
    'create_demo_header',
    'create_progress_bar',
    'highlight_code',
    'save_demo_output',
    'create_interface_diagram',
    'create_comparison_chart',
    'create_pragma_visualization',
    'export_visualization'
]