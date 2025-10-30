# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Centralized constants for component registry system.

Eliminates magic strings and provides single source of truth for
source names, component types, and default configurations.

Source Types:
    - Core namespace: 'brainsmith' - internal components loaded via direct import
    - Entry points: 'finn', etc. - discovered via pip package entry points
    - Filesystem: 'project', custom - loaded from configured directory paths
"""

# Core namespace reserved for brainsmith internal components
CORE_NAMESPACE = 'brainsmith'

# Standard source names
SOURCE_BRAINSMITH = 'brainsmith'
SOURCE_FINN = 'finn'
SOURCE_PROJECT = 'project'

# Known entry point sources (discovered at runtime, not filesystem-based)
# These are discovered via importlib.metadata.entry_points but we list known ones
# for validation purposes
KNOWN_ENTRY_POINTS = frozenset([SOURCE_FINN])

# Protected sources set - sources with special handling in source priority resolution
# Note: This is about lookup priority, not about filesystem paths. Only project
# has a configurable filesystem path.
PROTECTED_SOURCES = frozenset([SOURCE_BRAINSMITH, SOURCE_FINN, SOURCE_PROJECT])

# Default source resolution priority (first match wins)
DEFAULT_SOURCE_PRIORITY = [SOURCE_PROJECT, SOURCE_BRAINSMITH, SOURCE_FINN]

# Source module prefixes for auto-detection
# Maps module prefix -> source name
SOURCE_MODULE_PREFIXES = {
    f'{SOURCE_BRAINSMITH}.': SOURCE_BRAINSMITH,
    f'{SOURCE_FINN}.': SOURCE_FINN,
    'qonnx.': 'qonnx',
}
