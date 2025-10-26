# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Centralized constants for component registry system.

Eliminates magic strings and provides single source of truth for
source names, component types, and default configurations.
"""

# Protected plugin source names (cannot be overridden by users)
SOURCE_BRAINSMITH = 'brainsmith'
SOURCE_FINN = 'finn'
SOURCE_PROJECT = 'project'
SOURCE_USER = 'user'

# Protected sources set (immutable)
PROTECTED_SOURCES = frozenset([SOURCE_BRAINSMITH, SOURCE_FINN, SOURCE_PROJECT, SOURCE_USER])

# Default source resolution priority (first match wins)
DEFAULT_SOURCE_PRIORITY = [SOURCE_PROJECT, SOURCE_USER, SOURCE_BRAINSMITH, SOURCE_FINN]

# Source module prefixes for auto-detection
# Maps module prefix -> source name
SOURCE_MODULE_PREFIXES = {
    f'{SOURCE_BRAINSMITH}.': SOURCE_BRAINSMITH,
    f'{SOURCE_FINN}.': SOURCE_FINN,
    'qonnx.': 'qonnx',
}

# Component types (canonical singular forms)
COMPONENT_TYPES = ('step', 'kernel', 'backend')

# Component type plural->singular mapping (derived from COMPONENT_TYPES)
COMPONENT_TYPE_PLURALS = {f'{t}s': t for t in COMPONENT_TYPES}
