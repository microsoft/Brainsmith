# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Brainsmith kernel implementations.

Hardware kernels and backends for dataflow accelerators. Loaded during component
discovery to trigger @kernel and @backend decorator registration.

Access via registry:
    from brainsmith import get_kernel
    AddStreams = get_kernel('AddStreams')
"""

# Eager imports trigger decorator registration during discovery
# Each submodule defines __all__ to control exports
from .addstreams import *  # noqa: F403
from .channelwise import *  # noqa: F403
from .crop import *  # noqa: F403
from .duplicate_streams import *  # noqa: F403
from .elementwise_binary import *  # noqa: F403
from .layernorm import *  # noqa: F403
from .softmax import *  # noqa: F403
from .thresholding import *  # noqa: F403
