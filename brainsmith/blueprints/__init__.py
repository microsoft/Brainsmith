# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from brainsmith.blueprints.bert import BUILD_BERT_STEPS
from brainsmith.blueprints.finnloop import BUILD_FINNLOOP_STEPS

REGISTRY = {
    "bert": BUILD_BERT_STEPS,
    "finnloop": BUILD_FINNLOOP_STEPS,
}
