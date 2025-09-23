############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Base schema class for dataflow components."""

from abc import ABC, abstractmethod
from typing import List


class BaseSchema(ABC):
    """Base class for all Schema classes
    
    Schemas specify constraints, relationships, and validation rules.
    They define "what should be" rather than "what is".
    """
    
    @abstractmethod
    def validate(self) -> List[str]:
        """Validate the schema for internal consistency
        
        Returns:
            List of validation errors (empty if valid)
        """
        pass
