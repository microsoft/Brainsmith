"""Model-specific transforms."""

from .remove_bert_head import RemoveBertHead
from .remove_bert_tail import RemoveBertTail

__all__ = ["RemoveBertHead", "RemoveBertTail"]