"""HWCustomOp generator with direct KernelMetadata support."""
from typing import Dict, Any, List

from .base_v2 import GeneratorBase
from ..types.metadata import KernelMetadata
from ..types.rtl import ParameterCategory


class HWCustomOpGeneratorV2(GeneratorBase):
    """Generates HWCustomOp subclasses for FINN integration."""
    
    @property
    def name(self) -> str:
        return "hw_custom_op"
    
    @property
    def template_file(self) -> str:
        return "hw_custom_op_v2.py.j2"
    
    @property
    def output_pattern(self) -> str:
        return "{kernel_name}.py"
    
    def _get_specific_vars(self, metadata: KernelMetadata) -> Dict[str, Any]:
        """Get HWCustomOp-specific template variables."""
        return {
            # Only template-specific data that can't come from metadata
            'generation_timestamp': self._get_timestamp(),
        }
    
    
    
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for generation tracking."""
        from datetime import datetime
        return datetime.now().isoformat()
