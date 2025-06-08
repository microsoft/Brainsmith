"""
Blueprint metadata management.

Handles blueprint metadata including versioning, authorship,
and compatibility information.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
import json


class BlueprintMetadata:
    """
    Blueprint metadata management system.
    """
    
    def __init__(self, blueprint_name: str):
        """
        Initialize metadata manager.
        
        Args:
            blueprint_name: Name of the blueprint
        """
        self.blueprint_name = blueprint_name
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'author': 'unknown',
            'tags': [],
            'description': '',
            'compatibility': {
                'brainsmith_version': '>=0.4.0',
                'required_libraries': []
            }
        }
    
    def set_author(self, author: str):
        """Set blueprint author."""
        self.metadata['author'] = author
        self._update_timestamp()
    
    def add_tag(self, tag: str):
        """Add a tag to the blueprint."""
        if tag not in self.metadata['tags']:
            self.metadata['tags'].append(tag)
            self._update_timestamp()
    
    def set_description(self, description: str):
        """Set blueprint description."""
        self.metadata['description'] = description
        self._update_timestamp()
    
    def set_compatibility(self, brainsmith_version: str, required_libraries: List[str]):
        """Set compatibility requirements."""
        self.metadata['compatibility'] = {
            'brainsmith_version': brainsmith_version,
            'required_libraries': required_libraries
        }
        self._update_timestamp()
    
    def generate_hash(self, blueprint_content: Dict[str, Any]) -> str:
        """Generate content hash for the blueprint."""
        content_str = json.dumps(blueprint_content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def _update_timestamp(self):
        """Update the timestamp."""
        self.metadata['updated_at'] = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return self.metadata.copy()