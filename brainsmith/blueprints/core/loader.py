"""
Blueprint loading system.

Handles loading blueprints from various sources including files,
URLs, and embedded definitions.
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

from .blueprint import Blueprint

logger = logging.getLogger(__name__)


class BlueprintLoader:
    """
    Blueprint loading system with support for multiple formats and sources.
    """
    
    def __init__(self):
        """Initialize blueprint loader."""
        self.supported_extensions = {'.json', '.yaml', '.yml'}
        self.logger = logging.getLogger("brainsmith.blueprints.loader")
    
    def load(self, source: Union[str, Path, Dict[str, Any]]) -> Blueprint:
        """
        Load blueprint from various sources.
        
        Args:
            source: File path, URL, or blueprint dictionary
            
        Returns:
            Blueprint instance
            
        Raises:
            ValueError: If source format is invalid
            FileNotFoundError: If file source doesn't exist
        """
        if isinstance(source, dict):
            # Load from dictionary
            return self.load_from_dict(source)
        
        elif isinstance(source, (str, Path)):
            source_path = Path(source)
            
            if source_path.exists():
                # Load from file
                return self.load_from_file(source_path)
            else:
                # Try to interpret as JSON/YAML string
                return self.load_from_string(str(source))
        
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
    
    def load_from_file(self, file_path: Union[str, Path]) -> Blueprint:
        """
        Load blueprint from file.
        
        Args:
            file_path: Path to blueprint file
            
        Returns:
            Blueprint instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Blueprint file not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {file_path.suffix}")
        
        self.logger.info(f"Loading blueprint from file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if file_path.suffix.lower() == '.json':
                return self.load_from_json(content)
            elif file_path.suffix.lower() in {'.yaml', '.yml'}:
                return self.load_from_yaml(content)
            
        except Exception as e:
            raise ValueError(f"Failed to load blueprint from {file_path}: {e}")
    
    def load_from_string(self, content: str) -> Blueprint:
        """
        Load blueprint from string content.
        
        Args:
            content: Blueprint content as string
            
        Returns:
            Blueprint instance
        """
        content = content.strip()
        
        # Try JSON first
        if content.startswith('{'):
            try:
                return self.load_from_json(content)
            except:
                pass
        
        # Try YAML
        try:
            return self.load_from_yaml(content)
        except:
            pass
        
        raise ValueError("Unable to parse blueprint content as JSON or YAML")
    
    def load_from_json(self, json_content: str) -> Blueprint:
        """
        Load blueprint from JSON content.
        
        Args:
            json_content: JSON string
            
        Returns:
            Blueprint instance
        """
        try:
            data = json.loads(json_content)
            return self.load_from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    
    def load_from_yaml(self, yaml_content: str) -> Blueprint:
        """
        Load blueprint from YAML content.
        
        Args:
            yaml_content: YAML string
            
        Returns:
            Blueprint instance
        """
        try:
            data = yaml.safe_load(yaml_content)
            return self.load_from_dict(data)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
    
    def load_from_dict(self, data: Dict[str, Any]) -> Blueprint:
        """
        Load blueprint from dictionary.
        
        Args:
            data: Blueprint data dictionary
            
        Returns:
            Blueprint instance
        """
        if not isinstance(data, dict):
            raise ValueError("Blueprint data must be a dictionary")
        
        if 'name' not in data:
            raise ValueError("Blueprint must have a 'name' field")
        
        try:
            blueprint = Blueprint.from_dict(data)
            self.logger.info(f"Loaded blueprint: {blueprint.name} v{blueprint.version}")
            return blueprint
        except Exception as e:
            raise ValueError(f"Failed to create blueprint from data: {e}")
    
    def load_multiple(self, sources: List[Union[str, Path, Dict[str, Any]]]) -> List[Blueprint]:
        """
        Load multiple blueprints from multiple sources.
        
        Args:
            sources: List of blueprint sources
            
        Returns:
            List of Blueprint instances
        """
        blueprints = []
        errors = []
        
        for source in sources:
            try:
                blueprint = self.load(source)
                blueprints.append(blueprint)
            except Exception as e:
                error_msg = f"Failed to load blueprint from {source}: {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)
        
        if errors:
            self.logger.warning(f"Failed to load {len(errors)} blueprints")
        
        self.logger.info(f"Successfully loaded {len(blueprints)} blueprints")
        return blueprints
    
    def discover_blueprints(self, directory: Union[str, Path], 
                          recursive: bool = True) -> List[Blueprint]:
        """
        Discover and load all blueprints in a directory.
        
        Args:
            directory: Directory to search
            recursive: Whether to search recursively
            
        Returns:
            List of discovered blueprints
        """
        directory = Path(directory)
        
        if not directory.exists():
            self.logger.warning(f"Directory not found: {directory}")
            return []
        
        self.logger.info(f"Discovering blueprints in: {directory}")
        
        # Find blueprint files
        blueprint_files = []
        
        if recursive:
            for ext in self.supported_extensions:
                pattern = f"**/*{ext}"
                blueprint_files.extend(directory.glob(pattern))
        else:
            for ext in self.supported_extensions:
                pattern = f"*{ext}"
                blueprint_files.extend(directory.glob(pattern))
        
        self.logger.info(f"Found {len(blueprint_files)} potential blueprint files")
        
        # Load blueprints
        return self.load_multiple(blueprint_files)
    
    def save_blueprint(self, blueprint: Blueprint, file_path: Union[str, Path], 
                      format: str = 'json') -> bool:
        """
        Save blueprint to file.
        
        Args:
            blueprint: Blueprint to save
            file_path: Output file path
            format: Output format ('json' or 'yaml')
            
        Returns:
            True if successful
        """
        file_path = Path(file_path)
        
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate content
            if format.lower() == 'json':
                content = blueprint.to_json()
            elif format.lower() in {'yaml', 'yml'}:
                content = blueprint.to_yaml()
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Saved blueprint to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save blueprint to {file_path}: {e}")
            return False
    
    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """
        Validate a blueprint file without fully loading it.
        
        Args:
            file_path: Path to blueprint file
            
        Returns:
            True if file is valid
        """
        try:
            blueprint = self.load_from_file(file_path)
            return blueprint.validate()
        except Exception as e:
            self.logger.error(f"Validation failed for {file_path}: {e}")
            return False
    
    def get_blueprint_info(self, source: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Get basic information about a blueprint without full loading.
        
        Args:
            source: Blueprint source
            
        Returns:
            Blueprint info dictionary or None if failed
        """
        try:
            if isinstance(source, (str, Path)) and Path(source).exists():
                # Quick parse for basic info
                with open(source, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if Path(source).suffix.lower() == '.json':
                    data = json.loads(content)
                else:
                    data = yaml.safe_load(content)
                
                return {
                    'name': data.get('name', 'unknown'),
                    'version': data.get('version', '1.0.0'),
                    'description': data.get('description', ''),
                    'libraries': list(data.get('libraries', {}).keys()),
                    'file_path': str(source)
                }
            
        except Exception as e:
            self.logger.error(f"Failed to get info for {source}: {e}")
        
        return None