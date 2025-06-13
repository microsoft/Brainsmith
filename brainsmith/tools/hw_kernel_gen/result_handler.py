"""
Result handling system for unified generator.

This module provides classes for handling generation results and writing
generated files to the filesystem with proper organization and metadata.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .data import GenerationResult

logger = logging.getLogger(__name__)



class ResultHandler:
    """
    Handles generation results and file writing.
    
    Responsible for:
    - Writing generated files to organized directory structure
    - Creating metadata files with generation information
    - Handling file permissions and error recovery
    - Logging generation statistics
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize ResultHandler with output directory.
        
        Args:
            output_dir: Base directory for generated files
        """
        self.output_dir = Path(output_dir)
        self._ensure_output_directory()
        logger.info(f"Initialized ResultHandler with output directory: {self.output_dir}")
    
    def _ensure_output_directory(self) -> None:
        """Ensure output directory exists and is writable."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = self.output_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            
        except PermissionError:
            raise RuntimeError(f"No write permission for output directory: {self.output_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to create output directory {self.output_dir}: {e}")
    
    def write_result(self, result: GenerationResult) -> Path:
        """
        Write all generated files and metadata to output directory.
        
        Args:
            result: Generation result to write
            
        Returns:
            Path to the kernel-specific output directory
            
        Raises:
            RuntimeError: If file writing fails
        """
        try:
            logger.info(f"Writing generation result for {result.kernel_name}")
            
            # Create kernel-specific directory
            kernel_dir = self.output_dir / result.kernel_name
            kernel_dir.mkdir(exist_ok=True)
            
            files_written = []
            
            # Write all generated files
            for filename, content in result.generated_files.items():
                file_path = kernel_dir / filename
                try:
                    file_path.write_text(content, encoding='utf-8')
                    files_written.append(str(file_path))
                    logger.debug(f"Wrote file: {file_path}")
                except Exception as e:
                    error_msg = f"Failed to write file {filename}: {e}"
                    result.add_error(error_msg)
                    logger.error(error_msg)
            
            # Write generation metadata
            self._write_metadata(kernel_dir, result)
            
            # Write summary log
            self._write_summary_log(kernel_dir, result, files_written)
            
            logger.info(f"Successfully wrote {len(files_written)} files for {result.kernel_name}")
            return kernel_dir
            
        except Exception as e:
            error_msg = f"Failed to write result for {result.kernel_name}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _write_metadata(self, kernel_dir: Path, result: GenerationResult) -> None:
        """Write generation metadata to JSON file."""
        try:
            metadata = {
                "kernel_name": result.kernel_name,
                "source_file": str(result.source_file),
                "validation_passed": result.validation_passed,
                "success": result.is_success(),
                "errors": result.errors,
                "warnings": result.warnings,
                "generated_files": list(result.generated_files.keys()),
                "generation_time_ms": result.generation_time_ms,
                "summary": result.get_summary()
            }
            
            # Add template context info if available
            if result.template_context:
                metadata["template_context"] = {
                    "parameter_count": len(result.template_context.parameter_definitions),
                    "interface_count": len(result.template_context.interface_metadata),
                    "required_parameters": result.template_context.required_attributes,
                    "whitelisted_defaults": result.template_context.whitelisted_defaults
                }
            
            # Add kernel metadata info if available
            if result.kernel_metadata:
                metadata["kernel_metadata"] = {
                    "parameter_count": len(result.kernel_metadata.parameters),
                    "interface_count": len(result.kernel_metadata.interfaces),
                    "parameter_names": [p.name for p in result.kernel_metadata.parameters]
                }
            
            metadata_file = kernel_dir / "generation_metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2, default=str))
            
        except Exception as e:
            logger.warning(f"Failed to write metadata for {result.kernel_name}: {e}")
    
    def _write_summary_log(self, kernel_dir: Path, result: GenerationResult, files_written: List[str]) -> None:
        """Write human-readable summary log."""
        try:
            summary_lines = [
                f"Generation Summary for {result.kernel_name}",
                "=" * 50,
                f"Source File: {result.source_file}",
                f"Output Directory: {kernel_dir}",
                f"Success: {result.is_success()}",
                f"Validation Passed: {result.validation_passed}",
                f"Files Generated: {len(result.generated_files)}",
                ""
            ]
            
            if files_written:
                summary_lines.append("Generated Files:")
                for file_path in files_written:
                    summary_lines.append(f"  - {file_path}")
                summary_lines.append("")
            
            if result.errors:
                summary_lines.append("Errors:")
                for error in result.errors:
                    summary_lines.append(f"  - {error}")
                summary_lines.append("")
            
            if result.warnings:
                summary_lines.append("Warnings:")
                for warning in result.warnings:
                    summary_lines.append(f"  - {warning}")
                summary_lines.append("")
            
            if result.generation_time_ms:
                summary_lines.append(f"Generation Time: {result.generation_time_ms:.2f} ms")
            
            summary_file = kernel_dir / "generation_summary.txt"
            summary_file.write_text("\n".join(summary_lines))
            
        except Exception as e:
            logger.warning(f"Failed to write summary log for {result.kernel_name}: {e}")
    
    def cleanup_failed_generation(self, kernel_name: str) -> None:
        """Clean up files from a failed generation attempt."""
        try:
            kernel_dir = self.output_dir / kernel_name
            if kernel_dir.exists():
                import shutil
                shutil.rmtree(kernel_dir)
                logger.info(f"Cleaned up failed generation directory: {kernel_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up directory for {kernel_name}: {e}")
    
    def get_existing_results(self) -> List[str]:
        """Get list of existing kernel results in output directory."""
        try:
            if not self.output_dir.exists():
                return []
            
            return [
                d.name for d in self.output_dir.iterdir() 
                if d.is_dir() and not d.name.startswith('.')
            ]
        except Exception as e:
            logger.warning(f"Failed to list existing results: {e}")
            return []
    
    def load_result_metadata(self, kernel_name: str) -> Optional[Dict]:
        """Load generation metadata for a specific kernel."""
        try:
            metadata_file = self.output_dir / kernel_name / "generation_metadata.json"
            if metadata_file.exists():
                return json.loads(metadata_file.read_text())
            return None
        except Exception as e:
            logger.warning(f"Failed to load metadata for {kernel_name}: {e}")
            return None