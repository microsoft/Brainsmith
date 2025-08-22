"""
Simplified code generator for Brainsmith kernels.
"""

from pathlib import Path
from typing import Optional, Dict
from jinja2 import Environment, FileSystemLoader

from brainsmith.tools.kernel_integrator.metadata import KernelMetadata


class KernelGenerator:
    """Code generator for Brainsmith kernels."""
    
    ARTIFACTS = {
        'autohwcustomop': {
            'template': 'auto_hw_custom_op.py.j2',
            'filename': '{name}.py'
        },
        'rtlbackend': {
            'template': 'auto_rtl_backend.py.j2',
            'filename': '{name}_rtl.py'
        },
        'wrapper': {
            'template': 'rtl_wrapper.v.j2',
            'filename': '{name}_wrapper.v'
        }
    }
    
    def __init__(self, template_dir: Optional[Path] = None):
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
    
    def generate(self, artifact_type: str, kernel_metadata: KernelMetadata, output_dir: Path) -> Path:
        """Generate a single artifact and save to file."""
        config = self.ARTIFACTS[artifact_type]
        
        # Build context
        context = {'kernel_metadata': kernel_metadata}
        
        # Generate content
        try:
            template = self.env.get_template(config['template'])
            content = template.render(**context)
        except Exception as e:
            raise RuntimeError(f"Failed to generate {artifact_type}: {e}") from e
        
        # Write to file
        output_path = output_dir / config['filename'].format(name=kernel_metadata.name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding='utf-8')
        
        return output_path
    
    def generate_all(self, kernel_metadata: KernelMetadata, output_dir: Path) -> Dict[str, Path]:
        """Generate all artifacts and save to files."""
        output_dir = Path(output_dir)
        return {
            artifact_type: self.generate(artifact_type, kernel_metadata, output_dir)
            for artifact_type in self.ARTIFACTS
        }