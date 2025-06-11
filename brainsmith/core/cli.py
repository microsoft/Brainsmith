"""
Simple CLI for BrainSmith Core - Function-Focused Design

Provides direct access to the core forge() function with minimal complexity.
Aligned with "Functions Over Frameworks" design axiom.
"""

import click
import sys
from pathlib import Path

from .api import forge, validate_blueprint


@click.group()
@click.version_option(version="0.5.0", prog_name="brainsmith")
def brainsmith():
    """BrainSmith: Simple FPGA accelerator generation."""
    pass


@brainsmith.command(name="forge")
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('blueprint_path', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output directory')
def forge_cmd(model_path, blueprint_path, output):
    """Generate FPGA accelerator from model and blueprint."""
    try:
        click.echo(f"🔨 Forging accelerator...")
        click.echo(f"   Model: {model_path}")
        click.echo(f"   Blueprint: {blueprint_path}")
        
        result = forge(model_path, blueprint_path, output_dir=output)
        
        click.echo("✅ Forge completed successfully!")
        if output:
            click.echo(f"📁 Results saved to: {output}")
        
        # Display basic result info
        if isinstance(result, dict):
            if result.get('success'):
                click.echo("   Status: Success")
            else:
                click.echo("   Status: Completed with warnings")
    
    except Exception as e:
        click.echo(f"❌ Forge failed: {e}", err=True)
        sys.exit(1)


@brainsmith.command()
@click.argument('blueprint_path', type=click.Path(exists=True))
def validate(blueprint_path):
    """Validate blueprint configuration."""
    try:
        click.echo(f"🔍 Validating blueprint: {blueprint_path}")
        
        is_valid, errors = validate_blueprint(blueprint_path)
        
        if is_valid:
            click.echo("✅ Blueprint is valid")
        else:
            click.echo("❌ Blueprint validation failed:")
            for error in errors:
                click.echo(f"  • {error}")
            sys.exit(1)
    
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}", err=True)
        sys.exit(1)


# Alias forge_cmd as the default command
@brainsmith.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('blueprint_path', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output directory')
def run(model_path, blueprint_path, output):
    """Simple alias for forge_cmd."""
    ctx = click.get_current_context()
    ctx.invoke(forge_cmd, model_path=model_path, blueprint_path=blueprint_path, output=output)


if __name__ == '__main__':
    brainsmith()