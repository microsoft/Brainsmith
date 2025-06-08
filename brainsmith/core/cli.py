"""
Command-line interface using existing components.

This module provides the CLI for Brainsmith with hierarchical exit points
and extensible structure around existing functionality.
"""

import click
from pathlib import Path
import logging
import sys
from typing import Dict, Any

# Setup logging for CLI
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import API functions
from .api import (
    brainsmith_explore, brainsmith_roofline, brainsmith_dataflow_analysis,
    brainsmith_generate, validate_blueprint, brainsmith_workflow
)

@click.group()
@click.version_option(version="0.4.0", prog_name="brainsmith")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--quiet', '-q', is_flag=True, help='Suppress non-error output')
def brainsmith(verbose, quiet):
    """
    Brainsmith: Meta-toolchain for FPGA accelerator synthesis.
    
    Extensible structure using existing components with hierarchical exit points:
    - roofline: Quick analytical bounds using existing analysis tools
    - dataflow: Transform application and estimation using existing components  
    - generate: Full RTL/HLS generation using existing FINN flow
    
    Examples:
        brainsmith explore model.onnx blueprint.yaml --exit-point roofline
        brainsmith roofline model.onnx blueprint.yaml
        brainsmith generate model.onnx blueprint.yaml --output ./results
    """
    # Configure logging based on verbosity
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    elif quiet:
        logging.getLogger().setLevel(logging.ERROR)
    else:
        logging.getLogger().setLevel(logging.INFO)

@brainsmith.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('blueprint_path', type=click.Path(exists=True))
@click.option('--exit-point', '-e', 
              type=click.Choice(['roofline', 'dataflow_analysis', 'dataflow_generation']),
              default='dataflow_generation',
              help='Hierarchical exit point for exploration')
@click.option('--output', '-o', type=click.Path(), help='Output directory for results')
@click.option('--workflow', '-w', type=click.Choice(['fast', 'standard', 'comprehensive']),
              help='Predefined workflow type (overrides exit-point)')
def explore(model_path, blueprint_path, exit_point, output, workflow):
    """
    Explore design space using existing components with hierarchical exit points.
    
    Uses extensible structure around existing Brainsmith functionality:
    
    Exit Points:
    - roofline: Quick analytical performance bounds (30s)
    - dataflow_analysis: Transform + estimation without RTL (2min)  
    - dataflow_generation: Full RTL/HLS generation (10min)
    
    Examples:
        brainsmith explore model.onnx blueprint.yaml --exit-point roofline
        brainsmith explore model.onnx blueprint.yaml -e dataflow_analysis -o results/
        brainsmith explore model.onnx blueprint.yaml  # Full generation (default)
        brainsmith explore model.onnx blueprint.yaml --workflow fast  # Same as roofline
    """
    try:
        click.echo(f"🚀 Starting Brainsmith exploration...")
        click.echo(f"   Model: {model_path}")
        click.echo(f"   Blueprint: {blueprint_path}")
        
        # Use workflow if specified, otherwise use exit-point
        if workflow:
            click.echo(f"   Workflow: {workflow}")
            results, analysis = brainsmith_workflow(model_path, blueprint_path, workflow, output_dir=output)
            used_exit_point = analysis.get('exit_point', workflow)
        else:
            click.echo(f"   Exit point: {exit_point}")
            results, analysis = brainsmith_explore(model_path, blueprint_path, exit_point, output)
            used_exit_point = exit_point
        
        click.echo(f"✅ Exploration complete using existing components!")
        click.echo(f"   Exit point: {used_exit_point}")
        click.echo(f"   Method: {analysis.get('method', 'existing_tools')}")
        
        _display_summary_existing(analysis, used_exit_point)
        
        if output:
            click.echo(f"📁 Results saved to: {output}")
            
    except FileNotFoundError as e:
        click.echo(f"❌ File not found: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"❌ Invalid configuration: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Exploration failed: {e}")
        click.echo(f"❌ Exploration failed: {e}", err=True)
        sys.exit(1)

@brainsmith.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('blueprint_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
def roofline(model_path, blueprint_path, output):
    """
    Quick roofline analysis using existing analysis tools.
    
    Performs analytical performance bounds estimation without hardware
    generation. Fastest analysis option (~30 seconds).
    
    Uses existing analysis capabilities to provide:
    - Computational intensity analysis
    - Performance bounds estimation  
    - Memory bandwidth requirements
    - Quick feasibility assessment
    """
    try:
        click.echo("📊 Starting roofline analysis using existing tools...")
        click.echo(f"   Model: {model_path}")
        click.echo(f"   Blueprint: {blueprint_path}")
        
        results, analysis = brainsmith_roofline(model_path, blueprint_path, output)
        
        click.echo("✅ Roofline analysis complete!")
        _display_roofline_summary_existing(analysis)
        
        if output:
            click.echo(f"📁 Results saved to: {output}")
            
    except Exception as e:
        logger.error(f"Roofline analysis failed: {e}")
        click.echo(f"❌ Analysis failed: {e}", err=True)
        sys.exit(1)

@brainsmith.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('blueprint_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
def dataflow(model_path, blueprint_path, output):
    """
    Dataflow analysis using existing transforms and estimation.
    
    Applies existing model transforms and provides dataflow-level
    performance estimation without RTL generation (~2 minutes).
    
    Uses existing components to provide:
    - Model transformation using existing transforms from steps/
    - Kernel mapping to existing custom operations
    - Performance estimation without synthesis
    - Resource utilization estimates
    """
    try:
        click.echo("⚡ Starting dataflow analysis using existing transforms...")
        click.echo(f"   Model: {model_path}")
        click.echo(f"   Blueprint: {blueprint_path}")
        
        results, analysis = brainsmith_dataflow_analysis(model_path, blueprint_path, output)
        
        click.echo("✅ Dataflow analysis complete!")
        _display_dataflow_summary_existing(analysis)
        
        if output:
            click.echo(f"📁 Results saved to: {output}")
            
    except Exception as e:
        logger.error(f"Dataflow analysis failed: {e}")
        click.echo(f"❌ Analysis failed: {e}", err=True)
        sys.exit(1)

@brainsmith.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('blueprint_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
def generate(model_path, blueprint_path, output):
    """
    Full RTL/HLS generation using existing FINN flow.
    
    Performs complete optimization and hardware generation using
    existing DataflowBuildConfig workflow (~10 minutes).
    
    Uses existing components to provide:
    - Complete optimization using existing strategies from dse/
    - RTL/HLS generation using existing FINN DataflowBuildConfig
    - Synthesis results and performance metrics
    - Ready-to-use hardware implementation files
    """
    try:
        click.echo("🔧 Starting full generation using existing FINN flow...")
        click.echo(f"   Model: {model_path}")
        click.echo(f"   Blueprint: {blueprint_path}")
        
        results, analysis = brainsmith_generate(model_path, blueprint_path, output)
        
        click.echo("✅ Generation complete!")
        _display_generation_summary_existing(analysis)
        
        if output:
            click.echo(f"📁 Results saved to: {output}")
            
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        click.echo(f"❌ Generation failed: {e}", err=True)
        sys.exit(1)

@brainsmith.command()
@click.argument('blueprint_path', type=click.Path(exists=True))
def validate(blueprint_path):
    """
    Validate blueprint configuration for existing components.
    
    Checks blueprint YAML configuration to ensure:
    - Valid structure and required fields
    - References to existing components only
    - Proper parameter specifications
    - Compatibility with current Brainsmith capabilities
    """
    try:
        click.echo(f"🔍 Validating blueprint: {blueprint_path}")
        
        is_valid, errors = validate_blueprint(blueprint_path)
        
        if is_valid:
            click.echo("✅ Blueprint is valid and uses existing components only")
        else:
            click.echo("❌ Blueprint validation failed:")
            for error in errors:
                click.echo(f"  • {error}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        click.echo(f"❌ Validation failed: {e}", err=True)
        sys.exit(1)

@brainsmith.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('blueprint_path', type=click.Path(exists=True))
@click.option('--type', '-t', 'workflow_type',
              type=click.Choice(['fast', 'standard', 'comprehensive']),
              default='standard',
              help='Workflow type')
@click.option('--output', '-o', type=click.Path(), help='Output directory')
def workflow(model_path, blueprint_path, workflow_type, output):
    """
    Execute predefined workflows using existing components.
    
    Workflow Types:
    - fast: Roofline analysis only (~30s)
    - standard: Dataflow analysis (~2min) [default]  
    - comprehensive: Full generation (~10min)
    
    Each workflow uses only existing Brainsmith components while
    providing different levels of analysis depth and time investment.
    """
    try:
        click.echo(f"🔄 Executing {workflow_type} workflow...")
        click.echo(f"   Model: {model_path}")
        click.echo(f"   Blueprint: {blueprint_path}")
        
        results, analysis = brainsmith_workflow(model_path, blueprint_path, workflow_type, output_dir=output)
        
        click.echo(f"✅ {workflow_type.title()} workflow complete!")
        _display_workflow_summary(analysis, workflow_type)
        
        if output:
            click.echo(f"📁 Results saved to: {output}")
            
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        click.echo(f"❌ Workflow failed: {e}", err=True)
        sys.exit(1)

# Display helper functions

def _display_summary_existing(analysis: Dict, exit_point: str):
    """Display summary based on exit point using existing tools."""
    click.echo(f"  📊 Analysis method: {analysis.get('method', 'existing_tools')}")
    click.echo(f"  🔧 Components source: {analysis.get('components_source', 'existing_only')}")
    click.echo(f"  📚 Libraries used: {', '.join(analysis.get('libraries_status', {}).keys())}")
    
    if 'error' in analysis:
        click.echo(f"  ⚠️  Warning: {analysis['error']}")

def _display_roofline_summary_existing(analysis: Dict):
    """Display roofline summary using existing analysis."""
    roofline_data = analysis.get('roofline_specific', {})
    click.echo("  📊 Roofline Analysis Results:")
    click.echo(f"     • Analysis type: {roofline_data.get('analysis_type', 'roofline_bounds')}")
    click.echo(f"     • Performance bounds: {roofline_data.get('performance_bounds', 'computed')}")
    click.echo(f"     • Method: Existing analysis tools")
    
    # Display recommendations
    recommendations = roofline_data.get('recommendations', [])
    if recommendations:
        click.echo("  💡 Recommendations:")
        for rec in recommendations:
            click.echo(f"     • {rec}")

def _display_dataflow_summary_existing(analysis: Dict):
    """Display dataflow summary using existing components."""
    dataflow_data = analysis.get('dataflow_specific', {})
    click.echo("  ⚡ Dataflow Analysis Results:")
    click.echo(f"     • Analysis type: {dataflow_data.get('analysis_type', 'dataflow_estimation')}")
    click.echo(f"     • Transforms applied: Using existing transforms from steps/")
    click.echo(f"     • Kernel mapping: Using existing custom operations")
    click.echo(f"     • Performance estimation: Using existing estimation tools")
    
    # Display recommendations
    recommendations = dataflow_data.get('recommendations', [])
    if recommendations:
        click.echo("  💡 Recommendations:")
        for rec in recommendations:
            click.echo(f"     • {rec}")

def _display_generation_summary_existing(analysis: Dict):
    """Display generation summary using existing FINN flow."""
    generation_data = analysis.get('generation_specific', {})
    click.echo("  🔧 Generation Results:")
    click.echo(f"     • Analysis type: {generation_data.get('analysis_type', 'complete_generation')}")
    click.echo(f"     • RTL files: {generation_data.get('rtl_files_count', 0)}")
    click.echo(f"     • HLS files: {generation_data.get('hls_files_count', 0)}")
    click.echo(f"     • Synthesis status: {generation_data.get('synthesis_status', 'unknown')}")
    click.echo(f"     • Method: Existing FINN DataflowBuildConfig")
    
    # Display recommendations
    recommendations = generation_data.get('recommendations', [])
    if recommendations:
        click.echo("  💡 Recommendations:")
        for rec in recommendations:
            click.echo(f"     • {rec}")

def _display_workflow_summary(analysis: Dict, workflow_type: str):
    """Display workflow-specific summary."""
    click.echo(f"  🔄 Workflow Summary:")
    click.echo(f"     • Type: {workflow_type}")
    click.echo(f"     • Exit point: {analysis.get('exit_point', 'unknown')}")
    click.echo(f"     • Method: {analysis.get('method', 'existing_tools')}")
    click.echo(f"     • Components: Existing only")
    
    # Display workflow-specific information
    if workflow_type == 'fast':
        click.echo(f"     • Speed: Fastest analysis (~30s)")
        click.echo(f"     • Output: Performance bounds and feasibility")
    elif workflow_type == 'standard':
        click.echo(f"     • Speed: Balanced analysis (~2min)")
        click.echo(f"     • Output: Detailed estimation without RTL")
    elif workflow_type == 'comprehensive':
        click.echo(f"     • Speed: Complete analysis (~10min)")
        click.echo(f"     • Output: Ready-to-use RTL/HLS files")

# Convenience commands for common use cases

@brainsmith.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--device', '-d', default='xcvu9p-flga2104-2-i', help='Target FPGA device')
@click.option('--output', '-o', type=click.Path(), help='Output directory')
def quick(model_path, device, output):
    """
    Quick exploration using default configuration and existing components.
    
    Creates a minimal blueprint automatically and runs standard dataflow analysis.
    Useful for rapid prototyping and initial exploration.
    """
    try:
        click.echo(f"⚡ Quick exploration for: {model_path}")
        click.echo(f"   Target device: {device}")
        
        # Create minimal blueprint
        blueprint_content = _create_minimal_blueprint(device)
        blueprint_path = Path("quick_blueprint.yaml")
        
        with open(blueprint_path, 'w') as f:
            f.write(blueprint_content)
        
        click.echo(f"   Generated blueprint: {blueprint_path}")
        
        # Run standard workflow
        results, analysis = brainsmith_workflow(model_path, str(blueprint_path), 'standard', output_dir=output)
        
        click.echo("✅ Quick exploration complete!")
        _display_summary_existing(analysis, analysis.get('exit_point', 'dataflow_analysis'))
        
        # Cleanup temporary blueprint
        blueprint_path.unlink()
        
    except Exception as e:
        logger.error(f"Quick exploration failed: {e}")
        click.echo(f"❌ Quick exploration failed: {e}", err=True)
        sys.exit(1)

def _create_minimal_blueprint(device: str) -> str:
    """Create minimal blueprint YAML for quick exploration."""
    return f"""
name: "quick_exploration"
description: "Auto-generated minimal blueprint for quick exploration"

# Use existing components only
kernels:
  available: []

transforms:
  pipeline:
    - name: "streamlining"
      enabled: true
      source: "steps.streamlining"

hw_optimization:
  strategies:
    - name: "random_search"
      algorithm: "random"
      budget: 10
      source: "dse.simple.random"

finn_interface:
  legacy_config:
    fpga_part: "{device}"
    generate_outputs: ["estimate"]

constraints:
  target_device: "{device}"
  resource_limits:
    lut_utilization: 0.85

metadata:
  auto_generated: true
  components_used: "existing_only"
"""

if __name__ == '__main__':
    brainsmith()