"""
BrainSmith Libraries Registry Validation System

Critical vulnerability mitigation: Registry-Reality Desynchronization
Prevents drift between AVAILABLE_* dictionaries and filesystem reality.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Callable

logger = logging.getLogger(__name__)


def validate_all_registries() -> Dict[str, Any]:
    """
    Comprehensive validation of all library registries.
    
    Returns:
        {
            'status': 'healthy' | 'degraded' | 'critical',
            'libraries': {
                'kernels': {'errors': [], 'warnings': [], 'component_count': 2},
                'transforms': {'errors': [], 'warnings': [], 'component_count': 10},
                'analysis': {'errors': [], 'warnings': [], 'component_count': 3},
                'blueprints': {'errors': [], 'warnings': [], 'component_count': 2},
                'automation': {'status': 'no_registry'}
            },
            'summary': {
                'total_components': 17,
                'failed_components': 0,
                'missing_files': [],
                'drift_detected': False
            }
        }
    """
    report = {
        'status': 'healthy',
        'libraries': {},
        'summary': {
            'total_components': 0,
            'failed_components': 0,
            'missing_files': [],
            'drift_detected': False
        }
    }
    
    # Validate kernels library
    try:
        from .kernels import AVAILABLE_KERNELS
        from .kernels import get_kernel
        
        kernel_errors = validate_registry_integrity('kernels', AVAILABLE_KERNELS, get_kernel)
        report['libraries']['kernels'] = {
            'errors': kernel_errors,
            'warnings': [],
            'component_count': len(AVAILABLE_KERNELS)
        }
        
        if kernel_errors:
            report['summary']['failed_components'] += len(kernel_errors)
            
    except Exception as e:
        report['libraries']['kernels'] = {
            'errors': [f"Failed to validate kernels registry: {e}"],
            'warnings': [],
            'component_count': 0
        }
        report['summary']['failed_components'] += 1

    # Validate transforms library
    try:
        from .transforms import AVAILABLE_TRANSFORMS
        from .transforms import get_transform
        
        transform_errors = validate_registry_integrity('transforms', AVAILABLE_TRANSFORMS, get_transform)
        report['libraries']['transforms'] = {
            'errors': transform_errors,
            'warnings': [],
            'component_count': len(AVAILABLE_TRANSFORMS)
        }
        
        if transform_errors:
            report['summary']['failed_components'] += len(transform_errors)
            
    except Exception as e:
        report['libraries']['transforms'] = {
            'errors': [f"Failed to validate transforms registry: {e}"],
            'warnings': [],
            'component_count': 0
        }
        report['summary']['failed_components'] += 1

    # Validate analysis library
    try:
        from .analysis import AVAILABLE_ANALYSIS_TOOLS
        from .analysis import get_analysis_tool
        
        analysis_errors = validate_registry_integrity('analysis', AVAILABLE_ANALYSIS_TOOLS, get_analysis_tool)
        report['libraries']['analysis'] = {
            'errors': analysis_errors,
            'warnings': [],
            'component_count': len(AVAILABLE_ANALYSIS_TOOLS)
        }
        
        if analysis_errors:
            report['summary']['failed_components'] += len(analysis_errors)
            
    except Exception as e:
        report['libraries']['analysis'] = {
            'errors': [f"Failed to validate analysis registry: {e}"],
            'warnings': [],
            'component_count': 0
        }
        report['summary']['failed_components'] += 1

    # Validate blueprints library
    try:
        from .blueprints import AVAILABLE_BLUEPRINTS
        from .blueprints import get_blueprint
        
        blueprint_errors = validate_registry_integrity('blueprints', AVAILABLE_BLUEPRINTS, get_blueprint)
        report['libraries']['blueprints'] = {
            'errors': blueprint_errors,
            'warnings': [],
            'component_count': len(AVAILABLE_BLUEPRINTS)
        }
        
        if blueprint_errors:
            report['summary']['failed_components'] += len(blueprint_errors)
            
    except Exception as e:
        report['libraries']['blueprints'] = {
            'errors': [f"Failed to validate blueprints registry: {e}"],
            'warnings': [],
            'component_count': 0
        }
        report['summary']['failed_components'] += 1

    # Automation library has no registry (by design)
    report['libraries']['automation'] = {'status': 'no_registry'}

    # Calculate total components
    report['summary']['total_components'] = sum(
        lib.get('component_count', 0) for lib in report['libraries'].values()
        if isinstance(lib, dict) and 'component_count' in lib
    )

    # Check for drift in development mode
    if os.getenv('BRAINSMITH_DEV_MODE'):
        drift_report = suggest_registry_updates()
        if any(drift_report.values()):
            report['summary']['drift_detected'] = True
            for lib_name, drift in drift_report.items():
                if lib_name in report['libraries']:
                    report['libraries'][lib_name]['warnings'].extend([
                        f"Unregistered components: {drift.get('unregistered', [])}",
                        f"Orphaned entries: {drift.get('orphaned', [])}"
                    ])

    # Determine overall status
    if report['summary']['failed_components'] > 0:
        if report['summary']['failed_components'] >= report['summary']['total_components'] / 2:
            report['status'] = 'critical'
        else:
            report['status'] = 'degraded'
    elif report['summary']['drift_detected']:
        report['status'] = 'degraded'

    logger.info(f"Registry validation completed: {report['status']} status, "
               f"{report['summary']['total_components']} components, "
               f"{report['summary']['failed_components']} failures")

    return report


def validate_registry_integrity(library_name: str, registry: Dict, loader_func: Callable) -> List[str]:
    """
    Validate specific registry entries can load successfully.
    
    Args:
        library_name: Name of the library being validated
        registry: Registry dictionary to validate
        loader_func: Function to load/access components
        
    Returns:
        List of error messages for failed components
    """
    errors = []
    
    for component_name in registry.keys():
        try:
            # Try to load/access the component
            loader_func(component_name)
            logger.debug(f"✅ {library_name}.{component_name} loads successfully")
            
        except FileNotFoundError as e:
            error_msg = f"{library_name}.{component_name}: File not found - {e}"
            errors.append(error_msg)
            logger.error(error_msg)
            
        except ImportError as e:
            error_msg = f"{library_name}.{component_name}: Import error - {e}"
            errors.append(error_msg)
            logger.error(error_msg)
                
        except Exception as e:
            error_msg = f"{library_name}.{component_name}: Unexpected error - {e}"
            errors.append(error_msg)
            logger.error(error_msg)
    
    return errors


def suggest_registry_updates() -> Dict[str, Dict[str, List[str]]]:
    """
    Development mode: detect unregistered components and orphaned entries.
    
    Returns:
        {
            'kernels': {
                'unregistered': ['new_kernel_dir'],
                'orphaned': ['deleted_kernel_name'] 
            },
            'transforms': {
                'unregistered': [],
                'orphaned': []
            },
            # ... etc
        }
    """
    if not os.getenv('BRAINSMITH_DEV_MODE'):
        return {}
    
    suggestions = {}
    
    # Check kernels for filesystem vs registry mismatches
    try:
        from .kernels import AVAILABLE_KERNELS
        
        # Scan filesystem for kernel packages
        kernels_dir = Path(__file__).parent / 'kernels'
        discovered_kernels = set()
        
        if kernels_dir.exists():
            for item in kernels_dir.iterdir():
                if item.is_dir() and (item / 'kernel.yaml').exists():
                    # Skip special directories
                    if not item.name.startswith('.') and item.name not in ['__pycache__']:
                        discovered_kernels.add(item.name)
        
        registered_kernels = set(AVAILABLE_KERNELS.keys())
        
        suggestions['kernels'] = {
            'unregistered': list(discovered_kernels - registered_kernels),
            'orphaned': list(registered_kernels - discovered_kernels)
        }
        
        if suggestions['kernels']['unregistered']:
            logger.warning(f"DEV MODE: Unregistered kernels found: {suggestions['kernels']['unregistered']}")
        if suggestions['kernels']['orphaned']:
            logger.warning(f"DEV MODE: Orphaned kernel entries: {suggestions['kernels']['orphaned']}")
            
    except Exception as e:
        logger.error(f"Failed to check kernel drift: {e}")
        suggestions['kernels'] = {'unregistered': [], 'orphaned': []}

    # Similar checks could be added for other libraries with filesystem components
    # For now, transforms/analysis/blueprints have simpler structures
    
    return suggestions


def run_health_check() -> bool:
    """
    Quick health check - returns True if all registries are healthy.
    Used for CI/CD pipelines and development checks.
    """
    report = validate_all_registries()
    return report['status'] == 'healthy'


if __name__ == "__main__":
    """CLI interface for registry validation."""
    import sys
    
    print("🔍 BrainSmith Libraries Registry Validation")
    print("=" * 50)
    
    report = validate_all_registries()
    
    print(f"\n📊 Overall Status: {report['status'].upper()}")
    print(f"📦 Total Components: {report['summary']['total_components']}")
    print(f"❌ Failed Components: {report['summary']['failed_components']}")
    
    if report['summary']['drift_detected']:
        print(f"⚠️  Registry drift detected (see warnings)")
    
    print("\n📋 Library Details:")
    for lib_name, lib_report in report['libraries'].items():
        if isinstance(lib_report, dict) and 'component_count' in lib_report:
            status_icon = "✅" if not lib_report['errors'] else "❌"
            print(f"  {status_icon} {lib_name}: {lib_report['component_count']} components")
            
            for error in lib_report['errors']:
                print(f"    ❌ {error}")
            for warning in lib_report['warnings']:
                print(f"    ⚠️  {warning}")
        else:
            print(f"  ℹ️  {lib_name}: {lib_report.get('status', 'unknown')}")
    
    # Exit with appropriate code
    exit_code = 0 if report['status'] == 'healthy' else 1
    if report['status'] == 'critical':
        exit_code = 2
        
    print(f"\n🏁 Validation complete (exit code: {exit_code})")
    sys.exit(exit_code)