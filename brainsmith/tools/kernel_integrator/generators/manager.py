"""
Generator discovery and management for KI.

The GeneratorManager handles discovery of generator classes, loading templates,
and rendering output through the Jinja2 template engine.
"""

import glob
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from .base import GeneratorBase
from ..templates import TemplateContext

logger = logging.getLogger(__name__)


class GeneratorManagerError(Exception):
    """Custom exception for GeneratorManager errors."""
    pass


class GeneratorManager:
    """
    Manages generator discovery, loading, and template rendering.
    
    Automatically discovers generator classes from *_generator.py files
    and provides template rendering capabilities through Jinja2.
    """
    
    def __init__(self, generator_dir: Path, template_dir: Path):
        """
        Initialize GeneratorManager.
        
        Args:
            generator_dir: Directory containing generator Python files
            template_dir: Directory containing Jinja2 template files
        """
        self.generator_dir = Path(generator_dir)
        self.template_dir = Path(template_dir)
        self.generators: Dict[str, GeneratorBase] = {}
        
        # Initialize Jinja2 environment
        try:
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                trim_blocks=True,
                lstrip_blocks=True
            )
            
            # Add custom filters and tests
            self.jinja_env.filters['repr'] = repr
            self.jinja_env.tests['none'] = lambda x: x is None
            
            # Add global functions
            self.jinja_env.globals['enumerate'] = enumerate
            
            logger.info(f"Initialized Jinja2 environment with templates from {self.template_dir}")
        except Exception as e:
            raise GeneratorManagerError(f"Failed to initialize Jinja2 environment: {e}")
        
        # Discover generators
        self._discover_generators()
        logger.info(f"Discovered {len(self.generators)} generators")
    
    def _discover_generators(self) -> None:
        """Auto-discover generator definitions using package introspection."""
        # Import the generators package to ensure all generators are loaded
        try:
            from . import hw_custom_op_generator, rtl_wrapper_generator, rtl_backend_generator
            
            # Use introspection to find all GeneratorBase subclasses in current namespace
            import sys
            current_module = sys.modules[__name__]
            generators_package = sys.modules[current_module.__package__]
            
            # Find all GeneratorBase subclasses in the package
            for name in dir(generators_package):
                obj = getattr(generators_package, name)
                if (inspect.isclass(obj) and 
                    issubclass(obj, GeneratorBase) and
                    obj != GeneratorBase):
                    
                    try:
                        generator_instance = obj()
                        if generator_instance.validate():
                            if generator_instance.name in self.generators:
                                logger.warning(f"Duplicate generator name: {generator_instance.name}")
                            self.generators[generator_instance.name] = generator_instance
                            logger.debug(f"Registered generator: {generator_instance}")
                        else:
                            logger.warning(f"Invalid generator configuration: {obj.__name__}")
                    except Exception as e:
                        logger.error(f"Failed to instantiate generator {obj.__name__}: {e}")
                        
        except ImportError as e:
            logger.error(f"Failed to import generators package: {e}")
            # Fallback to dynamic discovery if package imports fail
            self._discover_generators_dynamically()
    
    def _discover_generators_dynamically(self) -> None:
        """Fallback: Dynamic generator discovery from files."""
        generator_pattern = str(self.generator_dir / "*_generator.py")
        
        for generator_file in glob.glob(generator_pattern):
            try:
                module = self._import_generator_module(generator_file)
                
                # Find GeneratorBase subclasses
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, GeneratorBase) and
                        obj != GeneratorBase):
                        
                        # Instantiate and validate generator
                        generator_instance = obj()
                        if generator_instance.validate():
                            if generator_instance.name in self.generators:
                                logger.warning(f"Duplicate generator name: {generator_instance.name}")
                            self.generators[generator_instance.name] = generator_instance
                            logger.debug(f"Registered generator: {generator_instance}")
                        else:
                            logger.warning(f"Invalid generator configuration: {obj.__name__}")
                            
            except Exception as e:
                logger.error(f"Failed to load generator from {generator_file}: {e}")
                continue
    
    def _import_generator_module(self, generator_file: str):
        """Import a generator module from file path."""
        module_name = Path(generator_file).stem
        
        # Add the parent directory to sys.path temporarily to help with imports
        import sys
        parent_dir = str(Path(generator_file).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, generator_file)
            if spec is None or spec.loader is None:
                raise GeneratorManagerError(f"Cannot load module from {generator_file}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            logger.error(f"Failed to import generator from {generator_file}: {e}")
            raise
    
    def list_generators(self) -> List[str]:
        """
        Get list of available generator names.
        
        Returns:
            List of generator names
        """
        return list(self.generators.keys())
    
    def get_generator(self, name: str) -> Optional[GeneratorBase]:
        """
        Get generator by name.
        
        Args:
            name: Generator name
            
        Returns:
            Generator instance or None if not found
        """
        return self.generators.get(name)
    
    def render_generator(
        self, 
        generator_name: str, 
        context: TemplateContext
    ) -> str:
        """
        Render template via generator with processed context.
        
        Args:
            generator_name: Name of generator to use
            context: Full template context
            
        Returns:
            Rendered template content
            
        Raises:
            GeneratorManagerError: If generator not found or rendering fails
        """
        if generator_name not in self.generators:
            available = ", ".join(self.generators.keys())
            raise GeneratorManagerError(
                f"Generator '{generator_name}' not found. Available: {available}"
            )
        
        generator = self.generators[generator_name]
        
        try:
            # Let generator process context
            processed_context = generator.process_context(context)
            
            # Convert TemplateContext to template variables
            template_vars = self._convert_context_to_template_vars(processed_context)
            
            # Get template file (with potential fallbacks)
            try:
                template_file = generator.get_template_file(self.jinja_env)
                template = self.jinja_env.get_template(template_file)
            except TemplateNotFound:
                raise GeneratorManagerError(
                    f"Template not found: {generator.template_file} for generator {generator_name}"
                )
            
            rendered_content = template.render(**template_vars)
            logger.debug(f"Successfully rendered {generator_name}")
            return rendered_content
            
        except Exception as e:
            logger.error(f"Failed to render {generator_name}: {e}")
            raise GeneratorManagerError(f"Rendering failed for {generator_name}: {e}")
    
    def get_output_filename(self, generator_name: str, kernel_name: str) -> str:
        """
        Get output filename for a generator.
        
        Args:
            generator_name: Name of generator
            kernel_name: Name of kernel
            
        Returns:
            Output filename
            
        Raises:
            GeneratorManagerError: If generator not found
        """
        if generator_name not in self.generators:
            raise GeneratorManagerError(f"Generator '{generator_name}' not found")
        
        return self.generators[generator_name].get_output_filename(kernel_name)
    
    def validate_templates(self) -> Dict[str, bool]:
        """
        Validate that all generator templates exist.
        
        Returns:
            Dictionary mapping generator names to template availability
        """
        status = {}
        
        for name, generator in self.generators.items():
            template_path = self.template_dir / generator.template_file
            status[name] = template_path.exists()
            
            if not status[name]:
                logger.warning(f"Template not found for {name}: {generator.template_file}")
        
        return status
    
    def _convert_context_to_template_vars(self, template_ctx: TemplateContext) -> Dict[str, Any]:
        """Convert TemplateContext to template variables for Jinja2."""
        from datetime import datetime
        from ..codegen_binding import SourceType
        
        # Extract variables that templates expect
        vars_dict = {
            # Basic info
            "kernel_name": template_ctx.module_name,
            "class_name": template_ctx.class_name,
            "source_file": str(template_ctx.source_file),
            "generation_timestamp": datetime.now().isoformat(),
            
            # Interfaces
            "interface_metadata": template_ctx.interface_metadata,
            "input_interfaces": template_ctx.input_interfaces,
            "output_interfaces": template_ctx.output_interfaces,
            "weight_interfaces": template_ctx.weight_interfaces,
            "config_interfaces": template_ctx.config_interfaces,
            "control_interfaces": template_ctx.control_interfaces,
            
            # Parameters
            "parameter_definitions": template_ctx.parameter_definitions,
            "exposed_parameters": template_ctx.exposed_parameters,
            "required_attributes": template_ctx.required_attributes,
            
            # Linked parameters from CodegenBinding
            "parameter_aliases": template_ctx.linked_parameters.get("aliases", {}),
            "derived_parameters": template_ctx.linked_parameters.get("derived", {}),
            "axilite_parameters": template_ctx.linked_parameters.get("axilite", {}),
            
            # Explicit mappings from CodegenBinding (compile-time)
            "explicit_datatype_attrs": self._generate_datatype_attributes_from_binding(template_ctx.codegen_binding),
            "explicit_parameter_assignments": self._generate_parameter_assignments_from_binding(template_ctx.codegen_binding),
            
            # Separate RTL-specific node attributes from datatype attributes
            "rtl_specific_nodeattrs": self._generate_rtl_specific_nodeattrs(template_ctx.codegen_binding),
            
            # Enum types for templates (still needed for some legacy logic)
            "SourceType": SourceType,
            
            # Flags
            "has_inputs": template_ctx.has_inputs,
            "has_outputs": template_ctx.has_outputs,
            "has_weights": template_ctx.has_weights,
            "has_config": template_ctx.has_config,
            
            # Other data
            "relationships": template_ctx.relationships,
            "internal_datatypes": template_ctx.internal_datatypes,
            "kernel_complexity": template_ctx.kernel_complexity,
            "kernel_type": template_ctx.kernel_type,
            "categorized_parameters": template_ctx.categorized_parameters,
            
            # SHAPE parameters for HWCustomOp node attributes
            "shape_nodeattrs": template_ctx.shape_nodeattrs,
        }
        
        return vars_dict
    
    
    def _generate_datatype_attributes_from_binding(self, codegen_binding) -> List[Dict[str, str]]:
        """Generate interface datatype attribute definitions from CodegenBinding."""
        datatype_attrs = []
        
        if not codegen_binding:
            return datatype_attrs
        
        # Generate datatype attributes for each interface with datatype bindings
        for interface_name, interface_binding in codegen_binding.interface_bindings.items():
            attr_name = f"{interface_name}DataType"
            datatype_attrs.append({
                "name": attr_name,
                "interface_name": interface_name,
                "attr_spec": ("s", True, "")
            })
        
        return datatype_attrs
    
    def _generate_parameter_assignments_from_binding(self, codegen_binding) -> List[Dict[str, str]]:
        """Generate explicit parameter assignment statements from CodegenBinding.
        
        For interface properties (BDIM/SDIM/DataType), generates assignments that
        use KernelModel. For algorithm parameters, uses node attributes.
        """
        from ..codegen_binding import SourceType, ParameterCategory
        from ..utils import create_parameter_assignment
        
        assignments = []
        
        if not codegen_binding:
            return assignments
        
        # Generate assignment for each parameter based on its source type and category
        for param_name, binding in codegen_binding.parameter_bindings.items():
            # Handle shape parameters (BDIM/SDIM) via KernelModel
            if binding.category == ParameterCategory.SHAPE:
                if binding.source.type == SourceType.INTERFACE_SHAPE:
                    interface_name = binding.source.interface_name
                    dimension_index = binding.source.dimension_index or 0
                    
                    # Determine if this is BDIM or SDIM based on parameter name
                    if "BDIM" in param_name:
                        # Find interface index and generate KernelModel access
                        assignment = create_parameter_assignment(
                            param_name,
                            f'str(self._get_interface_bdim("{interface_name}", {dimension_index}))',
                            f"Block dimension from KernelModel for {interface_name}"
                        )
                        assignments.append(assignment)
                    elif "SDIM" in param_name:
                        assignment = create_parameter_assignment(
                            param_name,
                            f'str(self._get_interface_sdim("{interface_name}", {dimension_index}))',
                            f"Stream dimension from KernelModel for {interface_name}"
                        )
                        assignments.append(assignment)
                elif binding.source.type == SourceType.NODEATTR:
                    # BDIM/SDIM exposed as node attributes but we'll use KernelModel
                    # We need to figure out which interface this belongs to
                    interface_name = None
                    for interface in codegen_binding.interface_bindings.values():
                        if param_name in interface.bdim_params or param_name in interface.sdim_params:
                            interface_name = interface.interface_name
                            break
                    
                    if interface_name and "BDIM" in param_name:
                        assignments.append({
                            "param": param_name,
                            "template_var": f"${param_name.upper()}$",
                            "assignment": f'str(self._get_interface_bdim("{interface_name}", 0))',
                            "comment": f"Block dimension from KernelModel for {interface_name}"
                        })
                    elif interface_name and "SDIM" in param_name:
                        assignments.append({
                            "param": param_name,
                            "template_var": f"${param_name.upper()}$",
                            "assignment": f'str(self._get_interface_sdim("{interface_name}", 0))',
                            "comment": f"Stream dimension from KernelModel for {interface_name}"
                        })
                    else:
                        # Fallback to node attribute
                        assignments.append({
                            "param": param_name,
                            "template_var": f"${param_name.upper()}$",
                            "assignment": f'str(self.get_nodeattr("{param_name}"))',
                            "comment": f"Shape parameter {param_name}"
                        })
            
            # Handle algorithm parameters via node attributes
            elif binding.category == ParameterCategory.ALGORITHM:
                if binding.source.type == SourceType.NODEATTR:
                    assignment = create_parameter_assignment(
                        param_name,
                        f'str(self.get_nodeattr("{param_name}"))',
                        f"Algorithm parameter {param_name}"
                    )
                    assignments.append(assignment)
                elif binding.source.type == SourceType.NODEATTR_ALIAS:
                    alias_name = binding.source.nodeattr_name
                    if alias_name:
                        assignment = create_parameter_assignment(
                            param_name,
                            f'str(self.get_nodeattr("{alias_name}"))',
                            f"Aliased parameter {param_name} from {alias_name}"
                        )
                        assignments.append(assignment)
                elif binding.source.type == SourceType.DERIVED:
                    expression = binding.source.expression
                    if expression:
                        assignment = create_parameter_assignment(
                            param_name,
                            f"str({expression})",
                            f"Derived parameter {param_name}"
                        )
                        assignments.append(assignment)
            
            # Handle datatype parameters via KernelModel
            elif binding.category == ParameterCategory.DATATYPE:
                if binding.source.type == SourceType.INTERFACE_DATATYPE:
                    interface_name = binding.source.interface_name
                    property_name = binding.source.property_name
                    
                    if property_name == "width":
                        assignments.append({
                            "param": param_name,
                            "template_var": f"${param_name.upper()}$",
                            "assignment": f'str(self._get_interface_width("{interface_name}"))',
                            "comment": f"Interface {interface_name} width from KernelModel"
                        })
                    elif property_name == "signed":
                        assignments.append({
                            "param": param_name, 
                            "template_var": f"${param_name.upper()}$",
                            "assignment": f'str(1 if self._get_interface_signed("{interface_name}") else 0)',
                            "comment": f"Interface {interface_name} signed from KernelModel"
                        })
            
            # Handle internal datatype parameters (still use node attributes for now)
            elif binding.category == ParameterCategory.INTERNAL:
                if binding.source.type == SourceType.INTERNAL_DATATYPE:
                    interface_name = binding.source.interface_name  # Internal datatype name
                    property_name = binding.source.property_name
                    
                    if property_name == "width":
                        assignments.append({
                            "param": param_name,
                            "template_var": f"${param_name.upper()}$",
                            "assignment": f'str(DataType[self.get_nodeattr("{interface_name}DataType")].bitwidth())',
                            "comment": f"Internal {interface_name} width parameter"
                        })
                    elif property_name == "signed":
                        assignments.append({
                            "param": param_name,
                            "template_var": f"${param_name.upper()}$", 
                            "assignment": f'str(1 if DataType[self.get_nodeattr("{interface_name}DataType")].signed() else 0)',
                            "comment": f"Internal {interface_name} signed parameter"
                        })
        
        return assignments
    
    def _generate_rtl_specific_nodeattrs(self, codegen_binding) -> Dict[str, tuple]:
        """Generate RTL-specific node attributes for RTLBackend.
        
        Only includes parameters that are exposed as NODEATTR or NODEATTR_ALIAS.
        Excludes all shape, datatype, and derived parameters.
        """
        from ..codegen_binding import SourceType, ParameterCategory
        
        rtl_nodeattrs = {}
        
        if not codegen_binding:
            return rtl_nodeattrs
        
        # Process parameter bindings to find exposed algorithm parameters
        for param_name, binding in codegen_binding.parameter_bindings.items():
            # Only include algorithm and control parameters that are exposed
            if binding.category not in [ParameterCategory.ALGORITHM, ParameterCategory.CONTROL]:
                continue
                
            if binding.source.type == SourceType.NODEATTR:
                # Direct node attribute
                # Determine type based on parameter name patterns
                if param_name.endswith("_PATH") or "PATH" in param_name:
                    # Path parameters are strings
                    rtl_nodeattrs[param_name] = ("s", False, '')
                else:
                    # Most parameters are integers
                    rtl_nodeattrs[param_name] = ("i", True, None)
                    
            elif binding.source.type == SourceType.NODEATTR_ALIAS:
                # Aliased node attribute - use the alias name
                alias_name = binding.source.nodeattr_name
                if alias_name:
                    # Determine type based on alias name
                    if alias_name.endswith("_PATH") or "PATH" in alias_name:
                        rtl_nodeattrs[alias_name] = ("s", False, '')
                    else:
                        rtl_nodeattrs[alias_name] = ("i", True, None)
        
        # Add internal datatype attributes if they exist
        # These are for internal mechanisms like accumulator, threshold
        for internal_name, internal_binding in codegen_binding.internal_bindings.items():
            attr_name = f"{internal_name}DataType"
            rtl_nodeattrs[attr_name] = ("s", False, "INT8")
        
        return rtl_nodeattrs