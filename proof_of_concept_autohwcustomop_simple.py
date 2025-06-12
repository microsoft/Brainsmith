"""
Simplified Proof of Concept: AutoHWCustomOp Subclass for Thresholding

This demonstrates what our template should generate, focusing on the
key architectural concepts without runtime issues.
"""

from typing import List, Dict, Any, Tuple


# === THIS IS WHAT THE TEMPLATE WOULD GENERATE ===

class ThresholdingAxi:  # Would inherit from AutoHWCustomOp
    """
    Auto-generated HWCustomOp for thresholding_axi kernel using DataflowModel.
    
    Key architectural points:
    1. Interface metadata is created from RTL analysis (ParsedKernelData)
    2. DataflowModel is created automatically in __init__ (by parent AutoHWCustomOp)
    3. We only override kernel-specific methods when needed
    4. Most functionality comes from DataflowModel via AutoHWCustomOp
    """
    
    def __init__(self, onnx_node, **kwargs):
        """
        Initialize with interface metadata extracted from RTL.
        
        The template generates this from ParsedKernelData:
        - Interfaces from RTL analysis become InterfaceMetadata objects
        - RTL parameters become node attributes
        - Pragma information drives chunking strategies
        """
        
        # This would be generated from ParsedKernelData.interfaces
        interface_metadata = [
            # From interface: s_axis_input (type=InterfaceType.INPUT)
            {
                "name": "s_axis_input",
                "interface_type": "INPUT",
                "allowed_datatypes": [
                    {"finn_type": "UINT8", "bit_width": 8, "signed": False},
                    {"finn_type": "INT8", "bit_width": 8, "signed": True},
                ],
                "chunking_strategy": "default",  # Could be from @brainsmith pragma
            },
            
            # From interface: s_axis_weights (type=InterfaceType.WEIGHT)
            {
                "name": "s_axis_weights", 
                "interface_type": "WEIGHT",
                "allowed_datatypes": [
                    {"finn_type": "UINT8", "bit_width": 8, "signed": False},
                    {"finn_type": "INT8", "bit_width": 8, "signed": True},
                ],
                "chunking_strategy": "default",
            },
            
            # From interface: m_axis_output (type=InterfaceType.OUTPUT)
            {
                "name": "m_axis_output",
                "interface_type": "OUTPUT", 
                "allowed_datatypes": [
                    {"finn_type": "UINT8", "bit_width": 8, "signed": False},
                ],
                "chunking_strategy": "default",
            },
        ]
        
        # In real implementation:
        # super().__init__(onnx_node, interface_metadata=interface_metadata, **kwargs)
        
        # This creates the DataflowModel internally!
        # The DataflowModel now handles:
        # - Shape calculations (normal/folded)
        # - Stream width computations
        # - Resource estimations
        # - Parallelism management
        
        # Store kernel info (from ParsedKernelData)
        self.kernel_name = "thresholding_axi"
        self.rtl_source = "examples/thresholding/thresholding_axi.sv"
        
        print(f"Created {self.__class__.__name__} with DataflowModel-driven architecture")
        print(f"  Kernel: {self.kernel_name}")
        print(f"  Source: {self.rtl_source}")
        print(f"  Interfaces: {len(interface_metadata)}")
    
    def get_nodeattr_types(self) -> Dict[str, Tuple]:
        """
        Define node attributes - combination of DataflowModel and kernel-specific.
        
        Template generates this from:
        - ParsedKernelData.parameters → RTL parameters
        - Parallelism analysis → PE, SIMD attributes
        - Algorithm inference → operation-specific attributes
        """
        
        # In real implementation: attrs = super().get_enhanced_nodeattr_types()
        # This would give us DataflowModel attributes:
        # - {interface_name}_parallel for each interface
        # - {interface_name}_dtype for each interface
        # - resource_estimation_mode
        # - enable_constraint_validation
        
        attrs = {
            # DataflowModel-provided attributes (from parent)
            "s_axis_input_parallel": ("i", False, 1),
            "s_axis_input_dtype": ("s", False, "UINT8"),
            "s_axis_weights_parallel": ("i", False, 1), 
            "s_axis_weights_dtype": ("s", False, "UINT8"),
            "m_axis_output_parallel": ("i", False, 1),
            "m_axis_output_dtype": ("s", False, "UINT8"),
            "resource_estimation_mode": ("s", False, "automatic"),
            "enable_constraint_validation": ("b", False, True),
            
            # RTL parameters (from ParsedKernelData.parameters)
            "N": ("i", False, 0),
            "C": ("i", False, 1),
            "PE": ("i", False, 1),
            "SIGNED": ("i", False, 1),
            "BIAS": ("i", False, 0),
            
            # Algorithm-specific (from TemplateContextGenerator analysis)
            "NumChannels": ("i", True, 1),  # Maps to C
            "numSteps": ("i", True, 1),     # Maps to N
            "ActVal": ("i", False, 0),      # Maps to BIAS
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
            "numInputVectors": ("ints", False, [1]),
        }
        
        return attrs
    
    # === OPTIONAL OVERRIDES ===
    # Most methods are handled by AutoHWCustomOp via DataflowModel:
    # - get_input_datatype() → from DataflowModel interfaces
    # - get_output_datatype() → from DataflowModel interfaces
    # - get_normal_input_shape() → from DataflowModel tensor reconstruction
    # - get_folded_input_shape() → from DataflowModel with parallelism
    # - get_instream_width() → from DataflowModel stream calculations
    # - get_outstream_width() → from DataflowModel stream calculations
    # - estimate_bram_usage() → from DataflowModel resource requirements
    # - estimate_lut_usage() → from DataflowModel resource requirements
    
    def get_exp_cycles(self) -> int:
        """
        Example of kernel-specific override.
        
        For many kernels, the default from DataflowModel is sufficient.
        But thresholding has a specific formula.
        """
        # In real implementation:
        # channels = self.get_nodeattr("NumChannels") 
        # pe = self.get_nodeattr("PE")
        # return (channels // pe) * num_vectors
        
        return 128  # Placeholder
    
    def verify_node(self):
        """Kernel-specific verification on top of DataflowModel validation."""
        # In real implementation:
        # super().verify_node()  # DataflowModel interface validation
        
        # Kernel-specific check
        print("  Verifying: NumChannels must be divisible by PE")
    
    def get_hw_compatible_threshold_tensor(self, orig_thres_matrix):
        """
        Example of kernel-specific method that wouldn't be in DataflowModel.
        
        This shows that we can still add operation-specific functionality
        while leveraging DataflowModel for all the standard operations.
        """
        print("  Preparing threshold tensor for hardware")
        return orig_thres_matrix  # Placeholder


# === DEMONSTRATION ===

def show_template_generation_concept():
    """Demonstrate the key concepts of template generation."""
    
    print("=== AutoHWCustomOp Template Generation Concept ===\n")
    
    print("INPUTS:")
    print("- ParsedKernelData from RTL Parser")
    print("  - name: 'thresholding_axi'")
    print("  - interfaces: {s_axis_input: INPUT, s_axis_weights: WEIGHT, m_axis_output: OUTPUT}")
    print("  - parameters: [N, C, PE, SIGNED, BIAS, ...]")
    print("  - pragmas: [@brainsmith bdim, @brainsmith datatype, ...]")
    print("")
    
    print("TEMPLATE GENERATION PROCESS:")
    print("1. TemplateContextGenerator analyzes ParsedKernelData")
    print("   - Extracts parallelism info (PE from parameters)")
    print("   - Infers algorithm type ('threshold' from name)")
    print("   - Maps RTL params to FINN attributes")
    print("")
    
    print("2. Template creates InterfaceMetadata list")
    print("   - Each RTL interface → InterfaceMetadata object")
    print("   - Interface types from RTL analysis") 
    print("   - Datatype constraints from pragmas or defaults")
    print("   - Chunking strategies from pragmas or defaults")
    print("")
    
    print("3. Template generates AutoHWCustomOp subclass")
    print("   - Passes InterfaceMetadata to parent __init__")
    print("   - Parent creates DataflowModel automatically")
    print("   - Only overrides kernel-specific methods")
    print("")
    
    # Create instance to show the result
    class FakeONNXNode:
        pass
    
    hw_op = ThresholdingAxi(FakeONNXNode())
    
    print("\n=== BENEFITS OF DATAFLOWMODEL-DRIVEN DESIGN ===")
    print("1. No manual shape calculations - DataflowModel handles it")
    print("2. No manual stream width code - DataflowModel computes it")
    print("3. No manual resource estimation - DataflowModel provides it")
    print("4. Parallelism changes automatically update all calculations")
    print("5. Interface validation is automatic")
    print("6. Standard FINN methods work out of the box")
    
    print("\n=== WHAT THE TEMPLATE NEEDS TO GENERATE ===")
    print("1. InterfaceMetadata list from ParsedKernelData.interfaces")
    print("2. Node attributes from ParsedKernelData.parameters + analysis")
    print("3. Optional kernel-specific method overrides")
    print("4. That's it! DataflowModel handles the rest")


if __name__ == "__main__":
    show_template_generation_concept()