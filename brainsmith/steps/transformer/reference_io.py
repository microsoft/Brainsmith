"""
Transformer-specific reference IO generation.
"""

from brainsmith.steps import register_step
import finn.core.onnx_exec as oxe
from qonnx.util.basic import gen_finn_dt_tensor
from qonnx.core.datatype import DataType
import numpy as np


@register_step(
    name="transformer.generate_reference_io",
    category="transformer",
    description="Generate reference IO pair for the model with head and tail removed"
)
def generate_reference_io_step(model, cfg):
    """
    This step is to generate a reference IO pair for the 
    onnx model where the head and the tail have been 
    chopped off.
    """
    input_m = model.graph.input[0]
    in_shape = [dim.dim_value for dim in input_m.type.tensor_type.shape.dim]
    in_tensor = gen_finn_dt_tensor(DataType["FLOAT32"], in_shape)
    np.save(cfg.output_dir+"/input.npy", in_tensor)

    input_t = { input_m.name : in_tensor}
    out_name = model.graph.output[0].name

    y_ref = oxe.execute_onnx(model, input_t, True)
    np.save(cfg.output_dir+"/expected_output.npy", y_ref[out_name])
    np.savez(cfg.output_dir+"/expected_context.npz", **y_ref) 
    return model