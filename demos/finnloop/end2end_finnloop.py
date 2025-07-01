import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
import brevitas.onnx as bo
from brainsmith.core.hw_compiler import forge
import onnx


class TwoMatMulModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoMatMulModule, self).__init__()
        self.matmul1 = qnn.QuantLinear(input_dim, hidden_dim, weight_quant=Int8WeightPerTensorFloat, bias=True)
        #self.relu1 = qnn.QuantReLU(act_quant=Int8ActPerTensorFloat)
        self.matmul2 = qnn.QuantLinear(hidden_dim, output_dim, weight_quant=Int8WeightPerTensorFloat, bias=True)
        #self.relu2 = qnn.QuantReLU(act_quant=Int8ActPerTensorFloat)

    def forward(self, x):
        x = self.matmul1(x)
        #x = self.relu1(x)
        x = self.matmul2(x)
        #x = self.relu2(x)
        return x

class MultiTwoMatMulModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, N):
        super(MultiTwoMatMulModule, self).__init__()
        self.layers = nn.ModuleList([TwoMatMulModule(input_dim, hidden_dim, output_dim) for _ in range(N)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Example usage:
input_dim = 10
hidden_dim = 20
output_dim = 10
N = 5

model = MultiTwoMatMulModule(input_dim, hidden_dim, output_dim, N)


# Formula: output = randn * scale + shift
# To generate values in range [min_val, max_val]:
min_val = -128  # Your desired minimum value
max_val = 127  # Your desired maximum value

# Calculate scale and shift
scale = (max_val - min_val) / 6  # Dividing by 6 covers ~99.7% of normal distribution
shift = (max_val + min_val) / 2  # Center point of the target range

# Generate random values in the specified range
input_tensor = torch.randn(1, input_dim) * scale + shift

print(f"Min value: {input_tensor.min().item()}")
print(f"Max value: {input_tensor.max().item()}")

#input_tensor = torch.randn(1, input_dim)
output_tensor = model(input_tensor)

with torch.no_grad():
    bo.export_qonnx(
        model,
        (input_tensor),
        "onnx_model.onnx",
        do_constant_folding=True,
        input_names=['input_ids'],
        opset_version=18,
        dynamo=True,
        optimize=True,
    )


class Args:
    def __init__(self):
        self.output = 'finnloop'
        self.fps = 1
        self.clk = 3.33
        self.param = '' #folding parameter config file
        self.stop_step = None
        self.fifodepth = 512
        self.fifosim_n_inferences = 2
        self.verification_atol = 1e-1
        self.split_large_fifos = True
        self.dcp = False
        self.board = "V80"
        self.save_intermediate = True
        self.num_hidden_layers = 5
        self.standalone_thresholds = True
        self.loop_body_hierarchy = ['','layers.0']
args = Args()

m = onnx.load("onnx_model.onnx")

forge('finnloop', m, args)

print(output_tensor)
