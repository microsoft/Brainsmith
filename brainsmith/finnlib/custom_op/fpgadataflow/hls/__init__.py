from brainsmith.finnlib.custom_op.fpgadataflow.hls.layernorm_hls import LayerNorm_hls
from brainsmith.finnlib.custom_op.fpgadataflow.hls.hwsoftmax_hls import HWSoftmax_hls
from brainsmith.finnlib.custom_op.fpgadataflow.hls.shuffle_hls import Shuffle_hls
from brainsmith.finnlib.custom_op.fpgadataflow.hls.crop_hls import Crop_hls

custom_op = dict()

# make sure new HLSCustomOp subclasses are imported here so that they get
# registered and plug in correctly into the infrastructure

custom_op["LayerNorm_hls"] = LayerNorm_hls
custom_op["HWSoftmax_hls"] = HWSoftmax_hls
custom_op["Shuffle_hls"] = Shuffle_hls
custom_op["Crop_hls"] = Crop_hls
