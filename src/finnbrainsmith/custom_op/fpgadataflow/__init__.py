from finnbrainsmith.custom_op.fpgadataflow.layernorm import LayerNorm
from finnbrainsmith.custom_op.fpgadataflow.quantsoftmax import QuantSoftmax
from finnbrainsmith.custom_op.fpgadataflow.hwsoftmax import HWSoftmax
from finnbrainsmith.custom_op.fpgadataflow.shuffle import Shuffle
from finnbrainsmith.custom_op.fpgadataflow.crop import Crop

custom_op = dict()

custom_op["LayerNorm"] = LayerNorm
custom_op["QuantSoftmax"] = QuantSoftmax
custom_op["HWSoftmax"] = HWSoftmax
custom_op["Shuffle"] = Shuffle
custom_op["Crop"] = Crop
