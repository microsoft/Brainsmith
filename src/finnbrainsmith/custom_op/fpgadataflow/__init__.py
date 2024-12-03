from finnbrainsmith.custom_op.fpgadataflow.layernorm import LayerNorm
from finnbrainsmith.custom_op.fpgadataflow.quantsoftmax import QuantSoftmax
from finnbrainsmith.custom_op.fpgadataflow.shuffle import Shuffle 

custom_op = dict()

custom_op["LayerNorm"] = LayerNorm
custom_op["QuantSoftmax"] = QuantSoftmax
custom_op["Shuffle"] = Shuffle 
