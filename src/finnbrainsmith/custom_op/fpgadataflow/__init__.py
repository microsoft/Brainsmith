from finnbrainsmith.custom_op.fpgadataflow.quantsoftmax import QuantSoftmax
from finnbrainsmith.custom_op.fpgadataflow.shuffle import Shuffle
from finnbrainsmith.custom_op.fpgadataflow.crop import Crop

custom_op = dict()

custom_op["QuantSoftmax"] = QuantSoftmax
custom_op["Shuffle"] = Shuffle
custom_op["Crop"] = Crop