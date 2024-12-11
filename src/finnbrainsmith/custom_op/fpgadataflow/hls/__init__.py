from finnbrainsmith.custom_op.fpgadataflow.hls.quantsoftmax_hls import QuantSoftmax_hls
from finnbrainsmith.custom_op.fpgadataflow.hls.shuffle_hls import Shuffle_hls
from finnbrainsmith.custom_op.fpgadataflow.hls.crop_hls import Crop_hls

custom_op = dict()

# make sure new HLSCustomOp subclasses are imported here so that they get
# registered and plug in correctly into the infrastructure
custom_op["QuantSoftmax_hls"] = QuantSoftmax_hls
custom_op["Shuffle_hls"] = Shuffle_hls
custom_op["Crop_hls"] = Crop_hls
