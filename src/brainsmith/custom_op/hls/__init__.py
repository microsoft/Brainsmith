from brainsmith.custom_op.hls.quantsoftmax_hls import QuantSoftmax_hls

custom_op = dict()

# make sure new HLSCustomOp subclasses are imported here so that they get
# registered and plug in correctly into the infrastructure
custom_op["QuantSoftmax_hls"] = QuantSoftmax_hls
