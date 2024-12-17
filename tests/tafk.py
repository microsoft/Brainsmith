import fpgadataflow.test_fpgadataflow_layernorm as t
# from qonnx.core.datatype import DataType

# ["cppsim", "rtlsim", "stitched_ip"]
# ["simd1", "simd2", "simd3", "simd4"]
exec_mode = "cppsim"
impl_style = "hls"
simd = "simd1"
#idt = ["FLOAT16"]
idt = "FLOAT32"
wdt = "FLOAT32"
bdt = "FLOAT32"
odt = "FLOAT32"
ifm_dim = (1, 1, 128, 384)
t.test_fpga_dataflow_layernorm(impl_style, exec_mode, simd, idt, wdt, bdt, odt, ifm_dim)
# tb.test_batchnorm_to_affine_cnv_w1a1()
