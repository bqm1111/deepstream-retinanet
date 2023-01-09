"""convert from onnx to serialize engine
https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#import_model_python
"""
from os import path as osp
import tensorrt as trt

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger(trt.Logger.Severity.WARNING)

# initialize TensorRT engine and parse ONNX model
builder = trt.Builder(TRT_LOGGER)
explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(explicit_batch)
parser = trt.OnnxParser(network, TRT_LOGGER)

# parse ONNX
curdir = osp.dirname(__file__)
onnx_path = osp.join(curdir, '../data/models/onnx/', 'glint360k_r50.onnx')
trt_path = osp.join(curdir, '../data/models/trt/', 'glint360k_r50.trt')
with open(onnx_path, 'rb') as model:
    print('Beginning ONNX file parsing')
    parser.parse(model.read())
print('Completed parsing of ONNX file')

# Builder config
config = builder.create_builder_config()
profile = builder.create_optimization_profile()
profile.set_shape(
    "conv1", min=(1, 3, 112, 112), opt=(16, 3, 112, 112), max=(32, 3, 112, 112)
)
config.add_optimization_profile(profile)
config.max_workspace_size = 1 << 30
# if builder.platform_has_fast_fp16:
#     config.set_flag(trt.BuilderFlag.FP16)

# generate TensorRT engine optimized for the target platform
print('Building an engine')
engine = builder.build_engine(network, config)
print("Completed creating Engine")
with open(trt_path, "wb") as f:
    f.write(engine.serialize())
    