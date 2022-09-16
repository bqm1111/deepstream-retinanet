import argparse

import tensorrt as trt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, help="Path to PTH file")
    parser.add_argument("--trt_path", type=str, help="Path to TRT file")
    parser.add_argument("--max_batch_size", type=int, help="Max batch size")
    args = parser.parse_args()

    # logger to capture errors, warnings, and other information during the build and inference phases
    TRT_LOGGER = trt.Logger(trt.Logger.Severity.WARNING)

    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # parse ONNX
    with open(args.onnx_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # ------------------------------------------------------------------
    # Builder config
    config = builder.create_builder_config()

    profile = builder.create_optimization_profile();
    profile.set_shape(
        "input.0", (1, 3, 128, 64), (args.max_batch_size//2, 3, 128, 64), 
        (args.max_batch_size, 3, 128, 64)
    ) 
    config.add_optimization_profile(profile)
    config.max_workspace_size = 1 << 30
    builder.max_batch_size = args.max_batch_size
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    # ------------------------------------------------------------------

    # generate TensorRT engine optimized for the target platform
    print('Building an engine')
    engine = builder.build_engine(network, config)
    print("Completed creating Engine")
    with open(args.trt_path, "wb") as f:
        f.write(engine.serialize())
