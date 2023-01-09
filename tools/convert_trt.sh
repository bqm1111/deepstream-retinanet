python3 convert_deepsort_detector_to_trt.py --onnx_path ../data/models/onnx/deepsort_detector.onnx --trt_path ../data/models/trt/deepsort_detector.trt --max_batch_size 24
python3 convert_deepsort_extractor_to_trt.py --onnx_path ../data/models/onnx/deepsort_extractor.onnx --trt_path ../data/models/trt/deepsort_extractor.trt --max_batch_size 24
python3 onnx_to_trt.py