import torch
import time
from trt_utilities import Engine

def export_trt(trt_path=None, onnx_path=None, use_fp16=True):
    if trt_path is None:
        trt_path = input("Enter the path to save the TensorRT engine (e.g ./realesrgan.engine): ")
    if onnx_path is None:
        onnx_path = input("Enter the path to the ONNX model (e.g ./realesrgan.onnx): ")

    engine = Engine(trt_path)

    torch.cuda.empty_cache()

    s = time.time()
    ret = engine.build(
        onnx_path,
        use_fp16,
        enable_preview=True,
        input_profile=[
            {"input": [(1,3,256,256), (1,3,1024,1024), (1,3,2048,2048)]}, # any sizes from 256x256 to 2048x2048
        ],
    )
    e = time.time()
    print(f"Time taken to build: {(e-s)} seconds")

    return ret

export_trt()
