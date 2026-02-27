import os
import sys
import subprocess
import re
from pathlib import Path
import folder_paths
import numpy as np
import torch
from comfy.utils import ProgressBar
from .trt_utilities import Engine
from .utilities import download_file, ColoredLogger, get_final_resolutions
import comfy.model_management as mm
import time
import json

# Auto-detect CUDA and install appropriate TensorRT packages
def _auto_install_tensorrt():
    """Auto-detect CUDA version and install appropriate TensorRT packages if needed"""
    try:
        # Check if TensorRT is already installed
        try:
            import tensorrt
            print("✅ TensorRT already installed")
            return True
        except ImportError:
            print("🔍 TensorRT not found, detecting CUDA version...")
        
        # Detect CUDA version
        cuda_version = None
        
        # Try nvcc command
        try:
            result = subprocess.run("nvcc --version", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                match = re.search(r"release (\d+\.\d+)", result.stdout)
                if match:
                    cuda_version = match.group(1)
                    print(f"✅ Detected CUDA version: {cuda_version}")
        except:
            pass
        
        # Try CUDA_PATH
        if not cuda_version and os.environ.get("CUDA_PATH"):
            nvcc_path = os.path.join(os.environ["CUDA_PATH"], "bin", "nvcc")
            if os.path.exists(nvcc_path):
                try:
                    result = subprocess.run(f"{nvcc_path} --version", shell=True, capture_output=True, text=True)
                    if result.returncode == 0:
                        match = re.search(r"release (\d+\.\d+)", result.stdout)
                        if match:
                            cuda_version = match.group(1)
                            print(f"✅ Detected CUDA via CUDA_PATH: {cuda_version}")
                except:
                    pass
        
        # Try CUDA_HOME
        if not cuda_version and os.environ.get("CUDA_HOME"):
            nvcc_path = os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc")
            if os.path.exists(nvcc_path):
                try:
                    result = subprocess.run(f"{nvcc_path} --version", shell=True, capture_output=True, text=True)
                    if result.returncode == 0:
                        match = re.search(r"release (\d+\.\d+)", result.stdout)
                        if match:
                            cuda_version = match.group(1)
                            print(f"✅ Detected CUDA via CUDA_HOME: {cuda_version}")
                except:
                    pass
        
        if not cuda_version:
            print("⚠️  Could not detect CUDA version automatically")
            print("Please run 'python install.py' manually to install TensorRT")
            return False
        
        # Install appropriate TensorRT packages
        major_version = int(cuda_version.split('.')[0])
        
        if major_version == 13:
            print("🚀 Installing CUDA 13 TensorRT packages (RTX 50 series)")
            packages = [
                "tensorrt_cu13==10.15.1.29",
                "tensorrt_cu13_bindings==10.15.1.29", 
                "tensorrt_cu13_libs==10.15.1.29",
                "cuda-toolkit>=13.0.0,<13.1.0"
            ]
        elif major_version == 12:
            print("🔧 Installing CUDA 12 TensorRT packages (RTX 30/40 series)")
            packages = [
                "tensorrt-cu12==10.13.3.9",
                "tensorrt-cu12-libs==10.13.3.9",
                "tensorrt-cu12-bindings==10.13.3.9",
                "cuda-toolkit>=12.8.0,<13.0.0"
            ]
        else:
            print(f"❌ Unsupported CUDA version: {cuda_version}")
            return False
        
        # Install packages
        for package in packages:
            print(f"Installing {package}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", package], capture_output=True)
            if result.returncode != 0:
                print(f"❌ Failed to install {package}")
                print(f"Error: {result.stderr.decode()}")
                return False
        
        print("✅ TensorRT installation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Auto-installation failed: {e}")
        print("Please run 'python install.py' manually to install TensorRT")
        return False

# Auto-detect CUDA toolkit and add DLL path before importing polygraphy
def _setup_cuda_dll_path():
    """Auto-detect CUDA toolkit and add cudart64 DLL path on Windows."""
    if not sys.platform.startswith("win"):
        return
    
    cuda_root = None
    
    # Check for CUDA_PATH or CUDA_HOME environment variables
    cuda_root = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    
    if not cuda_root:
        # Try default Windows install location
        program_files = os.environ.get("PROGRAMFILES")
        if program_files:
            cuda_base = Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
            if cuda_base.exists():
                # Find highest version directory
                versions = sorted([d for d in cuda_base.iterdir() if d.is_dir()], reverse=True)
                if versions:
                    cuda_root = str(versions[0])
    
    if cuda_root:
        cuda_path = Path(cuda_root)
        # CUDA 13.0+ puts cudart64 in bin/x64 subdirectory
        cuda_bin_x64 = cuda_path / "bin" / "x64"
        if cuda_bin_x64.exists() and any(cuda_bin_x64.glob("cudart64*.dll")):
            os.add_dll_directory(str(cuda_bin_x64))
            return
        # Fallback to regular bin directory for older CUDA versions
        cuda_bin = cuda_path / "bin"
        if cuda_bin.exists() and any(cuda_bin.glob("cudart64*.dll")):
            os.add_dll_directory(str(cuda_bin))
            return
    
    # CUDA toolkit not found - print warning with download link
    print("[ComfyUI-Upscaler-TensorRT] WARNING: CUDA toolkit not found.")
    print("    Set CUDA_PATH environment variable or install CUDA toolkit.")
    print("    Download: https://developer.nvidia.com/cuda-13-0-2-download-archive")

# Run auto-install and setup on module import
try:
    _auto_install_tensorrt()
    _setup_cuda_dll_path()
except Exception as e:
    print(f"[ComfyUI-Upscaler-TensorRT] Warning: Auto-installation failed: {e}")
    print("Please run 'python install.py' manually to install TensorRT")
    print("The node will continue loading, but TensorRT may not work properly")

try:
    import tensorrt
except ImportError as e:
    print(f"[ComfyUI-Upscaler-TensorRT] Error: TensorRT import failed: {e}")
    print("Please install TensorRT manually:")
    print("  CUDA 13: pip install tensorrt_cu13==10.15.1.29 tensorrt_cu13_bindings==10.15.1.29 tensorrt_cu13_libs==10.15.1.29")
    print("  CUDA 12: pip install tensorrt-cu12==10.13.3.9 tensorrt-cu12-libs==10.13.3.9 tensorrt-cu12-bindings==10.13.3.9")
    print("The node will continue loading, but TensorRT features will not be available")
    # Create a dummy tensorrt module to prevent further crashes
    import types
    tensorrt = types.ModuleType('tensorrt')
    tensorrt.__version__ = "not installed"

logger = ColoredLogger("ComfyUI-Upscaler-Tensorrt")

# Check if TensorRT is properly loaded
try:
    trt_version = tensorrt.__version__
    if trt_version == "not installed":
        logger.warning("TensorRT not properly installed - node functionality limited")
    else:
        logger.info(f"TensorRT {trt_version} loaded successfully")
except AttributeError:
    logger.warning("TensorRT version check failed - may not be properly installed")

IMAGE_DIM_MIN = 256
IMAGE_DIM_OPT = 512
IMAGE_DIM_MAX = 1280

# --- Function to load configuration ---
def load_node_config(config_filename="load_upscaler_config.json"):
    """Loads node configuration from a JSON file."""
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, config_filename)
    
    default_config = { # Fallback in case file is missing or corrupt
        "model": {
            "options": ["4x-UltraSharp"],
            "default": "4x-UltraSharp",
            "tooltip": "Default model (fallback from code)"
        },
        "precision": {
            "options": ["fp16", "fp32"],
            "default": "fp16",
            "tooltip": "Default precision (fallback from code)"
        }
    }

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Successfully loaded configuration from {config_filename}")
        return config
    except FileNotFoundError:
        logger.warning(f"Configuration file '{config_path}' not found. Using default fallback configuration.")
        return default_config
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from '{config_path}'. Using default fallback configuration.")
        return default_config
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading '{config_path}': {e}. Using default fallback.")
        return default_config

# --- Load the configuration once when the module is imported ---
LOAD_UPSCALER_NODE_CONFIG = load_node_config()


class UpscalerTensorrt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": f"Images to be upscaled. Resolution must be between {IMAGE_DIM_MIN} and {IMAGE_DIM_MAX} px"}),
                "upscaler_trt_model": ("UPSCALER_TRT_MODEL", {"tooltip": "Tensorrt model built and loaded"}),
                "resize_to": (["none", "custom", "HD", "FHD", "2k", "4k", "1x", "1.5x", "2x", "2.5x", "3x", "3.5x", "4x", "5x", "6x", "7x", "8x", "9x", "10x"], {"tooltip": "Resize the upscaled image to fixed resolutions, optional"}),
                "resize_width": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "resize_height": ("INT", {"default": 1024, "min": 1, "max": 8192}),
            }
        }
    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscaler_tensorrt"
    CATEGORY = "tensorrt"
    DESCRIPTION = "Upscale images with tensorrt"

    def upscaler_tensorrt(self, **kwargs):
        images = kwargs.get("images")
        upscaler_trt_model = kwargs.get("upscaler_trt_model")
        resize_to = kwargs.get("resize_to")

        images_bchw = images.permute(0, 3, 1, 2)
        B, C, H, W = images_bchw.shape

        for dim in (H, W):
            if dim > IMAGE_DIM_MAX or dim < IMAGE_DIM_MIN:
                raise ValueError(f"Input image dimensions fall outside of the supported range: {IMAGE_DIM_MIN} to {IMAGE_DIM_MAX} px!\nImage dimensions: {W}px by {H}px")

        if resize_to == "custom":
            final_width = kwargs.get("resize_width")
            final_height = kwargs.get("resize_height")
        else:
            final_width, final_height = get_final_resolutions(W, H, resize_to)

        logger.info(f"Upscaling {B} images from H:{H}, W:{W} to H:{H*4}, W:{W*4} | Final resolution: H:{final_height}, W:{final_width} | resize_to: {resize_to}")

        shape_dict = {
            "input": {"shape": (1, 3, H, W)},
            "output": {"shape": (1, 3, H*4, W*4)},
        }
        upscaler_trt_model.activate()
        upscaler_trt_model.allocate_buffers(shape_dict=shape_dict)

        cudaStream = torch.cuda.current_stream().cuda_stream
        pbar = ProgressBar(B)
        images_list = list(torch.split(images_bchw, split_size_or_sections=1))

        upscaled_frames = torch.empty((B, C, final_height, final_width), dtype=torch.float32, device=mm.intermediate_device())
        must_resize = W*4 != final_width or H*4 != final_height

        for i, img in enumerate(images_list):
            result = upscaler_trt_model.infer({"input": img}, cudaStream)
            result = result["output"]

            if must_resize:
                result = torch.nn.functional.interpolate(
                    result, 
                    size=(final_height, final_width),
                    mode='bicubic',
                    antialias=True
                )
            upscaled_frames[i] = result.to(mm.intermediate_device())
            pbar.update(1)

        output = upscaled_frames.permute(0, 2, 3, 1)
        upscaler_trt_model.reset()
        mm.soft_empty_cache()

        logger.info(f"Output shape: {output.shape}")
        return (output,)

class LoadUpscalerTensorrtModel:
    @classmethod
    def INPUT_TYPES(cls): # Changed 's' to 'cls' for convention
        # Use the pre-loaded configuration
        model_config = LOAD_UPSCALER_NODE_CONFIG.get("model", {})
        precision_config = LOAD_UPSCALER_NODE_CONFIG.get("precision", {})
        
        # Provide sensible defaults if keys are missing in the config (though load_node_config handles this broadly)
        model_options = model_config.get("options", ["4x-UltraSharp"])
        model_default = model_config.get("default", "4x-UltraSharp")
        model_tooltip = model_config.get("tooltip", "Select a model.")

        precision_options = precision_config.get("options", ["fp16", "fp32"])
        precision_default = precision_config.get("default", "fp16")
        precision_tooltip = precision_config.get("tooltip", "Select precision.")

        return {
            "required": {
                "model": (model_options, {"default": model_default, "tooltip": model_tooltip}),
                "precision": (precision_options, {"default": precision_default, "tooltip": precision_tooltip}),
            }
        }
    
    RETURN_NAMES = ("upscaler_trt_model",)
    RETURN_TYPES = ("UPSCALER_TRT_MODEL",)
    # FUNCTION = "main" # This was duplicated, removing
    CATEGORY = "tensorrt"
    DESCRIPTION = "Load tensorrt models, they will be built automatically if not found."
    FUNCTION = "load_upscaler_tensorrt_model" # This is the correct one
    
    def load_upscaler_tensorrt_model(self, model, precision):
        tensorrt_models_dir = os.path.join(folder_paths.models_dir, "tensorrt", "upscaler")
        onnx_models_dir = os.path.join(folder_paths.models_dir, "onnx")

        os.makedirs(tensorrt_models_dir, exist_ok=True)
        os.makedirs(onnx_models_dir, exist_ok=True)

        onnx_model_path = os.path.join(onnx_models_dir, f"{model}.onnx")
        
        engine_channel = 3
        engine_min_batch, engine_opt_batch, engine_max_batch = 1, 1, 1
        engine_min_h, engine_opt_h, engine_max_h = IMAGE_DIM_MIN, IMAGE_DIM_OPT, IMAGE_DIM_MAX
        engine_min_w, engine_opt_w, engine_max_w = IMAGE_DIM_MIN, IMAGE_DIM_OPT, IMAGE_DIM_MAX
        tensorrt_model_path = os.path.join(tensorrt_models_dir, f"{model}_{precision}_{engine_min_batch}x{engine_channel}x{engine_min_h}x{engine_min_w}_{engine_opt_batch}x{engine_channel}x{engine_opt_h}x{engine_opt_w}_{engine_max_batch}x{engine_channel}x{engine_max_h}x{engine_max_w}_{tensorrt.__version__}.trt")

        if not os.path.exists(tensorrt_model_path):
            if not os.path.exists(onnx_model_path):
                onnx_model_download_url = f"https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/{model}.onnx"
                logger.info(f"Downloading {onnx_model_download_url}")
                download_file(url=onnx_model_download_url, save_path=onnx_model_path)
            else:
                logger.info(f"Onnx model found at: {onnx_model_path}")

            logger.info(f"Building TensorRT engine for {onnx_model_path}: {tensorrt_model_path}")
            mm.soft_empty_cache()
            s = time.time()
            engine = Engine(tensorrt_model_path)
            engine.build(
                onnx_path=onnx_model_path,
                fp16= True if precision == "fp16" else False,
                input_profile=[
                    {"input": [(engine_min_batch,engine_channel,engine_min_h,engine_min_w), (engine_opt_batch,engine_channel,engine_opt_h,engine_min_w), (engine_max_batch,engine_channel,engine_max_h,engine_max_w)]},
                ],
            )
            e = time.time()
            logger.info(f"Time taken to build: {(e-s)} seconds")

        logger.info(f"Loading TensorRT engine: {tensorrt_model_path}")
        mm.soft_empty_cache()
        engine = Engine(tensorrt_model_path)
        engine.load()

        return (engine,)

NODE_CLASS_MAPPINGS = {
    "UpscalerTensorrt": UpscalerTensorrt,
    "LoadUpscalerTensorrtModel": LoadUpscalerTensorrtModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscalerTensorrt": "Upscaler Tensorrt ⚡",
    "LoadUpscalerTensorrtModel": "Load Upscale Tensorrt Model",
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]
