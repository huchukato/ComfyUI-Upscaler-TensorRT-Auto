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
    """Auto-detect CUDA version and install the matching TensorRT wheels.

    The NVIDIA CUDA Toolkit must already be installed on the system.
    This function installs only the TensorRT packages via pip.
    A marker file prevents repeated install attempts on every ComfyUI startup.
    """
    disable_auto_install = os.environ.get("DISABLE_TENSORRT_AUTO_INSTALL", "false").lower() == "true"
    if disable_auto_install:
        print("[ComfyUI-Upscaler-TensorRT] Auto-installation disabled via DISABLE_TENSORRT_AUTO_INSTALL")
        return True

    node_dir = Path(__file__).resolve().parent
    installed_marker = node_dir / ".tensorrt_auto_installed"
    failed_marker = node_dir / ".tensorrt_auto_install_failed"

    # Skip if we already installed successfully in a previous run AND tensorrt is importable.
    if installed_marker.exists():
        try:
            import tensorrt
            return True
        except ImportError:
            print("[ComfyUI-Upscaler-TensorRT] Install marker exists but tensorrt is not importable; retrying.")
            try:
                installed_marker.unlink()
            except Exception:
                pass

    # Avoid retrying too often after a failure (do not block every startup).
    if failed_marker.exists():
        try:
            last_fail = failed_marker.stat().st_mtime
            if time.time() - last_fail < 3600:
                print("[ComfyUI-Upscaler-TensorRT] Recent failed install attempt; skipping auto-install.")
                print("To retry now, delete the .tensorrt_auto_install_failed marker file or wait 1 hour.")
                return False
        except Exception:
            pass

    try:
        # Check if TensorRT is already installed.
        try:
            import tensorrt
            print("[ComfyUI-Upscaler-TensorRT] TensorRT already installed")
            installed_marker.touch()
            return True
        except ImportError:
            print("[ComfyUI-Upscaler-TensorRT] TensorRT not found, detecting CUDA version...")

        # Detect CUDA version
        cuda_version = None

        # Try nvcc command
        try:
            result = subprocess.run("nvcc --version", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                match = re.search(r"release (\d+\.\d+)", result.stdout)
                if match:
                    cuda_version = match.group(1)
                    print(f"[ComfyUI-Upscaler-TensorRT] Detected CUDA version: {cuda_version}")
        except Exception:
            pass

        # Try CUDA_PATH
        if not cuda_version and os.environ.get("CUDA_PATH"):
            nvcc_path = os.path.join(os.environ["CUDA_PATH"], "bin", "nvcc")
            if os.path.exists(nvcc_path):
                try:
                    result = subprocess.run(f'"{nvcc_path}" --version', shell=True, capture_output=True, text=True)
                    if result.returncode == 0:
                        match = re.search(r"release (\d+\.\d+)", result.stdout)
                        if match:
                            cuda_version = match.group(1)
                            print(f"[ComfyUI-Upscaler-TensorRT] Detected CUDA via CUDA_PATH: {cuda_version}")
                except Exception:
                    pass

        # Try CUDA_HOME
        if not cuda_version and os.environ.get("CUDA_HOME"):
            nvcc_path = os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc")
            if os.path.exists(nvcc_path):
                try:
                    result = subprocess.run(f'"{nvcc_path}" --version', shell=True, capture_output=True, text=True)
                    if result.returncode == 0:
                        match = re.search(r"release (\d+\.\d+)", result.stdout)
                        if match:
                            cuda_version = match.group(1)
                            print(f"[ComfyUI-Upscaler-TensorRT] Detected CUDA via CUDA_HOME: {cuda_version}")
                except Exception:
                    pass

        if not cuda_version:
            print("[ComfyUI-Upscaler-TensorRT] WARNING: Could not detect CUDA version automatically.")
            print("The NVIDIA CUDA Toolkit must be installed before TensorRT can work.")
            print("Please run 'python install.py' manually after installing CUDA.")
            failed_marker.touch()
            return False

        major_version = int(cuda_version.split('.')[0])

        if major_version == 13:
            print("[ComfyUI-Upscaler-TensorRT] Installing CUDA 13 TensorRT packages (RTX 50 series)")
            req_file = "requirements_cu13.txt"
        elif major_version == 12:
            print("[ComfyUI-Upscaler-TensorRT] Installing CUDA 12 TensorRT packages (RTX 30/40 series)")
            req_file = "requirements_cu12.txt"
        else:
            print(f"[ComfyUI-Upscaler-TensorRT] Unsupported CUDA version: {cuda_version}")
            failed_marker.touch()
            return False

        req_path = node_dir / req_file
        if not req_path.exists():
            print(f"[ComfyUI-Upscaler-TensorRT] Missing requirements file: {req_path}")
            failed_marker.touch()
            return False

        # Install base dependencies first, then the CUDA-specific TensorRT wheels.
        for req_name in ["requirements.txt", req_file]:
            req_file_path = node_dir / req_name
            if not req_file_path.exists():
                continue
            print(f"[ComfyUI-Upscaler-TensorRT] Installing from {req_name}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-input", "--prefer-binary", "-r", str(req_file_path)],
                capture_output=True
            )
            if result.returncode != 0:
                print(f"[ComfyUI-Upscaler-TensorRT] Failed to install {req_name}")
                print(result.stderr.decode(errors="replace"))
                failed_marker.touch()
                return False
            # Show pip stdout so the user can see progress / warnings.
            stdout = result.stdout.decode(errors="replace").strip()
            if stdout:
                print(stdout)

        installed_marker.touch()
        print("[ComfyUI-Upscaler-TensorRT] TensorRT installation completed successfully!")
        return True

    except Exception as e:
        print(f"[ComfyUI-Upscaler-TensorRT] Auto-installation failed: {e}")
        print("Please run 'python install.py' manually to install TensorRT")
        try:
            failed_marker.touch()
        except Exception:
            pass
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
IMAGE_DIM_MAX = 2048

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
                "images": ("IMAGE", {"tooltip": "Input images for upscaling"}),
                "upscaler_trt_model": ("UPSCALER_TRT_MODEL", {"tooltip": "Tensorrt model built and loaded"}),
                "resize_to": (["2x", "3x", "4x", "1080p", "2K", "4K", "custom"], {"default": "2x", "tooltip": "Target upscaling factor or fixed resolution preset"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1,
                                       "tooltip": "Number of images per GPU call. "
                                                  "Higher values improve throughput but use more VRAM. "
                                                  "Must be <= the batch_size used when building the engine. "
                                                  "Set to 1 for the safest behaviour."}),
            },
            "optional": {
                "resize_width": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 8, "tooltip": "Custom width (used when resize_to='custom')"}),
                "resize_height": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 8, "tooltip": "Custom height (used when resize_to='custom')"}),
            }
        }

    RETURN_NAMES = ("upscaled_images",)
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "⚡️ TensorRT/Upscaler"
    DESCRIPTION = "Upscale images using TensorRT acceleration."
    FUNCTION = "upscaler_tensorrt"

    def upscaler_tensorrt(self, **kwargs):
        images = kwargs.get("images")
        upscaler_trt_model = kwargs.get("upscaler_trt_model")
        resize_to = kwargs.get("resize_to")
        # Auto-detect upscale factor from the engine filename stored on the model object
        upscale_factor = getattr(upscaler_trt_model, "_upscale_factor", 4)
        batch_size = max(1, int(kwargs.get("batch_size", 1)))

        images_bchw = images.permute(0, 3, 1, 2)
        B, C, H, W = images_bchw.shape

        # Compute the desired final resolution using the *original* input dimensions.
        if resize_to == "custom":
            final_width = kwargs.get("resize_width")
            final_height = kwargs.get("resize_height")
        else:
            final_width, final_height = get_final_resolutions(W, H, resize_to, upscale_factor)

        # If the input is larger than the TensorRT engine max profile, resize it
        # down before upscaling. We keep the final target resolution so the output
        # is still resized to what the user asked for.
        if H > IMAGE_DIM_MAX or W > IMAGE_DIM_MAX:
            logger.warning(f"Input {W}x{H} exceeds max engine size {IMAGE_DIM_MAX}; resizing before upscale.")
            scale = IMAGE_DIM_MAX / max(H, W)
            new_H = int(H * scale)
            new_W = int(W * scale)
            # Ensure dimensions stay within bounds and are multiples of 8 (safe for ESRGAN).
            new_H = min(new_H, IMAGE_DIM_MAX) // 8 * 8
            new_W = min(new_W, IMAGE_DIM_MAX) // 8 * 8
            new_H = max(new_H, IMAGE_DIM_MIN)
            new_W = max(new_W, IMAGE_DIM_MIN)
            images_bchw = torch.nn.functional.interpolate(
                images_bchw,
                size=(new_H, new_W),
                mode='bicubic',
                antialias=True
            )
            H, W = new_H, new_W

        for dim in (H, W):
            if dim > IMAGE_DIM_MAX or dim < IMAGE_DIM_MIN:
                raise ValueError(f"Input image dimensions fall outside of the supported range: {IMAGE_DIM_MIN}x{IMAGE_DIM_MIN} to {IMAGE_DIM_MAX}x{IMAGE_DIM_MAX} px!\nImage dimensions: {W}px by {H}px")

        logger.info(f"Upscaling {B} images from H:{H}, W:{W} to H:{H*upscale_factor}, W:{W*upscale_factor} | Final resolution: H:{final_height}, W:{final_width} | resize_to: {resize_to} | scale: {upscale_factor}x | batch_size: {batch_size}")

        upscaler_trt_model.activate()
        cudaStream = torch.cuda.current_stream().cuda_stream
        pbar = ProgressBar(B)

        upscaled_frames = torch.empty((B, C, final_height, final_width), dtype=torch.float32, device=mm.intermediate_device())
        must_resize = W*upscale_factor != final_width or H*upscale_factor != final_height

        # Process in batches of batch_size
        for start_idx in range(0, B, batch_size):
            end_idx = min(start_idx + batch_size, B)
            current_batch_size = end_idx - start_idx
            batch = images_bchw[start_idx:end_idx]

            shape_dict = {
                "input": {"shape": (current_batch_size, 3, H, W)},
                "output": {"shape": (current_batch_size, 3, H*upscale_factor, W*upscale_factor)},
            }
            upscaler_trt_model.allocate_buffers(shape_dict=shape_dict)

            result = upscaler_trt_model.infer({"input": batch}, cudaStream)
            result = result["output"]

            if must_resize:
                result = torch.nn.functional.interpolate(
                    result,
                    size=(final_height, final_width),
                    mode='bicubic',
                    antialias=True
                )
            upscaled_frames[start_idx:end_idx] = result.to(mm.intermediate_device())
            pbar.update_absolute(end_idx)

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
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1,
                                       "tooltip": "Max batch size for the engine profile. "
                                                  "Higher values allow batched inference (multiple images per GPU call) "
                                                  "for better throughput, but use more VRAM and require a rebuild. "
                                                  "Set to 1 for the original behaviour."}),
            }
        }
    
    RETURN_NAMES = ("upscaler_trt_model",)
    RETURN_TYPES = ("UPSCALER_TRT_MODEL",)
    # FUNCTION = "main" # This was duplicated, removing
    CATEGORY = "⚡️ TensorRT/Upscaler"
    DESCRIPTION = "Load tensorrt models, they will be built automatically if not found."
    FUNCTION = "load_upscaler_tensorrt_model" # This is the correct one
    
    def load_upscaler_tensorrt_model(self, model, precision, batch_size=1):
        tensorrt_models_dir = os.path.join(folder_paths.models_dir, "tensorrt", "upscaler")
        onnx_models_dir = os.path.join(folder_paths.models_dir, "onnx")

        os.makedirs(tensorrt_models_dir, exist_ok=True)
        os.makedirs(onnx_models_dir, exist_ok=True)

        onnx_model_path = os.path.join(onnx_models_dir, f"{model}.onnx")

        # Auto-detect upscale factor from model name
        model_lower = model.lower()
        if model_lower.startswith("2x") or "x2plus" in model_lower or "x2" in model_lower:
            upscale_factor = 2
        elif model_lower.startswith("4x") or "x4plus" in model_lower or "x4" in model_lower:
            upscale_factor = 4
        else:
            upscale_factor = 4  # default to 4x for unknown models
        logger.info(f"Auto-detected upscale factor: {upscale_factor}x for model '{model}'")
        
        engine_channel = 3
        engine_min_batch = 1
        engine_opt_batch = max(1, batch_size)
        engine_max_batch = max(1, batch_size)
        engine_min_h, engine_opt_h, engine_max_h = IMAGE_DIM_MIN, IMAGE_DIM_OPT, IMAGE_DIM_MAX
        engine_min_w, engine_opt_w, engine_max_w = IMAGE_DIM_MIN, IMAGE_DIM_OPT, IMAGE_DIM_MAX
        tensorrt_model_path = os.path.join(tensorrt_models_dir, f"{model}_{precision}_{engine_min_batch}x{engine_channel}x{engine_min_h}x{engine_min_w}_{engine_opt_batch}x{engine_channel}x{engine_opt_h}x{engine_opt_w}_{engine_max_batch}x{engine_channel}x{engine_max_h}x{engine_max_w}_{tensorrt.__version__}.trt")

        if not os.path.exists(tensorrt_model_path):
            if not os.path.exists(onnx_model_path):
                # Try huchukato/garage/onnx/upscale first (dynamic shapes), then fallback to yuvraj108c
                onnx_model_download_url = f"https://huggingface.co/huchukato/garage/resolve/main/onnx/upscale/{model}.onnx"
                logger.info(f"Downloading {onnx_model_download_url}")
                try:
                    download_file(url=onnx_model_download_url, save_path=onnx_model_path)
                except Exception as e:
                    logger.warning(f"Failed to download from huchukato/garage: {e}")
                    onnx_model_download_url = f"https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/{model}.onnx"
                    logger.info(f"Fallback: Downloading {onnx_model_download_url}")
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
                    {"input": [(engine_min_batch,engine_channel,engine_min_h,engine_min_w), (engine_opt_batch,engine_channel,engine_opt_h,engine_opt_w), (engine_max_batch,engine_channel,engine_max_h,engine_max_w)]},
                ],
                enable_all_tactics=True,
            )
            e = time.time()
            logger.info(f"Time taken to build: {(e-s)} seconds")

        logger.info(f"Loading TensorRT engine: {tensorrt_model_path}")
        mm.soft_empty_cache()
        engine = Engine(tensorrt_model_path)
        engine.load()

        # Attach upscale factor so the runner node can auto-detect it
        engine._upscale_factor = upscale_factor

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
