# Update Log

## Version 1.1.3

- Removed the bogus `requires-comfyui >=1.0.0` constraint — ComfyUI versions are 0.x, and the mismatch was disabling the node pack in ComfyUI Manager.

## Version 1.1.2

- CUDA detection now invokes `nvcc --version` via argv lists instead of `shell=True` (registry `python_command_injection_risk` false positive; also general hardening).

## Version 1.1.1

- Fixed engine build failure on Blackwell (SM120) GPUs: when the fp16 Myelin tactic search fails (`No conv tactic found` / `Invalid Engine`, e.g. on 4xUltrasharp with TensorRT 10.15), the builder now deletes the partial engine and automatically retries in fp32. An existing fp32 engine is reused on subsequent runs. Manual workaround (setting `precision=fp32` in the node) is no longer needed.
- Publish workflow: changelog extraction rewritten in plain shell (registry scanner false positive on embedded Python).

## Version 1.1.0

- Added resolution presets (1080p / 2K / 4K) and fixed auto-install reliability.
- Reduced install time; no more PyPI cuda-toolkit download.
- Updated Comfy Node Registry metadata (publisher, description, icon, banner).
