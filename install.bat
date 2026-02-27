@echo off
REM Auto-install script for ComfyUI Upscaler TensorRT (Windows)
REM Detects CUDA version and installs appropriate requirements

echo 🔍 Detecting CUDA version...

REM Try nvcc command
nvcc --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=3" %%i in ('nvcc --version ^| find "release"') do (
        set CUDA_VERSION=%%i
        set CUDA_VERSION=!CUDA_VERSION:~0,-1!
    )
    if defined CUDA_VERSION (
        echo ✅ Found CUDA version: !CUDA_VERSION!
    )
)

REM Try CUDA_PATH if nvcc failed
if not defined CUDA_VERSION (
    if defined CUDA_PATH (
        if exist "%CUDA_PATH%\bin\nvcc.exe" (
            for /f "tokens=3" %%i in ('"%CUDA_PATH%\bin\nvcc.exe" --version ^| find "release"') do (
                set CUDA_VERSION=%%i
                set CUDA_VERSION=!CUDA_VERSION:~0,-1!
            )
            if defined CUDA_VERSION (
                echo ✅ Found CUDA version via CUDA_PATH: !CUDA_VERSION!
            )
        )
    )
)

REM Try CUDA_HOME if still failed
if not defined CUDA_VERSION (
    if defined CUDA_HOME (
        if exist "%CUDA_HOME%\bin\nvcc.exe" (
            for /f "tokens=3" %%i in ('"%CUDA_HOME%\bin\nvcc.exe" --version ^| find "release"') do (
                set CUDA_VERSION=%%i
                set CUDA_VERSION=!CUDA_VERSION:~0,-1!
            )
            if defined CUDA_VERSION (
                echo ✅ Found CUDA version via CUDA_HOME: !CUDA_VERSION!
            )
        )
    )
)

if not defined CUDA_VERSION (
    echo ⚠️  Could not detect CUDA version automatically
    echo Please ensure CUDA is installed and nvcc is in your PATH
    echo Or set CUDA_PATH or CUDA_HOME environment variables
    pause
    exit /b 1
)

REM Extract major version
for /f "tokens=1 delims=." %%i in ("!CUDA_VERSION!") do set MAJOR_VERSION=%%i

echo 📦 Installing requirements for CUDA !MAJOR_VERSION!...

if "!MAJOR_VERSION!"=="13" (
    echo 🚀 Using CUDA 13 requirements ^(RTX 50 series^)
    python -m pip install tensorrt_cu13==10.15.1.29 tensorrt_cu13_bindings==10.15.1.29 tensorrt_cu13_libs==10.15.1.29 "cuda-toolkit>=13.0.0,<13.1.0" polygraphy requests
) else if "!MAJOR_VERSION!"=="12" (
    echo 🔧 Using CUDA 12 requirements ^(RTX 30/40 series^)
    python -m pip install tensorrt-cu12==10.13.3.9 tensorrt-cu12-libs==10.13.3.9 tensorrt-cu12-bindings==10.13.3.9 "cuda-toolkit>=12.8.0,<13.0.0" polygraphy requests
) else (
    echo ❌ Unsupported CUDA version: !CUDA_VERSION!
    echo Supported versions: CUDA 12.x, CUDA 13.x
    pause
    exit /b 1
)

if %errorlevel% equ 0 (
    echo ✅ Installation completed successfully!
    echo 🎯 You can now use the ComfyUI Upscaler TensorRT node
) else (
    echo ❌ Installation failed!
)
pause
