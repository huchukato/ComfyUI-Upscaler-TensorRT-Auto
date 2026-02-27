#!/bin/bash

# Auto-install script for ComfyUI Upscaler TensorRT (Linux/macOS)
# Detects CUDA version and installs appropriate requirements

echo "🔍 Detecting CUDA version..."

# Try nvcc command
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1')
    if [ ! -z "$CUDA_VERSION" ]; then
        echo "✅ Found CUDA version: $CUDA_VERSION"
    fi
fi

# Try CUDA_PATH if nvcc failed
if [ -z "$CUDA_VERSION" ] && [ ! -z "$CUDA_PATH" ]; then
    if [ -f "$CUDA_PATH/bin/nvcc" ]; then
        CUDA_VERSION=$("$CUDA_PATH/bin/nvcc" --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1')
        if [ ! -z "$CUDA_VERSION" ]; then
            echo "✅ Found CUDA version via CUDA_PATH: $CUDA_VERSION"
        fi
    fi
fi

# Try CUDA_HOME if still failed
if [ -z "$CUDA_VERSION" ] && [ ! -z "$CUDA_HOME" ]; then
    if [ -f "$CUDA_HOME/bin/nvcc" ]; then
        CUDA_VERSION=$("$CUDA_HOME/bin/nvcc" --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1')
        if [ ! -z "$CUDA_VERSION" ]; then
            echo "✅ Found CUDA version via CUDA_HOME: $CUDA_VERSION"
        fi
    fi
fi

if [ -z "$CUDA_VERSION" ]; then
    echo "⚠️  Could not detect CUDA version automatically"
    echo "Please ensure CUDA is installed and nvcc is in your PATH"
    echo "Or set CUDA_PATH or CUDA_HOME environment variables"
    exit 1
fi

# Extract major version
MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d. -f1)

echo "📦 Installing requirements for CUDA $MAJOR_VERSION..."

if [ "$MAJOR_VERSION" = "13" ]; then
    echo "🚀 Using CUDA 13 requirements (RTX 50 series)"
    python3 -m pip install tensorrt_cu13==10.15.1.29 tensorrt_cu13_bindings==10.15.1.29 tensorrt_cu13_libs==10.15.1.29 "cuda-toolkit>=13.0.0,<13.1.0" polygraphy requests
elif [ "$MAJOR_VERSION" = "12" ]; then
    echo "🔧 Using CUDA 12 requirements (RTX 30/40 series)"
    python3 -m pip install tensorrt-cu12==10.13.3.9 tensorrt-cu12-libs==10.13.3.9 tensorrt-cu12-bindings==10.13.3.9 "cuda-toolkit>=12.8.0,<13.0.0" polygraphy requests
else
    echo "❌ Unsupported CUDA version: $CUDA_VERSION"
    echo "Supported versions: CUDA 12.x, CUDA 13.x"
    exit 1
fi

if [ $? -eq 0 ]; then
    echo "✅ Installation completed successfully!"
    echo "🎯 You can now use the ComfyUI Upscaler TensorRT node"
else
    echo "❌ Installation failed!"
    exit 1
fi
