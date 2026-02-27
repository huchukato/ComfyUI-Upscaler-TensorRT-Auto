#!/bin/bash

# FALLBACK install script for ComfyUI Upscaler TensorRT (Linux/macOS)
# 
# NOTE: This is a BACKUP script - auto-installation is recommended!
# 
# PRIMARY METHOD (Recommended):
# 1. pip install -r requirements.txt
# 2. Restart ComfyUI
# 3. Node auto-detects CUDA and installs TensorRT automatically
#
# Use this script ONLY if auto-installation fails!
#

echo "⚠️  FALLBACK INSTALLATION - Use only if auto-installation fails!"
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

# Install appropriate requirements based on CUDA version
if [ "$MAJOR_VERSION" = "13" ]; then
    echo "🚀 Installing CUDA 13 requirements (RTX 50 series)"
    echo "📦 Installing base dependencies + CUDA 13 TensorRT..."
    python3 -m pip install -r requirements.txt
    echo "📦 Installing CUDA 13 specific TensorRT packages..."
    python3 -m pip install -r requirements_cu13.txt
elif [ "$MAJOR_VERSION" = "12" ]; then
    echo "🔧 Installing CUDA 12 requirements (RTX 30/40 series)"
    echo "📦 Installing base dependencies + CUDA 12 TensorRT..."
    python3 -m pip install -r requirements.txt
    echo "📦 Installing CUDA 12 specific TensorRT packages..."
    python3 -m pip install -r requirements_cu12.txt
else
    echo "❌ Unsupported CUDA version: $CUDA_VERSION"
    echo "Supported versions: CUDA 12.x, CUDA 13.x"
    exit 1
fi

if [ $? -eq 0 ]; then
    echo "✅ Fallback installation completed!"
    echo "🎯 You can now use ComfyUI Upscaler TensorRT node"
    echo "💡 In the future, try auto-installation by just installing requirements.txt"
else
    echo "❌ Installation failed!"
    exit 1
fi
