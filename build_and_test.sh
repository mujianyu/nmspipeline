#!/bin/bash
# ============================================================================
# build_and_test.sh
# 1. Compile nmsFullPlugin.so from nmsFull.cu + nmsFullPlugin.cpp
# 2. Run insert_nms.py to create TwoStreamMPF_nms.onnx
# 3. Build TensorRT engine with trtexec + plugin
# 4. Benchmark inference speed
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Paths -------------------------------------------------------
CUDA_PATH="/usr/local/cuda-11.8"
TRT_INCLUDE="/opt/tensorrt/include"
TRT_LIB="/usr/lib/x86_64-linux-gnu"
TRT_BIN="/opt/tensorrt/targets/x86_64-linux-gnu/bin"
CUB_INCLUDE="/usr/local/cuda-11.8/targets/x86_64-linux/include"

NVCC="${CUDA_PATH}/bin/nvcc"

PLUGIN_SO="nmsFullPlugin.so"
ENGINE_FILE="TwoStreamMPF_nms.engine"
ONNX_NMS="TwoStreamMPF_nms.onnx"
ONNX_ORIG="TwoStreamMPF.onnx"

echo "========================================"
echo " Build & Test Pipeline"
echo "========================================"

# Step 1: Compile nmsFullPlugin.so -----------------------------
echo ""
echo "[1/4] Compiling nmsFullPlugin.so ..."

# Use g++ for the .cpp part (TRT plugin), nvcc for the .cu part (CUDA kernels)
# Strategy: compile .cu to .o with nvcc, compile .cpp to .o with g++, link together

# Compile nmsFull.cu -> nmsFull.cu.o
$NVCC -c nmsFull.cu -o nmsFull.cu.o \
    -I${TRT_INCLUDE} \
    -I${CUB_INCLUDE} \
    -Xcompiler -fPIC \
    -std=c++14 \
    -arch=sm_75 \
    -O3 \
    --expt-relaxed-constexpr

# Compile nmsFullPlugin.cpp -> nmsFullPlugin.cpp.o
$NVCC -c nmsFullPlugin.cpp -o nmsFullPlugin.cpp.o \
    -I${TRT_INCLUDE} \
    -I${CUB_INCLUDE} \
    -I. \
    -Xcompiler -fPIC \
    -std=c++14 \
    -O3

# Link into shared library
$NVCC -shared nmsFull.cu.o nmsFullPlugin.cpp.o \
    -o ${PLUGIN_SO} \
    -L${TRT_LIB} -lnvinfer \
    -lcudart

echo "  -> Created ${PLUGIN_SO}"

# Step 2: Insert NmsFull node into ONNX -----------------------
echo ""
echo "[2/4] Inserting NmsFull node into ONNX ..."
python3 insert_nms.py
echo "  -> Created ${ONNX_NMS}"

# Step 3: Build TensorRT Engine -------------------------------
echo ""
echo "[3/4] Building TensorRT engine ..."
${TRT_BIN}/trtexec \
    --onnx=${ONNX_NMS} \
    --plugins=${PLUGIN_SO} \
    --saveEngine=${ENGINE_FILE} \
    --fp16 \
    --verbose \
    2>&1 | tail -30

if [ -f "${ENGINE_FILE}" ]; then
    echo "  -> Engine saved: ${ENGINE_FILE}"
    ENGINE_SIZE=$(du -h ${ENGINE_FILE} | cut -f1)
    echo "  -> Engine size: ${ENGINE_SIZE}"
else
    echo "  -> ERROR: Engine build failed!"
    exit 1
fi

# Step 4: Benchmark -------------------------------------------
echo ""
echo "[4/4] Benchmarking inference speed ..."
echo ""
${TRT_BIN}/trtexec \
    --loadEngine=${ENGINE_FILE} \
    --iterations=1000 \
    --avgRuns=10 \
    2>&1 | grep -E "Throughput|Latency|mean|min|max|H2D|GPU|D2H|Total Host"

echo ""
echo "========================================"
echo " Done!"
echo "========================================"
