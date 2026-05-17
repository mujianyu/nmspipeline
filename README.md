# NMSFast — TensorRT NMS 集成方案

将 `nmsFull.cu` 的 CUDA NMS kernel 集成到 TensorRT 推理管线中，提供两种方案：

| 方案 | 描述 | 文件 |
|------|------|------|
| **A: GPU NMS 插件** | `nonBlockNmsFull` CUDA kernel 封装为 TensorRT IPluginV2IOExt，插入 ONNX 图后由 `trtexec` 编译为单一 engine | `nmsFullPlugin.h/.cpp`, `build_and_test.sh` |
| **B: CPU 传统 NMS** | 原始 ONNX 直接导出 TRT engine（输出 `[1,9,8400]`），在 CPU 上执行传统 per-class greedy NMS | `demo_cpu_nms.py` |

---

## 环境要求

| 组件 | 版本 | 路径 |
|------|------|------|
| CUDA | 11.8 | `/usr/local/cuda-11.8` |
| TensorRT | 8.6.1 | `/opt/tensorrt` |
| Python (conda) | 3.8 (env `t`) | `/root/miniconda3/envs/t` |
| Python 依赖 | tensorrt, pycuda, numpy, onnx | env `t` 中已安装 |

```bash
# 激活 conda 环境
/root/miniconda3/envs/t/bin/python -c "import tensorrt, pycuda, numpy; print('OK')"
```

---

## 方案 A：GPU NMS 插件（一键构建）

### 构建流程

```bash
cd /root/nmsfast && bash build_and_test.sh
```

脚本自动执行 4 步：

1. **编译插件** — `nmsFull.cu` + `nmsFullPlugin.cpp` → `nmsFullPlugin.so`
2. **插入 ONNX 节点** — 运行 `insert_nms.py`，生成 `TwoStreamMPF_nms.onnx`
3. **构建 TRT Engine** — `trtexec --onnx=TwoStreamMPF_nms.onnx --plugins=nmsFullPlugin.so --saveEngine=TwoStreamMPF_nms.engine --fp16`
4. **基准测试** — `trtexec --loadEngine=TwoStreamMPF_nms.engine --iterations=1000`

### 手动分布构建

```bash
# Step 1: 编译插件 .so
/usr/local/cuda-11.8/bin/nvcc -c nmsFull.cu -o nmsFull.cu.o \
    -I/opt/tensorrt/include -I/usr/local/cuda-11.8/targets/x86_64-linux/include \
    -Xcompiler -fPIC -std=c++14 -arch=sm_75 -O3 --expt-relaxed-constexpr

/usr/local/cuda-11.8/bin/nvcc -c nmsFullPlugin.cpp -o nmsFullPlugin.cpp.o \
    -I/opt/tensorrt/include -I/usr/local/cuda-11.8/targets/x86_64-linux/include \
    -I. -Xcompiler -fPIC -std=c++14 -O3

/usr/local/cuda-11.8/bin/nvcc -shared nmsFull.cu.o nmsFullPlugin.cpp.o \
    -o nmsFullPlugin.so -L/usr/lib/x86_64-linux-gnu -lnvinfer -lcudart

# Step 2: 插入 NmsFull 节点到 ONNX
python3 insert_nms.py

# Step 3: 构建 TRT engine
/opt/tensorrt/targets/x86_64-linux-gnu/bin/trtexec \
    --onnx=TwoStreamMPF_nms.onnx \
    --plugins=nmsFullPlugin.so \
    --saveEngine=TwoStreamMPF_nms.engine \
    --fp16

# Step 4: 基准测试
/opt/tensorrt/targets/x86_64-linux-gnu/bin/trtexec \
    --loadEngine=TwoStreamMPF_nms.engine \
    --iterations=1000 \
    --avgRuns=10
```

### Python 推理 (GPU Plugin)

```bash
# 加载 plugin engine 进行 Python 推理 + 计时
/root/miniconda3/envs/t/bin/python test_nms.py \
    --engine TwoStreamMPF_nms.engine \
    --iterations 100 \
    --warmup 20
```

### NMS 参数

在 [`insert_nms.py`](insert_nms.py) 中修改：

```python
SCORE_THRESHOLD  = 0.25   # 置信度阈值
IOU_THRESHOLD    = 0.45   # IoU 阈值
MAX_OUTPUT_BOXES = 300    # 最大输出框数
PER_BOX_STRIDE   = 9      # 4 bbox + 5 class scores
NUM_CLASSES      = 5      # 类别数
```

---

## 方案 B：CPU 传统 NMS

### 构建原始 TRT Engine（不含 NMS）

```bash
/opt/tensorrt/targets/x86_64-linux-gnu/bin/trtexec \
    --onnx=TwoStreamMPF.onnx \
    --saveEngine=TwoStreamMPF_raw.engine \
    --fp16
```

### 运行 CPU NMS Demo

```bash
/root/miniconda3/envs/t/bin/python demo_cpu_nms.py \
    --raw-engine TwoStreamMPF_raw.engine \
    --plugin-engine TwoStreamMPF_nms.engine \
    --iterations 100 \
    --warmup 20 \
    --score-threshold 0.25 \
    --iou-threshold 0.45 \
    --max-output 300
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--raw-engine` | `TwoStreamMPF_raw.engine` | 原始 TRT engine（输出 `[1,9,8400]`） |
| `--plugin-engine` | `TwoStreamMPF_nms.engine` | GPU NMS plugin engine（用于对比） |
| `--iterations` | `100` | 计时迭代次数 |
| `--warmup` | `20` | 预热迭代次数 |
| `--score-threshold` | `0.25` | NMS 置信度阈值 |
| `--iou-threshold` | `0.45` | NMS IoU 阈值 |
| `--max-output` | `300` | 每 batch 最大输出检测数 |

### 输出格式

| 输出 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| boxes | `[batch, 300, 4]` | float32 | `(x1, y1, x2, y2)` 角点格式 |
| scores | `[batch, 300]` | float32 | 检测置信度 |
| classes | `[batch, 300]` | int32 | 类别 ID (0-4) |
| counts | `[batch]` | int32 | 每 batch 有效检测数量 |

---

## 文件清单

| 文件 | 说明 |
|------|------|
| [`nmsFull.cu`](nmsFull.cu) | CUDA NMS kernel（已参数化 `per_box_stride` 和 `num_classes`） |
| [`nmsFullPlugin.h`](nmsFullPlugin.h) | TensorRT `IPluginV2IOExt` 插件头文件 |
| [`nmsFullPlugin.cpp`](nmsFullPlugin.cpp) | 插件实现（enqueue / serialize / workspace / REGISTER） |
| [`nmsFullPlugin.so`](nmsFullPlugin.so) | 编译后的插件动态库 (~777 KB) |
| [`insert_nms.py`](insert_nms.py) | 向 ONNX 图中插入 `NmsFull` 自定义算子 |
| [`build_and_test.sh`](build_and_test.sh) | 一键构建 + 测试脚本 |
| [`test_nms.py`](test_nms.py) | GPU Plugin engine Python 推理测试 |
| [`demo_cpu_nms.py`](demo_cpu_nms.py) | CPU NMS 方案（TRT 推理 + numpy NMS） |
| [`TwoStreamMPF.onnx`](TwoStreamMPF.onnx) | 原始 ONNX 模型（输入 `[1,6,640,640]`，输出 `[1,9,8400]`） |
| [`TwoStreamMPF_nms.onnx`](TwoStreamMPF_nms.onnx) | 插入 NmsFull 节点后的中间 ONNX（~15 MB） |
| [`TwoStreamMPF_nms.engine`](TwoStreamMPF_nms.engine) | GPU NMS plugin TRT engine（~11 MB, FP16） |
| [`TwoStreamMPF_raw.engine`](TwoStreamMPF_raw.engine) | 原始 TRT engine（输出 `[1,9,8400]`，FP16） |

---

## 性能对比

> 测试环境：TensorRT 8.6.1, FP16, GPU compute, input `[1,6,640,640]`

| 方案 | 吞吐量 | 平均延迟 | NMS 开销 |
|------|--------|----------|----------|
| **A: GPU NMS Plugin** | 258.0 FPS | 3.875 ms | 内嵌（~0 ms） |
| **B: CPU 传统 NMS** | 245.0 FPS | 4.081 ms | ~0.12 ms (CPU) |
| **TRT only (无 NMS)** | 254.5 FPS | 3.929 ms | — |

**结论**：对于 5 类 × 8400 框的规模，CPU NMS 仅增加 ~0.12ms 开销，与 GPU 插件方案性能几乎一致（1.05x 差距）。

---

## 常见问题

### Q: 如何修改 NMS 参数？

- **GPU Plugin 方案**：修改 [`insert_nms.py`](insert_nms.py) 中的常量，重新运行 `build_and_test.sh`。
- **CPU 方案**：通过 `demo_cpu_nms.py` 的 `--score-threshold` / `--iou-threshold` / `--max-output` 参数传入。

### Q: `trtexec` 报错找不到 plugin？

确保 `--plugins=nmsFullPlugin.so` 使用**绝对路径**或插件 `.so` 在当前目录。

### Q: 更换模型 / 不同类别数？

1. 修改 `nmsFull.cu` 中的 `num_classes` 参数调用
2. 修改 [`insert_nms.py`](insert_nms.py) 中的 `PER_BOX_STRIDE` 和 `NUM_CLASSES`
3. 修改 [`nmsFullPlugin.cpp`](nmsFullPlugin.cpp) 中 `NmsFullPluginCreator::createPlugin()` 的默认值
4. 重新编译 `.so` 和 engine

### Q: Python `import tensorrt` 失败？

使用 conda env `t`：
```bash
/root/miniconda3/envs/t/bin/python your_script.py
```

### Q: 想用真实输入而非随机数？

修改 `test_nms.py` 或 `demo_cpu_nms.py` 中 `inp["host"] = ...` 的部分，改为加载实际图像数据。
