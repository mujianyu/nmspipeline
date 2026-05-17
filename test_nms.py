#!/usr/bin/env python3
"""
test_nms.py
Loads the TensorRT engine (TwoStreamMPF_nms.engine) via tensorrt.Runtime,
runs CUDA inference with random input [1, 6, 640, 640],
measures latency breakdown, and reports results.

Usage:
    python3 test_nms.py [--engine ENGINE_PATH] [--iterations N] [--warmup N]
"""

import numpy as np
import os
import sys
import time
import argparse

# tensorrt and cuda imports — may need conda env "t" with TRT 8.6.1 pip installed
try:
    import tensorrt as trt
except ImportError:
    print("ERROR: tensorrt not installed. Install with:")
    print("  pip install /opt/tensorrt/python/tensorrt-8.6.1-cp3XX-none-linux_x86_64.whl")
    sys.exit(1)

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    print("ERROR: pycuda not installed. Install with: pip install pycuda")
    sys.exit(1)


def load_engine(engine_path):
    """Load a serialized TensorRT engine."""
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    engine = runtime.deserialize_cuda_engine(engine_data)
    return engine


def allocate_buffers(engine):
    """Allocate host and device buffers for all engine bindings."""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = engine.get_binding_dtype(i)
        shape = engine.get_binding_shape(i)
        size = trt.volume(shape) * dtype.itemsize
        # Allocate device memory
        d_buf = cuda.mem_alloc(size)
        # Allocate host memory (pinned for speed)
        h_buf = cuda.pagelocked_empty(trt.volume(shape), dtype=np.float32 if dtype == trt.float32 else np.int32)
        bindings.append(int(d_buf))
        if engine.binding_is_input(i):
            inputs.append({"name": name, "shape": shape, "dtype": dtype, "host": h_buf, "device": d_buf, "index": i})
        else:
            outputs.append({"name": name, "shape": shape, "dtype": dtype, "host": h_buf, "device": d_buf, "index": i})
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):
    """Execute inference."""
    # Transfer inputs to device
    for inp in inputs:
        cuda.memcpy_htod_async(inp["device"], inp["host"], stream)

    # Execute
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Transfer outputs back to host
    for out in outputs:
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)

    stream.synchronize()


def main():
    parser = argparse.ArgumentParser(description="Test NmsFull TensorRT Engine")
    parser.add_argument("--engine", default="TwoStreamMPF_nms.engine",
                        help="Path to TensorRT engine file")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of inference iterations for timing")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Number of warmup iterations")
    args = parser.parse_args()

    engine_path = args.engine
    if not os.path.isfile(engine_path):
        print(f"ERROR: Engine file not found: {engine_path}")
        print("Run build_and_test.sh first to generate the engine.")
        sys.exit(1)

    print("=" * 60)
    print(" NmsFull TensorRT Engine Test")
    print("=" * 60)
    print(f"Engine: {engine_path}")

    # Load engine
    print("\n[1] Loading engine ...")
    engine = load_engine(engine_path)

    # Print engine info
    print(f"  Num bindings: {engine.num_bindings}")
    print(f"  Inputs:")
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            dtype = engine.get_binding_dtype(i)
            print(f"    [{i}] {name}: shape={shape}, dtype={dtype}")

    print(f"  Outputs:")
    for i in range(engine.num_bindings):
        if not engine.binding_is_input(i):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            dtype = engine.get_binding_dtype(i)
            print(f"    [{i}] {name}: shape={shape}, dtype={dtype}")

    # Allocate buffers
    print("\n[2] Allocating buffers ...")
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()

    print(f"  Input tensors:")
    for inp in inputs:
        print(f"    {inp['name']}: {inp['shape']} ({inp['dtype']})")
    print(f"  Output tensors:")
    for out in outputs:
        print(f"    {out['name']}: {out['shape']} ({out['dtype']})")

    # Fill input with random data
    print("\n[3] Preparing input (random [1, 6, 640, 640]) ...")
    for inp in inputs:
        inp["host"] = np.random.randn(*inp["shape"]).astype(np.float32)
        print(f"  {inp['name']}: range [{inp['host'].min():.3f}, {inp['host'].max():.3f}]")

    # Set input shape if needed (for dynamic engines)
    for inp in inputs:
        context.set_binding_shape(inp["index"], inp["shape"])

    # Warmup
    print(f"\n[4] Warming up ({args.warmup} iterations) ...")
    for _ in range(args.warmup):
        do_inference(context, bindings, inputs, outputs, stream)

    # Benchmark
    print(f"[5] Benchmarking ({args.iterations} iterations) ...")
    timings = []
    for i in range(args.iterations):
        start = time.perf_counter()
        do_inference(context, bindings, inputs, outputs, stream)
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)  # ms

    timings = np.array(timings)
    avg_latency = np.mean(timings)
    min_latency = np.min(timings)
    max_latency = np.max(timings)
    std_latency = np.std(timings)
    fps = 1000.0 / avg_latency

    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  Average latency:  {avg_latency:.3f} ms")
    print(f"  Min latency:      {min_latency:.3f} ms")
    print(f"  Max latency:      {max_latency:.3f} ms")
    print(f"  Std deviation:    {std_latency:.3f} ms")
    print(f"  Throughput (FPS): {fps:.1f}")

    # Print output shapes and sample values
    print("\n" + "=" * 60)
    print("  Output Summary")
    print("=" * 60)
    for out in outputs:
        data = out["host"]
        name = out["name"]
        shape_val = out["shape"]
        dtype_val = out["dtype"]
        print(f"\n  {name} [{shape_val}] ({dtype_val}):")
        if dtype_val == trt.float32 or dtype_val == trt.DataType.FLOAT:
            print(f"    min={data.min():.5f}  max={data.max():.5f}  mean={data.mean():.5f}")
        else:
            print(f"    min={data.min()}  max={data.max()}  unique={len(np.unique(data))}")
        if data.size <= 20:
            print(f"    values={data}")
        elif data.ndim == 1:
            print(f"    first 20: {data[:20]}")
        elif data.ndim == 2:
            non_zero_rows = np.any(data != 0, axis=1)
            n_nonzero = non_zero_rows.sum()
            print(f"    non-zero rows: {n_nonzero}/{data.shape[0]}")
            if n_nonzero > 0 and n_nonzero <= 5:
                print(f"    non-zero data:\n{data[non_zero_rows]}")

    print("\n" + "=" * 60)
    print("  Test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
