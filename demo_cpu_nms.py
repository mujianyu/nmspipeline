#!/usr/bin/env python3
"""
demo_cpu_nms.py
===============
Pipeline: TRT raw model (output [1, 9, 8400]) → CPU traditional NMS → final detections.

Compares against the GPU NMS plugin approach (TwoStreamMPF_nms.engine).

Usage:
    /root/miniconda3/envs/t/bin/python demo_cpu_nms.py [--iterations 100] [--warmup 20]
"""

import numpy as np
import os
import sys
import time
import argparse
import ctypes

try:
    import tensorrt as trt
except ImportError:
    print("ERROR: tensorrt not installed")
    sys.exit(1)

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    print("ERROR: pycuda not installed")
    sys.exit(1)


# ============================================================================
# Traditional CPU NMS (multiclass)
# ============================================================================

def cpu_nms(raw_output, score_threshold=0.25, iou_threshold=0.45, max_output=300):
    """
    Traditional multiclass NMS on CPU.

    Args:
        raw_output:  numpy array [batch, per_box_stride, num_boxes]  (e.g. [1, 9, 8400])
                     Layout: raw[0:4, i] = bbox (cx, cy, w, h)
                             raw[4:9, i] = class scores (5 classes)
        score_threshold: minimum score to consider
        iou_threshold:   IoU threshold for suppression
        max_output:      maximum detections per batch item

    Returns:
        boxes_out:  [batch, max_output, 4]  (x1, y1, x2, y2)
        scores_out: [batch, max_output]
        classes_out:[batch, max_output]  (int32)
        counts_out: [batch]  (int32)
    """
    batch_size = raw_output.shape[0]
    num_classes = raw_output.shape[1] - 4  # per_box_stride - 4
    num_boxes = raw_output.shape[2]

    boxes_out = np.zeros((batch_size, max_output, 4), dtype=np.float32)
    scores_out = np.zeros((batch_size, max_output), dtype=np.float32)
    classes_out = np.zeros((batch_size, max_output), dtype=np.int32)
    counts_out = np.zeros((batch_size,), dtype=np.int32)

    for b in range(batch_size):
        # Extract data for this batch
        raw = raw_output[b]  # [9, 8400]

        # Convert center-format bbox to corner-format
        cx = raw[0, :]  # center x
        cy = raw[1, :]  # center y
        w  = raw[2, :]  # width
        h  = raw[3, :]  # height

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        detections = []  # (score, class, x1, y1, x2, y2)

        for cls in range(num_classes):
            scores = raw[4 + cls, :]  # [8400]
            mask = scores > score_threshold
            if not np.any(mask):
                continue

            # Get candidate boxes and scores
            cand_scores = scores[mask]
            cand_x1 = x1[mask]
            cand_y1 = y1[mask]
            cand_x2 = x2[mask]
            cand_y2 = y2[mask]

            # Sort by score descending
            order = np.argsort(cand_scores)[::-1]
            cand_scores = cand_scores[order]
            cand_x1 = cand_x1[order]
            cand_y1 = cand_y1[order]
            cand_x2 = cand_x2[order]
            cand_y2 = cand_y2[order]

            # Greedy NMS
            keep = []
            suppressed = np.zeros(len(cand_scores), dtype=bool)
            areas = (cand_x2 - cand_x1) * (cand_y2 - cand_y1)

            for i in range(len(cand_scores)):
                if suppressed[i]:
                    continue
                keep.append(i)

                if len(keep) >= max_output:
                    break

                # Compute IoU with remaining boxes
                inter_x1 = np.maximum(cand_x1[i], cand_x1[i+1:])
                inter_y1 = np.maximum(cand_y1[i], cand_y1[i+1:])
                inter_x2 = np.minimum(cand_x2[i], cand_x2[i+1:])
                inter_y2 = np.minimum(cand_y2[i], cand_y2[i+1:])

                inter_w = np.maximum(0.0, inter_x2 - inter_x1)
                inter_h = np.maximum(0.0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h

                union_area = areas[i] + areas[i+1:] - inter_area
                iou = np.where(union_area > 0, inter_area / union_area, 0.0)

                suppressed[i+1:] = suppressed[i+1:] | (iou > iou_threshold)

            for idx in keep:
                detections.append((
                    cand_scores[idx],
                    cls,
                    cand_x1[idx],
                    cand_y1[idx],
                    cand_x2[idx],
                    cand_y2[idx],
                ))

        # Sort all detections by score and take top max_output
        detections.sort(key=lambda x: x[0], reverse=True)
        detections = detections[:max_output]

        count = len(detections)
        counts_out[b] = count
        for i, det in enumerate(detections):
            scores_out[b, i] = det[0]
            classes_out[b, i] = det[1]
            boxes_out[b, i, 0] = det[2]
            boxes_out[b, i, 1] = det[3]
            boxes_out[b, i, 2] = det[4]
            boxes_out[b, i, 3] = det[5]

    return boxes_out, scores_out, classes_out, counts_out


# ============================================================================
# TRT Engine helpers
# ============================================================================

def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    return runtime.deserialize_cuda_engine(engine_data)


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = engine.get_binding_dtype(i)
        shape = engine.get_binding_shape(i)
        size = trt.volume(shape) * dtype.itemsize
        d_buf = cuda.mem_alloc(size)
        if dtype == trt.float32:
            h_buf = cuda.pagelocked_empty(trt.volume(shape), dtype=np.float32)
        elif dtype == trt.int32:
            h_buf = cuda.pagelocked_empty(trt.volume(shape), dtype=np.int32)
        else:
            h_buf = cuda.pagelocked_empty(trt.volume(shape), dtype=np.float32)
        bindings.append(int(d_buf))
        if engine.binding_is_input(i):
            inputs.append({"name": name, "shape": shape, "dtype": dtype,
                          "host": h_buf, "device": d_buf, "index": i})
        else:
            outputs.append({"name": name, "shape": shape, "dtype": dtype,
                           "host": h_buf, "device": d_buf, "index": i})
    return inputs, outputs, bindings, stream


def infer_raw(context, bindings, inputs, outputs, stream):
    for inp in inputs:
        cuda.memcpy_htod_async(inp["device"], inp["host"], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    for out in outputs:
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
    stream.synchronize()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TRT raw inference + CPU NMS demo")
    parser.add_argument("--raw-engine", default="TwoStreamMPF_raw.engine",
                        help="Raw TRT engine (output [1,9,8400])")
    parser.add_argument("--plugin-engine", default="TwoStreamMPF_nms.engine",
                        help="TRT engine with GPU NMS plugin")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--max-output", type=int, default=300)
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Check files
    # -----------------------------------------------------------------
    for fpath in [args.raw_engine]:
        if not os.path.isfile(fpath):
            print(f"ERROR: Engine file not found: {fpath}")
            print("Build it with:")
            print("  trtexec --onnx=TwoStreamMPF.onnx --saveEngine=TwoStreamMPF_raw.engine --fp16")
            sys.exit(1)

    print("=" * 70)
    print("  TRT Raw Inference + CPU Traditional NMS")
    print("=" * 70)
    print(f"  Raw engine:      {args.raw_engine}")
    print(f"  Score threshold: {args.score_threshold}")
    print(f"  IoU threshold:   {args.iou_threshold}")
    print(f"  Max output:      {args.max_output}")
    print(f"  Iterations:      {args.iterations}")
    print(f"  Warmup:          {args.warmup}")

    # -----------------------------------------------------------------
    # Load raw engine
    # -----------------------------------------------------------------
    print("\n[1] Loading raw TRT engine ...")
    raw_engine = load_engine(args.raw_engine)

    print("  Engine bindings:")
    for i in range(raw_engine.num_bindings):
        name = raw_engine.get_binding_name(i)
        shape = raw_engine.get_binding_shape(i)
        io = "IN " if raw_engine.binding_is_input(i) else "OUT"
        print(f"    [{i}] {io} {name}: shape={shape}")

    inputs, outputs, bindings, stream = allocate_buffers(raw_engine)
    context = raw_engine.create_execution_context()

    # -----------------------------------------------------------------
    # Prepare input
    # -----------------------------------------------------------------
    print("\n[2] Preparing input (random [1, 6, 640, 640]) ...")
    np.random.seed(42)
    for inp in inputs:
        inp["host"] = np.random.randn(*inp["shape"]).astype(np.float32)
        context.set_binding_shape(inp["index"], inp["shape"])
        print(f"  {inp['name']}: shape={inp['shape']}, range=[{inp['host'].min():.3f}, {inp['host'].max():.3f}]")

    # -----------------------------------------------------------------
    # Warmup
    # -----------------------------------------------------------------
    print(f"\n[3] Warming up ({args.warmup} iterations) ...")
    for _ in range(args.warmup):
        infer_raw(context, bindings, inputs, outputs, stream)

    # -----------------------------------------------------------------
    # Benchmark: end-to-end (TRT infer + CPU NMS)
    # -----------------------------------------------------------------
    print(f"[4] Benchmarking ({args.iterations} iterations) ...")

    # First, get raw output shape
    raw_shape = tuple(outputs[0]["shape"])  # e.g. (1, 9, 8400)

    trt_times = []
    nms_times = []
    total_times = []

    for _ in range(args.iterations):
        # Time TRT inference only
        t0 = time.perf_counter()
        infer_raw(context, bindings, inputs, outputs, stream)
        t1 = time.perf_counter()

        # Copy raw output to numpy
        raw_out = outputs[0]["host"].reshape(raw_shape).copy()

        # Time CPU NMS only
        t2 = time.perf_counter()
        boxes, scores, classes, counts = cpu_nms(
            raw_out,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            max_output=args.max_output,
        )
        t3 = time.perf_counter()

        trt_times.append((t1 - t0) * 1000.0)
        nms_times.append((t3 - t2) * 1000.0)
        total_times.append((t3 - t0) * 1000.0)

    trt_times = np.array(trt_times)
    nms_times = np.array(nms_times)
    total_times = np.array(total_times)

    print("\n" + "=" * 70)
    print("  Pipeline Timing (CPU NMS approach)")
    print("=" * 70)
    print(f"  TRT inference:  mean={trt_times.mean():.3f} ms  "
          f"min={trt_times.min():.3f}  max={trt_times.max():.3f}  "
          f"std={trt_times.std():.3f}")
    print(f"  CPU NMS:        mean={nms_times.mean():.3f} ms  "
          f"min={nms_times.min():.3f}  max={nms_times.max():.3f}  "
          f"std={nms_times.std():.3f}")
    print(f"  TOTAL end2end:  mean={total_times.mean():.3f} ms  "
          f"min={total_times.min():.3f}  max={total_times.max():.3f}  "
          f"std={total_times.std():.3f}")
    print(f"  Throughput:     {1000.0 / total_times.mean():.1f} FPS")

    # Show output summary
    print("\n" + "=" * 70)
    print("  Output Summary (CPU NMS)")
    print("=" * 70)
    print(f"  counts[0]: {counts[0]}")
    if counts[0] > 0:
        print(f"  Top-5 detections:")
        print(f"  {'#':<4} {'Class':<6} {'Score':<10} {'x1':<10} {'y1':<10} {'x2':<10} {'y2':<10}")
        for i in range(min(int(counts[0]), 10)):
            print(f"  {i:<4} {int(classes[0, i]):<6} "
                  f"{scores[0, i]:.6f}   "
                  f"{boxes[0, i, 0]:.4f}   {boxes[0, i, 1]:.4f}   "
                  f"{boxes[0, i, 2]:.4f}   {boxes[0, i, 3]:.4f}")

    # -----------------------------------------------------------------
    # Compare with GPU NMS plugin (if available)
    # -----------------------------------------------------------------
    plugin_so = "nmsFullPlugin.so"
    if os.path.isfile(args.plugin_engine) and os.path.isfile(plugin_so):
        print("\n" + "=" * 70)
        print("  Comparison: GPU NMS Plugin Engine")
        print("=" * 70)

        # Load plugin .so before deserializing engine.
        # The REGISTER_TENSORRT_PLUGIN macro's static constructor registers
        # NmsFullPluginCreator into TRT's global plugin registry upon library load.
        print(f"  Loading plugin: {plugin_so}")
        ctypes.cdll.LoadLibrary(os.path.abspath(plugin_so))

        plugin_engine = load_engine(args.plugin_engine)
        if plugin_engine is None:
            print("  ERROR: Failed to load plugin engine (plugin not registered?)")
        else:
            pin, pout, pb, pstream = allocate_buffers(plugin_engine)
            pctx = plugin_engine.create_execution_context()

            # Use same input data
            for inp in pin:
                inp["host"] = np.random.randn(*inp["shape"]).astype(np.float32)
                pctx.set_binding_shape(inp["index"], inp["shape"])

            # Warmup
            for _ in range(args.warmup):
                infer_raw(pctx, pb, pin, pout, pstream)

            # Benchmark GPU plugin
            gpu_times = []
            for _ in range(args.iterations):
                t0 = time.perf_counter()
                infer_raw(pctx, pb, pin, pout, pstream)
                t1 = time.perf_counter()
                gpu_times.append((t1 - t0) * 1000.0)

            gpu_times = np.array(gpu_times)
            print(f"  GPU Plugin (end2end): mean={gpu_times.mean():.3f} ms  "
                  f"min={gpu_times.min():.3f}  max={gpu_times.max():.3f}  "
                  f"std={gpu_times.std():.3f}")
            print(f"  GPU Plugin FPS:      {1000.0 / gpu_times.mean():.1f}")

            print(f"\n  Speed comparison:")
            ratio = total_times.mean() / gpu_times.mean()
            print(f"  Pipeline (CPU NMS total) / GPU Plugin total = {ratio:.2f}x")
            ratio_trt = trt_times.mean() / gpu_times.mean()
            print(f"  TRT-only                / GPU Plugin total = {ratio_trt:.2f}x  "
                  f"(GPU Plugin NMS adds ~{gpu_times.mean() - trt_times.mean():.2f} ms)")
            print(f"  CPU NMS overhead: {nms_times.mean():.3f} ms "
                  f"(vs GPU Plugin NMS overhead: {gpu_times.mean() - trt_times.mean():.2f} ms)")

    elif not os.path.isfile(args.plugin_engine):
        print(f"\n  (GPU plugin engine {args.plugin_engine} not found — skip comparison)")
    else:
        print(f"\n  ({plugin_so} not found — skip GPU plugin comparison)")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
