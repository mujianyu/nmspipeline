#!/usr/bin/env python3
"""
insert_nms.py
Inserts a custom NmsFull node after the original output0 of TwoStreamMPF.onnx.
The NmsFull node uses the "NmsFull" custom op that will be mapped to the
TensorRT NmsFullPlugin at engine build time.

Input:  TwoStreamMPF.onnx   output: "output0" [1, 9, 8400]
Output: TwoStreamMPF_nms.onnx  with NmsFull node consuming output0, producing:
        boxes  [batch, max_output_boxes, 4]
        scores [batch, max_output_boxes]
        classes[batch, max_output_boxes]
        counts [batch]
"""

import onnx
from onnx import helper, TensorProto
import sys
import os

ONNX_INPUT  = "TwoStreamMPF.onnx"
ONNX_OUTPUT = "TwoStreamMPF_nms.onnx"

# NmsFull node attributes
SCORE_THRESHOLD  = 0.25
IOU_THRESHOLD    = 0.45
MAX_OUTPUT_BOXES = 300
PER_BOX_STRIDE   = 9    # 4 bbox + 5 class scores
NUM_CLASSES      = 5

def main():
    print(f"Loading {ONNX_INPUT} ...")
    model = onnx.load(ONNX_INPUT)

    graph = model.graph

    # Find the original output node and its output tensor name
    original_output_names = [o.name for o in graph.output]
    print(f"Original outputs: {original_output_names}")

    if len(original_output_names) != 1:
        print(f"ERROR: Expected 1 output, got {len(original_output_names)}")
        sys.exit(1)

    original_output_name = original_output_names[0]

    # Find the node that produces the original output
    producer_node = None
    for node in graph.node:
        if original_output_name in node.output:
            producer_node = node
            break

    if producer_node is None:
        print(f"ERROR: Could not find node producing '{original_output_name}'")
        sys.exit(1)

    print(f"Producer of '{original_output_name}': {producer_node.op_type} ({producer_node.name})")

    # Output names for the NmsFull node
    boxes_name   = "nms_boxes"
    scores_name  = "nms_scores"
    classes_name = "nms_classes"
    counts_name  = "nms_counts"

    # Create the NmsFull custom node
    # It takes the original output as input and produces 4 outputs
    nms_node = helper.make_node(
        op_type="NmsFull",
        inputs=[original_output_name],
        outputs=[boxes_name, scores_name, classes_name, counts_name],
        name="nms_full_node",
        domain="",  # custom domain (empty = default)
        score_threshold=SCORE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        max_output_boxes=MAX_OUTPUT_BOXES,
        per_box_stride=PER_BOX_STRIDE,
        num_classes=NUM_CLASSES
    )

    graph.node.append(nms_node)
    print(f"Inserted NmsFull node: {nms_node.name}")

    # Add value_info for intermediate tensor (original output becomes intermediate)
    # output0 shape: [batch, 9, 8400]
    vi = helper.make_tensor_value_info(
        original_output_name,
        TensorProto.FLOAT,
        [1, PER_BOX_STRIDE, 8400]
    )
    graph.value_info.append(vi)
    print(f"Added value_info for intermediate: {original_output_name}")

    # Remove original outputs and add new outputs
    del graph.output[:]

    # New output 0: boxes [batch, max_output_boxes, 4]
    graph.output.append(helper.make_tensor_value_info(
        boxes_name, TensorProto.FLOAT, [1, MAX_OUTPUT_BOXES, 4]))

    # New output 1: scores [batch, max_output_boxes]
    graph.output.append(helper.make_tensor_value_info(
        scores_name, TensorProto.FLOAT, [1, MAX_OUTPUT_BOXES]))

    # New output 2: classes [batch, max_output_boxes]
    graph.output.append(helper.make_tensor_value_info(
        classes_name, TensorProto.INT32, [1, MAX_OUTPUT_BOXES]))

    # New output 3: counts [batch]
    graph.output.append(helper.make_tensor_value_info(
        counts_name, TensorProto.INT32, [1]))

    print("New outputs:")
    for o in graph.output:
        print(f"  {o.name}: {[d.dim_value for d in o.type.tensor_type.shape.dim]}")

    # Save
    onnx.save(model, ONNX_OUTPUT)
    print(f"\nSaved modified ONNX to: {ONNX_OUTPUT}")

    # Verify
    verify = onnx.load(ONNX_OUTPUT)
    print(f"Verification - outputs: {[o.name for o in verify.graph.output]}")
    nms_nodes = [n for n in verify.graph.node if n.op_type == "NmsFull"]
    print(f"Verification - NmsFull nodes: {len(nms_nodes)}")
    print("Done.")

if __name__ == "__main__":
    main()
