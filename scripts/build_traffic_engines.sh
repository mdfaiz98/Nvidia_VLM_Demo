#!/bin/bash
# Builds the /traffic cascade demo's detection models into models/ (next to
# coco_labels.txt, where the /traffic settings drawer lists them), one per
# compute unit for each Ultralytics ONNX export given:
#   <model>_dla0_fp16.engine  TensorRT on DLA core 0 - the Orin's counterpart to
#                             the IQ9's Hexagon NPU ("NPU" in the drawer);
#                             layers the DLA can't run fall back to the GPU
#   <model>_gpu_fp16.engine   TensorRT on the GPU
#   <model>.onnx              the export itself, run on the CPU by ONNX Runtime
# Engines are tied to this board's TensorRT version, so rebuild after a
# JetPack upgrade.
#
# ONNX export (once, in any Python env with ultralytics):
#   yolo export model=yolov8n.pt format=onnx imgsz=640 opset=17
#   yolo export model=yolo26n.pt format=onnx imgsz=640 opset=17
#
# Usage: scripts/build_traffic_engines.sh path/to/yolov8n.onnx [path/to/yolo26n.onnx ...]

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
[ $# -ge 1 ] || { echo "usage: $0 path/to/model.onnx [...]" >&2; exit 1; }
TRTEXEC=/usr/src/tensorrt/bin/trtexec
MODELS="$REPO_DIR/models"
mkdir -p "$MODELS"

for ONNX in "$@"; do
    NAME="$(basename "$ONNX" .onnx)"

    echo "== $NAME: DLA core 0, FP16 (GPU fallback for unsupported layers) =="
    "$TRTEXEC" --onnx="$ONNX" --saveEngine="$MODELS/${NAME}_dla0_fp16.engine" \
        --fp16 --useDLACore=0 --allowGPUFallback --skipInference 2>&1 | tail -n 1

    echo "== $NAME: GPU, FP16 =="
    "$TRTEXEC" --onnx="$ONNX" --saveEngine="$MODELS/${NAME}_gpu_fp16.engine" \
        --fp16 --skipInference 2>&1 | tail -n 1

    echo "== $NAME: CPU (ONNX Runtime) =="
    cp "$ONNX" "$MODELS/$NAME.onnx"
done

ls -l "$MODELS"
