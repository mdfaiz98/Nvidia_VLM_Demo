# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Vehicle Detection Module

Wraps a YOLO detector (YOLOv8n by default, or YOLO26n) on the compute unit
picked in the /traffic settings: a TensorRT engine - built from the
Ultralytics ONNX export by scripts/build_traffic_engines.sh - on one of the
Jetson AGX Orin's DLA cores (the Orin's counterpart to the IQ9's Hexagon NPU)
or on its GPU, or the ONNX model itself on the CPU via ONNX Runtime.
Same interface as the Qualcomm IQ9 demo's detector (Hexagon NPU via the QNN
TFLite delegate), so tracker.py and detection_processor.py are shared as-is.
Detection is cheap enough to run continuously; the VLM is the actual
bottleneck, so this module only feeds it cropped regions on demand rather
than every frame.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import tensorrt as trt
from cuda import cudart

logger = logging.getLogger(__name__)

# Default class filter for the traffic-monitoring demo: only report COCO
# classes relevant to a road/intersection scene, so the overlay/caption
# targets stay meaningful instead of flagging every pedestrian, bench, etc.
DEFAULT_TARGET_CLASSES = {"car", "motorcycle", "bus", "truck"}

# Ultralytics letterbox fill value
LETTERBOX_PAD = 114


def _cuda_check(result):
    """cuda-python calls return (err, *values); raise on error, return the values."""
    err, *values = result
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {cudart.cudaGetErrorString(err)[1].decode()}")
    return values[0] if len(values) == 1 else values


@dataclass
class Detection:
    label: str
    score: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in original frame pixel coords


class YoloDetector:
    """YOLO object detector on a Jetson DLA core or GPU (TensorRT .engine) or
    on the CPU (ONNX Runtime, .onnx).

    YOLO26's Ultralytics export is end-to-end (NMS-free): one output of
    [1, N, 6] rows (x1, y1, x2, y2 in 640x640 model-input space, score,
    class index). YOLOv8's classic [1, 4 + classes, anchors] head is decoded too.
    Per-class NMS is still applied, then a class-agnostic dedupe, so the
    tracker sees one box per physical vehicle either way.
    """

    INPUT_SIZE = 640  # fixed at export time

    # Cumulative inference time on the DLA, process-wide. The system monitor
    # turns it into a DLA busy % for /traffic's telemetry - jtop has no DLA
    # utilization counter. Only touched from the detection thread.
    dla_busy_seconds: float = 0.0

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        conf_threshold: float = 0.45,
        nms_iou_threshold: float = 0.7,
        target_classes: Optional[Set[str]] = None,
    ):
        """
        Args:
            model_path: Path to a TensorRT .engine (built for this board and
                TensorRT version - engines aren't portable; "dla<N>" in the
                file name marks a DLA build), or an .onnx model to run on
                the CPU.
            labels_path: Path to a newline-separated COCO class labels file
                (index order must match the model's class output).
            conf_threshold: Minimum class score to keep a candidate box.
            nms_iou_threshold: IoU threshold for suppressing overlapping
                boxes of the same class.
            target_classes: Set of label strings to report; other classes
                are detected but filtered out. Defaults to
                DEFAULT_TARGET_CLASSES (vehicle classes only).
        """
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.target_classes = (
            target_classes if target_classes is not None else DEFAULT_TARGET_CLASSES
        )

        with open(labels_path) as f:
            self.labels = [line.strip() for line in f if line.strip()]

        start = time.perf_counter()
        if model_path.endswith(".onnx"):
            self.backend = "cpu"
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self._ort_input = self.session.get_inputs()[0].name
            input_shape = tuple(self.session.get_inputs()[0].shape)
        else:
            self.backend = "dla" if "dla" in Path(model_path).stem.lower() else "gpu"
            self._load_engine(model_path)
            input_shape = self._inputs[0]["host"].shape

        expected_shape = (1, 3, self.INPUT_SIZE, self.INPUT_SIZE)
        if input_shape != expected_shape:
            raise ValueError(
                f"Unexpected model input shape {input_shape}, expected {expected_shape} "
                "(NCHW, as Ultralytics exports it at imgsz=640)."
            )

        logger.info(
            f"YoloDetector ready: {Path(model_path).name} on {self.backend.upper()} "
            f"(loaded in {time.perf_counter() - start:.2f}s), {len(self.labels)} classes, "
            f"filtering to {sorted(self.target_classes)}"
        )

    def _load_engine(self, model_path: str) -> None:
        """Deserialize a TensorRT engine and bind one host + device buffer per I/O tensor."""
        self._trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self._trt_logger, "")
        with open(model_path, "rb") as f:
            self.engine = trt.Runtime(self._trt_logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Could not load TensorRT engine {model_path}")
        self.context = self.engine.create_execution_context()
        self.stream = _cuda_check(cudart.cudaStreamCreate())

        # One host + device buffer per I/O tensor, bound once
        self._inputs, self._outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = np.zeros(shape, dtype=dtype)
            device = _cuda_check(cudart.cudaMalloc(host.nbytes))
            self.context.set_tensor_address(name, device)
            io = (self._inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self._outputs)
            io.append({"name": name, "host": host, "device": device})

    def _letterbox(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """Resize frame to fit within INPUT_SIZE x INPUT_SIZE preserving aspect
        ratio, centered on a grey canvas - the Ultralytics preprocessing, so
        decoded box coordinates map back to the original frame correctly."""
        h, w = frame.shape[:2]
        scale = min(self.INPUT_SIZE / w, self.INPUT_SIZE / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (self.INPUT_SIZE - new_w) // 2
        pad_y = (self.INPUT_SIZE - new_h) // 2
        canvas = np.full((self.INPUT_SIZE, self.INPUT_SIZE, 3), LETTERBOX_PAD, dtype=np.uint8)
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
        return canvas, scale, pad_x, pad_y

    def _infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run the model on one NCHW float32 tensor; returns the first output."""
        if self.backend == "cpu":
            return self.session.run(None, {self._ort_input: input_tensor})[0]
        # TensorRT: copy in, run the engine, copy out
        inp, out = self._inputs[0], self._outputs[0]
        np.copyto(inp["host"], input_tensor.astype(inp["host"].dtype, copy=False))
        kind_h2d = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        kind_d2h = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        _cuda_check(cudart.cudaMemcpyAsync(inp["device"], inp["host"].ctypes.data, inp["host"].nbytes, kind_h2d, self.stream))
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT inference failed")
        _cuda_check(cudart.cudaMemcpyAsync(out["host"].ctypes.data, out["device"], out["host"].nbytes, kind_d2h, self.stream))
        _cuda_check(cudart.cudaStreamSynchronize(self.stream))
        return out["host"]

    def _decode(self, output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """-> boxes [N, 4] xyxy in model-input space, scores [N], class_idx [N]"""
        out = output[0].astype(np.float32)
        if out.shape[-1] == 6:  # end-to-end head: x1, y1, x2, y2, score, class
            return out[:, :4], out[:, 4], out[:, 5].astype(np.int64)
        # classic head: [4 + classes, anchors] of cx, cy, w, h, class scores
        out = out.T
        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        class_scores = out[:, 4:]
        class_idx = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(len(class_idx)), class_idx]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        return boxes, scores, class_idx

    def detect(self, rgb_frame: np.ndarray) -> List[Detection]:
        """Run detection on a single RGB frame (H, W, 3), uint8.

        Returns detections filtered to self.target_classes, in original
        frame pixel coordinates, after confidence thresholding and NMS.
        """
        orig_h, orig_w = rgb_frame.shape[:2]
        canvas, scale, pad_x, pad_y = self._letterbox(rgb_frame)

        input_tensor = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
        t0 = time.perf_counter()
        output = self._infer(input_tensor)
        if self.backend == "dla":
            YoloDetector.dla_busy_seconds += time.perf_counter() - t0

        boxes, scores, class_idx = self._decode(output)

        keep = scores >= self.conf_threshold
        if not np.any(keep):
            return []
        boxes, scores, class_idx = boxes[keep], scores[keep], class_idx[keep]

        # cv2.dnn.NMSBoxes wants [x, y, w, h] rects, not xyxy
        nms_boxes = np.stack(
            [boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]],
            axis=1,
        )

        detections: List[Detection] = []
        for cls in np.unique(class_idx):
            label = self.labels[int(cls)] if int(cls) < len(self.labels) else str(int(cls))
            if label not in self.target_classes:
                continue

            cls_mask = class_idx == cls
            idxs = cv2.dnn.NMSBoxes(
                nms_boxes[cls_mask].tolist(),
                scores[cls_mask].tolist(),
                self.conf_threshold,
                self.nms_iou_threshold,
            )
            if len(idxs) == 0:
                continue

            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            for i in np.array(idxs).flatten():
                x1, y1, x2, y2 = cls_boxes[i]
                # Undo letterbox: remove the pad offset, then rescale to the original frame
                x1 = (x1 - pad_x) / scale
                y1 = (y1 - pad_y) / scale
                x2 = (x2 - pad_x) / scale
                y2 = (y2 - pad_y) / scale
                x1 = max(0, min(orig_w, x1))
                y1 = max(0, min(orig_h, y1))
                x2 = max(0, min(orig_w, x2))
                y2 = max(0, min(orig_h, y2))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue  # box lies (almost) entirely outside the frame
                detections.append(
                    Detection(
                        label=label,
                        score=float(cls_scores[i]),
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                    )
                )

        return self._dedupe_cross_class(detections)

    def _dedupe_cross_class(
        self, detections: List[Detection], iou_threshold: float = 0.5
    ) -> List[Detection]:
        """cv2.dnn.NMSBoxes above only suppresses overlapping boxes within
        the SAME class, so a vehicle the model is genuinely torn on (e.g.
        borderline car/truck) can survive NMS as two overlapping boxes with
        different labels on the same physical object. This final
        class-agnostic pass keeps only the single highest-confidence box per
        overlapping region, regardless of label, which is what a downstream
        tracker needs to treat one physical object as one track."""
        ordered = sorted(detections, key=lambda d: d.score, reverse=True)
        kept: List[Detection] = []
        for d in ordered:
            if all(self._iou(d.bbox, k.bbox) < iou_threshold for k in kept):
                kept.append(d)
        return kept

    @staticmethod
    def _iou(box_a, box_b) -> float:
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / float(area_a + area_b - inter)


if __name__ == "__main__":
    # Standalone smoke test: run against an image, confirm the engine loads,
    # and print detections + latency.
    #   .venv/bin/python -m live_vlm_webui.detector IMAGE [--model ...] [--all-classes]
    import argparse

    models_dir = Path(__file__).resolve().parents[2] / "models"
    parser = argparse.ArgumentParser(description="Standalone YoloDetector smoke test")
    parser.add_argument("image", help="Path to a test image")
    parser.add_argument("--model", default=str(models_dir / "yolov8n_dla0_fp16.engine"))
    parser.add_argument("--labels", default=str(models_dir / "coco_labels.txt"))
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Report all COCO classes instead of just the vehicle-class filter",
    )
    parser.add_argument("--conf", type=float, default=0.45)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    target_classes = None if args.all_classes else DEFAULT_TARGET_CLASSES
    detector = YoloDetector(
        model_path=args.model,
        labels_path=args.labels,
        conf_threshold=args.conf,
        target_classes=target_classes if target_classes is not None else set(),
    )
    if args.all_classes:
        detector.target_classes = set(detector.labels)

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # First call includes one-time warmup; run more for a steady-state latency.
    for i in range(5):
        start = time.perf_counter()
        dets = detector.detect(rgb)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"Run {i + 1}: {elapsed_ms:.1f} ms, {len(dets)} detections")

    print(f"\n{args.image} -> {len(dets)} detections (backend={detector.backend}):")
    for d in dets:
        print(f"  {d.label:12s} score={d.score:.2f} bbox={d.bbox}")
