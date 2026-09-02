"""ONNX YOLOv8/v9 bbox detection for the 'adetailer' inpaint-mask backend.

Vendors the letterbox/NMS/rescale pipeline ultralytics normally provides, so
the face/hand detect-head models (extras/inpaint_mask.py's 'adetailer'
dispatch) run on plain onnxruntime with no ultralytics dependency, mirroring
extras/wd14tagger.py's InferenceSession usage.

The numpy-only helpers (compute_letterbox_params, letterbox_image,
postprocess_predictions) are unit-testable without a model file. The
letterbox tensor/ratio/pads contract and postprocess_predictions'
num_extra_coeffs / kept_indices parameters are FWDF-199's extension points
for decoding the -seg models' mask-coefficient columns and un-letterboxing
their prototype masks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from onnxruntime import InferenceSession

# ultralytics predictor default. Not exposed via ADetailerOptions -- keeps
# the option surface FWDF-198 exposes to the UI minimal.
NMS_IOU_THRESHOLD = 0.7

LETTERBOX_PAD_COLOR = (114, 114, 114)

# Keyed by resolved model path, mirroring extras/wd14tagger.py's
# global_model module-level cache (:23,44-48).
_session_cache: dict[str, InferenceSession] = {}


@dataclass(slots=True)
class DetectionResult:
    """Detections from one detect_bboxes() call, in source-image coordinates.

    masks is unused by this ticket's bbox-only scope; it exists so FWDF-199
    can attach per-detection segmentation masks without changing this shape.
    """
    boxes: np.ndarray  # (N, 4) xyxy, source coords
    scores: np.ndarray  # (N,)
    masks: np.ndarray | None = field(default=None)


def compute_letterbox_params(src_h: int, src_w: int, target: int = 640) -> tuple[float, float, float]:
    """Aspect-preserving fit of an src_h x src_w image into a target x target
    square. Returns (ratio, pad_x, pad_y) where pad_x/pad_y are the padding
    applied to each side (half of the total padding) once the resized image
    is centered on the square canvas."""
    ratio = min(target / src_h, target / src_w)
    new_w, new_h = round(src_w * ratio), round(src_h * ratio)
    pad_x = (target - new_w) / 2
    pad_y = (target - new_h) / 2
    return ratio, pad_x, pad_y


def letterbox_image(image_rgb: np.ndarray, target: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize+pad image_rgb (HWC uint8, RGB) onto a target x target canvas
    and return (tensor, ratio, pads). tensor is float32 NCHW (1, 3, target,
    target) scaled to [0, 1] (YOLO export expects RGB, 0-1). ratio/pads are
    part of the public contract -- FWDF-199 reuses them to un-letterbox
    segmentation masks."""
    src_h, src_w = image_rgb.shape[0], image_rgb.shape[1]
    ratio, pad_x, pad_y = compute_letterbox_params(src_h, src_w, target)
    new_w, new_h = round(src_w * ratio), round(src_h * ratio)

    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target, target, 3), LETTERBOX_PAD_COLOR, dtype=np.uint8)
    top, left = round(pad_y), round(pad_x)
    canvas[top:top + new_h, left:left + new_w] = resized

    tensor = canvas.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]  # (1, 3, target, target)
    return tensor, ratio, (pad_x, pad_y)


def _empty_postprocess_result() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
    )


def postprocess_predictions(
    output0: np.ndarray,
    ratio: float,
    pads: tuple[float, float],
    src_shape: tuple[int, int],
    confidence: float,
    iou: float,
    num_extra_coeffs: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a raw YOLOv8/v9 detect head output0 (1, 4+nc+num_extra_coeffs,
    num_anchors) into (boxes_xyxy, scores, kept_indices), sorted by score
    descending, in src_shape (src_h, src_w) coordinates.

    kept_indices index the rows of output0[0].T (num_anchors, C) that
    survived the confidence filter and NMS -- FWDF-199 gathers its 32
    trailing mask-coefficient columns with the same indices, so the keep-set
    applies to full per-anchor rows, not just boxes/scores.
    """
    pred = output0[0].T  # (num_anchors, C)
    num_classes = pred.shape[1] - 4 - num_extra_coeffs

    boxes_cxcywh = pred[:, :4]
    class_scores = pred[:, 4:4 + num_classes]
    scores = class_scores.max(axis=1)

    confidence_mask = scores > confidence
    if not np.any(confidence_mask):
        return _empty_postprocess_result()

    kept_row_indices = np.nonzero(confidence_mask)[0]
    filtered_boxes_cxcywh = boxes_cxcywh[confidence_mask]
    filtered_scores = scores[confidence_mask]

    cx, cy, w, h = (filtered_boxes_cxcywh[:, i] for i in range(4))
    x = cx - w / 2
    y = cy - h / 2
    boxes_xywh = np.stack([x, y, w, h], axis=1)

    nms_indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), filtered_scores.tolist(), confidence, iou)
    nms_indices = np.array(nms_indices).flatten().astype(np.int64)
    if nms_indices.size == 0:
        return _empty_postprocess_result()

    kept_indices = kept_row_indices[nms_indices]
    boxes_xywh = boxes_xywh[nms_indices]
    scores = filtered_scores[nms_indices]

    pad_x, pad_y = pads
    x1 = (boxes_xywh[:, 0] - pad_x) / ratio
    y1 = (boxes_xywh[:, 1] - pad_y) / ratio
    x2 = (boxes_xywh[:, 0] + boxes_xywh[:, 2] - pad_x) / ratio
    y2 = (boxes_xywh[:, 1] + boxes_xywh[:, 3] - pad_y) / ratio

    src_h, src_w = src_shape
    x1 = np.clip(x1, 0, src_w)
    x2 = np.clip(x2, 0, src_w)
    y1 = np.clip(y1, 0, src_h)
    y2 = np.clip(y2, 0, src_h)

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    order = np.argsort(-scores)
    return boxes_xyxy[order], scores[order].astype(np.float32), kept_indices[order]


def _get_session(model_path: str) -> InferenceSession:
    session = _session_cache.get(model_path)
    if session is None:
        session = InferenceSession(model_path, providers=['CPUExecutionProvider'])
        _session_cache[model_path] = session
    return session


def detect_bboxes(image_rgb: np.ndarray, model_path: str, confidence: float) -> DetectionResult:
    """Run a bbox-only adetailer ONNX model over image_rgb (HWC uint8, RGB)
    and return detections in source-image coordinates.

    Model capability is derived from the loaded session, not the filename:
    a single output is a detect head (bbox-only, handled here); a second
    output is the seg-proto tensor, which is FWDF-199's scope.
    """
    session = _get_session(model_path)
    if len(session.get_outputs()) != 1:
        raise ValueError("segmentation adetailer models require FWDF-199")

    src_h, src_w = image_rgb.shape[0], image_rgb.shape[1]
    tensor, ratio, pads = letterbox_image(image_rgb)

    input_name = session.get_inputs()[0].name
    output0 = session.run(None, {input_name: tensor})[0]

    boxes, scores, _kept_indices = postprocess_predictions(
        output0, ratio, pads, (src_h, src_w), confidence, NMS_IOU_THRESHOLD
    )
    return DetectionResult(boxes=boxes, scores=scores)
