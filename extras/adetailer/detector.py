"""ONNX YOLOv8/v9 bbox/segmentation detection for the 'adetailer' inpaint-mask
backend.

Vendors the letterbox/NMS/rescale pipeline ultralytics normally provides, so
the face/hand detect-head models and the person/deepfashion2 segmentation
models (extras/inpaint_mask.py's 'adetailer' dispatch) run on plain
onnxruntime with no ultralytics dependency, mirroring extras/wd14tagger.py's
InferenceSession usage. Mask-prototype decoding reimplements ultralytics
ops.process_mask(upsample=True) in numpy/cv2.

The numpy-only helpers (compute_letterbox_params, letterbox_image,
postprocess_predictions, decode_masks) are unit-testable without a model
file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import cv2
import numpy as np
from onnxruntime import InferenceSession

# ultralytics predictor default. Not exposed via ADetailerOptions -- keeps
# the option surface FWDF-198 exposes to the UI minimal.
NMS_IOU_THRESHOLD = 0.7

LETTERBOX_PAD_COLOR = (114, 114, 114)

# YOLOv8-seg mask-prototype coefficient count (output0's trailing columns,
# output1's channel count). Fixed by the ultralytics seg export format.
SEG_MASK_COEFFICIENTS = 32


class _CachedSession(NamedTuple):
    """An onnxruntime session plus its capability, resolved once at load
    time so detect_bboxes never re-inspects the session per call."""
    session: InferenceSession
    has_masks: bool


# Keyed by resolved model path, mirroring extras/wd14tagger.py's
# global_model module-level cache (:23,44-48).
_session_cache: dict[str, _CachedSession] = {}


@dataclass(slots=True)
class DetectionResult:
    """Detections from one detect_bboxes() call, in source-image coordinates.

    masks is populated only for segmentation-capable models (a 2-output
    session whose second output is the mask-prototype tensor); bbox-only
    models (a single-output session) leave it None.
    """
    boxes: np.ndarray  # (N, 4) xyxy, source coords
    scores: np.ndarray  # (N,)
    masks: np.ndarray | None = field(default=None)  # (N, src_h, src_w) bool


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
    reused by decode_masks to un-letterbox segmentation masks."""
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


def decode_masks(
    protos: np.ndarray,
    coefficients: np.ndarray,
    letterbox_boxes: np.ndarray,
    ratio: float,
    pads: tuple[float, float],
    src_shape: tuple[int, int],
) -> np.ndarray:
    """Decode YOLOv8-seg mask prototypes into per-detection boolean masks in
    source-image coordinates, reimplementing ultralytics
    ops.process_mask(upsample=True) without torch or ultralytics.

    protos: (32, proto_h, proto_w) mask-prototype tensor (output1[0]).
        proto_h/proto_w are 1/4 of the letterbox canvas size ultralytics
        seg exports use (e.g. 160 for a 640 canvas).
    coefficients: (N, 32) per-detection mask coefficients, gathered from
        output0's trailing columns with postprocess_predictions' kept_indices.
    letterbox_boxes: (N, 4) xyxy boxes in the padded letterbox canvas (the
        square onnxruntime input space), NOT the un-letterboxed source
        coordinates postprocess_predictions returns.
    ratio, pads: from letterbox_image(), used to strip padding and rescale
        the decoded masks back to src_shape.
    src_shape: (src_h, src_w) of the original source image.

    Returns (N, src_h, src_w) bool masks, one per row of `coefficients`.
    """
    num_detections = coefficients.shape[0]
    src_h, src_w = src_shape
    if num_detections == 0:
        return np.empty((0, src_h, src_w), dtype=bool)

    num_coeffs, proto_h, proto_w = protos.shape
    letterbox_target = proto_h * 4  # protos are 1/4 the letterbox canvas resolution
    proto_scale = proto_h / letterbox_target

    raw_masks = coefficients @ protos.reshape(num_coeffs, -1)  # (N, proto_h*proto_w)
    proto_masks = 1.0 / (1.0 + np.exp(-raw_masks))
    proto_masks = proto_masks.reshape(num_detections, proto_h, proto_w)

    new_h = round(src_h * ratio)
    new_w = round(src_w * ratio)
    pad_x, pad_y = pads
    left, top = round(pad_x), round(pad_y)

    masks = np.zeros((num_detections, src_h, src_w), dtype=bool)
    for i in range(num_detections):
        x1, y1, x2, y2 = letterbox_boxes[i]

        # Crop to the detection's box, scaled into proto space.
        px1 = int(np.clip(np.floor(x1 * proto_scale), 0, proto_w))
        py1 = int(np.clip(np.floor(y1 * proto_scale), 0, proto_h))
        px2 = int(np.clip(np.ceil(x2 * proto_scale), 0, proto_w))
        py2 = int(np.clip(np.ceil(y2 * proto_scale), 0, proto_h))

        cropped = np.zeros_like(proto_masks[i])
        cropped[py1:py2, px1:px2] = proto_masks[i, py1:py2, px1:px2]

        # Upsample to the full letterbox canvas, strip padding, resize to source.
        upsampled = cv2.resize(cropped, (letterbox_target, letterbox_target), interpolation=cv2.INTER_LINEAR)
        valid_region = upsampled[top:top + new_h, left:left + new_w]
        resized_to_source = cv2.resize(valid_region, (src_w, src_h), interpolation=cv2.INTER_LINEAR)

        masks[i] = resized_to_source > 0.5

    return masks


def _to_letterbox_space(boxes_xyxy_src: np.ndarray, ratio: float, pads: tuple[float, float]) -> np.ndarray:
    """Inverse of postprocess_predictions' un-letterbox step: maps
    source-image xyxy boxes back into the padded letterbox canvas, so
    decode_masks can crop mask prototypes in that same coordinate space."""
    pad_x, pad_y = pads
    letterbox_boxes = boxes_xyxy_src * ratio
    letterbox_boxes[:, [0, 2]] += pad_x
    letterbox_boxes[:, [1, 3]] += pad_y
    return letterbox_boxes


def _derive_has_masks(session: InferenceSession) -> bool:
    """A seg-capable session has a second output: the mask-prototype tensor,
    rank-4 with 32 channels. No filename parsing."""
    outputs = session.get_outputs()
    if len(outputs) != 2:
        return False
    proto_shape = outputs[1].shape
    return len(proto_shape) == 4 and proto_shape[1] == SEG_MASK_COEFFICIENTS


def _get_session(model_path: str) -> _CachedSession:
    cached = _session_cache.get(model_path)
    if cached is None:
        session = InferenceSession(model_path, providers=['CPUExecutionProvider'])
        cached = _CachedSession(session=session, has_masks=_derive_has_masks(session))
        _session_cache[model_path] = cached
    return cached


def detect_bboxes(image_rgb: np.ndarray, model_path: str, confidence: float) -> DetectionResult:
    """Run an adetailer ONNX model (bbox-only or segmentation) over
    image_rgb (HWC uint8, RGB) and return detections in source-image
    coordinates. Segmentation-capable models additionally populate
    DetectionResult.masks with per-detection boolean masks.
    """
    cached = _get_session(model_path)
    session = cached.session

    src_h, src_w = image_rgb.shape[0], image_rgb.shape[1]
    tensor, ratio, pads = letterbox_image(image_rgb)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    output0 = outputs[0]

    num_extra_coeffs = SEG_MASK_COEFFICIENTS if cached.has_masks else 0
    boxes, scores, kept_indices = postprocess_predictions(
        output0, ratio, pads, (src_h, src_w), confidence, NMS_IOU_THRESHOLD, num_extra_coeffs
    )

    if not cached.has_masks:
        return DetectionResult(boxes=boxes, scores=scores)

    if len(boxes) == 0:
        return DetectionResult(boxes=boxes, scores=scores, masks=np.empty((0, src_h, src_w), dtype=bool))

    protos = outputs[1][0]  # (32, proto_h, proto_w)
    coefficients = output0[0].T[kept_indices, -SEG_MASK_COEFFICIENTS:]
    letterbox_boxes = _to_letterbox_space(boxes.copy(), ratio, pads)
    masks = decode_masks(protos, coefficients, letterbox_boxes, ratio, pads, (src_h, src_w))
    return DetectionResult(boxes=boxes, scores=scores, masks=masks)
