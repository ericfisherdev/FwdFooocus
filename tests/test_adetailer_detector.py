"""Tests for extras.adetailer.detector's pure-numpy letterbox/NMS/rescale
helpers (FWDF-197). These run without any ONNX model file -- detect_bboxes'
session-dependent path is exercised indirectly through
tests/test_inpaint_mask_adetailer.py by monkeypatching detect_bboxes itself.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from extras.adetailer import detector  # noqa: E402


class TestComputeLetterboxParams:
    def test_square_source_has_ratio_one_and_zero_pad(self):
        ratio, pad_x, pad_y = detector.compute_letterbox_params(640, 640, target=640)

        assert ratio == 1.0
        assert pad_x == 0.0
        assert pad_y == 0.0

    def test_wide_source_pads_top_and_bottom(self):
        # 1280x640 (h x w) -> fit width, pad height
        ratio, pad_x, pad_y = detector.compute_letterbox_params(640, 1280, target=640)

        assert ratio == 0.5
        assert pad_x == 0.0
        assert pad_y == 160.0  # (640 - 640*0.5) / 2

    def test_tall_source_pads_left_and_right(self):
        # 1280x640 (h x w) -> fit height, pad width
        ratio, pad_x, pad_y = detector.compute_letterbox_params(1280, 640, target=640)

        assert ratio == 0.5
        assert pad_y == 0.0
        assert pad_x == 160.0


class TestLetterboxImage:
    def test_output_tensor_shape_dtype_and_range(self):
        image = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)

        tensor, ratio, pads = detector.letterbox_image(image, target=640)

        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0
        assert ratio == 1.0  # 640 wide already fits
        assert pads == (0.0, 80.0)

    def test_pad_regions_are_gray_114(self):
        image = np.full((320, 640, 3), 255, dtype=np.uint8)

        tensor, ratio, pads = detector.letterbox_image(image, target=640)

        # top padding rows should be 114/255 everywhere
        pad_rows = round(pads[1])
        expected = 114 / 255.0
        assert np.allclose(tensor[0, :, :pad_rows, :], expected, atol=1e-6)


class TestPostprocessPredictions:
    def _make_output0(self, boxes_cxcywh, scores):
        """Builds a synthetic (1, 5, N) detect-head tensor (nc=1, no extras)."""
        num_anchors = len(scores)
        output0 = np.zeros((1, 5, num_anchors), dtype=np.float32)
        for i, (cx, cy, w, h) in enumerate(boxes_cxcywh):
            output0[0, 0, i] = cx
            output0[0, 1, i] = cy
            output0[0, 2, i] = w
            output0[0, 3, i] = h
            output0[0, 4, i] = scores[i]
        return output0

    def test_filters_by_confidence(self):
        output0 = self._make_output0(
            boxes_cxcywh=[(320, 320, 100, 100), (100, 100, 50, 50)],
            scores=[0.9, 0.1],
        )

        boxes, scores, kept_indices = detector.postprocess_predictions(
            output0, ratio=1.0, pads=(0.0, 0.0), src_shape=(640, 640),
            confidence=0.3, iou=0.7,
        )

        assert len(boxes) == 1
        assert scores[0] == np.float32(0.9)
        assert kept_indices[0] == 0

    def test_empty_result_when_nothing_passes_confidence(self):
        output0 = self._make_output0(
            boxes_cxcywh=[(320, 320, 100, 100)],
            scores=[0.05],
        )

        boxes, scores, kept_indices = detector.postprocess_predictions(
            output0, ratio=1.0, pads=(0.0, 0.0), src_shape=(640, 640),
            confidence=0.3, iou=0.7,
        )

        assert boxes.shape == (0, 4)
        assert scores.shape == (0,)
        assert kept_indices.shape == (0,)

    def test_nms_suppresses_overlapping_boxes_keeping_higher_score(self):
        output0 = self._make_output0(
            boxes_cxcywh=[
                (320, 320, 100, 100),  # box A, high score
                (325, 325, 100, 100),  # box B, heavily overlaps A, lower score
            ],
            scores=[0.9, 0.8],
        )

        boxes, scores, kept_indices = detector.postprocess_predictions(
            output0, ratio=1.0, pads=(0.0, 0.0), src_shape=(640, 640),
            confidence=0.3, iou=0.5,
        )

        assert len(boxes) == 1
        assert scores[0] == np.float32(0.9)
        assert kept_indices[0] == 0

    def test_rescales_to_source_coordinates_undoing_letterbox(self):
        # A box centered at (320, 320) size 100x100 in a 640x640 letterboxed
        # frame that was produced from a 640x480 source (ratio=0.75, pad_y=80).
        output0 = self._make_output0(
            boxes_cxcywh=[(320, 320, 100, 100)],
            scores=[0.9],
        )
        ratio = 0.75
        pads = (0.0, 80.0)
        src_shape = (480, 640)  # (h, w)

        boxes, _, _ = detector.postprocess_predictions(
            output0, ratio=ratio, pads=pads, src_shape=src_shape,
            confidence=0.3, iou=0.7,
        )

        expected_x1 = (320 - 50 - pads[0]) / ratio
        expected_y1 = (320 - 50 - pads[1]) / ratio
        expected_x2 = (320 + 50 - pads[0]) / ratio
        expected_y2 = (320 + 50 - pads[1]) / ratio

        assert np.allclose(boxes[0], [expected_x1, expected_y1, expected_x2, expected_y2], atol=1e-3)

    def test_clips_to_source_bounds(self):
        # Box extends past the top-left corner once un-letterboxed.
        output0 = self._make_output0(
            boxes_cxcywh=[(10, 10, 100, 100)],
            scores=[0.9],
        )

        boxes, _, _ = detector.postprocess_predictions(
            output0, ratio=1.0, pads=(0.0, 0.0), src_shape=(640, 640),
            confidence=0.3, iou=0.7,
        )

        assert boxes[0][0] >= 0.0
        assert boxes[0][1] >= 0.0

    def test_results_sorted_by_score_descending(self):
        output0 = self._make_output0(
            boxes_cxcywh=[(100, 100, 40, 40), (500, 500, 40, 40), (300, 300, 40, 40)],
            scores=[0.4, 0.95, 0.6],
        )

        _, scores, _ = detector.postprocess_predictions(
            output0, ratio=1.0, pads=(0.0, 0.0), src_shape=(640, 640),
            confidence=0.3, iou=0.7,
        )

        assert list(scores) == sorted(scores, reverse=True)

    def test_kept_indices_reference_original_anchor_rows(self):
        output0 = self._make_output0(
            boxes_cxcywh=[(320, 320, 100, 100), (100, 100, 50, 50), (500, 500, 60, 60)],
            scores=[0.2, 0.9, 0.85],
        )

        _, _, kept_indices = detector.postprocess_predictions(
            output0, ratio=1.0, pads=(0.0, 0.0), src_shape=(640, 640),
            confidence=0.3, iou=0.7,
        )

        assert set(kept_indices.tolist()) == {1, 2}
