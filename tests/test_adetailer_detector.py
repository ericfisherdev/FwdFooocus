"""Tests for extras.adetailer.detector's pure-numpy letterbox/NMS/rescale
helpers (FWDF-197) and mask-prototype decoding (FWDF-199). These run without
any ONNX model file -- detect_bboxes' session-dependent path is exercised
here with a stubbed InferenceSession injected directly into the module's
session cache (and indirectly through tests/test_inpaint_mask_adetailer.py
by monkeypatching detect_bboxes itself).
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


class TestDecodeMasks:
    def test_large_positive_coefficient_yields_exactly_that_block(self):
        # 8x8 proto grid -> 32x32 letterbox canvas. A 4x4 bright block sits
        # in the middle of an otherwise strongly-negative background.
        protos = np.full((1, 8, 8), -10.0, dtype=np.float32)
        protos[0, 2:6, 2:6] = 10.0
        coefficients = np.array([[1.0]], dtype=np.float32)
        letterbox_boxes = np.array([[0, 0, 32, 32]], dtype=np.float32)  # full-canvas box

        masks = detector.decode_masks(
            protos, coefficients, letterbox_boxes, ratio=1.0, pads=(0.0, 0.0), src_shape=(32, 32)
        )

        assert masks.shape == (1, 32, 32)
        assert masks[0, 12:20, 12:20].all()  # well inside the block (proto rows/cols 2:6 -> pixels 8:24)
        assert not masks[0, 0:6, 0:6].any()  # well outside the block

    def test_all_negative_coefficient_yields_empty_mask(self):
        protos = np.full((1, 8, 8), 10.0, dtype=np.float32)  # positive prototype plane
        coefficients = np.array([[-1.0]], dtype=np.float32)  # negative coefficient flips the sign
        letterbox_boxes = np.array([[0, 0, 32, 32]], dtype=np.float32)

        masks = detector.decode_masks(
            protos, coefficients, letterbox_boxes, ratio=1.0, pads=(0.0, 0.0), src_shape=(32, 32)
        )

        assert not masks.any()

    def test_crop_to_box_zeroes_pixels_outside_the_box(self):
        protos = np.full((1, 8, 8), 10.0, dtype=np.float32)  # positive everywhere
        coefficients = np.array([[1.0]], dtype=np.float32)
        # Box covers only the left half of the 32x32 canvas.
        letterbox_boxes = np.array([[0, 0, 16, 32]], dtype=np.float32)

        masks = detector.decode_masks(
            protos, coefficients, letterbox_boxes, ratio=1.0, pads=(0.0, 0.0), src_shape=(32, 32)
        )

        assert masks[0, :, 0:14].all()  # inside the box
        assert not masks[0, :, 18:32].any()  # outside the box, zeroed by the crop

    def test_unletterbox_maps_known_proto_space_block_to_expected_source_region(self):
        # 4x4 proto grid -> 16x16 letterbox canvas. pad_y=4 is an exact
        # multiple of the 4x proto->canvas scale factor, so the padded rows
        # align cleanly with whole proto rows.
        protos = np.zeros((1, 4, 4), dtype=np.float32)
        protos[0, 0, :] = 10.0   # top pad row -- must be stripped, not leak into output
        protos[0, 1, :] = 10.0   # bright content row -- must appear in output
        protos[0, 2, :] = -10.0  # dark content row -- must appear (as False) in output
        protos[0, 3, :] = 10.0   # bottom pad row -- must be stripped
        coefficients = np.array([[1.0]], dtype=np.float32)
        letterbox_boxes = np.array([[0, 0, 16, 16]], dtype=np.float32)  # no additional box crop

        masks = detector.decode_masks(
            protos, coefficients, letterbox_boxes, ratio=1.0, pads=(0.0, 4.0), src_shape=(8, 16)
        )

        assert masks.shape == (1, 8, 16)
        assert masks[0, 1:3, :].all()  # source rows from the bright content row (proto row 1)
        assert not masks[0, 5:7, :].any()  # source rows from the dark content row (proto row 2)

    def test_zero_detections_returns_empty_array_with_source_shape(self):
        protos = np.zeros((1, 8, 8), dtype=np.float32)
        coefficients = np.empty((0, 1), dtype=np.float32)
        letterbox_boxes = np.empty((0, 4), dtype=np.float32)

        masks = detector.decode_masks(
            protos, coefficients, letterbox_boxes, ratio=1.0, pads=(0.0, 0.0), src_shape=(32, 32)
        )

        assert masks.shape == (0, 32, 32)


class _FakeIO:
    """Stands in for onnxruntime's NodeArg (only .name/.shape are used)."""
    def __init__(self, name=None, shape=None):
        self.name = name
        self.shape = shape


class _FakeSession:
    """Stands in for onnxruntime.InferenceSession so detect_bboxes can be
    exercised without a real ONNX model file."""
    def __init__(self, outputs):
        self._outputs = outputs

    def get_inputs(self):
        return [_FakeIO(name='images')]

    def get_outputs(self):
        return [_FakeIO(shape=list(output.shape)) for output in self._outputs]

    def run(self, output_names, feed):
        return self._outputs


class TestDetectBboxesSessionCapability:
    def test_stubbed_single_output_session_still_produces_rectangle_masks(self, monkeypatch):
        """Regression: bbox-only (single-output) models must be unaffected
        by the seg-decode path added for -seg models."""
        output0 = np.zeros((1, 5, 1), dtype=np.float32)
        output0[0, :, 0] = [320, 320, 100, 100, 0.9]  # cx, cy, w, h, score

        model_path = 'fake-bbox-model.onnx'
        monkeypatch.setitem(
            detector._session_cache, model_path,
            detector._CachedSession(session=_FakeSession([output0]), has_masks=False),
        )

        image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = detector.detect_bboxes(image, model_path, confidence=0.3)

        assert result.masks is None
        assert len(result.boxes) == 1
        assert len(result.scores) == 1

    def test_stubbed_two_output_session_populates_segmentation_masks(self, monkeypatch):
        """Integration: a seg-capable (2-output) session must decode its
        mask-coefficient columns and prototype tensor into DetectionResult.masks."""
        nc = 1
        num_extra_coeffs = detector.SEG_MASK_COEFFICIENTS
        output0 = np.zeros((1, 4 + nc + num_extra_coeffs, 1), dtype=np.float32)
        output0[0, 0, 0] = 320  # cx
        output0[0, 1, 0] = 320  # cy
        output0[0, 2, 0] = 200  # w
        output0[0, 3, 0] = 200  # h
        output0[0, 4, 0] = 0.9  # class score
        output0[0, 5:, 0] = 1.0  # mask coefficients

        proto_size = 160  # 1/4 of the default 640 letterbox canvas
        protos = np.full((1, num_extra_coeffs, proto_size, proto_size), -1.0, dtype=np.float32)
        # Box in source coords is [220, 220, 420, 420]; proto space is 1/4 of that.
        protos[0, :, 55:105, 55:105] = 1.0  # sum over 32 channels = +32 -> sigmoid ~= 1

        model_path = 'fake-seg-model.onnx'
        monkeypatch.setitem(
            detector._session_cache, model_path,
            detector._CachedSession(session=_FakeSession([output0, protos]), has_masks=True),
        )

        image = np.zeros((640, 640, 3), dtype=np.uint8)  # square -> ratio=1.0, no padding
        result = detector.detect_bboxes(image, model_path, confidence=0.3)

        assert result.masks is not None
        assert result.masks.shape == (1, 640, 640)
        assert result.masks[0, 300:340, 300:340].all()  # well inside the detected box
        assert not result.masks[0, 0:50, 0:50].any()  # well outside the detected box
