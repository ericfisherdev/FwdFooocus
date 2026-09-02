"""Tests for extras.inpaint_mask's 'adetailer' dispatch branch (FWDF-197).

detect_bboxes and modules.config.download_adetailer_model are monkeypatched
throughout -- no ONNX model file is downloaded or run in CI.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

_original_argv = sys.argv
sys.argv = [sys.argv[0]]
import extras.inpaint_mask as inpaint_mask  # noqa: E402
import modules.config  # noqa: E402
from extras.adetailer.detector import DetectionResult  # noqa: E402
sys.argv = _original_argv


@pytest.fixture(autouse=True)
def stub_model_download(monkeypatch):
    monkeypatch.setattr(modules.config, 'download_adetailer_model', lambda model_name: f'/fake/{model_name}.onnx')


class TestADetailerOptionsDefaults:
    def test_defaults(self):
        options = inpaint_mask.ADetailerOptions(model_name='face_yolov9c')

        assert options.model_name == 'face_yolov9c'
        assert options.confidence == 0.3
        assert options.max_detections == 0
        assert options.box_erode_or_dilate == 0

    def test_explicit_values_stick(self):
        options = inpaint_mask.ADetailerOptions(
            model_name='hand_yolov9c', confidence=0.5, max_detections=2, box_erode_or_dilate=4
        )

        assert options.model_name == 'hand_yolov9c'
        assert options.confidence == 0.5
        assert options.max_detections == 2
        assert options.box_erode_or_dilate == 4


def _image(h=100, w=100):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestGenerateMaskFromImageADetailerDispatch:
    def test_dispatches_to_adetailer_and_resolves_model_path(self, monkeypatch):
        seen = {}

        def fake_detect_bboxes(image, model_path, confidence):
            seen['model_path'] = model_path
            seen['confidence'] = confidence
            return DetectionResult(boxes=np.empty((0, 4)), scores=np.empty((0,)))

        monkeypatch.setattr(inpaint_mask, 'detect_bboxes', fake_detect_bboxes)

        options = inpaint_mask.ADetailerOptions(model_name='face_yolov9c', confidence=0.42)
        inpaint_mask.generate_mask_from_image(_image(), mask_model='adetailer', adetailer_options=options)

        assert seen['model_path'] == '/fake/face_yolov9c.onnx'
        assert seen['confidence'] == 0.42

    def test_zero_detections_returns_all_black_mask_and_zero_count(self, monkeypatch):
        monkeypatch.setattr(
            inpaint_mask, 'detect_bboxes',
            lambda image, model_path, confidence: DetectionResult(boxes=np.empty((0, 4)), scores=np.empty((0,)))
        )

        options = inpaint_mask.ADetailerOptions(model_name='face_yolov9c')
        mask, count, second, third = inpaint_mask.generate_mask_from_image(
            _image(50, 50), mask_model='adetailer', adetailer_options=options
        )

        assert count == 0
        assert second == 0
        assert third == 0
        assert mask.shape == (50, 50, 3)
        assert np.all(mask == 0)

    def test_detection_fills_box_region_into_mask(self, monkeypatch):
        boxes = np.array([[10, 10, 30, 30]], dtype=np.float32)
        monkeypatch.setattr(
            inpaint_mask, 'detect_bboxes',
            lambda image, model_path, confidence: DetectionResult(boxes=boxes, scores=np.array([0.9]))
        )

        options = inpaint_mask.ADetailerOptions(model_name='face_yolov9c')
        mask, count, _, _ = inpaint_mask.generate_mask_from_image(
            _image(50, 50), mask_model='adetailer', adetailer_options=options
        )

        assert count == 1
        region = mask[10:30, 10:30]
        assert np.all(region == 255)
        outside = mask[0:10, 0:10]
        assert np.all(outside == 0)

    def test_max_detections_truncates_applied_boxes(self, monkeypatch):
        # Two disjoint boxes, sorted by score descending as detect_bboxes guarantees.
        boxes = np.array([[0, 0, 10, 10], [40, 40, 50, 50]], dtype=np.float32)
        monkeypatch.setattr(
            inpaint_mask, 'detect_bboxes',
            lambda image, model_path, confidence: DetectionResult(boxes=boxes, scores=np.array([0.9, 0.8]))
        )

        options = inpaint_mask.ADetailerOptions(model_name='face_yolov9c', max_detections=1)
        mask, count, _, _ = inpaint_mask.generate_mask_from_image(
            _image(60, 60), mask_model='adetailer', adetailer_options=options
        )

        # detection_count reports the raw detections, but only the top box is painted.
        assert count == 2
        assert np.all(mask[0:10, 0:10] == 255)
        assert np.all(mask[40:50, 40:50] == 0)

    def test_box_erode_or_dilate_expands_mask_region(self, monkeypatch):
        boxes = np.array([[20, 20, 30, 30]], dtype=np.float32)
        monkeypatch.setattr(
            inpaint_mask, 'detect_bboxes',
            lambda image, model_path, confidence: DetectionResult(boxes=boxes, scores=np.array([0.9]))
        )

        options = inpaint_mask.ADetailerOptions(model_name='face_yolov9c', box_erode_or_dilate=5)
        mask, _, _, _ = inpaint_mask.generate_mask_from_image(
            _image(60, 60), mask_model='adetailer', adetailer_options=options
        )

        # Dilated by 5px on each side: region [15:35, 15:35] should now be filled,
        # including pixels just outside the original [20:30, 20:30] box.
        assert mask[16, 16][0] == 255
        assert mask[34, 34][0] == 255

    def test_negative_erode_or_dilate_shrinks_mask_region(self, monkeypatch):
        boxes = np.array([[20, 20, 40, 40]], dtype=np.float32)
        monkeypatch.setattr(
            inpaint_mask, 'detect_bboxes',
            lambda image, model_path, confidence: DetectionResult(boxes=boxes, scores=np.array([0.9]))
        )

        options = inpaint_mask.ADetailerOptions(model_name='face_yolov9c', box_erode_or_dilate=-5)
        mask, _, _, _ = inpaint_mask.generate_mask_from_image(
            _image(60, 60), mask_model='adetailer', adetailer_options=options
        )

        # Eroded by 5px on each side: region right at the original edge is now unfilled.
        assert mask[21, 21][0] == 0
        assert mask[30, 30][0] == 255

    def test_none_options_does_not_dispatch_to_adetailer(self, monkeypatch):
        called = {}
        monkeypatch.setattr(
            inpaint_mask, 'detect_bboxes',
            lambda *a, **k: called.setdefault('detect_bboxes_called', True)
        )
        # mask_model='adetailer' with no options must not reach the adetailer
        # branch at all (mirrors the existing sam_options-is-None fallthrough).
        try:
            inpaint_mask.generate_mask_from_image(_image(), mask_model='adetailer', adetailer_options=None)
        except Exception:
            pass  # the rembg fallthrough's own behavior is out of this test's scope

        assert 'detect_bboxes_called' not in called
