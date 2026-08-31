"""Tests for modules.async_worker's FWDF-155 capability-driven inpaint gate:
_inpaint_family_lacks_engine_head() decides whether apply_inpaint() must
skip modules/inpaint_worker.py's SDXL-specific InpaintHead patch().

apply_inpaint() itself is a closure nested inside modules.async_worker.worker()
(worker() never returns -- it ends in an infinite task-polling loop -- and its
heavy pipeline/inpaint_worker imports are deliberately deferred inside worker()
rather than at module scope, per tests/test_zimage_pipeline.py's docstring
about modules.default_pipeline being expensive to import). Extracting it to
module level, or calling worker() directly, is out of this ticket's scope, so
this file tests the capability-check helper apply_inpaint() delegates to --
the actual new decision logic -- directly. See tests/test_patch_inpaint_masking.py
for direct verification of the mechanism that keeps masking correct once that
patch() call is skipped.
"""
import base64
import io
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

_original_argv = sys.argv
sys.argv = [sys.argv[0]]
try:

    _installed_stub_names = []

    # Stub only when the real module (or an earlier stub) isn't already loaded:
    # unconditionally overwriting sys.modules would silently hand later test
    # files a None-returning generate_mask_from_image depending on run order.
    if 'extras.inpaint_mask' not in sys.modules:
        _inpaint_mask_stub = types.ModuleType('extras.inpaint_mask')
        _inpaint_mask_stub.generate_mask_from_image = lambda *_args, **_kwargs: None
        _inpaint_mask_stub.SAMOptions = object
        sys.modules['extras.inpaint_mask'] = _inpaint_mask_stub
        _installed_stub_names.append('extras.inpaint_mask')

    import transformers  # noqa: E402,F401  (forces the real torchvision-unavailable check first)

    _torchvision_available = True
    try:
        import torchvision  # noqa: F401
    except ImportError:
        _torchvision_available = False

    if not _torchvision_available:
        _functional_stub = types.ModuleType('torchvision.transforms.functional')
        _functional_stub.InterpolationMode = object
        _functional_stub.rotate = lambda *_args, **_kwargs: None
        _transforms_stub = types.ModuleType('torchvision.transforms')
        _transforms_stub.functional = _functional_stub
        _torchvision_stub = types.ModuleType('torchvision')
        _torchvision_stub.transforms = _transforms_stub
        sys.modules['torchvision'] = _torchvision_stub
        sys.modules['torchvision.transforms'] = _transforms_stub
        sys.modules['torchvision.transforms.functional'] = _functional_stub
        _installed_stub_names.extend(
            ['torchvision', 'torchvision.transforms', 'torchvision.transforms.functional'])

    from modules import async_worker  # noqa: E402
    from modules.model_family import ModelFamily  # noqa: E402

finally:
    sys.argv = _original_argv
    # Restore immediately after the modules under test captured their
    # imports — pytest may import later test modules before any fixture
    # teardown runs, and they must not observe the stand-ins.
    for _name in _installed_stub_names:
        sys.modules.pop(_name, None)
    _installed_stub_names.clear()


@pytest.fixture
def fake_family(monkeypatch):
    """Lets a test pin modules.model_family_detection.get_family()'s result
    without touching a real checkpoint file, mirroring the pattern already
    used in tests/test_zimage_pipeline.py."""
    holder = {'family': ModelFamily.SDXL}
    monkeypatch.setattr(
        async_worker.modules.model_family_detection, 'get_family',
        lambda _name: holder['family']
    )
    return holder




class TestInpaintFamilyLacksEngineHead:
    def test_true_for_family_without_inpaint_engine_support(self, fake_family):
        fake_family['family'] = ModelFamily.Z_IMAGE

        assert async_worker._inpaint_family_lacks_engine_head('z_image_turbo.safetensors') is True

    def test_false_for_family_with_inpaint_engine_support(self, fake_family):
        fake_family['family'] = ModelFamily.SDXL

        assert async_worker._inpaint_family_lacks_engine_head('sdxl_base.safetensors') is False

    def test_delegates_family_detection_to_base_model_name(self, monkeypatch):
        seen = {}

        def fake_get_family(name):
            seen['name'] = name
            return ModelFamily.SDXL

        monkeypatch.setattr(async_worker.modules.model_family_detection, 'get_family', fake_get_family)

        async_worker._inpaint_family_lacks_engine_head('some_checkpoint.safetensors')

        assert seen['name'] == 'some_checkpoint.safetensors'

    def test_matches_live_registry_for_unknown_family(self, fake_family):
        # UNKNOWN aliases SDXL's capability entry (modules.model_family):
        # an undetectable checkpoint must keep today's SDXL-engine behavior.
        fake_family['family'] = ModelFamily.UNKNOWN

        assert async_worker._inpaint_family_lacks_engine_head('mystery.safetensors') is False


def _eraser_data_url(alpha):
    """Build a data:image/png;base64,... string whose alpha channel is
    `alpha` (a 2D uint8 array), matching what javascript/inpaint_eraser.js's
    offscreen-canvas toDataURL('image/png') export produces."""
    rgb = np.zeros(alpha.shape + (3,), dtype=np.uint8)
    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode='RGBA').save(buf, format='PNG')
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


class TestDecodeEraserMask:
    def test_empty_string_returns_none(self):
        assert async_worker.decode_eraser_mask('', width=4, height=4) is None

    def test_none_input_returns_none(self):
        assert async_worker.decode_eraser_mask(None, width=4, height=4) is None

    def test_malformed_base64_returns_none(self):
        assert async_worker.decode_eraser_mask('data:image/png;base64,not-valid-base64!!', width=4, height=4) is None

    def test_valid_alpha_decodes_to_binarized_mask_of_requested_size(self):
        alpha = np.zeros((4, 4), dtype=np.uint8)
        alpha[1:3, 1:3] = 255
        data_url = _eraser_data_url(alpha)

        decoded = async_worker.decode_eraser_mask(data_url, width=4, height=4)

        assert decoded.shape == (4, 4)
        assert decoded.dtype == np.uint8
        assert np.array_equal(decoded, alpha)

    def test_partial_alpha_binarizes_at_127(self):
        alpha = np.full((2, 2), 100, dtype=np.uint8)  # below the >127 threshold
        alpha[0, 0] = 200  # above it
        data_url = _eraser_data_url(alpha)

        decoded = async_worker.decode_eraser_mask(data_url, width=2, height=2)

        assert decoded[0, 0] == 255
        assert decoded[0, 1] == 0
        assert decoded[1, 0] == 0
        assert decoded[1, 1] == 0

    def test_resizes_to_requested_dimensions(self):
        alpha = np.full((4, 4), 255, dtype=np.uint8)
        data_url = _eraser_data_url(alpha)

        decoded = async_worker.decode_eraser_mask(data_url, width=8, height=2)

        assert decoded.shape == (2, 8)


class TestApplyEraserMask:
    def test_no_eraser_mask_leaves_mask_untouched(self):
        mask = np.full((3, 3), 255, dtype=np.uint8)

        result = async_worker.apply_eraser_mask(mask, None)

        assert np.array_equal(result, np.full((3, 3), 255, dtype=np.uint8))

    def test_erased_pixels_become_zero_others_untouched(self):
        mask = np.full((3, 3), 255, dtype=np.uint8)
        eraser = np.zeros((3, 3), dtype=np.uint8)
        eraser[1, 1] = 255

        result = async_worker.apply_eraser_mask(mask, eraser)

        assert result[1, 1] == 0
        assert np.array_equal(result[eraser == 0], np.full(8, 255, dtype=np.uint8))

    def test_advanced_canvas_eraser_does_not_affect_main_canvas_mask(self):
        # Mirrors apply_image_input(): the main-canvas mask and the
        # advanced-canvas mask are separate arrays until np.maximum combines
        # them, so erasing one must never mutate the other.
        main_mask = np.full((3, 3), 255, dtype=np.uint8)
        advanced_mask = np.full((3, 3), 255, dtype=np.uint8)
        advanced_eraser = np.full((3, 3), 255, dtype=np.uint8)

        advanced_mask = async_worker.apply_eraser_mask(advanced_mask, advanced_eraser)
        combined = np.maximum(main_mask, advanced_mask)

        assert np.array_equal(main_mask, np.full((3, 3), 255, dtype=np.uint8))
        assert np.array_equal(advanced_mask, np.zeros((3, 3), dtype=np.uint8))
        # The main canvas's own strokes still select the combined mask even
        # though the advanced canvas was fully erased.
        assert np.array_equal(combined, np.full((3, 3), 255, dtype=np.uint8))

    def test_invert_after_erase_selects_the_erased_region(self):
        mask = np.full((3, 3), 255, dtype=np.uint8)
        eraser = np.zeros((3, 3), dtype=np.uint8)
        eraser[1, 1] = 255

        mask = async_worker.apply_eraser_mask(mask, eraser)
        inverted = 255 - mask

        assert inverted[1, 1] == 255
        assert np.array_equal(inverted[eraser == 0], np.zeros(8, dtype=np.uint8))


if __name__ == '__main__':
    # Pytest-style test classes (no unittest.TestCase inheritance) need the
    # pytest runner for direct execution; unittest.main() would collect zero.
    raise SystemExit(pytest.main([__file__]))
