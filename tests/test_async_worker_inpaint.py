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
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

_original_argv = sys.argv
sys.argv = [sys.argv[0]]
try:

    # Stub only when the real module (or an earlier stub) isn't already loaded:
    # unconditionally overwriting sys.modules would silently hand later test
    # files a None-returning generate_mask_from_image depending on run order.
    if 'extras.inpaint_mask' not in sys.modules:
        _inpaint_mask_stub = types.ModuleType('extras.inpaint_mask')
        _inpaint_mask_stub.generate_mask_from_image = lambda *_args, **_kwargs: None
        _inpaint_mask_stub.SAMOptions = object
        sys.modules['extras.inpaint_mask'] = _inpaint_mask_stub

    import transformers  # noqa: E402,F401  (forces the real torchvision-unavailable check first)

    _torchvision_available = True
    try:
        import torchvision  # noqa: F401
    except ImportError:
        _torchvision_available = False

    _installed_stub_names = []
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



@pytest.fixture(scope='module', autouse=True)
def _remove_installed_stubs_after_module():
    """Pop the torchvision stand-ins this module installed once its tests
    finish, so they cannot leak into later test modules in the session."""
    yield
    for name in _installed_stub_names:
        sys.modules.pop(name, None)


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


if __name__ == '__main__':
    # Pytest-style test classes (no unittest.TestCase inheritance) need the
    # pytest runner for direct execution; unittest.main() would collect zero.
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__]))
