"""Tests for modules.async_worker's FWDF-156 follow-up ControlNet routing
fix: `_controlnet_family()` and `_controlnet_type_supported()` decide which
checkpoint loader ControlNet routing uses and whether a given ControlNet
type (e.g. CPDS) is downloaded/processed at all, both keyed off the
REQUESTED checkpoint (`async_task.base_model_name`) rather than
`pipeline.model_base.family` (the currently *loaded* model, stale on a
base-model family switch until `process_prompt()`'s later
`pipeline.refresh_everything()` call catches up).

`apply_image_input()`/`apply_control_nets()`/`process_task()` themselves are
closures nested inside `modules.async_worker.worker()` (see
tests/test_async_worker_inpaint.py's docstring for why they cannot be
called directly in a unit test); this file tests the module-level helpers
those closures now delegate every family/type routing decision to -- the
actual fixed logic -- directly.
"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

_original_argv = sys.argv
sys.argv = [sys.argv[0]]
try:

    _installed_stub_names = []

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
    for _name in _installed_stub_names:
        sys.modules.pop(_name, None)
    _installed_stub_names.clear()


@pytest.fixture
def fake_family(monkeypatch):
    """Lets a test pin modules.model_family_detection.get_family()'s result
    without touching a real checkpoint file, mirroring the pattern already
    used in tests/test_zimage_pipeline.py and tests/test_async_worker_inpaint.py."""
    holder = {'family': ModelFamily.SDXL}
    monkeypatch.setattr(
        async_worker.modules.model_family_detection, 'get_family',
        lambda _name: holder['family']
    )
    return holder


class TestControlnetFamily:
    def test_delegates_to_model_family_detection_get_family(self, monkeypatch):
        seen = {}

        def fake_get_family(name):
            seen['name'] = name
            return ModelFamily.Z_IMAGE

        monkeypatch.setattr(async_worker.modules.model_family_detection, 'get_family', fake_get_family)

        result = async_worker._controlnet_family('some_checkpoint.safetensors')

        assert seen['name'] == 'some_checkpoint.safetensors'
        assert result is ModelFamily.Z_IMAGE

    def test_reads_the_requested_checkpoint_not_pipeline_state(self, fake_family, monkeypatch):
        # The whole point of this helper: it must not consult
        # pipeline.model_base at all. Simulate the pipeline "holding" SDXL
        # (the previous request's family) while the requested checkpoint is
        # Z-Image -- the routing decision must follow the requested name.
        fake_family['family'] = ModelFamily.Z_IMAGE

        result = async_worker._controlnet_family('z_image_turbo.safetensors')

        assert result == ModelFamily.Z_IMAGE


class TestControlnetTypeSupported:
    def test_sdxl_supports_canny_and_cpds(self, fake_family):
        fake_family['family'] = ModelFamily.SDXL

        assert async_worker._controlnet_type_supported('sdxl_base.safetensors', 'canny') is True
        assert async_worker._controlnet_type_supported('sdxl_base.safetensors', 'cpds') is True

    def test_z_image_supports_canny_but_not_cpds(self, fake_family):
        # CPDS has no published DiT equivalent (modules/model_family.py's
        # _build_z_image_capabilities docstring) -- a Z-Image task must
        # never download or process a CPDS checkpoint.
        fake_family['family'] = ModelFamily.Z_IMAGE

        assert async_worker._controlnet_type_supported('z_image_turbo.safetensors', 'canny') is True
        assert async_worker._controlnet_type_supported('z_image_turbo.safetensors', 'cpds') is False

    def test_unknown_family_matches_sdxl_cpds_support(self, fake_family):
        # UNKNOWN aliases SDXL's capability entry -- an undetectable
        # checkpoint must keep today's SDXL CPDS behavior unchanged.
        fake_family['family'] = ModelFamily.UNKNOWN

        assert async_worker._controlnet_type_supported('mystery.safetensors', 'cpds') is True

    def test_switching_the_requested_family_changes_the_result_for_the_same_type(self, fake_family):
        # Simulates the exact bug scenario: consecutive requests for
        # different base_model_name values must each get their own,
        # independently-correct CPDS gate -- not whichever family happened
        # to be loaded first.
        fake_family['family'] = ModelFamily.SDXL
        assert async_worker._controlnet_type_supported('sdxl_base.safetensors', 'cpds') is True

        fake_family['family'] = ModelFamily.Z_IMAGE
        assert async_worker._controlnet_type_supported('z_image_turbo.safetensors', 'cpds') is False


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__]))
