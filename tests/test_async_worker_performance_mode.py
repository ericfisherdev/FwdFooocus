"""Tests for AsyncTask's performance-label resolution (FWDF-152).

The Gradio radio and the new-UI API both hand AsyncTask a performance *label*:
a legacy `Performance` value (Quality, Speed, ...) or, once the checkpoint's
family swaps the radio's choices, a family mode such as Z-Image's `Turbo` or
Krea 2's `Raw`. AsyncTask used to run `Performance(label)` and raise
`ValueError: 'Turbo' is not a valid Performance` for the latter; it now
resolves the label against the requested checkpoint's capability entry.
"""
import sys
import types
from pathlib import Path
from unittest.mock import patch

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
        _inpaint_mask_stub.ADetailerOptions = object
        sys.modules['extras.inpaint_mask'] = _inpaint_mask_stub
        _installed_stub_names.append('extras.inpaint_mask')

    import transformers  # noqa: E402,F401  (forces the real torchvision-unavailable check first)

    try:
        import torchvision  # noqa: F401
    except ImportError:
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

    import modules.config as config  # noqa: E402
    from modules import async_worker, flags  # noqa: E402
    from modules.model_family import ModelFamily  # noqa: E402
    from new_ui.app import _build_generate_args  # noqa: E402
finally:
    sys.argv = _original_argv
    for _name in _installed_stub_names:
        sys.modules.pop(_name, None)
    _installed_stub_names.clear()


@pytest.fixture
def family_of_requested_checkpoint(monkeypatch):
    holder = {'family': ModelFamily.SDXL}
    monkeypatch.setattr(
        async_worker.modules.model_family_detection, 'get_family', lambda _name: holder['family'])
    return holder


def _task_for(base_model_name: str = 'checkpoint.safetensors') -> 'async_worker.AsyncTask':
    task = async_worker.AsyncTask(args=[])
    task.base_model_name = base_model_name
    return task


class TestApplyPerformanceMode:
    def test_legacy_label_on_sdxl_keeps_the_legacy_enum_and_steps(self, family_of_requested_checkpoint):
        task = _task_for()

        task._apply_performance_mode('Lightning')

        assert task.performance_selection is flags.Performance.LIGHTNING
        assert task.performance_label == 'Lightning'
        assert task.steps == task.original_steps == flags.Performance.LIGHTNING.steps()
        assert task.performance_mode.steps_uov == flags.Performance.LIGHTNING.steps_uov()

    @pytest.mark.parametrize('label', [member.value for member in flags.Performance])
    def test_every_legacy_label_matches_the_enum_it_names(self, family_of_requested_checkpoint, label):
        task = _task_for()

        task._apply_performance_mode(label)

        member = flags.Performance(label)
        assert task.performance_selection is member
        assert (task.steps, task.performance_mode.steps_uov) == (member.steps(), member.steps_uov())

    @pytest.mark.parametrize('family, label, steps', [
        (ModelFamily.Z_IMAGE, 'Turbo', 9),
        (ModelFamily.KREA2_RAW, 'Raw', 52),
        (ModelFamily.KREA2_TURBO, 'Turbo', 8),
    ])
    def test_family_label_resolves_without_raising(self, family_of_requested_checkpoint, family, label, steps):
        family_of_requested_checkpoint['family'] = family
        task = _task_for()

        task._apply_performance_mode(label)

        assert task.performance_label == label
        assert task.steps == task.original_steps == steps
        assert task.performance_mode.steps_uov == steps

    def test_family_label_maps_to_the_loraless_unrestricted_legacy_member(self, family_of_requested_checkpoint):
        # SDXL-only logic (performance LoRAs, Lightning/LCM/Hyper-SD defaults)
        # switches on performance_selection; a family mode must trigger none.
        family_of_requested_checkpoint['family'] = ModelFamily.KREA2_RAW
        task = _task_for()

        task._apply_performance_mode('Raw')

        assert task.performance_selection is flags.Performance.SPEED
        assert task.performance_selection.lora_filename() is None
        assert not flags.Performance.has_restricted_features(task.performance_selection)

    def test_label_of_another_family_raises_value_error(self, family_of_requested_checkpoint):
        family_of_requested_checkpoint['family'] = ModelFamily.KREA2_TURBO

        with pytest.raises(ValueError):
            _task_for()._apply_performance_mode('Raw')

    @pytest.mark.parametrize('label', ['Extreme Speed', 'Lightning', 'Hyper-SD'])
    def test_sdxl_accelerator_lora_labels_are_rejected_for_krea2(self, family_of_requested_checkpoint, label):
        family_of_requested_checkpoint['family'] = ModelFamily.KREA2_RAW

        with pytest.raises(ValueError):
            _task_for()._apply_performance_mode(label)

    def test_unknown_label_raises_value_error(self, family_of_requested_checkpoint):
        with pytest.raises(ValueError):
            _task_for()._apply_performance_mode('Warp Speed')

    def test_resolves_against_the_requested_checkpoint(self, monkeypatch):
        seen = []

        def get_family(name):
            seen.append(name)
            return ModelFamily.KREA2_TURBO

        monkeypatch.setattr(async_worker.modules.model_family_detection, 'get_family', get_family)

        _task_for('krea2_turbo_bf16.safetensors')._apply_performance_mode('Turbo')

        assert seen == ['krea2_turbo_bf16.safetensors']


class TestTaskConstructionFromGenerateArgs:
    """The same positional args both UIs build, run through the real
    AsyncTask.__init__, so the label's position in the arg list and the
    base_model_name that selects the family stay in step."""

    def _task(self, family, body):
        with patch.object(config, 'default_max_lora_number', 0), \
                patch.object(config, 'default_controlnet_image_count', 0), \
                patch.object(config, 'default_enhance_tabs', 0), \
                patch('modules.model_family_detection.get_family', return_value=family), \
                patch.object(config, 'model_filenames', ['checkpoint.safetensors']):
            args = _build_generate_args({'base_model_name': 'checkpoint.safetensors', **body})
            return async_worker.AsyncTask(args=args)

    @pytest.mark.parametrize('family, label, steps', [
        (ModelFamily.Z_IMAGE, 'Turbo', 9),
        (ModelFamily.KREA2_RAW, 'Raw', 52),
        (ModelFamily.KREA2_TURBO, 'Turbo', 8),
    ])
    def test_family_modes_build_a_task(self, family, label, steps):
        task = self._task(family, {'performance_selection': label})

        assert task.performance_label == label
        assert task.steps == steps

    def test_krea2_turbo_task_pins_cfg_and_drops_the_negative_prompt(self):
        task = self._task(ModelFamily.KREA2_TURBO, {'cfg_scale': 7.0, 'negative_prompt': 'blurry'})

        assert task.cfg_scale == 1.0
        assert task.negative_prompt == ''

    def test_krea2_raw_task_defaults_to_cfg_3_5(self):
        task = self._task(ModelFamily.KREA2_RAW, {})

        assert task.cfg_scale == 3.5
        assert task.steps == 52

    def test_sdxl_task_is_unchanged(self):
        task = self._task(ModelFamily.SDXL, {'performance_selection': 'Quality'})

        assert task.performance_selection is flags.Performance.QUALITY
        assert task.steps == flags.Performance.QUALITY.steps()


class TestDownstreamUsesTheResolvedMode:
    def test_session_snapshot_records_the_family_label(self, family_of_requested_checkpoint):
        family_of_requested_checkpoint['family'] = ModelFamily.KREA2_TURBO
        task = _task_for()
        task._apply_performance_mode('Turbo')
        for name in ('prompt', 'negative_prompt', 'style_selections', 'refiner_model_name', 'vae_name',
                     'sampler_name', 'scheduler_name', 'cfg_scale', 'image_number', 'sharpness',
                     'seed', 'aspect_ratios_selection'):
            setattr(task, name, None)
        task.loras = []

        snapshot = async_worker._build_session_state(task)

        assert snapshot['performance'] == 'Turbo'
        assert snapshot['steps'] == 8
