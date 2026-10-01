"""Real-weight generation smoke tests for Krea 2 Raw and Turbo (FWDF-160).

Every test here loads a 12B checkpoint, the Qwen3-VL-4B text encoder and the
Qwen Image VAE, so the module is skipped unless `FWDF_RUN_MODEL_TESTS=1` is set
and the files exist (`tests/conftest.py`, `tests/_model_gate.py`). In a normal
model-free `pytest tests/` run all of them report "skipped" with the reason.

Run locally:

    FWDF_RUN_MODEL_TESTS=1 python -m pytest tests/test_krea2_generation_smoke.py -m requires_models

Prerequisites beyond the three Krea 2 files: one checkpoint of each variant
whose file name contains `krea2` (the gate only discovers those) and resolves
to that variant (`krea2_raw*` / `krea2_turbo*`, or a `krea2_variant_overrides`
entry for such a name), and the prompt-expansion model that
`refresh_everything` always loads. A variant with no checkpoint skips its tests.

Budget (24 GB card): Raw is 52 steps with two passes per step, roughly several
minutes per image; Turbo is 8 single-pass steps. Each variant is loaded once
and unloaded at module teardown. The 2K tests at the bottom are the slowest
and are the empirical check for which registry aspect ratios fit in memory.

This module is where FWDF-152's empirical open items are verified:
  - negative prompt changes Raw output but not Turbo's
  - Turbo skips the unconditional pass (cond_scale == 1.0)
  - live previews render for the 16-channel latent
  - every aspect ratio the registry offers actually generates
"""
import contextlib
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

_original_argv = sys.argv
sys.argv = [sys.argv[0]]

import modules.config  # noqa: E402
from modules.model_family import FamilyCapabilities, ModelFamily, get_capabilities  # noqa: E402
from tests._model_gate import find_krea2_checkpoints  # noqa: E402

sys.argv = _original_argv

pytestmark = pytest.mark.requires_models

SEED = 12345
PROMPT = 'a red fox sitting in a snowy forest, golden hour, detailed fur'
CONTRASTING_NEGATIVE_PROMPT = 'blurry, monochrome, low quality'
MEGAPIXEL_SIDE = 1024

MIN_PIXEL_STD = 1.0
RAW_NEGATIVE_PROMPT_MIN_MEAN_DIFFERENCE = 2.0
TURBO_NEGATIVE_PROMPT_MAX_MEAN_DIFFERENCE = 0.5


@dataclass
class Krea2Session:
    """One loaded Krea 2 variant plus the registry values to drive it with, so
    a smoke test cannot drift from the registry."""

    family: ModelFamily
    capabilities: FamilyCapabilities
    pipeline: object

    def generate(self, positive_prompt, negative_prompt='', width=MEGAPIXEL_SIDE, height=MEGAPIXEL_SIDE,
                 disable_preview=True, callback=None):
        """One image as a (height, width, 3) uint8 array."""
        positive = self.pipeline.clip_encode([positive_prompt])
        negative = self.pipeline.clip_encode([negative_prompt])
        steps = self.capabilities.default_steps
        images = self.pipeline.process_diffusion(
            positive_cond=positive, negative_cond=negative,
            steps=steps, switch=steps, width=width, height=height, image_seed=SEED,
            callback=callback,
            sampler_name=self.capabilities.sampler_names[0],
            scheduler_name=self.capabilities.scheduler_names[0],
            cfg_scale=self.capabilities.default_cfg,
            disable_preview=disable_preview,
        )
        assert len(images) == 1
        return images[0]


def _checkpoint_for(family):
    import modules.model_family_detection as detection
    for name in find_krea2_checkpoints(modules.config.paths_checkpoints):
        if detection.get_family(name) is family:
            return name
    pytest.skip(f'no checkpoint under {modules.config.paths_checkpoints} resolves to {family.value}')


@contextlib.contextmanager
def _loaded_pipeline(family):
    """Loads `family`'s checkpoint through the real pipeline and unloads it on
    exit so two 12B models are never resident together. Deliberately avoids
    `initialize_default_pipeline()`, which would load the configured SDXL model."""
    import ldm_patched.modules.model_management as model_management
    from tests._default_pipeline_doubles import install_default_pipeline_test_doubles

    restore_stubs = install_default_pipeline_test_doubles()
    try:
        import modules.default_pipeline as pipeline
        import modules.patch as fork_patch

        fork_patch.patch_all()
        pid = os.getpid()
        # Sharpness is an SDXL-only heuristic (capabilities.supports_sharpness is False).
        fork_patch.patch_settings[pid] = fork_patch.PatchSettings(sharpness=0.0)
        pipeline.refresh_everything(refiner_model_name='None', base_model_name=_checkpoint_for(family), loras=[])
        try:
            yield pipeline
        finally:
            pipeline.clear_all_caches()
            model_management.unload_all_models()
            fork_patch.patch_settings.pop(pid, None)
            gc.collect()
            model_management.soft_empty_cache()
    finally:
        restore_stubs()


@pytest.fixture(scope='module')
def raw_session():
    with _loaded_pipeline(ModelFamily.KREA2_RAW) as pipeline:
        yield Krea2Session(ModelFamily.KREA2_RAW, get_capabilities(ModelFamily.KREA2_RAW), pipeline)


@pytest.fixture(scope='module')
def turbo_session():
    with _loaded_pipeline(ModelFamily.KREA2_TURBO) as pipeline:
        yield Krea2Session(ModelFamily.KREA2_TURBO, get_capabilities(ModelFamily.KREA2_TURBO), pipeline)


@contextlib.contextmanager
def _record_unconditional_arguments():
    """Yields the list of `uncond` arguments the sampler hands to
    `calc_cond_uncond_batch` while active: None means the unconditional pass
    was skipped (cond_scale == 1.0)."""
    import modules.patch as fork_patch

    recorded = []
    original = fork_patch.calc_cond_uncond_batch

    def recording(model, cond, uncond, *args, **kwargs):
        recorded.append(uncond)
        return original(model, cond, uncond, *args, **kwargs)

    fork_patch.calc_cond_uncond_batch = recording
    try:
        yield recorded
    finally:
        fork_patch.calc_cond_uncond_batch = original


def _assert_is_a_real_image(image, width, height):
    assert image.shape == (height, width, 3)
    assert image.dtype == np.uint8
    # A NaN latent decodes to a flat image; a healthy generation has structure.
    assert image.std() > MIN_PIXEL_STD


def _mean_absolute_difference(first, second):
    return float(np.abs(first.astype(int) - second.astype(int)).mean())


class TestRawGeneration:
    def test_text_to_image(self, raw_session):
        _assert_is_a_real_image(raw_session.generate(PROMPT), MEGAPIXEL_SIDE, MEGAPIXEL_SIDE)

    def test_negative_prompt_changes_the_output(self, raw_session):
        without = raw_session.generate(PROMPT, negative_prompt='')
        with_negative = raw_session.generate(PROMPT, negative_prompt=CONTRASTING_NEGATIVE_PROMPT)
        assert _mean_absolute_difference(without, with_negative) > RAW_NEGATIVE_PROMPT_MIN_MEAN_DIFFERENCE

    def test_runs_the_unconditional_pass(self, raw_session):
        with _record_unconditional_arguments() as unconditional_arguments:
            raw_session.generate(PROMPT, negative_prompt=CONTRASTING_NEGATIVE_PROMPT)
        assert unconditional_arguments
        assert all(uncond is not None for uncond in unconditional_arguments)


class TestTurboGeneration:
    def test_text_to_image(self, turbo_session):
        _assert_is_a_real_image(turbo_session.generate(PROMPT), MEGAPIXEL_SIDE, MEGAPIXEL_SIDE)

    def test_negative_prompt_is_inert(self, turbo_session):
        """Documents the registry's "effectively CFG-free" claim. If this
        measurement turns out false, record the observed difference and relax
        to a skip with a reason rather than deleting the test."""
        without = turbo_session.generate(PROMPT, negative_prompt='')
        with_negative = turbo_session.generate(PROMPT, negative_prompt=CONTRASTING_NEGATIVE_PROMPT)
        assert _mean_absolute_difference(without, with_negative) < TURBO_NEGATIVE_PROMPT_MAX_MEAN_DIFFERENCE

    def test_skips_the_unconditional_pass(self, turbo_session):
        with _record_unconditional_arguments() as unconditional_arguments:
            turbo_session.generate(PROMPT, negative_prompt=CONTRASTING_NEGATIVE_PROMPT)
        assert unconditional_arguments
        assert all(uncond is None for uncond in unconditional_arguments)

    def test_live_preview_renders_every_step(self, turbo_session):
        previews = []
        turbo_session.generate(PROMPT, disable_preview=False,
                               callback=lambda step, x0, x, total_steps, preview: previews.append(preview))
        assert previews
        assert all(preview is not None and preview.ndim == 3 and preview.shape[-1] == 3
                   for preview in previews)


def _registry_aspect_ratios():
    return get_capabilities(ModelFamily.KREA2_TURBO).aspect_ratios


@pytest.mark.parametrize('aspect_ratio', _registry_aspect_ratios())
def test_every_registry_aspect_ratio_generates_on_turbo(turbo_session, aspect_ratio):
    """The registry offers 1MP and 2K sizes; this is the empirical check that
    each one fits in memory. A size that fails here should be pruned from
    `krea2_aspect_ratios` rather than left in the UI to OOM."""
    width, height = (int(side) for side in aspect_ratio.split('*'))
    _assert_is_a_real_image(turbo_session.generate(PROMPT, width=width, height=height), width, height)
