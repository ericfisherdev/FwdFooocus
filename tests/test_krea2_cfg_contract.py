"""Model-free checks that Krea 2's guidance registry values do what the
registry comment claims (FWDF-160, closing FWDF-152's unverified items).

- Turbo's pinned CFG of exactly 1.0 skips the unconditional pass in BOTH the
  installed `samplers.sampling_function` and the fork's `patched_sampling_function`; Raw's default does
  not. The checks read cfg values from the registry, so changing a value there
  without revisiting the sampler contract fails here.
- A Gradio slider whose minimum equals its maximum (Turbo's `cfg_range` is
  `(1.0, 1.0)`) constructs and round-trips its bounds. Whether a browser
  *renders* such a slider is not testable here.
"""
import math
import os
import sys
import types
from pathlib import Path

import gradio as gr
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

_original_argv = sys.argv
sys.argv = [sys.argv[0]]

import modules.config  # noqa: E402
from ldm_patched.modules import samplers  # noqa: E402
from modules import family_ui_gates  # noqa: E402
from modules.model_family import ModelFamily, get_capabilities  # noqa: E402

sys.argv = _original_argv

from tests._default_pipeline_doubles import install_default_pipeline_test_doubles  # noqa: E402

KREA2_FAMILIES = [ModelFamily.KREA2_RAW, ModelFamily.KREA2_TURBO]

_COND_PREDICTION = torch.full((1, 16, 16, 16), 2.0)
_UNCOND_PREDICTION = torch.full((1, 16, 16, 16), 1.0)


@pytest.fixture(scope='module')
def fork_patch():
    """`modules.patch`, imported under the torchvision stand-in the other
    pipeline-level tests use (see tests/_default_pipeline_doubles.py)."""
    restore = install_default_pipeline_test_doubles()
    try:
        import modules.patch as patch_module
    except Exception:
        restore()
        raise
    yield patch_module
    restore()


class _BatchSpy:
    """Stands in for `calc_cond_uncond_batch`, recording the `uncond`
    argument it receives. Returns fixed predictions so CFG arithmetic is
    deterministic."""

    def __init__(self):
        self.uncond_arguments = []

    def __call__(self, model, cond, uncond, x, timestep, model_options):
        self.uncond_arguments.append(uncond)
        return _COND_PREDICTION.clone(), _UNCOND_PREDICTION.clone()


def _install_spy(monkeypatch):
    """Routes both `calc_cond_uncond_batch` bindings to one spy. `patch_all()`
    (run at `modules.async_worker` import in a full session) replaces
    `samplers.sampling_function` with the fork's version, which reads
    `modules.patch`'s own binding and its per-process settings, so a test that
    wants "the stock sampler" cannot assume which one is installed."""
    spy = _BatchSpy()
    monkeypatch.setattr(samplers, 'calc_cond_uncond_batch', spy)
    fork_patch = sys.modules.get('modules.patch')
    if fork_patch is not None:
        monkeypatch.setattr(fork_patch, 'calc_cond_uncond_batch', spy)
        monkeypatch.setitem(fork_patch.patch_settings, os.getpid(), types.SimpleNamespace(
            eps_record=None, sharpness=0.0, global_diffusion_progress=0.0, adaptive_cfg=7.0))
    return spy


def _sample_with_installed_sampler(monkeypatch, cond_scale):
    spy = _install_spy(monkeypatch)
    x = torch.zeros(1, 16, 16, 16)
    result = samplers.sampling_function(None, x, torch.tensor([1.0]), uncond=['negative'], cond=['positive'],
                                        cond_scale=cond_scale, model_options={})
    return spy, result


def _sample_with_fork_sampler(fork_patch, monkeypatch, cond_scale):
    spy = _install_spy(monkeypatch)
    x = torch.zeros(1, 16, 16, 16)
    result = fork_patch.patched_sampling_function(None, x, torch.tensor([1.0]), uncond=['negative'],
                                                  cond=['positive'], cond_scale=cond_scale, model_options={})
    return spy, result


class TestTurboCfgSkipsTheUnconditionalPass:
    def test_turbo_registry_cfg_is_exactly_the_sampler_skip_condition(self):
        caps = get_capabilities(ModelFamily.KREA2_TURBO)
        for cfg in (caps.default_cfg, *caps.cfg_range):
            assert math.isclose(cfg, 1.0)

    def test_installed_sampler_runs_only_the_conditional_pass_at_turbo_cfg(self, monkeypatch):
        cond_scale = get_capabilities(ModelFamily.KREA2_TURBO).default_cfg
        spy, result = _sample_with_installed_sampler(monkeypatch, cond_scale)
        assert spy.uncond_arguments == [None]
        assert torch.equal(result, _COND_PREDICTION)

    def test_fork_sampler_runs_only_the_conditional_pass_at_turbo_cfg(self, fork_patch, monkeypatch):
        cond_scale = get_capabilities(ModelFamily.KREA2_TURBO).default_cfg
        spy, result = _sample_with_fork_sampler(fork_patch, monkeypatch, cond_scale)
        assert spy.uncond_arguments == [None]
        assert torch.equal(result, _COND_PREDICTION)


class TestRawCfgRunsTheUnconditionalPass:
    def test_installed_sampler_passes_the_negative_cond_at_raw_default_cfg(self, monkeypatch):
        cond_scale = get_capabilities(ModelFamily.KREA2_RAW).default_cfg
        spy, _ = _sample_with_installed_sampler(monkeypatch, cond_scale)
        assert spy.uncond_arguments == [['negative']]

    def test_fork_sampler_passes_the_negative_cond_at_raw_default_cfg(self, fork_patch, monkeypatch):
        cond_scale = get_capabilities(ModelFamily.KREA2_RAW).default_cfg
        spy, _ = _sample_with_fork_sampler(fork_patch, monkeypatch, cond_scale)
        assert spy.uncond_arguments == [['negative']]

    def test_raw_cfg_range_excludes_nothing_below_the_skip_value(self):
        low, high = get_capabilities(ModelFamily.KREA2_RAW).cfg_range
        assert low >= 1.0
        assert high > 1.0


class TestSingleValueCfgSliderConstructs:
    """Turbo hides the negative prompt and pins CFG; the Gradio slider is then
    built with minimum == maximum."""

    def test_family_gate_returns_the_pinned_value_for_any_requested_cfg(self):
        caps = get_capabilities(ModelFamily.KREA2_TURBO)
        for requested in (0.0, 1.0, 7.0, 30.0):
            assert family_ui_gates.guidance_scale_range_and_value(caps, requested) == (1.0, 1.0, 1.0)

    def test_slider_with_equal_bounds_keeps_its_bounds_and_value(self):
        minimum, maximum, value = family_ui_gates.guidance_scale_range_and_value(
            get_capabilities(ModelFamily.KREA2_TURBO), 7.0)
        slider = gr.Slider(label='Guidance Scale', minimum=minimum, maximum=maximum, step=0.01, value=value)
        config = slider.get_config()
        assert (config['minimum'], config['maximum'], config['value']) == (1.0, 1.0, 1.0)

    def test_update_payload_with_equal_bounds_is_accepted(self):
        update = gr.update(minimum=1.0, maximum=1.0, value=1.0)
        assert (update['minimum'], update['maximum'], update['value']) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize('family', KREA2_FAMILIES)
def test_guidance_slider_range_stays_inside_the_global_flag_bounds(family):
    """The slider is constructed with the global `guidance_scale_range`; a
    family's own range must never fall outside it or Gradio would clamp the
    registry default."""
    import modules.flags
    caps = get_capabilities(family)
    low, high = modules.flags.guidance_scale_range
    assert low <= caps.cfg_range[0] <= caps.default_cfg <= caps.cfg_range[1] <= high
