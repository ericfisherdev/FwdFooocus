"""Tests for modules.patch.patched_KSamplerX0Inpaint_forward's masking
mechanism (FWDF-155).

modules/patch.py's patch_all() replaces
ldm_patched.modules.samplers.KSamplerX0Inpaint.forward with
patched_KSamplerX0Inpaint_forward *unconditionally* at modules.async_worker
import time. That replacement never reads its own `denoise_mask` parameter --
it enforces the inpaint mask purely from the module-global
modules.inpaint_worker.current_task.latent/.latent_mask (set by
InpaintWorker.load_latent(), which modules/async_worker.py's apply_inpaint()
calls for every family, unconditionally). This is the mechanism that actually
keeps the unmasked region pixel-preserved during denoising for *any* family --
SDXL or DiT alike -- since it operates via plain tensor broadcasting between
a (B, 1, H, W) mask and a (B, C, H, W) latent, with no channel-count
assumption anywhere in the function body.

This is why FWDF-155's fix only needs to skip the SDXL-specific InpaintHead
patch() call for families without a learned inpaint-engine head (see
modules/async_worker.py's _inpaint_family_lacks_engine_head()) -- the masking
itself already works for a 16-channel DiT latent with zero additional
plumbing.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

_original_argv = sys.argv
sys.argv = [sys.argv[0]]
try:

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

    from modules import inpaint_worker, patch  # noqa: E402

finally:
    sys.argv = _original_argv



@pytest.fixture(scope='module', autouse=True)
def _remove_installed_stubs_after_module():
    """Pop the torchvision stand-ins this module installed once its tests
    finish, so they cannot leak into later test modules in the session."""
    yield
    for name in _installed_stub_names:
        sys.modules.pop(name, None)


class _FakeRealModel:
    """Stands in for the unwrapped model (real_model.model), whose
    process_latent_in() is architecture-generic (ldm_patched.modules.model_base
    .BaseModel.process_latent_in() delegates to self.latent_format.process_in(),
    inherited unchanged by ZImage(BaseModel)). Identity here since this test
    only needs to prove the mask broadcast is channel-agnostic, not exercise
    a real latent format's normalization.
    """

    def process_latent_in(self, latent):
        return latent


class _FakeInnerModel:
    """Stands in for CFGNoisePredictor: self.inner_model.inner_model is the
    real model above; calling self.inner_model(...) returns a fixed,
    distinguishable tensor representing "what the denoiser predicted"."""

    def __init__(self, output):
        self.inner_model = _FakeRealModel()
        self._output = output

    def __call__(self, x, sigma, cond, uncond, cond_scale, model_options, seed):
        return self._output


@pytest.fixture(autouse=True)
def _reset_inpaint_current_task():
    original = inpaint_worker.current_task
    yield
    inpaint_worker.current_task = original


def _run_forward(channels, mask_covers_last_columns):
    """Builds a (1, channels, 4, 4) latent scenario: the last
    `mask_covers_last_columns` columns are masked (should regenerate to the
    model's output); the rest are unmasked (should be pinned to the original
    fill latent) -- and calls the real, patched forward."""
    batch, height, width = 1, 4, 4
    fill_latent = torch.full((batch, channels, height, width), 2.0)
    mask = torch.zeros((batch, 1, height, width))
    if mask_covers_last_columns > 0:
        mask[:, :, :, -mask_covers_last_columns:] = 1.0

    generated_output = torch.full((batch, channels, height, width), 9.0)

    inpaint_worker.current_task = types.SimpleNamespace(latent=fill_latent, latent_mask=mask)

    self_obj = types.SimpleNamespace(inner_model=_FakeInnerModel(generated_output))
    x = torch.zeros((batch, channels, height, width))
    sigma = torch.tensor([1.0])

    out = patch.patched_KSamplerX0Inpaint_forward(
        self_obj, x, sigma, uncond=[], cond=[], cond_scale=1.0, denoise_mask=None, seed=0)

    return out, fill_latent, generated_output, mask


class TestMaskingIsChannelAgnostic:
    """The mask broadcast must hold for both SDXL's 4-channel latents and a
    DiT family's 16-channel latents -- no channel-count literal anywhere in
    the masking math."""

    def test_sixteen_channel_latent_unmasked_region_pinned_to_original(self):
        out, fill_latent, generated_output, mask = _run_forward(channels=16, mask_covers_last_columns=2)

        self.assertions_hold(out, fill_latent, generated_output, mask)

    def test_four_channel_latent_unmasked_region_pinned_to_original(self):
        out, fill_latent, generated_output, mask = _run_forward(channels=4, mask_covers_last_columns=2)

        self.assertions_hold(out, fill_latent, generated_output, mask)

    @staticmethod
    def assertions_hold(out, fill_latent, generated_output, mask):
        assert out.shape == fill_latent.shape
        masked = mask.expand_as(out).bool()
        # Masked region: hard-pinned to the model's generated prediction.
        assert torch.allclose(out[masked], generated_output[masked])
        # Unmasked region: hard-pinned to the original fill latent, i.e.
        # pixel/latent-preserved regardless of what the model produced there.
        assert torch.allclose(out[~masked], fill_latent[~masked])


class TestNoOpWithoutActiveInpaintTask:
    """When no inpainting is in progress, the patch must be a pure passthrough
    to the wrapped model -- unaffected by this ticket's change."""

    def test_returns_inner_model_output_directly(self):
        inpaint_worker.current_task = None

        generated_output = torch.full((1, 16, 4, 4), 5.0)
        self_obj = types.SimpleNamespace(inner_model=_FakeInnerModel(generated_output))
        x = torch.zeros((1, 16, 4, 4))
        sigma = torch.tensor([1.0])

        out = patch.patched_KSamplerX0Inpaint_forward(
            self_obj, x, sigma, uncond=[], cond=[], cond_scale=1.0, denoise_mask=None, seed=0)

        assert torch.equal(out, generated_output)


if __name__ == '__main__':
    import unittest
    unittest.main()
