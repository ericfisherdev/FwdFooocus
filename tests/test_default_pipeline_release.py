"""Tests for FWDF-187's model-release path in modules.default_pipeline:

- _release_model(): evicts every patcher owned by a StableDiffusionModel from
  ldm_patched's resident-model cache (unload_model_clones per patcher).
- refresh_base_model()/refresh_refiner_model(): release the outgoing
  checkpoint's patchers, rebind the global to an empty StableDiffusionModel,
  gc.collect(), and soft_empty_cache() -- in that order -- BEFORE loading the
  replacement. Without this, the outgoing checkpoint's sampling clones stay
  pinned in current_loaded_models until the incoming checkpoint's first load
  evicts them lazily, so both checkpoints are resident in RAM/VRAM at once
  during a swap (the multi-minute stall this ticket fixes).
- refresh_everything(): the synthetic-refiner branch releases a stale
  synthetic model_refiner (which aliases model_base's own patchers) before
  refresh_base_model() swaps model_base out from under it.

Import setup and StableDiffusionModel/UNet doubles are shared with
tests/test_zimage_pipeline.py via tests/_default_pipeline_doubles.py, rather
than duplicated here.
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

_original_argv = sys.argv
sys.argv = [sys.argv[0]]

import modules.config  # noqa: E402

sys.argv = _original_argv

from tests._default_pipeline_doubles import (  # noqa: E402
    FakeStableDiffusionModel as _FakeStableDiffusionModel,
    FakeUnet as _FakeUnet,
    install_default_pipeline_test_doubles as _install_default_pipeline_test_doubles,
)


class _FakeEncoderWrapper:
    """Stands in for CLIP/VAE wrappers: default_pipeline._release_model()
    reaches through `.patcher` on clip/clip_with_lora/clip_vision/vae."""

    def __init__(self, patcher=None):
        self.patcher = patcher if patcher is not None else object()


@pytest.fixture(scope='module')
def default_pipeline():
    """The real modules.default_pipeline, imported with just enough test
    doubles for the import to succeed. initialize_default_pipeline() is
    never called here, so model_base/model_refiner/final_* stay at their
    None/empty defaults unless a test sets them directly."""
    restore = _install_default_pipeline_test_doubles()
    try:
        import modules.default_pipeline as pipeline
    except Exception:
        restore()
        raise
    yield pipeline
    restore()


@pytest.fixture(autouse=True)
def _stub_unload_model_clones(default_pipeline, monkeypatch):
    """unload_model_clones() walks ldm_patched's real current_loaded_models
    list; stubbing it isolates these tests from that global process-wide
    state and lets tests assert exactly which patchers were released."""
    mock = MagicMock()
    monkeypatch.setattr(
        default_pipeline.ldm_patched.modules.model_management, 'unload_model_clones', mock)
    return mock


@pytest.fixture(autouse=True)
def _stub_gc_and_cache(default_pipeline, monkeypatch):
    gc_mock = MagicMock()
    cache_mock = MagicMock()
    monkeypatch.setattr(default_pipeline, 'gc', gc_mock)
    monkeypatch.setattr(
        default_pipeline.ldm_patched.modules.model_management, 'soft_empty_cache', cache_mock)
    return gc_mock, cache_mock


# ---------------------------------------------------------------------------
# _release_model()
# ---------------------------------------------------------------------------


class TestReleaseModel:
    def test_unloads_every_owned_patcher(self, default_pipeline, _stub_unload_model_clones):
        unet = _FakeUnet()
        unet_with_lora = _FakeUnet()
        clip = _FakeEncoderWrapper()
        clip_with_lora = _FakeEncoderWrapper()
        clip_vision = _FakeEncoderWrapper()
        vae = _FakeEncoderWrapper()

        model = _FakeStableDiffusionModel(unet=unet, vae=vae, clip=clip)
        model.unet_with_lora = unet_with_lora
        model.clip_with_lora = clip_with_lora
        model.clip_vision = clip_vision

        default_pipeline._release_model(model)

        released = [call.args[0] for call in _stub_unload_model_clones.call_args_list]
        assert released == [
            unet, unet_with_lora, clip.patcher, clip_with_lora.patcher,
            clip_vision.patcher, vae.patcher,
        ]

    def test_skips_absent_components_without_error(self, default_pipeline, _stub_unload_model_clones):
        unet = _FakeUnet()
        model = _FakeStableDiffusionModel(unet=unet)
        model.unet_with_lora = None

        default_pipeline._release_model(model)

        released = [call.args[0] for call in _stub_unload_model_clones.call_args_list]
        assert released == [unet]

    def test_empty_model_releases_nothing(self, default_pipeline, _stub_unload_model_clones):
        default_pipeline._release_model(default_pipeline.core.StableDiffusionModel())

        _stub_unload_model_clones.assert_not_called()


# ---------------------------------------------------------------------------
# refresh_base_model(): release-before-load ordering
# ---------------------------------------------------------------------------


def _pin_resolve_checkpoint_path(pipeline, monkeypatch, mapping):
    """resolve_checkpoint_path() normally searches modules.config's real
    checkpoint directories -- pin it to a fixed name->path mapping so these
    tests don't depend on filesystem/config state, matching each fixture's
    hardcoded StableDiffusionModel.filename values."""
    monkeypatch.setattr(pipeline, 'resolve_checkpoint_path',
                         lambda name, *_a, **_k: mapping[name])


class TestRefreshBaseModelReleaseOrdering:
    def test_releases_outgoing_model_before_loading_replacement(
            self, default_pipeline, monkeypatch, _stub_unload_model_clones, _stub_gc_and_cache):
        pipeline = default_pipeline
        gc_mock, cache_mock = _stub_gc_and_cache

        old_unet = _FakeUnet()
        old_model = _FakeStableDiffusionModel(
            unet=old_unet, filename='/models/checkpoints/old.safetensors', vae_filename=None)
        old_model.unet_with_lora = None
        pipeline.model_base = old_model
        _pin_resolve_checkpoint_path(pipeline, monkeypatch,
                                      {'new.safetensors': '/models/checkpoints/new.safetensors'})

        call_order = []
        _stub_unload_model_clones.side_effect = lambda *_a, **_k: call_order.append('release')
        gc_mock.collect.side_effect = lambda: call_order.append('gc')
        cache_mock.side_effect = lambda: call_order.append('cache')

        new_model = _FakeStableDiffusionModel(unet=_FakeUnet(), filename='/models/checkpoints/new.safetensors')
        load_model_mock = MagicMock(side_effect=lambda *_a, **_k: call_order.append('load') or new_model)
        monkeypatch.setattr(pipeline.core, 'load_model', load_model_mock)
        monkeypatch.setattr(pipeline.modules.model_family_detection, 'get_family',
                             lambda name: pipeline.modules.model_family.ModelFamily.SDXL)

        pipeline.refresh_base_model('new.safetensors')

        assert call_order == ['release', 'gc', 'cache', 'load']
        _stub_unload_model_clones.assert_called_once_with(old_unet)
        assert pipeline.model_base is new_model

    def test_no_op_reload_does_not_release_or_load(
            self, default_pipeline, monkeypatch, _stub_unload_model_clones, _stub_gc_and_cache):
        pipeline = default_pipeline
        gc_mock, cache_mock = _stub_gc_and_cache

        current = _FakeStableDiffusionModel(
            unet=_FakeUnet(), filename='/models/checkpoints/same.safetensors', vae_filename=None)
        pipeline.model_base = current
        _pin_resolve_checkpoint_path(pipeline, monkeypatch,
                                      {'same.safetensors': '/models/checkpoints/same.safetensors'})

        load_model_mock = MagicMock()
        monkeypatch.setattr(pipeline.core, 'load_model', load_model_mock)
        monkeypatch.setattr(pipeline.modules.model_family_detection, 'get_family',
                             lambda name: pipeline.modules.model_family.ModelFamily.SDXL)

        pipeline.refresh_base_model('same.safetensors')

        _stub_unload_model_clones.assert_not_called()
        gc_mock.collect.assert_not_called()
        cache_mock.assert_not_called()
        load_model_mock.assert_not_called()
        assert pipeline.model_base is current


# ---------------------------------------------------------------------------
# refresh_refiner_model(): release-before-load ordering
# ---------------------------------------------------------------------------


class TestRefreshRefinerModelReleaseOrdering:
    def test_releases_outgoing_refiner_before_loading_replacement(
            self, default_pipeline, monkeypatch, _stub_unload_model_clones, _stub_gc_and_cache):
        pipeline = default_pipeline
        gc_mock, cache_mock = _stub_gc_and_cache

        old_unet = _FakeUnet()
        old_refiner = _FakeStableDiffusionModel(
            unet=old_unet, filename='/models/checkpoints/old_refiner.safetensors')
        old_refiner.unet_with_lora = None
        pipeline.model_refiner = old_refiner
        _pin_resolve_checkpoint_path(
            pipeline, monkeypatch, {'new_refiner.safetensors': '/models/checkpoints/new_refiner.safetensors'})

        call_order = []
        _stub_unload_model_clones.side_effect = lambda *_a, **_k: call_order.append('release')
        gc_mock.collect.side_effect = lambda: call_order.append('gc')
        cache_mock.side_effect = lambda: call_order.append('cache')

        new_refiner = _FakeStableDiffusionModel(
            unet=_FakeUnet(), filename='/models/checkpoints/new_refiner.safetensors')
        load_model_mock = MagicMock(side_effect=lambda *_a, **_k: call_order.append('load') or new_refiner)
        monkeypatch.setattr(pipeline.core, 'load_model', load_model_mock)

        pipeline.refresh_refiner_model('new_refiner.safetensors')

        assert call_order == ['release', 'gc', 'cache', 'load']
        _stub_unload_model_clones.assert_called_once_with(old_unet)
        assert pipeline.model_refiner is new_refiner

    def test_switching_to_none_still_releases_outgoing_refiner(
            self, default_pipeline, monkeypatch, _stub_unload_model_clones, _stub_gc_and_cache):
        pipeline = default_pipeline
        gc_mock, cache_mock = _stub_gc_and_cache

        old_unet = _FakeUnet()
        old_refiner = _FakeStableDiffusionModel(
            unet=old_unet, filename='/models/checkpoints/old_refiner.safetensors')
        old_refiner.unet_with_lora = None
        pipeline.model_refiner = old_refiner

        pipeline.refresh_refiner_model('None')

        _stub_unload_model_clones.assert_called_once_with(old_unet)
        gc_mock.collect.assert_called_once()
        cache_mock.assert_called_once()
        assert isinstance(pipeline.model_refiner, pipeline.core.StableDiffusionModel)
        assert pipeline.model_refiner.filename is None

    def test_does_not_evict_live_base_when_outgoing_refiner_aliases_it(
            self, default_pipeline, monkeypatch, _stub_unload_model_clones, _stub_gc_and_cache):
        """A synthetic refiner's unet IS model_base.unet (same object,
        assigned by synthesize_refiner_model()) -- unload_model_clones()
        matches by that identity, so releasing it here would evict the
        still-current base rather than just the (non-existent) refiner."""
        pipeline = default_pipeline
        gc_mock, cache_mock = _stub_gc_and_cache

        live_base_unet = _FakeUnet()
        pipeline.model_base = _FakeStableDiffusionModel(unet=live_base_unet)
        pipeline.model_base.unet_with_lora = None
        pipeline.model_refiner = _FakeStableDiffusionModel(
            unet=live_base_unet, filename='/models/checkpoints/old_base.safetensors')
        pipeline.model_refiner.unet_with_lora = None
        _pin_resolve_checkpoint_path(
            pipeline, monkeypatch, {'real_refiner.safetensors': '/models/checkpoints/real_refiner.safetensors'})

        new_refiner = _FakeStableDiffusionModel(
            unet=_FakeUnet(), filename='/models/checkpoints/real_refiner.safetensors')
        monkeypatch.setattr(pipeline.core, 'load_model', MagicMock(return_value=new_refiner))

        pipeline.refresh_refiner_model('real_refiner.safetensors')

        _stub_unload_model_clones.assert_not_called()
        gc_mock.collect.assert_called_once()
        cache_mock.assert_called_once()
        assert pipeline.model_refiner is new_refiner
        assert pipeline.model_base.unet is live_base_unet


# ---------------------------------------------------------------------------
# refresh_everything(): synthetic refiner released before base model swap
# ---------------------------------------------------------------------------


class TestRefreshEverythingSyntheticRefinerRelease:
    def test_base_swap_releases_old_base_exactly_once_via_refresh_base_model(
            self, default_pipeline, monkeypatch, _stub_unload_model_clones, _stub_gc_and_cache):
        """synthesize_refiner_model() sets model_refiner.unet = model_base.unet
        -- the SAME object, not a copy -- so this models that aliasing
        faithfully (unlike a naive test using two distinct fakes, which
        would not catch the live-base eviction bug this guards against)."""
        pipeline = default_pipeline
        gc_mock, cache_mock = _stub_gc_and_cache

        old_base_unet = _FakeUnet()
        pipeline.model_base = _FakeStableDiffusionModel(
            unet=old_base_unet, filename='/models/checkpoints/old_base.safetensors', vae_filename=None)
        pipeline.model_base.unet_with_lora = None
        pipeline.model_refiner = _FakeStableDiffusionModel(
            unet=old_base_unet, filename='/models/checkpoints/old_base.safetensors')
        pipeline.model_refiner.unet_with_lora = None

        sdxl_family = pipeline.modules.model_family.ModelFamily.SDXL
        monkeypatch.setattr(pipeline.modules.model_family_detection, 'get_family',
                             lambda name: sdxl_family)
        monkeypatch.setattr(
            pipeline.modules.model_family, 'get_capabilities',
            lambda family: types.SimpleNamespace(supports_refiner=True))

        new_base = _FakeStableDiffusionModel(
            unet=_FakeUnet(), vae=object(), clip=object(),
            filename='/models/checkpoints/new_base.safetensors')
        monkeypatch.setattr(pipeline.core, 'load_model', MagicMock(return_value=new_base))
        monkeypatch.setattr(pipeline, 'refresh_loras', MagicMock())
        monkeypatch.setattr(pipeline, 'assert_model_integrity', MagicMock())
        monkeypatch.setattr(pipeline, 'prepare_text_encoder', MagicMock())
        monkeypatch.setattr(pipeline, 'clear_all_caches', MagicMock())

        pipeline.refresh_everything(
            refiner_model_name='None', base_model_name='new_base.safetensors',
            loras=[], use_synthetic_refiner=True)

        released = [call.args[0] for call in _stub_unload_model_clones.call_args_list]
        # The synthetic refiner ALIASES the outgoing base's own UNet object,
        # so refresh_everything() must not explicitly release model_refiner
        # (that would just be releasing the base's own patchers under a
        # different name) -- the old base's UNet is released exactly once,
        # by refresh_base_model()'s own _release_model(model_base) below.
        assert released == [old_base_unet]
        assert pipeline.model_base is new_base
        # synthesize_refiner_model() re-aliases model_refiner to the new
        # base model's own (unset, in this fake) clip/vae/unet.
        assert pipeline.model_refiner.unet is new_base.unet

    def test_repeat_request_with_unchanged_base_does_not_evict_live_base(
            self, default_pipeline, monkeypatch, _stub_unload_model_clones, _stub_gc_and_cache):
        """The common inpaint/upscale/image-prompt path: same base checkpoint,
        synthetic refiner, request after request. Releasing the stale
        synthetic refiner here would evict the still-resident base (it
        aliases the base's own UNet) even though nothing is swapping --
        forcing a needless CPU<->GPU reupload on every single generation."""
        pipeline = default_pipeline

        base_unet = _FakeUnet()
        pipeline.model_base = _FakeStableDiffusionModel(
            unet=base_unet, vae=object(), clip=object(),
            filename='/models/checkpoints/base.safetensors', vae_filename=None)
        pipeline.model_base.unet_with_lora = None
        pipeline.model_refiner = _FakeStableDiffusionModel(
            unet=base_unet, filename='/models/checkpoints/base.safetensors')
        pipeline.model_refiner.unet_with_lora = None

        sdxl_family = pipeline.modules.model_family.ModelFamily.SDXL
        monkeypatch.setattr(pipeline.modules.model_family_detection, 'get_family',
                             lambda name: sdxl_family)
        monkeypatch.setattr(
            pipeline.modules.model_family, 'get_capabilities',
            lambda family: types.SimpleNamespace(supports_refiner=True))
        monkeypatch.setattr(pipeline, 'resolve_checkpoint_path',
                             lambda name, *_a, **_k: '/models/checkpoints/base.safetensors')
        load_model_mock = MagicMock()
        monkeypatch.setattr(pipeline.core, 'load_model', load_model_mock)
        monkeypatch.setattr(pipeline, 'refresh_loras', MagicMock())
        monkeypatch.setattr(pipeline, 'assert_model_integrity', MagicMock())
        monkeypatch.setattr(pipeline, 'prepare_text_encoder', MagicMock())
        monkeypatch.setattr(pipeline, 'clear_all_caches', MagicMock())

        pipeline.refresh_everything(
            refiner_model_name='None', base_model_name='base.safetensors',
            loras=[], use_synthetic_refiner=True)

        _stub_unload_model_clones.assert_not_called()
        load_model_mock.assert_not_called()
        assert pipeline.model_base.unet is base_unet
