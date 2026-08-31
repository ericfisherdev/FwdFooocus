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

Mirrors tests/test_zimage_pipeline.py's approach to importing the real,
unmodified modules.default_pipeline with just enough test doubles for the
import to succeed in a torch/torchvision-optional sandbox.
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


class _FakeStableDiffusionModel:
    """Stands in for modules.core.StableDiffusionModel: a lightweight surface
    for direct per-test construction of model_base/model_refiner fixtures."""

    def __init__(self, unet=None, vae=None, clip=None, clip_vision=None,
                 filename=None, vae_filename=None):
        self.unet = unet
        self.vae = vae
        self.clip = clip
        self.clip_vision = clip_vision
        self.filename = filename
        self.vae_filename = vae_filename
        self.unet_with_lora = unet
        self.clip_with_lora = clip

    def refresh_loras(self, loras):
        pass


class _FakeUnet:
    def __init__(self, model=None):
        self.model = model if model is not None else object()


class _FakeEncoderWrapper:
    """Stands in for CLIP/VAE wrappers: default_pipeline._release_model()
    reaches through `.patcher` on clip/clip_with_lora/clip_vision/vae."""

    def __init__(self, patcher=None):
        self.patcher = patcher if patcher is not None else object()


def _install_default_pipeline_test_doubles():
    """Install a torchvision stand-in (when torchvision isn't installed) so
    the real modules.default_pipeline can be imported. Returns a zero-arg
    callable that restores the prior state.

    Mirrors tests/test_zimage_pipeline.py's helper of the same name.
    """
    import transformers  # noqa: F401  (forces the real torchvision-unavailable check first)

    restore_actions = []

    torchvision_available = True
    try:
        import torchvision  # noqa: F401
    except ImportError:
        torchvision_available = False

    if not torchvision_available:
        stub_names = ('torchvision', 'torchvision.transforms', 'torchvision.transforms.functional')
        for name in stub_names:
            assert name not in sys.modules, f'unexpected pre-existing stub conflict for {name}'
        functional_stub = types.ModuleType('torchvision.transforms.functional')
        functional_stub.InterpolationMode = object
        functional_stub.rotate = lambda *a, **k: None
        transforms_stub = types.ModuleType('torchvision.transforms')
        transforms_stub.functional = functional_stub
        torchvision_stub = types.ModuleType('torchvision')
        torchvision_stub.transforms = transforms_stub

        sys.modules['torchvision'] = torchvision_stub
        sys.modules['torchvision.transforms'] = transforms_stub
        sys.modules['torchvision.transforms.functional'] = functional_stub
        restore_actions.append(lambda: [sys.modules.pop(n, None) for n in stub_names])

    def _restore():
        for action in reversed(restore_actions):
            action()

    return _restore


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


# ---------------------------------------------------------------------------
# refresh_everything(): synthetic refiner released before base model swap
# ---------------------------------------------------------------------------


class TestRefreshEverythingSyntheticRefinerRelease:
    def test_releases_stale_synthetic_refiner_before_base_model_swap(
            self, default_pipeline, monkeypatch, _stub_unload_model_clones, _stub_gc_and_cache):
        pipeline = default_pipeline
        gc_mock, cache_mock = _stub_gc_and_cache

        stale_refiner_unet = _FakeUnet()
        old_base_unet = _FakeUnet()
        pipeline.model_refiner = _FakeStableDiffusionModel(
            unet=stale_refiner_unet, filename='/models/checkpoints/old_base.safetensors')
        pipeline.model_refiner.unet_with_lora = None
        pipeline.model_base = _FakeStableDiffusionModel(
            unet=old_base_unet, filename='/models/checkpoints/old_base.safetensors', vae_filename=None)
        pipeline.model_base.unet_with_lora = None

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
        # The stale synthetic refiner's UNet must be released before the
        # OLD base model's own UNet -- proving refresh_everything() clears
        # the synthetic model_refiner first, rather than leaving it aliasing
        # patchers that refresh_base_model() is about to swap out from
        # under it. Neither call releases the incoming new_base.unet.
        assert released == [stale_refiner_unet, old_base_unet]
        assert pipeline.model_base is new_base
        # synthesize_refiner_model() re-aliases model_refiner to the new
        # base model's own (unset, in this fake) clip/vae/unet.
        assert pipeline.model_refiner.unet is new_base.unet
