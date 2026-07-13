"""Tests for FWDF-127's family-aware pipeline wiring in
modules.default_pipeline:

- refresh_base_model(): Z-Image companion acquisition + Qwen3-4B wiring,
  SDXL path left untouched.
- assert_model_integrity(): family-aware validation, SDXL path byte-identical.
- set_clip_skip(): no-op guard for non-CLIP encoders.
- refresh_everything(): refiner assembly gated by the capability registry.

modules.default_pipeline unconditionally imports a torchvision-only
inpainting architecture chain, which torchvision isn't installed in every
dev/test environment; this file installs the same torchvision stand-in as
tests/test_text_encoder.py to let the real, unmodified pipeline functions run
end-to-end. (It used to also survive a module-level refresh_everything(...)
call that loaded a real checkpoint + GPT-2 prompt expansion model -- FWDF-167
moved that into an explicit initialize_default_pipeline(), which this file
never calls, so importing modules.default_pipeline here no longer touches
disk for model weights.)
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
# Captured before any fixture installs the modules.core stub, so tests that
# need the REAL class (e.g. the refresh_loras invariant) are order-immune.
from modules.core import StableDiffusionModel as RealStableDiffusionModel  # noqa: E402

sys.argv = _original_argv


class _FakeStableDiffusionModel:
    """Stands in for modules.core.StableDiffusionModel: a lightweight
    surface for direct per-test construction of model_base/model_refiner
    fixtures, decoupled from real model loading."""

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
    def __init__(self, model):
        self.model = model


def _install_default_pipeline_test_doubles():
    """Install a torchvision stand-in (when torchvision isn't installed) so
    the real modules.default_pipeline can be imported. Returns a zero-arg
    callable that restores the prior state.

    Mirrors tests/test_text_encoder.py's helper of the same name -- see that
    module's docstring for why ordering matters.
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
    doubles (see _install_default_pipeline_test_doubles) for the import to
    succeed. initialize_default_pipeline() is never called here (FWDF-167),
    so model_base/model_refiner/final_* stay at their None/empty defaults
    unless a test sets them directly -- pipeline.core is the real
    modules.core module."""
    restore = _install_default_pipeline_test_doubles()
    try:
        import modules.default_pipeline as pipeline
    except Exception:
        restore()
        raise
    yield pipeline
    restore()


@pytest.fixture
def z_image_family(default_pipeline):
    return default_pipeline.modules.model_family.ModelFamily.Z_IMAGE


@pytest.fixture
def sdxl_family(default_pipeline):
    return default_pipeline.modules.model_family.ModelFamily.SDXL


# ---------------------------------------------------------------------------
# refresh_base_model(): family routing
# ---------------------------------------------------------------------------


class TestRefreshBaseModelZImageRouting:
    def test_wires_qwen3_encoder_and_acquires_companions(self, default_pipeline, monkeypatch, z_image_family):
        pipeline = default_pipeline

        fake_model = _FakeStableDiffusionModel(
            unet=_FakeUnet(object()), vae=object(), clip=None,
            filename='/models/checkpoints/z_image_turbo.safetensors', vae_filename=None,
        )
        load_model_mock = MagicMock(return_value=fake_model)
        monkeypatch.setattr(pipeline.core, 'load_model', load_model_mock)
        monkeypatch.setattr(pipeline.modules.model_family_detection, 'get_family',
                             lambda name: z_image_family)

        vae_download_mock = MagicMock(return_value='/models/vae/ae.safetensors')
        monkeypatch.setattr(pipeline.modules.config, 'z_image_vae_path',
                             MagicMock(return_value='/models/vae/ae.safetensors'))
        text_encoder_download_mock = MagicMock(return_value='/models/text_encoders/qwen_3_4b.safetensors')
        monkeypatch.setattr(pipeline.modules.config, 'downloading_z_image_vae', vae_download_mock)
        monkeypatch.setattr(pipeline.modules.config, 'downloading_z_image_text_encoder',
                             text_encoder_download_mock)

        sentinel_encoder = object()
        load_qwen3_mock = MagicMock(return_value=sentinel_encoder)
        monkeypatch.setattr(pipeline.modules.qwen3_text_encoder, 'load_qwen3_text_encoder', load_qwen3_mock)

        pipeline.model_base = pipeline.core.StableDiffusionModel()

        pipeline.refresh_base_model('z_image_turbo.safetensors')

        vae_download_mock.assert_called_once()
        load_model_mock.assert_called_once()
        args, kwargs = load_model_mock.call_args
        assert args[1] == '/models/vae/ae.safetensors'

        text_encoder_download_mock.assert_called_once()
        load_qwen3_mock.assert_called_once()
        assert pipeline.model_base.clip is sentinel_encoder
        assert pipeline.model_base.family == z_image_family

    def test_ignores_user_selected_vae_dropdown(self, default_pipeline, monkeypatch, z_image_family):
        """Z-Image's VAE is a mandatory companion download, not overridable
        (supports_vae_override=False in the capability registry): a
        vae_name argument must not change which VAE gets loaded."""
        pipeline = default_pipeline

        load_model_mock = MagicMock(return_value=_FakeStableDiffusionModel(unet=_FakeUnet(object())))
        monkeypatch.setattr(pipeline.core, 'load_model', load_model_mock)
        monkeypatch.setattr(pipeline.modules.model_family_detection, 'get_family',
                             lambda name: z_image_family)
        monkeypatch.setattr(pipeline.modules.config, 'downloading_z_image_vae',
                             MagicMock(return_value='/models/vae/ae.safetensors'))
        monkeypatch.setattr(pipeline.modules.config, 'z_image_vae_path',
                             MagicMock(return_value='/models/vae/ae.safetensors'))
        monkeypatch.setattr(pipeline.modules.config, 'downloading_z_image_text_encoder', MagicMock())
        monkeypatch.setattr(pipeline.modules.qwen3_text_encoder, 'load_qwen3_text_encoder', MagicMock())

        pipeline.model_base = pipeline.core.StableDiffusionModel()

        pipeline.refresh_base_model('z_image_turbo.safetensors', vae_name='some_other_vae.safetensors')

        args, kwargs = load_model_mock.call_args
        assert args[1] == '/models/vae/ae.safetensors'


class TestRefreshBaseModelSdxlRoutingUnchanged:
    def test_does_not_touch_z_image_companions_or_qwen3(self, default_pipeline, monkeypatch, sdxl_family):
        pipeline = default_pipeline

        original_clip = object()
        fake_model = _FakeStableDiffusionModel(
            unet=_FakeUnet(object()), vae=object(), clip=original_clip,
            filename='/models/checkpoints/sdxl_base.safetensors', vae_filename=None,
        )
        load_model_mock = MagicMock(return_value=fake_model)
        monkeypatch.setattr(pipeline.core, 'load_model', load_model_mock)
        monkeypatch.setattr(pipeline.modules.model_family_detection, 'get_family',
                             lambda name: sdxl_family)

        vae_download_mock = MagicMock()
        text_encoder_download_mock = MagicMock()
        load_qwen3_mock = MagicMock()
        monkeypatch.setattr(pipeline.modules.config, 'downloading_z_image_vae', vae_download_mock)
        monkeypatch.setattr(pipeline.modules.config, 'downloading_z_image_text_encoder',
                             text_encoder_download_mock)
        monkeypatch.setattr(pipeline.modules.qwen3_text_encoder, 'load_qwen3_text_encoder', load_qwen3_mock)

        pipeline.model_base = pipeline.core.StableDiffusionModel()

        pipeline.refresh_base_model('sdxl_base.safetensors')

        vae_download_mock.assert_not_called()
        text_encoder_download_mock.assert_not_called()
        load_qwen3_mock.assert_not_called()
        assert pipeline.model_base.clip is original_clip
        assert pipeline.model_base.family == sdxl_family

    def test_selected_vae_dropdown_still_resolved_via_folder_lookup(self, default_pipeline, monkeypatch,
                                                                      sdxl_family):
        pipeline = default_pipeline

        load_model_mock = MagicMock(return_value=_FakeStableDiffusionModel(unet=_FakeUnet(object())))
        monkeypatch.setattr(pipeline.core, 'load_model', load_model_mock)
        monkeypatch.setattr(pipeline.modules.model_family_detection, 'get_family',
                             lambda name: sdxl_family)

        pipeline.model_base = pipeline.core.StableDiffusionModel()

        pipeline.refresh_base_model('sdxl_base.safetensors', vae_name='my_vae.safetensors')

        expected_vae_path = pipeline.get_file_from_folder_list('my_vae.safetensors', pipeline.modules.config.path_vae)
        args, kwargs = load_model_mock.call_args
        assert args[1] == expected_vae_path


# ---------------------------------------------------------------------------
# refresh_controlnets(): explicit family routing (FWDF-156 follow-up fix)
#
# refresh_controlnets() used to dispatch on pipeline.model_base.family --
# the currently *loaded* model's family, stale until refresh_everything()
# (called later, from process_prompt() in modules/async_worker.py) catches
# up. These tests pin model_base.family to one family while passing a
# *different* family explicitly, proving routing follows the argument, not
# pipeline state.
# ---------------------------------------------------------------------------


class TestRefreshControlnetsFamilyRouting:
    @pytest.fixture(autouse=True)
    def _reset_controlnet_cache(self, default_pipeline):
        """loaded_ControlNets is a module-level cache keyed by path; a path
        reused across tests must not short-circuit the loader-dispatch logic
        this test class exists to verify."""
        default_pipeline.loaded_ControlNets = {}
        yield
        default_pipeline.loaded_ControlNets = {}

    def test_z_image_family_argument_chooses_zimage_loader_while_model_base_holds_sdxl(
            self, default_pipeline, monkeypatch, sdxl_family, z_image_family):
        pipeline = default_pipeline
        pipeline.model_base = pipeline.core.StableDiffusionModel()
        pipeline.model_base.family = sdxl_family  # stale: previous request's family

        load_zimage_mock = MagicMock(return_value=object())
        load_sdxl_mock = MagicMock(return_value=object())
        monkeypatch.setattr(pipeline.core, 'load_controlnet_zimage', load_zimage_mock, raising=False)
        monkeypatch.setattr(pipeline.core, 'load_controlnet', load_sdxl_mock, raising=False)

        pipeline.refresh_controlnets(['/models/controlnet/z_image_canny.safetensors'], z_image_family)

        load_zimage_mock.assert_called_once_with('/models/controlnet/z_image_canny.safetensors')
        load_sdxl_mock.assert_not_called()

    def test_sdxl_family_argument_chooses_sdxl_loader_while_model_base_holds_z_image(
            self, default_pipeline, monkeypatch, sdxl_family, z_image_family):
        pipeline = default_pipeline
        pipeline.model_base = pipeline.core.StableDiffusionModel()
        pipeline.model_base.family = z_image_family  # stale: previous request's family

        load_zimage_mock = MagicMock(return_value=object())
        load_sdxl_mock = MagicMock(return_value=object())
        monkeypatch.setattr(pipeline.core, 'load_controlnet_zimage', load_zimage_mock, raising=False)
        monkeypatch.setattr(pipeline.core, 'load_controlnet', load_sdxl_mock, raising=False)

        pipeline.refresh_controlnets(['/models/controlnet/canny_sdxl.safetensors'], sdxl_family)

        load_sdxl_mock.assert_called_once_with('/models/controlnet/canny_sdxl.safetensors')
        load_zimage_mock.assert_not_called()

    def test_none_paths_are_skipped_regardless_of_family(self, default_pipeline, monkeypatch, z_image_family):
        pipeline = default_pipeline

        load_zimage_mock = MagicMock(return_value=object())
        monkeypatch.setattr(pipeline.core, 'load_controlnet_zimage', load_zimage_mock, raising=False)

        pipeline.refresh_controlnets([None, None], z_image_family)

        load_zimage_mock.assert_not_called()
        assert pipeline.loaded_ControlNets == {}

    def test_cached_path_is_reused_without_reloading(self, default_pipeline, monkeypatch, sdxl_family):
        pipeline = default_pipeline
        cached_controlnet = object()
        pipeline.loaded_ControlNets = {'/models/controlnet/canny_sdxl.safetensors': cached_controlnet}

        load_sdxl_mock = MagicMock(return_value=object())
        monkeypatch.setattr(pipeline.core, 'load_controlnet', load_sdxl_mock, raising=False)

        pipeline.refresh_controlnets(['/models/controlnet/canny_sdxl.safetensors'], sdxl_family)

        load_sdxl_mock.assert_not_called()
        assert pipeline.loaded_ControlNets['/models/controlnet/canny_sdxl.safetensors'] is cached_controlnet


# ---------------------------------------------------------------------------
# assert_model_integrity(): family-aware validation
# ---------------------------------------------------------------------------


class TestAssertModelIntegrity:
    def test_sdxl_path_passes_for_sdxl_model(self, default_pipeline, sdxl_family):
        from ldm_patched.modules.model_base import SDXL

        pipeline = default_pipeline
        pipeline.model_base = _FakeStableDiffusionModel(unet=_FakeUnet(SDXL.__new__(SDXL)), clip=object(), vae=object())
        pipeline.model_base.family = sdxl_family

        assert pipeline.assert_model_integrity() is True

    def test_sdxl_path_raises_for_non_sdxl_model(self, default_pipeline, sdxl_family):
        pipeline = default_pipeline
        pipeline.model_base = _FakeStableDiffusionModel(unet=_FakeUnet(object()), clip=object(), vae=object())
        pipeline.model_base.family = sdxl_family

        with pytest.raises(NotImplementedError):
            pipeline.assert_model_integrity()

    def test_z_image_path_passes_when_dit_clip_and_vae_all_present(self, default_pipeline, z_image_family):
        from ldm_patched.modules.model_base import ZImage

        pipeline = default_pipeline
        pipeline.model_base = _FakeStableDiffusionModel(
            unet=_FakeUnet(ZImage.__new__(ZImage)), clip=object(), vae=object(),
        )
        pipeline.model_base.family = z_image_family

        assert pipeline.assert_model_integrity() is True

    def test_z_image_path_raises_for_wrong_dit_class(self, default_pipeline, z_image_family):
        from ldm_patched.modules.model_base import SDXL

        pipeline = default_pipeline
        pipeline.model_base = _FakeStableDiffusionModel(unet=_FakeUnet(SDXL.__new__(SDXL)), clip=object(), vae=object())
        pipeline.model_base.family = z_image_family

        with pytest.raises(NotImplementedError):
            pipeline.assert_model_integrity()

    def test_z_image_path_raises_when_clip_missing(self, default_pipeline, z_image_family):
        from ldm_patched.modules.model_base import ZImage

        pipeline = default_pipeline
        pipeline.model_base = _FakeStableDiffusionModel(
            unet=_FakeUnet(ZImage.__new__(ZImage)), clip=None, vae=object(),
        )
        pipeline.model_base.family = z_image_family

        with pytest.raises(NotImplementedError):
            pipeline.assert_model_integrity()

    def test_z_image_path_raises_when_vae_missing(self, default_pipeline, z_image_family):
        from ldm_patched.modules.model_base import ZImage

        pipeline = default_pipeline
        pipeline.model_base = _FakeStableDiffusionModel(
            unet=_FakeUnet(ZImage.__new__(ZImage)), clip=object(), vae=None,
        )
        pipeline.model_base.family = z_image_family

        with pytest.raises(NotImplementedError):
            pipeline.assert_model_integrity()

    def test_missing_family_attribute_falls_back_to_sdxl_check(self, default_pipeline):
        """model_base objects constructed before FWDF-127 (or by any code
        path that doesn't set .family) must keep today's exact behavior."""
        from ldm_patched.modules.model_base import SDXL

        pipeline = default_pipeline
        pipeline.model_base = _FakeStableDiffusionModel(unet=_FakeUnet(SDXL.__new__(SDXL)), clip=object(), vae=object())
        assert not hasattr(pipeline.model_base, 'family')

        assert pipeline.assert_model_integrity() is True


# ---------------------------------------------------------------------------
# set_clip_skip(): no-op guard for non-CLIP encoders
# ---------------------------------------------------------------------------


class TestSetClipSkip:
    def test_calls_clip_layer_for_clip_like_encoder(self, default_pipeline):
        pipeline = default_pipeline
        clip_like = MagicMock()
        pipeline.final_clip = clip_like

        pipeline.set_clip_skip(3)

        clip_like.clip_layer.assert_called_once_with(-3)

    def test_no_op_for_encoder_without_clip_layer(self, default_pipeline):
        pipeline = default_pipeline

        class NoClipLayerEncoder:
            pass

        encoder = NoClipLayerEncoder()
        pipeline.final_clip = encoder

        pipeline.set_clip_skip(3)  # must not raise

    def test_no_op_when_final_clip_is_none(self, default_pipeline):
        pipeline = default_pipeline
        pipeline.final_clip = None

        pipeline.set_clip_skip(3)  # must not raise


# ---------------------------------------------------------------------------
# refresh_everything(): refiner assembly gated by capability registry
# ---------------------------------------------------------------------------


class TestRefreshEverythingRefinerCapabilityGate:
    def _stub_refresh_everything_collaborators(self, pipeline, monkeypatch):
        monkeypatch.setattr(pipeline, 'refresh_base_model', MagicMock())
        monkeypatch.setattr(pipeline, 'refresh_loras', MagicMock())
        monkeypatch.setattr(pipeline, 'assert_model_integrity', MagicMock(return_value=True))
        monkeypatch.setattr(pipeline, 'prepare_text_encoder', MagicMock())
        monkeypatch.setattr(pipeline, 'clear_all_caches', MagicMock())
        pipeline.model_base = _FakeStableDiffusionModel(unet=_FakeUnet(object()), clip=object(), vae=object())
        pipeline.model_refiner = _FakeStableDiffusionModel(unet=_FakeUnet(object()), vae=object())
        # refresh_everything() only constructs FooocusExpansion() when
        # final_expansion is still None; pin a sentinel so this direct call
        # doesn't try to load the real GPT-2 prompt-expansion model from disk.
        monkeypatch.setattr(pipeline, 'final_expansion', object())

    def test_refiner_forced_to_none_when_family_does_not_support_it(self, default_pipeline, monkeypatch,
                                                                      z_image_family):
        pipeline = default_pipeline
        monkeypatch.setattr(pipeline.modules.model_family_detection, 'get_family', lambda name: z_image_family)

        refresh_refiner_mock = MagicMock()
        synthesize_mock = MagicMock()
        monkeypatch.setattr(pipeline, 'refresh_refiner_model', refresh_refiner_mock)
        monkeypatch.setattr(pipeline, 'synthesize_refiner_model', synthesize_mock)
        self._stub_refresh_everything_collaborators(pipeline, monkeypatch)

        pipeline.refresh_everything(
            refiner_model_name='some_refiner.safetensors',
            base_model_name='z_image_turbo.safetensors',
            loras=[],
            use_synthetic_refiner=True,
        )

        refresh_refiner_mock.assert_called_once_with('None')
        synthesize_mock.assert_not_called()

    def test_refiner_untouched_when_family_supports_it(self, default_pipeline, monkeypatch, sdxl_family):
        pipeline = default_pipeline
        monkeypatch.setattr(pipeline.modules.model_family_detection, 'get_family', lambda name: sdxl_family)

        refresh_refiner_mock = MagicMock()
        monkeypatch.setattr(pipeline, 'refresh_refiner_model', refresh_refiner_mock)
        self._stub_refresh_everything_collaborators(pipeline, monkeypatch)

        pipeline.refresh_everything(
            refiner_model_name='sdxl_refiner.safetensors',
            base_model_name='sdxl_base.safetensors',
            loras=[],
        )

        refresh_refiner_mock.assert_called_once_with('sdxl_refiner.safetensors')


class TestRefreshLorasCloneInvariant:
    def test_patchable_but_unclonable_encoder_is_rejected(self, monkeypatch):
        """An encoder with add_patches() but no clone() would accumulate LoRA
        patches on the shared instance across refreshes; refresh_loras() must
        fail loudly instead of aliasing."""
        import torch.nn as nn

        class _PatchableUnclonableClip:
            cond_stage_model = nn.Linear(2, 2)

            def add_patches(self, patches, weight):
                return []

        class _FakeUnet:
            model = nn.Linear(2, 2)

            def clone(self):
                return self

        # Bypass __init__ (it builds LoRA key maps from real model configs);
        # the invariant under test lives in refresh_loras().
        model = RealStableDiffusionModel.__new__(RealStableDiffusionModel)
        model.unet = _FakeUnet()
        model.vae = object()
        model.clip = _PatchableUnclonableClip()
        model.clip_vision = None
        model.filename = 'fake.safetensors'
        model.vae_filename = None
        model.unet_with_lora = model.unet
        model.clip_with_lora = model.clip
        model.visited_loras = ''
        model.lora_key_map_unet = {}
        model.lora_key_map_clip = {}

        with pytest.raises(TypeError, match="add_patches\\(\\) without clone\\(\\)"):
            model.refresh_loras([('some_lora.safetensors', 1.0)])
