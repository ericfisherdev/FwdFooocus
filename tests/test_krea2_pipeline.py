"""Tests for Krea 2's pipeline assembly (FWDF-152).

`modules.default_pipeline` assembles multi-file families (Z-Image, Krea 2)
from a family-keyed table rather than per-family `if` branches, so Krea 2
needs no pipeline gate of its own. These tests exercise that table through the
real `refresh_base_model()` / `assert_model_integrity()` with config helpers,
the model loader and the text-encoder loader monkeypatched, mirroring
tests/test_zimage_pipeline.py (whose module docstring explains the
torchvision stand-in the shared fixtures install).
"""
import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

_original_argv = sys.argv
sys.argv = [sys.argv[0]]

import modules.config  # noqa: E402
from modules.model_family import ModelFamily  # noqa: E402

sys.argv = _original_argv

from tests._default_pipeline_doubles import (  # noqa: E402
    FakeStableDiffusionModel as _FakeStableDiffusionModel,
    FakeUnet as _FakeUnet,
    install_default_pipeline_test_doubles as _install_default_pipeline_test_doubles,
)

KREA2_FAMILIES = [ModelFamily.KREA2_RAW, ModelFamily.KREA2_TURBO]


@pytest.fixture(scope='module')
def default_pipeline():
    restore = _install_default_pipeline_test_doubles()
    try:
        import modules.default_pipeline as pipeline
    except Exception:
        restore()
        raise
    yield pipeline
    restore()


@pytest.fixture
def krea2_companion_doubles(default_pipeline, monkeypatch):
    """Monkeypatches the Krea 2 companion-file helpers and the encoder loader,
    returning the mocks so a test can assert on how they were used."""
    pipeline = default_pipeline
    doubles = types_namespace(
        vae_path=MagicMock(return_value='/models/vae/qwen_image_vae.safetensors'),
        download_vae=MagicMock(return_value='/models/vae/qwen_image_vae.safetensors'),
        download_text_encoder=MagicMock(return_value='/models/text_encoders/qwen3vl_4b_bf16.safetensors'),
        load_text_encoder=MagicMock(return_value=object()),
    )
    monkeypatch.setattr(pipeline.modules.config, 'qwen_image_vae_path', doubles.vae_path)
    monkeypatch.setattr(pipeline.modules.config, 'downloading_qwen_image_vae', doubles.download_vae)
    monkeypatch.setattr(pipeline.modules.config, 'downloading_krea2_text_encoder', doubles.download_text_encoder)
    monkeypatch.setattr(pipeline.modules.krea2_text_encoder, 'load_krea2_text_encoder', doubles.load_text_encoder)
    return doubles


def types_namespace(**values):
    import types
    return types.SimpleNamespace(**values)


def _loaded_model(clip=None):
    return _FakeStableDiffusionModel(
        unet=_FakeUnet(object()), vae=object(), clip=clip,
        filename='/models/checkpoints/krea2.safetensors', vae_filename=None,
    )


class TestRefreshBaseModelKrea2Routing:
    @pytest.mark.parametrize('family', KREA2_FAMILIES)
    def test_wires_the_qwen3_vl_encoder_and_acquires_companions(
            self, default_pipeline, monkeypatch, krea2_companion_doubles, family):
        pipeline = default_pipeline
        load_model_mock = MagicMock(return_value=_loaded_model())
        monkeypatch.setattr(pipeline.core, 'load_model', load_model_mock)
        monkeypatch.setattr(pipeline.modules.model_family_detection, 'get_family', lambda name: family)
        pipeline.model_base = pipeline.core.StableDiffusionModel()

        pipeline.refresh_base_model('krea2.safetensors')

        krea2_companion_doubles.download_vae.assert_called_once()
        krea2_companion_doubles.download_text_encoder.assert_called_once()
        krea2_companion_doubles.load_text_encoder.assert_called_once()
        args, _ = load_model_mock.call_args
        assert args[1] == '/models/vae/qwen_image_vae.safetensors'
        assert pipeline.model_base.clip is krea2_companion_doubles.load_text_encoder.return_value
        assert pipeline.model_base.family == family

    def test_ignores_the_user_selected_vae_dropdown(
            self, default_pipeline, monkeypatch, krea2_companion_doubles):
        pipeline = default_pipeline
        load_model_mock = MagicMock(return_value=_loaded_model())
        monkeypatch.setattr(pipeline.core, 'load_model', load_model_mock)
        monkeypatch.setattr(pipeline.modules.model_family_detection, 'get_family',
                            lambda name: ModelFamily.KREA2_TURBO)
        pipeline.model_base = pipeline.core.StableDiffusionModel()

        pipeline.refresh_base_model('krea2.safetensors', vae_name='some_other_vae.safetensors')

        args, _ = load_model_mock.call_args
        assert args[1] == '/models/vae/qwen_image_vae.safetensors'

    def test_keeps_an_encoder_the_checkpoint_loader_already_built(
            self, default_pipeline, monkeypatch, krea2_companion_doubles):
        """If supported_models.Krea2.clip_target() ever returns a real target,
        core.load_model() builds the encoder itself and the table must not
        replace it."""
        pipeline = default_pipeline
        built_by_checkpoint_loader = object()
        monkeypatch.setattr(pipeline.core, 'load_model',
                            MagicMock(return_value=_loaded_model(clip=built_by_checkpoint_loader)))
        monkeypatch.setattr(pipeline.modules.model_family_detection, 'get_family',
                            lambda name: ModelFamily.KREA2_RAW)
        pipeline.model_base = pipeline.core.StableDiffusionModel()

        pipeline.refresh_base_model('krea2.safetensors')

        krea2_companion_doubles.load_text_encoder.assert_not_called()
        assert pipeline.model_base.clip is built_by_checkpoint_loader

    def test_unchanged_checkpoint_short_circuits_before_any_download(
            self, default_pipeline, monkeypatch, krea2_companion_doubles):
        pipeline = default_pipeline
        monkeypatch.setattr(pipeline.modules.model_family_detection, 'get_family',
                            lambda name: ModelFamily.KREA2_RAW)
        monkeypatch.setattr(pipeline, 'resolve_checkpoint_path', lambda *args: '/models/checkpoints/krea2.safetensors')
        load_model_mock = MagicMock()
        monkeypatch.setattr(pipeline.core, 'load_model', load_model_mock)
        pipeline.model_base = _FakeStableDiffusionModel(
            filename='/models/checkpoints/krea2.safetensors',
            vae_filename='/models/vae/qwen_image_vae.safetensors')

        pipeline.refresh_base_model('krea2.safetensors')

        load_model_mock.assert_not_called()
        krea2_companion_doubles.download_vae.assert_not_called()
        krea2_companion_doubles.download_text_encoder.assert_not_called()

    def test_sdxl_does_not_touch_krea2_companions(self, default_pipeline, monkeypatch, krea2_companion_doubles):
        pipeline = default_pipeline
        original_clip = object()
        monkeypatch.setattr(pipeline.core, 'load_model', MagicMock(return_value=_loaded_model(clip=original_clip)))
        monkeypatch.setattr(pipeline.modules.model_family_detection, 'get_family',
                            lambda name: ModelFamily.SDXL)
        pipeline.model_base = pipeline.core.StableDiffusionModel()

        pipeline.refresh_base_model('sdxl_base.safetensors')

        krea2_companion_doubles.download_vae.assert_not_called()
        krea2_companion_doubles.download_text_encoder.assert_not_called()
        krea2_companion_doubles.load_text_encoder.assert_not_called()
        assert pipeline.model_base.clip is original_clip


class TestAssertModelIntegrityKrea2:
    @pytest.fixture
    def krea2_model_class(self):
        from ldm_patched.modules.model_base import Krea2
        return Krea2

    @pytest.mark.parametrize('family', KREA2_FAMILIES)
    def test_passes_when_dit_clip_and_vae_are_present(self, default_pipeline, krea2_model_class, family):
        pipeline = default_pipeline
        pipeline.model_base = _FakeStableDiffusionModel(
            unet=_FakeUnet(krea2_model_class.__new__(krea2_model_class)), clip=object(), vae=object())
        pipeline.model_base.family = family

        assert pipeline.assert_model_integrity() is True

    def test_rejects_a_z_image_dit(self, default_pipeline):
        from ldm_patched.modules.model_base import ZImage
        pipeline = default_pipeline
        pipeline.model_base = _FakeStableDiffusionModel(
            unet=_FakeUnet(ZImage.__new__(ZImage)), clip=object(), vae=object())
        pipeline.model_base.family = ModelFamily.KREA2_RAW

        with pytest.raises(NotImplementedError, match='Krea 2 base model did not load'):
            pipeline.assert_model_integrity()

    def test_requires_the_text_encoder(self, default_pipeline, krea2_model_class):
        pipeline = default_pipeline
        pipeline.model_base = _FakeStableDiffusionModel(
            unet=_FakeUnet(krea2_model_class.__new__(krea2_model_class)), clip=None, vae=object())
        pipeline.model_base.family = ModelFamily.KREA2_TURBO

        with pytest.raises(NotImplementedError, match='Qwen3-VL-4B text encoder'):
            pipeline.assert_model_integrity()

    def test_requires_the_vae(self, default_pipeline, krea2_model_class):
        pipeline = default_pipeline
        pipeline.model_base = _FakeStableDiffusionModel(
            unet=_FakeUnet(krea2_model_class.__new__(krea2_model_class)), clip=object(), vae=None)
        pipeline.model_base.family = ModelFamily.KREA2_TURBO

        with pytest.raises(NotImplementedError, match='standalone VAE'):
            pipeline.assert_model_integrity()

    def test_z_image_messages_keep_their_wording(self, default_pipeline):
        from ldm_patched.modules.model_base import ZImage
        pipeline = default_pipeline
        pipeline.model_base = _FakeStableDiffusionModel(
            unet=_FakeUnet(ZImage.__new__(ZImage)), clip=None, vae=object())
        pipeline.model_base.family = ModelFamily.Z_IMAGE

        with pytest.raises(NotImplementedError) as excinfo:
            pipeline.assert_model_integrity()

        assert str(excinfo.value) == 'Z-Image requires the Qwen3-4B text encoder to be loaded; none was assembled.'


class TestFamilyAssemblyTable:
    def test_krea2_variants_share_one_assembly_row(self, default_pipeline):
        table = default_pipeline._FAMILY_ASSEMBLY
        assert table[ModelFamily.KREA2_RAW] is table[ModelFamily.KREA2_TURBO]

    def test_single_file_families_have_no_row(self, default_pipeline):
        for family in (ModelFamily.SDXL, ModelFamily.SD15, ModelFamily.UNKNOWN):
            assert family not in default_pipeline._FAMILY_ASSEMBLY

    def test_the_two_gates_contain_no_per_family_branches(self, default_pipeline):
        """Extensibility contract: supporting Krea 2 added a table row, not
        an `elif family == ...` to either pipeline gate."""
        for gate in (default_pipeline.refresh_base_model, default_pipeline.assert_model_integrity,
                     default_pipeline._integrity_error):
            source = inspect.getsource(inspect.unwrap(gate))
            for member in ModelFamily:
                if member is not ModelFamily.UNKNOWN:  # the "not detected" default is not a branch
                    assert f'ModelFamily.{member.name}' not in source, (gate.__name__, member.name)
