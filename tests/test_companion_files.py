"""Tests for the companion-file registry (modules.companion_files, FWDF-151)."""

import importlib
import os
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# args_manager calls parse_args() at import time, which chokes on pytest's
# argv. Patch sys.argv before any project modules are imported.
_original_argv = sys.argv
sys.argv = [sys.argv[0]]

import modules.companion_files as companion_files  # noqa: E402
import modules.config  # noqa: E402
from modules.model_family import ModelFamily  # noqa: E402

sys.argv = _original_argv


@pytest.fixture(autouse=True)
def no_hf_mirror(monkeypatch):
    """Keep URL assertions independent of the developer's HF_MIRROR."""
    monkeypatch.delenv('HF_MIRROR', raising=False)


class TestGetCompanions:
    @pytest.mark.parametrize('family', [ModelFamily.SDXL, ModelFamily.SD15, ModelFamily.UNKNOWN])
    def test_single_file_families_have_no_companions(self, family):
        assert companion_files.get_companions(family) is None

    def test_z_image_entry_delegates_to_existing_z_image_helpers(self):
        companions = companion_files.get_companions(ModelFamily.Z_IMAGE)

        assert companions is not None
        assert companions.vae_path() == modules.config.z_image_vae_path()
        assert companions.text_encoder_path() == os.path.join(
            modules.config.path_text_encoders, 'qwen_3_4b.safetensors')

    def test_krea2_entry_points_at_krea2_files(self):
        companions = companion_files.get_companions(ModelFamily.KREA2)

        assert companions is not None
        assert companions.text_encoder_path() == modules.config.krea2_text_encoder_path()
        assert companions.vae_path() == modules.config.qwen_image_vae_path()


@pytest.mark.parametrize('family', [ModelFamily.Z_IMAGE, ModelFamily.KREA2])
class TestEnsureSuccess:
    def test_ensure_text_encoder_returns_path_after_verified_download(self, family):
        companions = companion_files.get_companions(family)
        with patch('modules.config.load_file_from_url') as mock_load:
            result = companions.ensure_text_encoder()

        mock_load.assert_called_once()
        assert result == companions.text_encoder_path()

    def test_ensure_vae_returns_path_after_verified_download(self, family):
        companions = companion_files.get_companions(family)
        with patch('modules.config.load_file_from_url') as mock_load:
            result = companions.ensure_vae()

        mock_load.assert_called_once()
        assert result == companions.vae_path()


_DOWNLOAD_FAILURES = [
    RuntimeError('SHA256 mismatch'),
    urllib.error.URLError('network unreachable'),
    urllib.error.HTTPError('http://x', 404, 'Not Found', {}, None),
    OSError('disk full'),
]


@pytest.mark.parametrize('failure', _DOWNLOAD_FAILURES, ids=lambda f: type(f).__name__)
class TestEnsureFailure:
    def test_krea2_text_encoder_failure_is_actionable(self, failure):
        companions = companion_files.get_companions(ModelFamily.KREA2)
        with patch('modules.config.load_file_from_url', side_effect=failure):
            with pytest.raises(companion_files.MissingCompanionFileError) as excinfo:
                companions.ensure_text_encoder()

        error = excinfo.value
        assert error.file_name == 'qwen3vl_4b_bf16.safetensors'
        assert error.directory == modules.config.path_text_encoders
        assert error.url == modules.config.KREA2_TEXT_ENCODER_URL
        for detail in (error.file_name, error.directory, error.url):
            assert detail in str(error)
        assert error.__cause__ is failure

    def test_krea2_vae_failure_is_actionable(self, failure):
        companions = companion_files.get_companions(ModelFamily.KREA2)
        with patch('modules.config.load_file_from_url', side_effect=failure):
            with pytest.raises(companion_files.MissingCompanionFileError) as excinfo:
                companions.ensure_vae()

        error = excinfo.value
        assert error.file_name == 'qwen_image_vae.safetensors'
        assert error.directory == modules.config.path_vae
        assert error.url == modules.config.QWEN_IMAGE_VAE_URL
        for detail in (error.file_name, error.directory, error.url):
            assert detail in str(error)

    def test_z_image_failure_is_actionable(self, failure):
        companions = companion_files.get_companions(ModelFamily.Z_IMAGE)
        with patch('modules.config.load_file_from_url', side_effect=failure):
            with pytest.raises(companion_files.MissingCompanionFileError) as excinfo:
                companions.ensure_vae()

        assert excinfo.value.url == modules.config.Z_IMAGE_VAE_URL
        assert excinfo.value.file_name == 'ae.safetensors'


def test_error_names_the_mirror_url_actually_requested(monkeypatch):
    monkeypatch.setenv('HF_MIRROR', 'https://hf-mirror.example/')
    companions = companion_files.get_companions(ModelFamily.KREA2)
    with patch('modules.config.load_file_from_url', side_effect=OSError('unreachable')):
        with pytest.raises(companion_files.MissingCompanionFileError) as excinfo:
            companions.ensure_vae()

    assert excinfo.value.url == modules.config.QWEN_IMAGE_VAE_URL.replace(
        'https://huggingface.co', 'https://hf-mirror.example', 1)
    assert 'hf-mirror.example' in str(excinfo.value)
    assert 'huggingface.co' not in str(excinfo.value)


def test_error_names_canonical_url_when_no_mirror_is_set(monkeypatch):
    monkeypatch.delenv('HF_MIRROR', raising=False)
    companions = companion_files.get_companions(ModelFamily.KREA2)
    with patch('modules.config.load_file_from_url', side_effect=OSError('unreachable')):
        with pytest.raises(companion_files.MissingCompanionFileError) as excinfo:
            companions.ensure_text_encoder()

    assert excinfo.value.url == modules.config.KREA2_TEXT_ENCODER_URL


def test_unrelated_exceptions_are_not_swallowed():
    companions = companion_files.get_companions(ModelFamily.KREA2)
    with patch('modules.config.load_file_from_url', side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            companions.ensure_vae()


def test_importing_module_performs_no_download():
    with patch('modules.model_loader.load_file_from_url') as mock_loader, \
            patch('modules.config.load_file_from_url') as mock_config_loader:
        importlib.reload(companion_files)

    mock_loader.assert_not_called()
    mock_config_loader.assert_not_called()
