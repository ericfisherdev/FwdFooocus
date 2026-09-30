"""Tests for the Krea 2 companion file (text encoder + Qwen Image VAE)
acquisition entries added to modules.config (FWDF-151)."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# args_manager calls parse_args() at import time, which chokes on pytest's
# argv. Patch sys.argv before any project modules are imported.
_original_argv = sys.argv
sys.argv = [sys.argv[0]]

import modules.config  # noqa: E402

sys.argv = _original_argv

PINNED_COMMIT = 'eb1eddd3983a54678545a9b2c178c5853b30f7be'


@pytest.fixture(autouse=True)
def reset_license_note_guard():
    with patch.object(modules.config, '_krea2_license_note_shown', False):
        yield


class TestDownloadingKrea2TextEncoder:
    def test_downloads_into_path_text_encoders_with_verification(self):
        with patch('modules.config.load_file_from_url') as mock_load:
            result = modules.config.downloading_krea2_text_encoder()

        mock_load.assert_called_once()
        _, kwargs = mock_load.call_args
        assert kwargs['model_dir'] == modules.config.path_text_encoders
        assert kwargs['file_name'] == 'qwen3vl_4b_bf16.safetensors'
        assert kwargs['expected_sha256'] == (
            '36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34'
        )
        assert kwargs['expected_size'] == 8875719384
        assert 'Comfy-Org/Krea-2' in kwargs['url']
        assert kwargs['url'].endswith('/text_encoders/qwen3vl_4b_bf16.safetensors')
        assert result == os.path.join(modules.config.path_text_encoders, 'qwen3vl_4b_bf16.safetensors')
        assert result == modules.config.krea2_text_encoder_path()

    def test_url_is_pinned_to_a_commit_not_a_moving_branch(self):
        with patch('modules.config.load_file_from_url') as mock_load:
            modules.config.downloading_krea2_text_encoder()

        _, kwargs = mock_load.call_args
        assert f'/resolve/{PINNED_COMMIT}/' in kwargs['url']
        assert '/resolve/main/' not in kwargs['url']


class TestDownloadingQwenImageVae:
    def test_downloads_into_path_vae_with_verification(self):
        with patch('modules.config.load_file_from_url') as mock_load:
            result = modules.config.downloading_qwen_image_vae()

        mock_load.assert_called_once()
        _, kwargs = mock_load.call_args
        assert kwargs['model_dir'] == modules.config.path_vae
        assert kwargs['file_name'] == 'qwen_image_vae.safetensors'
        assert kwargs['expected_sha256'] == (
            'a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f'
        )
        assert kwargs['expected_size'] == 253806246
        assert 'Comfy-Org/Krea-2' in kwargs['url']
        assert kwargs['url'].endswith('/vae/qwen_image_vae.safetensors')
        assert result == os.path.join(modules.config.path_vae, 'qwen_image_vae.safetensors')
        assert result == modules.config.qwen_image_vae_path()

    def test_url_is_pinned_to_a_commit_not_a_moving_branch(self):
        with patch('modules.config.load_file_from_url') as mock_load:
            modules.config.downloading_qwen_image_vae()

        _, kwargs = mock_load.call_args
        assert f'/resolve/{PINNED_COMMIT}/' in kwargs['url']
        assert '/resolve/main/' not in kwargs['url']


class TestKrea2LicenseNote:
    def test_printed_exactly_once_across_both_downloads(self, capsys):
        with patch('modules.config.load_file_from_url'):
            modules.config.downloading_krea2_text_encoder()
            modules.config.downloading_qwen_image_vae()
            modules.config.downloading_krea2_text_encoder()

        out = capsys.readouterr().out
        assert out.count(modules.config.KREA2_LICENSE_NOTE) == 1

    def test_note_cites_the_krea2_community_license(self):
        assert 'Krea 2 Community License' in modules.config.KREA2_LICENSE_NOTE
        assert 'LICENSE.pdf' in modules.config.KREA2_LICENSE_NOTE


class TestPathHelpersAreSideEffectFree:
    def test_path_helpers_do_not_download_or_print_license(self, capsys):
        with patch('modules.config.load_file_from_url') as mock_load:
            modules.config.krea2_text_encoder_path()
            modules.config.qwen_image_vae_path()

        mock_load.assert_not_called()
        assert modules.config.KREA2_LICENSE_NOTE not in capsys.readouterr().out
