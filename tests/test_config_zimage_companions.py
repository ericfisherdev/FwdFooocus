"""Tests for the Z-Image companion file (text encoder + VAE) acquisition
entries added to modules.config (FWDF-126)."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# args_manager calls parse_args() at import time, which chokes on pytest's
# argv. Patch sys.argv before any project modules are imported.
_original_argv = sys.argv
sys.argv = [sys.argv[0]]

import modules.config  # noqa: E402

sys.argv = _original_argv


class TestPathTextEncoders:
    def test_path_text_encoders_is_configured_directory(self):
        assert os.path.isdir(modules.config.path_text_encoders)

    def test_path_text_encoders_is_distinct_from_path_vae(self):
        assert os.path.abspath(modules.config.path_text_encoders) != os.path.abspath(modules.config.path_vae)


class TestDownloadingZImageTextEncoder:
    def test_downloads_into_path_text_encoders_with_verification(self):
        with patch('modules.config.load_file_from_url') as mock_load:
            mock_load.return_value = os.path.join(modules.config.path_text_encoders, 'qwen_3_4b.safetensors')
            result = modules.config.downloading_z_image_text_encoder()

        mock_load.assert_called_once()
        _, kwargs = mock_load.call_args
        assert kwargs['model_dir'] == modules.config.path_text_encoders
        assert kwargs['file_name'] == 'qwen_3_4b.safetensors'
        assert kwargs['expected_sha256'] == (
            '6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a'
        )
        assert kwargs['expected_size'] == 8044982048
        assert 'Comfy-Org/z_image_turbo' in kwargs['url']
        assert 'split_files/text_encoders/qwen_3_4b.safetensors' in kwargs['url']
        assert result == os.path.join(modules.config.path_text_encoders, 'qwen_3_4b.safetensors')

    def test_url_is_pinned_to_a_commit_not_a_moving_branch(self):
        with patch('modules.config.load_file_from_url') as mock_load:
            mock_load.return_value = 'unused'
            modules.config.downloading_z_image_text_encoder()

        _, kwargs = mock_load.call_args
        assert '/resolve/d24c4cf2a0cd98a42f23467e27e3d76ee9438b8e/' in kwargs['url']
        assert '/resolve/main/' not in kwargs['url']


class TestDownloadingZImageVae:
    def test_downloads_into_path_vae_with_verification(self):
        with patch('modules.config.load_file_from_url') as mock_load:
            mock_load.return_value = os.path.join(modules.config.path_vae, 'ae.safetensors')
            result = modules.config.downloading_z_image_vae()

        mock_load.assert_called_once()
        _, kwargs = mock_load.call_args
        assert kwargs['model_dir'] == modules.config.path_vae
        assert kwargs['file_name'] == 'ae.safetensors'
        assert kwargs['expected_sha256'] == (
            'afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38'
        )
        assert kwargs['expected_size'] == 335304388
        assert 'Comfy-Org/z_image_turbo' in kwargs['url']
        assert 'split_files/vae/ae.safetensors' in kwargs['url']
        assert result == os.path.join(modules.config.path_vae, 'ae.safetensors')

    def test_url_is_pinned_to_a_commit_not_a_moving_branch(self):
        with patch('modules.config.load_file_from_url') as mock_load:
            mock_load.return_value = 'unused'
            modules.config.downloading_z_image_vae()

        _, kwargs = mock_load.call_args
        assert '/resolve/d24c4cf2a0cd98a42f23467e27e3d76ee9438b8e/' in kwargs['url']
        assert '/resolve/main/' not in kwargs['url']
