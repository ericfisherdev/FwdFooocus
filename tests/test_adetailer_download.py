"""Tests for the ADetailer ONNX model download registry added to
modules.config (FWDF-196)."""

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
import modules.flags  # noqa: E402

sys.argv = _original_argv


class TestAdetailerModelRegistry:
    def test_flags_list_is_subset_of_registry(self):
        for model_name in modules.flags.inpaint_mask_adetailer_model:
            assert model_name in modules.config._adetailer_model_registry

    def test_registry_entries_have_valid_sha256_and_positive_size(self):
        for release_url, file_name, expected_sha256, expected_size in modules.config._adetailer_model_registry.values():
            assert release_url.startswith('https://github.com/ericfisherdev/FwdFooocus/releases/download/')
            assert len(expected_sha256) == 64
            assert all(c in '0123456789abcdef' for c in expected_sha256)
            assert expected_size > 0
            assert file_name.endswith('.onnx')

    def test_registry_contains_all_exported_models(self):
        assert set(modules.config._adetailer_model_registry.keys()) == {
            'face_yolov9c', 'hand_yolov9c', 'person_yolov8m-seg', 'deepfashion2_yolov8s-seg',
            'anzhc_face-seg',
        }

    def test_anzhc_model_downloads_from_its_own_agpl_release(self):
        with patch('modules.config.load_file_from_url') as mock_load:
            mock_load.return_value = 'unused'
            modules.config.download_adetailer_model('anzhc_face-seg')

        _, kwargs = mock_load.call_args
        assert kwargs['url'] == (
            'https://github.com/ericfisherdev/FwdFooocus/releases/download/'
            'adetailer-onnx-anzhc-v1/anzhc_face_seg_640_v4_y11n.onnx'
        )
        assert kwargs['expected_size'] == 11626379


class TestDownloadAdetailerModel:
    def test_raises_for_unknown_model_name(self):
        try:
            modules.config.download_adetailer_model('nope')
            assert False, "expected ValueError"
        except ValueError as e:
            assert 'nope' in str(e)

    def test_downloads_into_path_adetailer_with_verification(self):
        with patch('modules.config.load_file_from_url') as mock_load:
            mock_load.return_value = os.path.join(modules.config.path_adetailer, 'face_yolov9c.onnx')
            result = modules.config.download_adetailer_model('face_yolov9c')

        mock_load.assert_called_once()
        _, kwargs = mock_load.call_args
        assert kwargs['model_dir'] == modules.config.path_adetailer
        assert kwargs['file_name'] == 'face_yolov9c.onnx'
        assert kwargs['expected_sha256'] == (
            '1e05f810e80903a85cc32104460cd468d9fa90f7d2f9dfeb2fea94fcd412f71d'
        )
        assert kwargs['expected_size'] == 101632702
        assert kwargs['url'] == (
            'https://github.com/ericfisherdev/FwdFooocus/releases/download/'
            'adetailer-onnx-v1/face_yolov9c.onnx'
        )
        assert result == os.path.join(modules.config.path_adetailer, 'face_yolov9c.onnx')

    def test_every_registry_model_invokes_load_file_from_url_with_verification_kwargs(self):
        for model_name in modules.config._adetailer_model_registry:
            with patch('modules.config.load_file_from_url') as mock_load:
                mock_load.return_value = 'unused'
                modules.config.download_adetailer_model(model_name)

            mock_load.assert_called_once()
            _, kwargs = mock_load.call_args
            assert kwargs['expected_sha256'] is not None
            assert kwargs['expected_size'] is not None
