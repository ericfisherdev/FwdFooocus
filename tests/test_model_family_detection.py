"""Unit tests for checkpoint model family detection."""

import os
import shutil
import struct
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import safetensors.torch  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open as real_safe_open  # noqa: E402

# args_manager calls parse_args() at import time, which chokes on pytest's
# argv (modules.model_family_detection imports modules.config, which imports
# args_manager). Patch sys.argv before any project modules are imported.
# Mirrors the convention in tests/test_model_family.py.
_original_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    from modules import model_family_detection  # noqa: E402
    from modules.model_family import ModelFamily  # noqa: E402
finally:
    sys.argv = _original_argv


_SDXL_KEYS = [
    'model.diffusion_model.input_blocks.0.0.weight',
    'model.diffusion_model.label_emb.0.0.weight',
]
_SD15_KEYS = [
    'model.diffusion_model.input_blocks.0.0.weight',
]
_Z_IMAGE_KEYS = [
    'model.diffusion_model.x_embedder.weight',
    'model.diffusion_model.cap_embedder.mlp.weight',
]
_KREA2_KEYS = [
    'model.diffusion_model.txtfusion.projector.weight',
]
_UNRELATED_KEYS = [
    'some.unrelated.tensor.weight',
]


def _write_checkpoint(path, tensor_names, tensor_shape=(2, 2)):
    """Write a minimal synthetic safetensors file with the given tensor names."""
    state_dict = {name: torch.zeros(*tensor_shape) for name in tensor_names}
    safetensors.torch.save_file(state_dict, str(path))


class _CheckpointTestCase(unittest.TestCase):
    """Shared fixture: an isolated checkpoint directory wired into modules.config."""

    def setUp(self):
        self.checkpoint_dir = tempfile.mkdtemp()
        self._original_paths_checkpoints = model_family_detection.modules.config.paths_checkpoints
        self._original_path_fast_checkpoints = model_family_detection.modules.config.path_fast_checkpoints
        self._original_default_base_model = model_family_detection.modules.config.default_base_model
        model_family_detection.modules.config.paths_checkpoints = [self.checkpoint_dir]
        model_family_detection.modules.config.path_fast_checkpoints = None
        model_family_detection.modules.config.default_base_model = None
        model_family_detection._family_cache.clear()

    def tearDown(self):
        model_family_detection.modules.config.paths_checkpoints = self._original_paths_checkpoints
        model_family_detection.modules.config.path_fast_checkpoints = self._original_path_fast_checkpoints
        model_family_detection.modules.config.default_base_model = self._original_default_base_model
        model_family_detection._family_cache.clear()
        shutil.rmtree(self.checkpoint_dir, ignore_errors=True)

    def _checkpoint_path(self, filename):
        return os.path.join(self.checkpoint_dir, filename)


class TestFamilyDetection(_CheckpointTestCase):
    """get_family() discriminant coverage, aligned with the FWDF-116 detection registry."""

    def test_detects_sdxl(self):
        _write_checkpoint(self._checkpoint_path('sdxl.safetensors'), _SDXL_KEYS)
        self.assertIs(model_family_detection.get_family('sdxl.safetensors'), ModelFamily.SDXL)

    def test_detects_sd15(self):
        _write_checkpoint(self._checkpoint_path('sd15.safetensors'), _SD15_KEYS)
        self.assertIs(model_family_detection.get_family('sd15.safetensors'), ModelFamily.SD15)

    def test_detects_z_image(self):
        _write_checkpoint(self._checkpoint_path('z_image.safetensors'), _Z_IMAGE_KEYS)
        self.assertIs(model_family_detection.get_family('z_image.safetensors'), ModelFamily.Z_IMAGE)

    def test_detects_krea2(self):
        _write_checkpoint(self._checkpoint_path('krea2.safetensors'), _KREA2_KEYS)
        self.assertIs(model_family_detection.get_family('krea2.safetensors'), ModelFamily.KREA2)

    def test_unknown_for_unrecognized_keys(self):
        _write_checkpoint(self._checkpoint_path('mystery.safetensors'), _UNRELATED_KEYS)
        self.assertIs(model_family_detection.get_family('mystery.safetensors'), ModelFamily.UNKNOWN)

    def test_z_image_requires_both_discriminant_keys(self):
        # x_embedder alone (no cap_embedder.*) must not be mistaken for Z_IMAGE.
        _write_checkpoint(
            self._checkpoint_path('partial.safetensors'),
            ['model.diffusion_model.x_embedder.weight'],
        )
        self.assertIs(model_family_detection.get_family('partial.safetensors'), ModelFamily.UNKNOWN)

    def test_missing_file_returns_unknown_without_raising(self):
        self.assertIs(
            model_family_detection.get_family('does-not-exist.safetensors'),
            ModelFamily.UNKNOWN,
        )

    def test_malformed_file_returns_unknown_without_raising(self):
        garbage_path = self._checkpoint_path('garbage.safetensors')
        with open(garbage_path, 'wb') as f:
            f.write(b'not a safetensors file' * 10)
        self.assertIs(model_family_detection.get_family('garbage.safetensors'), ModelFamily.UNKNOWN)


class _SpyOpen:
    """Wraps the real safe_open, counting calls that would materialize tensor bytes.

    keys() delegates to the real implementation (so detection still works);
    get_tensor()/get_slice() would only ever be called by code that reads
    actual tensor data, which get_family() must never do.
    """

    tensor_read_calls = 0

    def __init__(self, path, framework):
        self._real = real_safe_open(path, framework=framework)

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._real.__exit__(exc_type, exc_val, exc_tb)

    def keys(self):
        return self._real.keys()

    def get_tensor(self, name):
        type(self).tensor_read_calls += 1
        return self._real.get_tensor(name)

    def get_slice(self, name):
        type(self).tensor_read_calls += 1
        return self._real.get_slice(name)


class TestHeaderOnlyRead(_CheckpointTestCase):
    """Detection must read only the safetensors header, never tensor data."""

    def setUp(self):
        super().setUp()
        self._original_safe_open = model_family_detection.safe_open
        _SpyOpen.tensor_read_calls = 0

    def tearDown(self):
        model_family_detection.safe_open = self._original_safe_open
        super().tearDown()

    def test_detection_never_materializes_tensor_data(self):
        _write_checkpoint(self._checkpoint_path('sdxl.safetensors'), _SDXL_KEYS)
        model_family_detection.safe_open = _SpyOpen

        family = model_family_detection.get_family('sdxl.safetensors')

        self.assertIs(family, ModelFamily.SDXL)
        self.assertEqual(_SpyOpen.tensor_read_calls, 0)

    def test_detection_ignores_corrupted_tensor_payload(self):
        # Overwrite the tensor payload (everything after the header) with
        # garbage of the same length. Detection must still succeed since it
        # never reads this region of the file.
        path = self._checkpoint_path('sdxl.safetensors')
        _write_checkpoint(path, _SDXL_KEYS)

        with open(path, 'r+b') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            f.seek(8 + header_len)
            payload_len = len(f.read())
            f.seek(8 + header_len)
            f.write(b'\xff' * payload_len)

        self.assertIs(model_family_detection.get_family('sdxl.safetensors'), ModelFamily.SDXL)


class TestCaching(_CheckpointTestCase):
    """get_family() caches by (path, mtime, size), invalidating on either change."""

    def setUp(self):
        super().setUp()
        self._original_read_state_dict_keys = model_family_detection._read_state_dict_keys
        self._read_call_count = 0

        def counting_read(path):
            self._read_call_count += 1
            return self._original_read_state_dict_keys(path)

        model_family_detection._read_state_dict_keys = counting_read

    def tearDown(self):
        model_family_detection._read_state_dict_keys = self._original_read_state_dict_keys
        super().tearDown()

    def test_repeat_calls_hit_cache(self):
        _write_checkpoint(self._checkpoint_path('sdxl.safetensors'), _SDXL_KEYS)

        first = model_family_detection.get_family('sdxl.safetensors')
        second = model_family_detection.get_family('sdxl.safetensors')

        self.assertIs(first, ModelFamily.SDXL)
        self.assertIs(second, ModelFamily.SDXL)
        self.assertEqual(self._read_call_count, 1)

    def test_cache_invalidates_when_mtime_changes(self):
        path = self._checkpoint_path('checkpoint.safetensors')
        _write_checkpoint(path, _SDXL_KEYS)

        first = model_family_detection.get_family('checkpoint.safetensors')
        self.assertEqual(self._read_call_count, 1)

        stat_before = os.stat(path)
        os.utime(path, (stat_before.st_atime, stat_before.st_mtime + 5))

        second = model_family_detection.get_family('checkpoint.safetensors')

        self.assertEqual(self._read_call_count, 2)
        self.assertIs(first, ModelFamily.SDXL)
        self.assertIs(second, ModelFamily.SDXL)

    def test_cache_invalidates_when_size_changes(self):
        path = self._checkpoint_path('checkpoint.safetensors')
        _write_checkpoint(path, _SDXL_KEYS)

        first = model_family_detection.get_family('checkpoint.safetensors')
        self.assertIs(first, ModelFamily.SDXL)
        self.assertEqual(self._read_call_count, 1)

        # Rewrite with different discriminant keys and a much larger tensor
        # payload, guaranteeing a different file size.
        _write_checkpoint(path, _Z_IMAGE_KEYS, tensor_shape=(64, 64))

        second = model_family_detection.get_family('checkpoint.safetensors')

        self.assertEqual(self._read_call_count, 2)
        self.assertIs(second, ModelFamily.Z_IMAGE)


class TestCorruptCheckpointError(_CheckpointTestCase):
    """_read_state_dict_keys() wraps safetensors parse failures in a specific type."""

    def test_raises_on_malformed_file(self):
        garbage_path = self._checkpoint_path('garbage.safetensors')
        with open(garbage_path, 'wb') as f:
            f.write(b'not a safetensors file' * 10)

        with self.assertRaises(model_family_detection.CorruptCheckpointError):
            model_family_detection._read_state_dict_keys(garbage_path)


class TestSessionStateKey(_CheckpointTestCase):
    """session_state_id() shares the UNKNOWN -> config-string fallback rule."""

    def test_returns_family_value_when_detected(self):
        _write_checkpoint(self._checkpoint_path('sdxl.safetensors'), _SDXL_KEYS)
        self.assertEqual(model_family_detection.session_state_id('sdxl.safetensors'), 'sdxl')

    def test_falls_back_to_config_default_base_model_when_unknown(self):
        _write_checkpoint(self._checkpoint_path('mystery.safetensors'), _UNRELATED_KEYS)
        model_family_detection.modules.config.default_base_model = 'pony'

        self.assertEqual(model_family_detection.session_state_id('mystery.safetensors'), 'pony')

    def test_returns_unknown_literal_when_unknown_and_no_config_default(self):
        _write_checkpoint(self._checkpoint_path('mystery.safetensors'), _UNRELATED_KEYS)
        model_family_detection.modules.config.default_base_model = None

        self.assertEqual(model_family_detection.session_state_id('mystery.safetensors'), 'unknown')

    def test_detected_family_takes_priority_over_config_default(self):
        _write_checkpoint(self._checkpoint_path('sdxl.safetensors'), _SDXL_KEYS)
        model_family_detection.modules.config.default_base_model = 'pony'

        self.assertEqual(model_family_detection.session_state_id('sdxl.safetensors'), 'sdxl')


class TestCacheBoundedness(unittest.TestCase):
    def setUp(self):
        model_family_detection._family_cache.clear()

    def tearDown(self):
        model_family_detection._family_cache.clear()

    def test_in_place_update_replaces_entry_instead_of_accumulating(self):
        """Rewriting the same checkpoint path must keep exactly one cache
        entry for it (latest fingerprint wins), not one per fingerprint."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, 'model.safetensors')
            safetensors.torch.save_file({'input_blocks.0.0.weight': torch.zeros(2)}, path)
            with mock.patch.object(
                model_family_detection.modules.config, 'paths_checkpoints', [tmp_dir]
            ), mock.patch.object(
                model_family_detection.modules.config, 'path_fast_checkpoints', None
            ):
                model_family_detection.get_family('model.safetensors')
                # Rewrite in place with different content/size.
                safetensors.torch.save_file({'input_blocks.0.0.weight': torch.zeros(64),
                                            'label_emb.0.0.weight': torch.zeros(4, 4)}, path)
                os.utime(path, (1, 1))
                model_family_detection.get_family('model.safetensors')

        self.assertEqual(len(model_family_detection._family_cache), 1)


if __name__ == '__main__':
    unittest.main()


class TestReadErrorsNeverEscape(unittest.TestCase):
    def setUp(self):
        model_family_detection._family_cache.clear()

    def tearDown(self):
        model_family_detection._family_cache.clear()

    def test_file_vanishing_between_stat_and_open_returns_unknown(self):
        """get_family()'s never-raises contract must hold even when the file
        disappears after os.stat() succeeds (TOCTOU) — safe_open()'s OSError
        is converted, not propagated."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, 'model.safetensors')
            safetensors.torch.save_file({'input_blocks.0.0.weight': torch.zeros(2)}, path)
            real_stat = os.stat

            def stat_then_delete(p, *a, **k):
                result = real_stat(p, *a, **k)
                if p == os.path.abspath(path) or p == path:
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
                return result

            with mock.patch.object(
                model_family_detection.modules.config, 'paths_checkpoints', [tmp_dir]
            ), mock.patch.object(
                model_family_detection.modules.config, 'path_fast_checkpoints', None
            ), mock.patch('modules.model_family_detection.os.stat', side_effect=stat_then_delete):
                family = model_family_detection.get_family('model.safetensors')

        self.assertIs(family, ModelFamily.UNKNOWN)
