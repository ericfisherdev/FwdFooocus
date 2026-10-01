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
        model_family_detection.modules.config.paths_checkpoints = [self.checkpoint_dir]
        model_family_detection.modules.config.path_fast_checkpoints = None
        model_family_detection._family_cache.clear()

    def tearDown(self):
        model_family_detection.modules.config.paths_checkpoints = self._original_paths_checkpoints
        model_family_detection.modules.config.path_fast_checkpoints = self._original_path_fast_checkpoints
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


class TestKrea2VariantResolution(_CheckpointTestCase):
    """Raw and Turbo share one state-dict layout, so the header only proves
    "Krea 2"; the variant comes from the config override or the file name."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(model_family_detection.modules.config, 'krea2_variant_overrides', {})
        self.overrides = patcher.start()
        self.addCleanup(patcher.stop)

    def _family_of(self, filename):
        _write_checkpoint(self._checkpoint_path(filename), _KREA2_KEYS)
        return model_family_detection.get_family(filename)

    def test_header_alone_defaults_to_raw(self):
        self.assertIs(model_family_detection._detect_family_from_keys(frozenset(_KREA2_KEYS)), ModelFamily.KREA2_RAW)

    def test_turbo_filename_resolves_to_turbo(self):
        self.assertIs(self._family_of('krea2_turbo_bf16.safetensors'), ModelFamily.KREA2_TURBO)

    def test_raw_filename_resolves_to_raw(self):
        self.assertIs(self._family_of('krea2_raw_bf16.safetensors'), ModelFamily.KREA2_RAW)

    def test_filename_match_is_case_insensitive(self):
        self.assertIs(self._family_of('Krea2-TURBO.safetensors'), ModelFamily.KREA2_TURBO)

    def test_only_the_basename_is_searched(self):
        # A directory called "turbo" must not decide the variant of a file
        # whose own name says raw.
        os.makedirs(self._checkpoint_path('turbo'))
        self.assertIs(self._family_of(os.path.join('turbo', 'krea2_raw.safetensors')), ModelFamily.KREA2_RAW)

    def test_ambiguous_filename_falls_back_to_raw_with_a_warning(self):
        with self.assertLogs(model_family_detection.logger, level='WARNING') as logs:
            family = self._family_of('krea2.safetensors')
        self.assertIs(family, ModelFamily.KREA2_RAW)
        self.assertIn('krea2_variant_overrides', logs.output[0])

    def test_filename_naming_both_variants_is_ambiguous(self):
        with self.assertLogs(model_family_detection.logger, level='WARNING'):
            family = self._family_of('krea2_raw_to_turbo_merge.safetensors')
        self.assertIs(family, ModelFamily.KREA2_RAW)

    def test_variant_names_inside_longer_words_do_not_count(self):
        # 'drawing' contains 'raw' but names nothing, so the name is ambiguous.
        with self.assertLogs(model_family_detection.logger, level='WARNING') as logs:
            family = self._family_of('krea2_distilled_drawing.safetensors')
        self.assertIs(family, ModelFamily.KREA2_RAW)
        self.assertIn('krea2_variant_overrides', logs.output[0])

    def test_a_variant_word_next_to_an_unrelated_word_still_resolves(self):
        with self.assertNoLogs(model_family_detection.logger, level='WARNING'):
            family = self._family_of('krea2_turbo_drawing.safetensors')
        self.assertIs(family, ModelFamily.KREA2_TURBO)

    def test_a_digit_is_a_word_boundary(self):
        with self.assertNoLogs(model_family_detection.logger, level='WARNING'):
            self.assertIs(self._family_of('krea2raw.safetensors'), ModelFamily.KREA2_RAW)
            self.assertIs(self._family_of('krea2turbo.safetensors'), ModelFamily.KREA2_TURBO)

    def test_a_clear_filename_does_not_warn(self):
        with self.assertNoLogs(model_family_detection.logger, level='WARNING'):
            self._family_of('krea2_turbo.safetensors')

    def test_override_beats_the_filename(self):
        self.overrides['krea2_raw_bf16.safetensors'] = 'turbo'
        self.assertIs(self._family_of('krea2_raw_bf16.safetensors'), ModelFamily.KREA2_TURBO)

    def test_override_resolves_an_otherwise_ambiguous_filename_without_warning(self):
        self.overrides['krea2.safetensors'] = 'turbo'
        with self.assertNoLogs(model_family_detection.logger, level='WARNING'):
            family = self._family_of('krea2.safetensors')
        self.assertIs(family, ModelFamily.KREA2_TURBO)

    def test_override_may_be_keyed_by_basename_for_a_subfolder_checkpoint(self):
        self.overrides['krea2_final.safetensors'] = 'turbo'
        os.makedirs(self._checkpoint_path('krea'))
        self.assertIs(self._family_of(os.path.join('krea', 'krea2_final.safetensors')), ModelFamily.KREA2_TURBO)

    def test_override_for_another_file_does_not_apply(self):
        self.overrides['other.safetensors'] = 'turbo'
        with self.assertLogs(model_family_detection.logger, level='WARNING'):
            family = self._family_of('krea2.safetensors')
        self.assertIs(family, ModelFamily.KREA2_RAW)

    def test_filename_heuristic_never_reclassifies_other_families(self):
        _write_checkpoint(self._checkpoint_path('turbo_xl.safetensors'), _SDXL_KEYS)
        self.assertIs(model_family_detection.get_family('turbo_xl.safetensors'), ModelFamily.SDXL)
        _write_checkpoint(self._checkpoint_path('z_image_turbo.safetensors'), _Z_IMAGE_KEYS)
        self.assertIs(model_family_detection.get_family('z_image_turbo.safetensors'), ModelFamily.Z_IMAGE)


class TestKrea2VariantOverridesConfig(unittest.TestCase):
    """The `krea2_variant_overrides` config item and its validator."""

    def test_defaults_to_an_empty_mapping(self):
        self.assertEqual(model_family_detection.modules.config.krea2_variant_overrides, {})

    def test_every_config_variant_has_a_family(self):
        self.assertEqual(set(model_family_detection.modules.config.KREA2_VARIANTS),
                         set(model_family_detection._KREA2_VARIANT_FAMILIES))

    def test_validator_accepts_filename_to_variant_mappings(self):
        is_valid = model_family_detection.modules.config.is_valid_krea2_variant_overrides
        self.assertTrue(is_valid({}))
        self.assertTrue(is_valid({'a.safetensors': 'raw', 'b.safetensors': 'turbo'}))

    def test_validator_rejects_malformed_values(self):
        is_valid = model_family_detection.modules.config.is_valid_krea2_variant_overrides
        for value in ({'a.safetensors': 'fast'}, {'a.safetensors': None}, {1: 'raw'}, ['a.safetensors'], 'turbo', None):
            with self.subTest(value=value):
                self.assertFalse(is_valid(value))


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
