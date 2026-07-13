import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch


class TestResolveCheckpointPath(unittest.TestCase):
    def setUp(self):
        self.slow_dir = tempfile.mkdtemp()
        self.fast_dir = tempfile.mkdtemp()
        # Create a fake checkpoint on the slow drive
        self.checkpoint_name = 'model.safetensors'
        self.slow_path = os.path.join(self.slow_dir, self.checkpoint_name)
        with open(self.slow_path, 'wb') as f:
            f.write(b'\x00' * 1024)

    def tearDown(self):
        shutil.rmtree(self.slow_dir, ignore_errors=True)
        shutil.rmtree(self.fast_dir, ignore_errors=True)

    def test_feature_disabled_returns_normal_path(self):
        from modules.fast_checkpoint import resolve_checkpoint_path
        result = resolve_checkpoint_path(
            self.checkpoint_name, [self.slow_dir], fast_path=None
        )
        self.assertEqual(result, self.slow_path)

    def test_copies_to_fast_drive_on_first_use(self):
        from modules.fast_checkpoint import resolve_checkpoint_path
        result = resolve_checkpoint_path(
            self.checkpoint_name, [self.slow_dir], fast_path=self.fast_dir
        )
        expected_fast = os.path.join(self.fast_dir, self.checkpoint_name)
        self.assertEqual(result, expected_fast)
        self.assertTrue(os.path.isfile(expected_fast))

    def test_uses_existing_fast_copy_when_unchanged(self):
        from modules.fast_checkpoint import resolve_checkpoint_path
        # Pre-populate fast drive with an up-to-date copy of the source
        # (same mtime/size, as a real cached copy would have).
        fast_file = os.path.join(self.fast_dir, self.checkpoint_name)
        shutil.copy2(self.slow_path, fast_file)

        with patch('modules.fast_checkpoint._copy_to_fast_drive') as mock_copy:
            result = resolve_checkpoint_path(
                self.checkpoint_name, [self.slow_dir], fast_path=self.fast_dir
            )

        self.assertEqual(result, fast_file)
        mock_copy.assert_not_called()

    def test_revalidates_when_source_mtime_changed(self):
        from modules.fast_checkpoint import resolve_checkpoint_path
        fast_file = os.path.join(self.fast_dir, self.checkpoint_name)
        shutil.copy2(self.slow_path, fast_file)

        # Same size and content, but a newer mtime (e.g. re-downloaded in place).
        future = time.time() + 100
        os.utime(self.slow_path, (future, future))

        result = resolve_checkpoint_path(
            self.checkpoint_name, [self.slow_dir], fast_path=self.fast_dir
        )
        self.assertEqual(result, fast_file)
        self.assertEqual(
            os.stat(fast_file).st_mtime_ns, os.stat(self.slow_path).st_mtime_ns
        )

    def test_revalidates_when_source_size_changed(self):
        from modules.fast_checkpoint import resolve_checkpoint_path
        fast_file = os.path.join(self.fast_dir, self.checkpoint_name)
        shutil.copy2(self.slow_path, fast_file)

        new_content = b'\x02' * 2048
        with open(self.slow_path, 'wb') as f:
            f.write(new_content)

        result = resolve_checkpoint_path(
            self.checkpoint_name, [self.slow_dir], fast_path=self.fast_dir
        )
        self.assertEqual(result, fast_file)
        with open(fast_file, 'rb') as f:
            self.assertEqual(f.read(), new_content)

    def test_source_missing_returns_existing_fast_copy(self):
        from modules.fast_checkpoint import resolve_checkpoint_path
        fast_file = os.path.join(self.fast_dir, self.checkpoint_name)
        shutil.copy2(self.slow_path, fast_file)

        os.remove(self.slow_path)  # Source no longer exists

        result = resolve_checkpoint_path(
            self.checkpoint_name, [self.slow_dir], fast_path=self.fast_dir
        )
        self.assertEqual(result, fast_file)
        self.assertTrue(os.path.isfile(fast_file))

    def test_rejects_absolute_checkpoint_path(self):
        from modules.fast_checkpoint import resolve_checkpoint_path
        abs_evil = os.path.join(self.slow_dir, 'evil.safetensors')
        with open(abs_evil, 'wb') as f:
            f.write(b'\x00' * 16)

        result = resolve_checkpoint_path(
            abs_evil, [self.slow_dir], fast_path=self.fast_dir
        )
        self.assertEqual(result, abs_evil)
        self.assertFalse(result.startswith(self.fast_dir))

    def test_rejects_parent_directory_traversal(self):
        from modules.fast_checkpoint import resolve_checkpoint_path
        result = resolve_checkpoint_path(
            '../evil.safetensors', [self.slow_dir], fast_path=self.fast_dir
        )
        self.assertFalse(result.startswith(self.fast_dir))

    def test_falls_back_on_copy_failure(self):
        from modules.fast_checkpoint import resolve_checkpoint_path
        # Use a non-writable path to force copy failure
        bad_fast_dir = '/proc/fake_nonexistent_dir'
        result = resolve_checkpoint_path(
            self.checkpoint_name, [self.slow_dir], fast_path=bad_fast_dir
        )
        # Should fall back to slow path
        self.assertEqual(result, self.slow_path)

    def test_checkpoint_not_found_anywhere(self):
        from modules.fast_checkpoint import resolve_checkpoint_path
        result = resolve_checkpoint_path(
            'nonexistent.safetensors', [self.slow_dir], fast_path=self.fast_dir
        )
        # Should return constructed path in first checkpoint dir (existing behavior)
        expected = os.path.abspath(os.path.realpath(
            os.path.join(self.slow_dir, 'nonexistent.safetensors')
        ))
        self.assertEqual(result, expected)

    def test_subdirectory_checkpoint_preserves_structure(self):
        from modules.fast_checkpoint import resolve_checkpoint_path
        # Create a checkpoint in a subdirectory on the slow drive
        subdir = os.path.join(self.slow_dir, 'sdxl')
        os.makedirs(subdir)
        sub_checkpoint = os.path.join(subdir, 'model.safetensors')
        with open(sub_checkpoint, 'wb') as f:
            f.write(b'\x00' * 256)

        result = resolve_checkpoint_path(
            'sdxl/model.safetensors', [self.slow_dir], fast_path=self.fast_dir
        )
        expected_fast = os.path.join(self.fast_dir, 'sdxl', 'model.safetensors')
        self.assertEqual(result, expected_fast)
        self.assertTrue(os.path.isfile(expected_fast))


if __name__ == '__main__':
    unittest.main()
