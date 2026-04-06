import os
import shutil
import tempfile
import unittest


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

    def test_uses_existing_fast_copy(self):
        from modules.fast_checkpoint import resolve_checkpoint_path
        # Pre-populate fast drive
        fast_file = os.path.join(self.fast_dir, self.checkpoint_name)
        with open(fast_file, 'wb') as f:
            f.write(b'\x01' * 512)  # Different content/size

        result = resolve_checkpoint_path(
            self.checkpoint_name, [self.slow_dir], fast_path=self.fast_dir
        )
        self.assertEqual(result, fast_file)
        # Verify it was NOT overwritten (still 512 bytes)
        self.assertEqual(os.path.getsize(fast_file), 512)

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
