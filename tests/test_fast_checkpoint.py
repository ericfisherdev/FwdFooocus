import os
import shutil
import tempfile
import threading
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


class TestConcurrentCopies(unittest.TestCase):
    """Concurrent resolve_checkpoint_path calls for one checkpoint (FWDF-206)."""

    JOIN_TIMEOUT = 10

    def setUp(self):
        self.slow_dir = tempfile.mkdtemp()
        self.fast_dir = tempfile.mkdtemp()
        self.checkpoint_name = 'model.safetensors'
        self.slow_path = os.path.join(self.slow_dir, self.checkpoint_name)
        self.fast_file = os.path.join(self.fast_dir, self.checkpoint_name)
        with open(self.slow_path, 'wb') as f:
            f.write(b'\x01' * 4096)
        self.copy_started = threading.Event()
        self.copy_calls = []
        self.real_copy2 = shutil.copy2

    def tearDown(self):
        shutil.rmtree(self.slow_dir, ignore_errors=True)
        shutil.rmtree(self.fast_dir, ignore_errors=True)

    def _slow_copy2(self, source, destination, **kwargs):
        self.copy_calls.append(destination)
        self.copy_started.set()
        time.sleep(0.3)
        return self.real_copy2(source, destination, **kwargs)

    def _resolve_in_thread(self, results, index):
        from modules.fast_checkpoint import resolve_checkpoint_path
        results[index] = resolve_checkpoint_path(
            self.checkpoint_name, [self.slow_dir], fast_path=self.fast_dir
        )

    def _join(self, threads):
        for thread in threads:
            thread.join(timeout=self.JOIN_TIMEOUT)
            self.assertFalse(thread.is_alive(), 'resolver thread did not finish')

    def test_concurrent_requests_copy_once_and_never_warn(self):
        results = [None, None]
        first = threading.Thread(target=self._resolve_in_thread, args=(results, 0))
        second = threading.Thread(target=self._resolve_in_thread, args=(results, 1))

        with patch('modules.fast_checkpoint.shutil.copy2', side_effect=self._slow_copy2), \
                self.assertNoLogs('modules.fast_checkpoint', level='WARNING'):
            first.start()
            self.assertTrue(self.copy_started.wait(timeout=self.JOIN_TIMEOUT))
            second.start()
            self._join([first, second])

        self.assertEqual(results, [self.fast_file, self.fast_file])
        self.assertEqual(len(self.copy_calls), 1)
        self.assertEqual(os.path.getsize(self.fast_file), os.path.getsize(self.slow_path))

    def test_final_path_absent_until_copy_completes(self):
        results = [None]
        tmp_fully_written = threading.Event()

        def copy2_then_signal(source, destination, **kwargs):
            self._slow_copy2(source, destination, **kwargs)
            tmp_fully_written.set()

        copier = threading.Thread(target=self._resolve_in_thread, args=(results, 0))
        with patch('modules.fast_checkpoint.shutil.copy2', side_effect=copy2_then_signal):
            copier.start()
            self.assertTrue(self.copy_started.wait(timeout=self.JOIN_TIMEOUT))
            observations = []
            while not tmp_fully_written.is_set():
                exists = os.path.exists(self.fast_file)
                # The rename only happens after the signal, so an observation
                # taken while the signal is still unset predates it.
                if not tmp_fully_written.is_set():
                    observations.append(exists)
                time.sleep(0.01)
            self._join([copier])

        self.assertTrue(observations)
        self.assertFalse(any(observations))
        self.assertEqual(results, [self.fast_file])
        self.assertEqual(os.path.getsize(self.fast_file), os.path.getsize(self.slow_path))

    def test_tmp_name_is_unique_per_call(self):
        from modules.fast_checkpoint import resolve_checkpoint_path
        with patch('modules.fast_checkpoint.shutil.copy2', side_effect=self.real_copy2) as spy:
            resolve_checkpoint_path(self.checkpoint_name, [self.slow_dir], fast_path=self.fast_dir)
            future = time.time() + 100
            os.utime(self.slow_path, (future, future))  # make the cached copy stale
            resolve_checkpoint_path(self.checkpoint_name, [self.slow_dir], fast_path=self.fast_dir)

        tmp_paths = [call.args[1] for call in spy.call_args_list]
        self.assertEqual(len(tmp_paths), 2)
        self.assertNotEqual(tmp_paths[0], tmp_paths[1])
        for tmp_path in tmp_paths:
            self.assertNotEqual(tmp_path, self.fast_file + '.tmp')
            self.assertEqual(os.path.dirname(tmp_path), self.fast_dir)

    def test_failed_copy_removes_only_its_own_tmp(self):
        from modules.fast_checkpoint import resolve_checkpoint_path
        stray_tmp = os.path.join(self.fast_dir, 'model.safetensors.other.tmp')
        with open(stray_tmp, 'wb') as f:
            f.write(b'in-flight data of another copier')

        with patch('modules.fast_checkpoint.shutil.copy2', side_effect=OSError('disk full')):
            result = resolve_checkpoint_path(
                self.checkpoint_name, [self.slow_dir], fast_path=self.fast_dir
            )

        self.assertEqual(result, self.slow_path)
        self.assertEqual(os.listdir(self.fast_dir), ['model.safetensors.other.tmp'])
        with open(stray_tmp, 'rb') as f:
            self.assertEqual(f.read(), b'in-flight data of another copier')

    def test_concurrent_requests_for_different_checkpoints_do_not_serialize(self):
        from modules.fast_checkpoint import resolve_checkpoint_path
        other_name = 'other.safetensors'
        with open(os.path.join(self.slow_dir, other_name), 'wb') as f:
            f.write(b'\x02' * 2048)

        both_copying = threading.Barrier(2, timeout=self.JOIN_TIMEOUT)

        def rendezvous_copy2(source, destination, **kwargs):
            both_copying.wait()  # only released if both copies are in flight together
            return self.real_copy2(source, destination, **kwargs)

        errors = []

        def resolve(name):
            try:
                resolve_checkpoint_path(name, [self.slow_dir], fast_path=self.fast_dir)
            except threading.BrokenBarrierError as e:
                errors.append(e)

        threads = [threading.Thread(target=resolve, args=(name,))
                   for name in (self.checkpoint_name, other_name)]
        with patch('modules.fast_checkpoint.shutil.copy2', side_effect=rendezvous_copy2):
            for thread in threads:
                thread.start()
            self._join(threads)

        self.assertEqual(errors, [])
        self.assertTrue(os.path.isfile(self.fast_file))
        self.assertTrue(os.path.isfile(os.path.join(self.fast_dir, other_name)))


if __name__ == '__main__':
    unittest.main()
