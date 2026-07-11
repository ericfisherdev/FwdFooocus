import hashlib
import os
import shutil
import tempfile
import unittest
from unittest import mock
from unittest.mock import patch

import modules.model_loader as model_loader
from modules.model_loader import load_file_from_url


def _write_file(path, content: bytes):
    with open(path, 'wb') as f:
        f.write(content)


class TestLoadFileFromUrl(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_skips_download_when_file_exists_and_no_verification_requested(self):
        cached_file = os.path.join(self.tmp_dir, 'model.safetensors')
        _write_file(cached_file, b'existing content')

        with patch('torch.hub.download_url_to_file') as mock_download:
            result = load_file_from_url(
                url='https://huggingface.co/foo/resolve/main/model.safetensors',
                model_dir=self.tmp_dir,
                file_name='model.safetensors',
            )

        mock_download.assert_not_called()
        self.assertEqual(result, cached_file)

    def test_downloads_when_file_missing(self):
        cached_file = os.path.join(self.tmp_dir, 'model.safetensors')

        def fake_download(url, dst, progress=True):
            _write_file(dst, b'downloaded content')

        with patch('torch.hub.download_url_to_file', side_effect=fake_download) as mock_download:
            result = load_file_from_url(
                url='https://huggingface.co/foo/resolve/main/model.safetensors',
                model_dir=self.tmp_dir,
                file_name='model.safetensors',
            )

        mock_download.assert_called_once()
        self.assertEqual(result, cached_file)
        self.assertTrue(os.path.exists(cached_file))

    def test_infers_file_name_from_url_when_not_provided(self):
        def fake_download(url, dst, progress=True):
            _write_file(dst, b'content')

        with patch('torch.hub.download_url_to_file', side_effect=fake_download):
            result = load_file_from_url(
                url='https://huggingface.co/foo/resolve/main/inferred.safetensors',
                model_dir=self.tmp_dir,
            )

        self.assertEqual(os.path.basename(result), 'inferred.safetensors')

    def test_creates_model_dir_if_missing(self):
        nested_dir = os.path.join(self.tmp_dir, 'nested', 'dir')

        def fake_download(url, dst, progress=True):
            _write_file(dst, b'content')

        with patch('torch.hub.download_url_to_file', side_effect=fake_download):
            result = load_file_from_url(
                url='https://huggingface.co/foo/resolve/main/model.safetensors',
                model_dir=nested_dir,
                file_name='model.safetensors',
            )

        self.assertTrue(os.path.isdir(nested_dir))
        self.assertTrue(os.path.exists(result))

    def test_existing_file_matching_hash_and_size_is_not_redownloaded(self):
        content = b'a' * 128
        cached_file = os.path.join(self.tmp_dir, 'model.safetensors')
        _write_file(cached_file, content)
        expected_sha256 = hashlib.sha256(content).hexdigest()

        with patch('torch.hub.download_url_to_file') as mock_download:
            result = load_file_from_url(
                url='https://huggingface.co/foo/resolve/main/model.safetensors',
                model_dir=self.tmp_dir,
                file_name='model.safetensors',
                expected_sha256=expected_sha256,
                expected_size=len(content),
            )

        mock_download.assert_not_called()
        self.assertEqual(result, cached_file)

    def test_existing_file_with_hash_mismatch_is_deleted_and_redownloaded(self):
        cached_file = os.path.join(self.tmp_dir, 'model.safetensors')
        _write_file(cached_file, b'corrupted content')
        good_content = b'good content'
        expected_sha256 = hashlib.sha256(good_content).hexdigest()

        def fake_download(url, dst, progress=True):
            _write_file(dst, good_content)

        with patch('torch.hub.download_url_to_file', side_effect=fake_download) as mock_download:
            result = load_file_from_url(
                url='https://huggingface.co/foo/resolve/main/model.safetensors',
                model_dir=self.tmp_dir,
                file_name='model.safetensors',
                expected_sha256=expected_sha256,
            )

        mock_download.assert_called_once()
        with open(result, 'rb') as f:
            self.assertEqual(f.read(), good_content)

    def test_existing_file_with_size_mismatch_is_deleted_and_redownloaded(self):
        cached_file = os.path.join(self.tmp_dir, 'model.safetensors')
        _write_file(cached_file, b'short')
        good_content = b'the correct, longer content'

        def fake_download(url, dst, progress=True):
            _write_file(dst, good_content)

        with patch('torch.hub.download_url_to_file', side_effect=fake_download) as mock_download:
            result = load_file_from_url(
                url='https://huggingface.co/foo/resolve/main/model.safetensors',
                model_dir=self.tmp_dir,
                file_name='model.safetensors',
                expected_size=len(good_content),
            )

        mock_download.assert_called_once()
        with open(result, 'rb') as f:
            self.assertEqual(f.read(), good_content)

    def test_raises_when_freshly_downloaded_file_fails_hash_verification(self):
        expected_sha256 = hashlib.sha256(b'the real content').hexdigest()

        def fake_download(url, dst, progress=True):
            _write_file(dst, b'wrong content')

        with patch('torch.hub.download_url_to_file', side_effect=fake_download):
            with self.assertRaises(RuntimeError) as ctx:
                load_file_from_url(
                    url='https://huggingface.co/foo/resolve/main/model.safetensors',
                    model_dir=self.tmp_dir,
                    file_name='model.safetensors',
                    expected_sha256=expected_sha256,
                )

        self.assertIn('SHA256 mismatch', str(ctx.exception))

    def test_raises_when_freshly_downloaded_file_fails_size_verification(self):
        def fake_download(url, dst, progress=True):
            _write_file(dst, b'short')

        with patch('torch.hub.download_url_to_file', side_effect=fake_download):
            with self.assertRaises(RuntimeError) as ctx:
                load_file_from_url(
                    url='https://huggingface.co/foo/resolve/main/model.safetensors',
                    model_dir=self.tmp_dir,
                    file_name='model.safetensors',
                    expected_size=999,
                )

        self.assertIn('Size mismatch', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()


class TestVerificationMarker(unittest.TestCase):
    """A successful verification writes a .verified sidecar so warm starts
    skip re-hashing multi-gigabyte files (size still re-checked)."""

    def _write(self, path, data=b'hello'):
        with open(path, 'wb') as f:
            f.write(data)

    def test_marker_written_after_successful_cached_verification(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, 'model.bin')
            self._write(path)
            sha = hashlib.sha256(b'hello').hexdigest()
            with mock.patch('torch.hub.download_url_to_file') as dl:
                model_loader.load_file_from_url(
                    'https://huggingface.co/x/resolve/abc/model.bin',
                    model_dir=tmp_dir, expected_sha256=sha, expected_size=5)
            dl.assert_not_called()
            self.assertTrue(os.path.exists(path + '.verified'))

    def test_marker_skips_rehash_on_warm_start(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, 'model.bin')
            self._write(path)
            sha = hashlib.sha256(b'hello').hexdigest()
            kwargs = dict(model_dir=tmp_dir, expected_sha256=sha, expected_size=5)
            url = 'https://huggingface.co/x/resolve/abc/model.bin'
            with mock.patch('torch.hub.download_url_to_file'):
                model_loader.load_file_from_url(url, **kwargs)
                with mock.patch.object(model_loader, '_sha256_of_file') as hasher:
                    model_loader.load_file_from_url(url, **kwargs)
                    hasher.assert_not_called()

    def test_marker_for_different_expectation_does_not_skip(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, 'model.bin')
            self._write(path)
            sha = hashlib.sha256(b'hello').hexdigest()
            url = 'https://huggingface.co/x/resolve/abc/model.bin'
            def fake_download(u, dest, progress=True):
                self._write(dest)

            with mock.patch('torch.hub.download_url_to_file', side_effect=fake_download):
                model_loader.load_file_from_url(url, model_dir=tmp_dir, expected_sha256=sha, expected_size=5)
                # New expectation (different pin) must re-verify, not trust the old marker.
                with self.assertRaises(RuntimeError):
                    model_loader.load_file_from_url(url, model_dir=tmp_dir,
                                                    expected_sha256='0' * 64, expected_size=5)

    def test_size_change_invalidates_marker(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, 'model.bin')
            self._write(path)
            sha = hashlib.sha256(b'hello').hexdigest()
            url = 'https://huggingface.co/x/resolve/abc/model.bin'
            with mock.patch('torch.hub.download_url_to_file') as dl:
                model_loader.load_file_from_url(url, model_dir=tmp_dir, expected_sha256=sha, expected_size=5)
                self._write(path, b'hello-truncation-changed')

                def restore(u, dest, progress=True):
                    self._write(dest)

                dl.side_effect = restore
                model_loader.load_file_from_url(url, model_dir=tmp_dir, expected_sha256=sha, expected_size=5)
                dl.assert_called_once()
