import hashlib
import os
from urllib.parse import urlparse
from typing import Optional


def _sha256_of_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _describe_mismatch(
        cached_file: str,
        expected_sha256: Optional[str],
        expected_size: Optional[int],
) -> Optional[str]:
    """Return a human-readable description of the first verification failure
    found for `cached_file`, or None if it matches all provided expectations.
    """
    if expected_size is not None:
        actual_size = os.path.getsize(cached_file)
        if actual_size != expected_size:
            return (f'Size mismatch for "{cached_file}": '
                     f'expected {expected_size} bytes, got {actual_size} bytes.')
    if expected_sha256 is not None:
        actual_sha256 = _sha256_of_file(cached_file)
        if actual_sha256.lower() != expected_sha256.lower():
            return (f'SHA256 mismatch for "{cached_file}": '
                     f'expected {expected_sha256}, got {actual_sha256}.')
    return None


def _verification_marker(cached_file: str) -> str:
    return cached_file + '.verified'


def _has_valid_verification_marker(
        cached_file: str,
        expected_sha256: Optional[str],
        expected_size: Optional[int],
) -> bool:
    """True if a sidecar marker records a prior successful verification of
    this exact (sha256, size) expectation AND the file's on-disk size still
    matches the size recorded at verification time.

    Skips re-hashing multi-gigabyte files on every warm start. The disk-size
    re-check uses the size recorded when the marker was written, so it guards
    against truncation/replacement even for callers that only supply
    `expected_sha256`.
    """
    marker = _verification_marker(cached_file)
    if not os.path.exists(marker):
        return False
    try:
        with open(marker, 'r', encoding='utf-8') as f:
            recorded = f.read().strip()
    except OSError:
        return False
    expected_prefix = f'sha256={expected_sha256} size={expected_size} disk_size='
    if not recorded.startswith(expected_prefix):
        return False
    try:
        recorded_disk_size = int(recorded[len(expected_prefix):])
    except ValueError:
        return False
    try:
        return os.path.getsize(cached_file) == recorded_disk_size
    except OSError:
        return False


def _write_verification_marker(
        cached_file: str,
        expected_sha256: Optional[str],
        expected_size: Optional[int],
) -> None:
    disk_size = os.path.getsize(cached_file)
    with open(_verification_marker(cached_file), 'w', encoding='utf-8') as f:
        f.write(f'sha256={expected_sha256} size={expected_size} disk_size={disk_size}')


def _remove_verification_state(cached_file: str) -> None:
    """Delete a cached file and its sidecar marker, tolerating absence."""
    for path in (cached_file, _verification_marker(cached_file)):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        expected_size: Optional[int] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    If `expected_sha256` and/or `expected_size` are provided, an existing cached
    file that fails verification is deleted and re-downloaded; a freshly
    downloaded file that still fails verification raises a `RuntimeError`
    naming the file and the source URL so the caller can retry or investigate.
    A successful verification writes a `.verified` sidecar so subsequent calls
    skip re-hashing the (potentially multi-gigabyte) file; only its size is
    re-checked on warm starts.

    Returns the path to the downloaded file.
    """
    domain = os.environ.get("HF_MIRROR", "https://huggingface.co").rstrip('/')
    url = str.replace(url, "https://huggingface.co", domain, 1)
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))

    verification_requested = expected_sha256 is not None or expected_size is not None

    if os.path.exists(cached_file) and verification_requested:
        if not _has_valid_verification_marker(cached_file, expected_sha256, expected_size):
            mismatch = _describe_mismatch(cached_file, expected_sha256, expected_size)
            if mismatch:
                print(f'{mismatch} Deleting cached file and re-downloading from "{url}".')
                _remove_verification_state(cached_file)
            else:
                _write_verification_marker(cached_file, expected_sha256, expected_size)

    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress)

        if verification_requested:
            mismatch = _describe_mismatch(cached_file, expected_sha256, expected_size)
            if mismatch:
                # Leave nothing behind that a later call could mistake for a
                # verified download.
                _remove_verification_state(cached_file)
                raise RuntimeError(
                    f'{mismatch} The download from "{url}" appears corrupted. '
                    f'It has been deleted; try again, or check your network/mirror.'
                )
            _write_verification_marker(cached_file, expected_sha256, expected_size)

    return cached_file
