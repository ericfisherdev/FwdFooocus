"""
Fast Checkpoint Storage Module

Caches checkpoint files on a fast drive for quicker loading.
On first use, copies a checkpoint from the slow drive to the fast drive.
Subsequent loads use the cached fast copy.
"""

import logging
import os
import shutil
import tempfile
import threading
import time

logger = logging.getLogger(__name__)

# One lock per fast-drive destination so concurrent callers (worker thread,
# Gradio request threads, FastAPI threadpool) copy a checkpoint once and the
# rest wait for the finished file. The map holds one tiny entry per distinct
# checkpoint path and is never evicted. In-process locking is sufficient: the
# app is a single process.
_copy_locks_guard = threading.Lock()
_copy_locks: dict[str, threading.Lock] = {}


def _destination_lock(fast_file: str) -> threading.Lock:
    """Return the lock serialising copies to `fast_file`, creating it on first use."""
    with _copy_locks_guard:
        return _copy_locks.setdefault(fast_file, threading.Lock())


def _find_in_folders(name: str, folders: list[str]) -> str:
    """
    Search a list of folders for a file by name.

    Mirrors the behaviour of modules.util.get_file_from_folder_list without
    pulling in that module's heavy transitive dependencies (numpy, torch, …).

    Returns the absolute real path of the first match, or the constructed path
    in the first folder when the file does not exist in any of the folders.
    """
    if not isinstance(folders, list):
        folders = [folders]

    for folder in folders:
        candidate = os.path.abspath(os.path.realpath(os.path.join(folder, name)))
        if os.path.isfile(candidate):
            return candidate

    return os.path.abspath(os.path.realpath(os.path.join(folders[0], name)))


def resolve_checkpoint_path(
    checkpoint_name: str,
    checkpoint_folders: list[str],
    fast_path: str | None = None,
) -> str:
    """
    Resolve the path for a checkpoint, caching it on the fast drive if configured.

    Concurrent calls for the same checkpoint are single-flight: one copies,
    the others block until it finishes and then serve the finished file.

    When a fast copy already exists, it is revalidated against the source
    checkpoint using a cheap (mtime, size) comparison before being served.
    If the source has changed (re-download, in-place edit), the fast copy is
    refreshed; if the source no longer exists, the existing fast copy is
    served as-is rather than treated as an error.

    Args:
        checkpoint_name: Checkpoint filename or relative path (e.g. 'model.safetensors').
        checkpoint_folders: List of directories to search for checkpoints.
        fast_path: Path to the fast checkpoint cache directory, or None if disabled.

    Returns:
        Absolute path to the checkpoint file (on fast drive if available,
        otherwise from the original location).
    """
    if fast_path is None:
        return _find_in_folders(checkpoint_name, checkpoint_folders)

    # Validate that the destination stays inside the fast cache root
    safe_name = os.path.normpath(checkpoint_name)
    if os.path.isabs(safe_name) or safe_name.startswith('..' + os.sep) or safe_name == '..':
        logger.warning(f"Refusing unsafe checkpoint path for fast cache: {checkpoint_name}")
        return _find_in_folders(checkpoint_name, checkpoint_folders)

    fast_root = os.path.abspath(os.path.realpath(fast_path))
    fast_file = os.path.abspath(os.path.realpath(os.path.join(fast_root, safe_name)))
    if os.path.commonpath([fast_root, fast_file]) != fast_root:
        logger.warning(f"Resolved fast-cache path escapes cache root: {checkpoint_name}")
        return _find_in_folders(checkpoint_name, checkpoint_folders)

    with _destination_lock(fast_file):
        return _serve_fast_copy(checkpoint_name, checkpoint_folders, fast_file)


def _serve_fast_copy(
    checkpoint_name: str,
    checkpoint_folders: list[str],
    fast_file: str,
) -> str:
    """
    Return the path to serve for `checkpoint_name`, copying to `fast_file` if needed.

    Must be called with the destination lock for `fast_file` held, so the
    exists / staleness / copy decision is atomic with respect to other callers.
    """
    original_path = _find_in_folders(checkpoint_name, checkpoint_folders)

    if os.path.isfile(fast_file):
        if not os.path.isfile(original_path):
            # Source is gone; the fast copy is all that remains.
            return fast_file

        if _fast_copy_is_stale(original_path, fast_file):
            logger.info(
                f"Fast-cache copy is stale, refreshing: {checkpoint_name}"
            )
            return _copy_to_fast_drive(original_path, fast_file)

        return fast_file

    if not os.path.isfile(original_path):
        return original_path

    return _copy_to_fast_drive(original_path, fast_file)


def _fast_copy_is_stale(source_path: str, fast_file: str) -> bool:
    """
    Check whether a fast-drive copy is out of date relative to its source.

    Uses a cheap stat-based (mtime, size) comparison rather than hashing file
    contents, keeping the common unchanged-source path fast.

    Args:
        source_path: Path to the original checkpoint file.
        fast_file: Path to the cached copy on the fast drive.

    Returns:
        True if the source's mtime or size differs from the fast copy's.
    """
    source_stat = os.stat(source_path)
    fast_stat = os.stat(fast_file)
    return (
        source_stat.st_mtime_ns != fast_stat.st_mtime_ns
        or source_stat.st_size != fast_stat.st_size
    )


def _copy_to_fast_drive(source_path: str, dest_path: str) -> str:
    """
    Copy a checkpoint file to the fast drive using atomic write.

    Two invariants hold regardless of concurrency:
    - Every call writes to its own unique temporary file, so copiers never
      truncate, rename, or delete one another's in-flight data.
    - `dest_path` is only ever created by `os.replace` of a fully written and
      closed file, so no reader can open a partial file at the final path.

    Args:
        source_path: Path to the original checkpoint file.
        dest_path: Target path on the fast drive.

    Returns:
        dest_path on success, source_path on failure.
    """
    tmp_path = None
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        file_size_mb = os.path.getsize(source_path) / (1024 * 1024)
        logger.info(
            f"Copying checkpoint to fast storage: "
            f"{os.path.basename(source_path)} ({file_size_mb:.0f} MB)"
        )

        tmp_path = _create_unique_tmp_file(dest_path)
        start_time = time.time()
        shutil.copy2(source_path, tmp_path)
        os.replace(tmp_path, dest_path)
        elapsed = time.time() - start_time

        logger.info(
            f"Checkpoint cached on fast storage in {elapsed:.1f}s: {dest_path}"
        )
        return dest_path

    except OSError as e:
        logger.warning(
            f"Failed to cache checkpoint on fast storage: {e}. "
            f"Loading from original location."
        )
        if tmp_path is not None:
            _remove_quietly(tmp_path)
        return source_path


def _create_unique_tmp_file(dest_path: str) -> str:
    """
    Create an empty, uniquely named temporary file beside `dest_path`.

    The file lives in the destination's directory so the later `os.replace`
    stays on one filesystem (and therefore atomic). The descriptor is closed
    immediately because `shutil.copy2` reopens by path; `copy2` also copies
    the source's mode bits onto it, so the final file keeps the source's
    permissions.

    Raises:
        OSError: if the temporary file cannot be created.
    """
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(dest_path),
        prefix=os.path.basename(dest_path) + '.',
        suffix='.tmp',
    )
    os.close(fd)
    return tmp_path


def _remove_quietly(path: str) -> None:
    """Remove `path`, ignoring a file that is already gone or cannot be removed."""
    try:
        os.remove(path)
    except OSError:
        pass
