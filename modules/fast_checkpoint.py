"""
Fast Checkpoint Storage Module

Caches checkpoint files on a fast drive for quicker loading.
On first use, copies a checkpoint from the slow drive to the fast drive.
Subsequent loads use the cached fast copy.
"""

import logging
import os
import shutil
import time

logger = logging.getLogger(__name__)


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

    fast_file = os.path.join(fast_path, checkpoint_name)

    if os.path.isfile(fast_file):
        return fast_file

    original_path = _find_in_folders(checkpoint_name, checkpoint_folders)

    if not os.path.isfile(original_path):
        return original_path

    return _copy_to_fast_drive(original_path, fast_file)


def _copy_to_fast_drive(source_path: str, dest_path: str) -> str:
    """
    Copy a checkpoint file to the fast drive using atomic write.

    Args:
        source_path: Path to the original checkpoint file.
        dest_path: Target path on the fast drive.

    Returns:
        dest_path on success, source_path on failure.
    """
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        file_size_mb = os.path.getsize(source_path) / (1024 * 1024)
        logger.info(
            f"Copying checkpoint to fast storage: "
            f"{os.path.basename(source_path)} ({file_size_mb:.0f} MB)"
        )

        tmp_path = dest_path + '.tmp'
        start_time = time.time()
        shutil.copy2(source_path, tmp_path)
        os.rename(tmp_path, dest_path)
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
        # Clean up partial tmp file
        tmp_path = dest_path + '.tmp'
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return source_path
