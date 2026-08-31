"""
Image List Management Module (FWDF-188)

Mirrors modules/lora_presets.py's directory-listing idiom, but each "list"
is a directory of saved image copies plus its own log.html rather than a
single JSON file. Saving an image to a list is additive: the image and its
default per-date log.html entry (modules/private_logger.log) are left
untouched -- this module writes a *second* copy of the file and a second
log entry under the list's own directory.

UI-agnostic on purpose (no gradio import) so both the legacy Gradio UI
(webui.py, this ticket) and the new FastAPI UI (new_ui/, a follow-up
ticket) can call it directly. The output root is injected as a parameter
(root_dir) rather than read from modules.config deep inside each function,
so callers own config lookups and tests never need modules.config.
"""
import os
import re
import shutil
from typing import Optional

from PIL import Image

import modules.meta_parser
from modules.private_logger import append_log_entry

# Mirrors modules/lora_presets.py's sanitize_preset_name: invalid filesystem
# characters are replaced rather than rejected, matching this ticket's
# acceptance criterion that invalid characters in a list name are
# sanitized, not refused outright.
_INVALID_NAME_CHARS = re.compile(r'[\\/:*?"<>|]')

# In-process registry: absolute source image path -> (metadata, task) as
# passed to private_logger.log() for that image. Populated by
# record_metadata() from modules.async_worker.save_and_log right after each
# image is written, so a later "Save to List" click can write a full
# log.html entry without re-deriving anything. Does not survive a process
# restart -- see get_metadata()'s embedded-metadata fallback.
_metadata_registry: dict[str, tuple[list, Optional[dict]]] = {}


def sanitize_list_name(name: str) -> str:
    """Sanitize a list name to be safe for filesystem use."""
    sanitized = _INVALID_NAME_CHARS.sub('_', name)
    sanitized = sanitized.strip('. ')
    if not sanitized:
        sanitized = 'unnamed_list'
    return sanitized


def _resolve_confined(target: str, root_dir: str) -> Optional[str]:
    """Resolve target and root_dir through symlinks and verify true path
    containment via os.path.commonpath.

    This is the authoritative safety check -- mirrors
    modules/wildcard_ui.py's _resolve_confined(): os.path.commonpath on
    os.path.realpath'd inputs catches a symlink inside root_dir pointing
    outside it, which a plain os.path.normpath (no symlink following) or a
    bare string-prefix comparison would miss. Returns the resolved path
    when confined, else None.
    """
    real_root = os.path.realpath(root_dir)
    real_target = os.path.realpath(target)
    if os.path.commonpath([real_root, real_target]) != real_root:
        return None
    return real_target


def get_list_dir(name: str, root_dir: str) -> Optional[str]:
    """Resolve name to its confined, absolute list directory under
    root_dir, or None if the sanitized name resolves outside root_dir
    (e.g. via a symlink planted inside root_dir)."""
    safe_name = sanitize_list_name(name)
    target = os.path.join(root_dir, safe_name)

    real_target = _resolve_confined(target, root_dir)
    if real_target is None:
        return None

    # CodeQL's py/path-injection recognized containment pattern (normalize,
    # then verify the startswith-prefix), applied to the already
    # symlink-resolved real_target -- mirrors wildcard_ui.py's
    # resolve_wildcard_path/write_wildcard, which was hardened the same way.
    base_path = os.path.normpath(os.path.realpath(root_dir))
    full_path = os.path.normpath(real_target)
    if not full_path.startswith(base_path + os.sep):
        return None

    return full_path


def list_image_lists(root_dir: str) -> list[str]:
    """List all existing image lists (subdirectories of root_dir), sorted
    alphabetically."""
    try:
        return sorted(
            entry for entry in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, entry))
        )
    except OSError as e:
        print(f'Error listing image lists: {e}')
        return []


def list_exists(name: str, root_dir: str) -> bool:
    list_dir = get_list_dir(name, root_dir)
    return list_dir is not None and os.path.isdir(list_dir)


def record_metadata(path: str, metadata: list, task: Optional[dict]) -> None:
    """Record the metadata/task used to save `path` so a later Save to List
    click can write a full log entry without re-deriving anything. Called
    from modules.async_worker.save_and_log right after each image is
    written."""
    _metadata_registry[os.path.abspath(path)] = (metadata, task)


def get_metadata(path: str) -> tuple[Optional[list], Optional[dict]]:
    """Return (metadata, task) previously recorded for path via
    record_metadata(), or fall back to the image's own embedded metadata
    when the in-process registry has no entry (e.g. after a server
    restart). Returns (None, None) if neither source yields usable
    metadata -- callers should fall back to a reduced entry rather than
    raise."""
    entry = _metadata_registry.get(os.path.abspath(path))
    if entry is not None:
        return entry

    try:
        with Image.open(path) as image:
            parameters, _metadata_scheme = modules.meta_parser.read_info_from_image(image)
    except (OSError, ValueError) as e:
        print(f'Failed to read embedded metadata from {path}: {e}')
        return None, None

    if not parameters:
        return None, None

    if isinstance(parameters, dict):
        metadata = [(key, key, value) for key, value in parameters.items()]
    else:
        metadata = [('Parameters', 'parameters', parameters)]

    return metadata, None


def save_image_to_list(name: str, source_image_path: str, root_dir: str,
                        metadata: Optional[list] = None,
                        task: Optional[dict] = None) -> tuple[bool, str]:
    """Save (copy) source_image_path into the named list, creating the list
    directory if needed, and append a log.html entry for it.

    Additive: the caller's default-output copy and its date-folder
    log.html entry (private_logger.log) are untouched -- this only ever
    writes into root_dir/<sanitized name>/. Saving the same filename twice
    into one list overwrites the copy (exact bytes, via shutil.copy2) but
    does not duplicate the log entry.
    """
    if not os.path.isfile(source_image_path):
        return False, f"Source image not found: {source_image_path}"

    list_dir = get_list_dir(name, root_dir)
    if list_dir is None:
        return False, f"Invalid list name: {name!r}"

    list_name = os.path.basename(list_dir)

    try:
        os.makedirs(list_dir, exist_ok=True)

        only_name = os.path.basename(source_image_path)
        dest_path = os.path.join(list_dir, only_name)
        already_saved = os.path.exists(dest_path)

        # shutil.copy2 preserves exact bytes -- including any embedded
        # metadata -- rather than re-encoding. The numpy array that
        # produced the image no longer exists at click time, so re-running
        # private_logger.log() with an output-folder override is not an
        # option here.
        shutil.copy2(source_image_path, dest_path)

        if already_saved:
            return True, f"Updated '{only_name}' in list '{list_name}'"

        entry_metadata, entry_task = metadata, task
        if entry_metadata is None:
            entry_metadata, entry_task = get_metadata(source_image_path)
        if entry_metadata is None:
            entry_metadata = [('Filename', 'filename', only_name)]

        html_path = os.path.join(list_dir, 'log.html')
        append_log_entry(html_path, only_name, entry_metadata, task=entry_task, title_suffix=list_name)

        return True, f"Saved '{only_name}' to list '{list_name}'"
    except OSError as e:
        return False, f"Failed to save image to list '{list_name}': {e}"
