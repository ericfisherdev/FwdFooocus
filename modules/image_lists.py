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
import contextlib
import os
import re
import shutil
import stat
import threading
from typing import Optional, Union

from PIL import Image

import modules.meta_parser
from modules.private_logger import append_log_entry

# Mirrors modules/lora_presets.py's sanitize_preset_name: invalid filesystem
# characters are replaced rather than rejected, matching this ticket's
# acceptance criterion that invalid characters in a list name are
# sanitized, not refused outright. Includes ASCII control characters
# (\x00-\x1f, \x7f) -- legal in a POSIX filename but a nuisance on disk and
# a log-forgery aid once they flow into title_suffix/status text/dropdown
# entries verbatim.
_INVALID_NAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f\x7f]')

# In-process registry: absolute source image path -> (metadata, task) as
# passed to private_logger.log() for that image. Populated by
# record_metadata() from modules.async_worker.save_and_log right after each
# image is written, so a later "Save to List" click can write a full
# log.html entry without re-deriving anything. Does not survive a process
# restart -- see get_metadata()'s embedded-metadata fallback.
_metadata_registry: dict[str, tuple[list, Optional[dict]]] = {}

# Serializes the copy-plus-log section of save_image_to_list: the webui
# Save to List handlers run queue=False, so Gradio can invoke them from
# concurrent request threads. append_log_entry does a read-prepend-rewrite
# of the whole log.html (backed by the shared private_logger.log_cache),
# and the already_saved dedup check is a plain os.path.exists -- neither is
# safe without a lock serializing the whole read-check-write section for a
# given process. One process-wide lock is enough: private_logger.log() (the
# per-date, single-worker-thread path) never contends with this.
_save_lock = threading.Lock()


def sanitize_list_name(name: str) -> str:
    """Sanitize a list name to be safe for filesystem use."""
    sanitized = _INVALID_NAME_CHARS.sub('_', name)
    sanitized = sanitized.strip('. ')
    if not sanitized:
        sanitized = 'unnamed_list'
    return sanitized


def _confine(target: str, base_dir: str) -> Optional[str]:
    """Resolve target and base_dir through symlinks and verify true path
    containment, returning the confined absolute path or None.

    Two checks, both required -- mirrors modules/wildcard_ui.py's
    _resolve_confined() plus its inline containment guard:

    1. os.path.commonpath on os.path.realpath'd inputs is the authoritative
       check: it catches a symlink inside base_dir pointing outside it,
       which a plain os.path.normpath (no symlink following) or a bare
       string-prefix comparison would miss.
    2. CodeQL's py/path-injection recognized containment pattern
       (normalize, then verify the startswith-prefix), applied to the
       already symlink-resolved path immediately before the caller's
       filesystem operation. Kept alongside check 1, not instead of it --
       normpath alone does not resolve symlinks, so it cannot replace the
       realpath+commonpath check.

    target need not exist yet (os.path.realpath tolerates a missing final
    component), so this is safe to call before a create/write as well as
    before a read.
    """
    real_base = os.path.realpath(base_dir)
    real_target = os.path.realpath(target)
    if os.path.commonpath([real_base, real_target]) != real_base:
        return None

    base_path = os.path.normpath(real_base)
    full_path = os.path.normpath(real_target)
    if not full_path.startswith(base_path + os.sep):
        return None

    return full_path


def get_list_dir(name: str, root_dir: str) -> Optional[str]:
    """Resolve name to its confined, absolute list directory under
    root_dir, or None if the sanitized name resolves outside root_dir
    (e.g. via a symlink planted inside root_dir, or a name containing '..'
    or a path separator)."""
    safe_name = sanitize_list_name(name)
    target = os.path.join(root_dir, safe_name)
    return _confine(target, root_dir)


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
                        source_root_dirs: Union[str, list[str]],
                        metadata: Optional[list] = None,
                        task: Optional[dict] = None) -> tuple[bool, str]:
    """Save (copy) source_image_path into the named list, creating the list
    directory if needed, and append a log.html entry for it.

    source_root_dirs confines source_image_path the same way root_dir
    confines the list name -- source_image_path must resolve inside at
    least one of them (first match wins), or the save is refused before
    any filesystem call touches it. A single str is also accepted. Callers
    pass every app-controlled directory a gallery path can point into:
    modules.config.path_outputs normally, plus modules.config.temp_path
    when --disable-image-log or a non-persisted intermediate routed the
    image there instead (see modules/private_logger.py's log()). Without
    this confinement, an unvalidated path flowed straight into
    os.path.isfile/os.open.

    Additive: the caller's default-output copy and its date-folder
    log.html entry (private_logger.log) are untouched -- this only ever
    writes into root_dir/<sanitized name>/. Saving the same filename twice
    into one list overwrites the copy (exact bytes) but does not duplicate
    the log entry.
    """
    roots = [source_root_dirs] if isinstance(source_root_dirs, str) else list(source_root_dirs)
    real_source_path = None
    for root in roots:
        real_source_path = _confine(source_image_path, root)
        if real_source_path is not None:
            break
    if real_source_path is None:
        return False, f"Source image outside allowed directory: {source_image_path}"

    if not os.path.isfile(real_source_path):
        return False, f"Source image not found: {source_image_path}"

    list_dir = get_list_dir(name, root_dir)
    if list_dir is None:
        return False, f"Invalid list name: {name!r}"

    list_name = os.path.basename(list_dir)
    only_name = os.path.basename(real_source_path)

    # dest_path is a fresh os.path.join even though list_dir is already
    # confined -- re-confine it here too so the guard sits immediately at
    # the sink for every path this function writes to, not just the ones
    # constructed a function call away.
    real_dest_path = _confine(os.path.join(list_dir, only_name), list_dir)
    if real_dest_path is None:
        return False, f"Invalid destination path for '{only_name}' in list '{list_name}'"

    # Serializes makedirs, the already_saved dedup check, the copy, and
    # append_log_entry's log.html read-modify-write -- see _save_lock's
    # module-level comment. Held across the whole section, not just the
    # write, so two concurrent saves of different files can't both read
    # already_saved=False and both append a log entry, and so two
    # concurrent saves of the *same* file can't interleave their
    # append_log_entry calls and lose one entry to the other's rewrite.
    with _save_lock:
        try:
            os.makedirs(list_dir, exist_ok=True)

            already_saved = os.path.exists(real_dest_path)

            # Preserve exact bytes -- including any embedded metadata --
            # rather than re-encoding. The numpy array that produced the
            # image no longer exists at click time, so re-running
            # private_logger.log() with an output-folder override is not
            # an option here.
            #
            # Written to a temp name next to real_dest_path, then
            # os.replace'd into place, rather than truncated in place, so
            # a failed copy (disk full, crash) can never leave a partial
            # real_dest_path behind -- in-place O_TRUNC would otherwise
            # make every retry see already_saved=True from the partial
            # file and permanently skip the log entry. os.replace also
            # replaces (rather than follows) a symlink at the destination.
            #
            # mode/timestamps are applied through the still-open fd
            # (os.fchmod/os.utime), not by path afterwards -- a by-path
            # shutil.copystat call here would re-resolve real_dest_path
            # and could follow a symlink swapped in after the fd write,
            # reopening the exact TOCTOU window O_NOFOLLOW below closes.
            #
            # O_NOFOLLOW makes symlink-rejection atomic with the open
            # itself instead of just advisory, mirroring
            # modules/wildcard_ui.py's write_wildcard.
            tmp_dest_path = real_dest_path + '.tmp'
            open_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_NOFOLLOW', 0)
            dest_fd = os.open(tmp_dest_path, open_flags, 0o644)  # OSError caught below

            try:
                with open(real_source_path, 'rb') as src_file, open(dest_fd, 'wb') as dest_file:
                    shutil.copyfileobj(src_file, dest_file)
                    source_stat = os.stat(real_source_path)
                    os.fchmod(dest_file.fileno(), stat.S_IMODE(source_stat.st_mode))
                    dest_file.flush()
                    os.utime(dest_file.fileno(),
                             ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
                os.replace(tmp_dest_path, real_dest_path)
            except OSError:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_dest_path)
                raise

            if already_saved:
                return True, f"Updated '{only_name}' in list '{list_name}'"

            entry_metadata, entry_task = metadata, task
            if entry_metadata is None:
                entry_metadata, entry_task = get_metadata(real_source_path)
            if entry_metadata is None:
                entry_metadata = [('Filename', 'filename', only_name)]

            html_path = os.path.join(list_dir, 'log.html')
            append_log_entry(html_path, only_name, entry_metadata, task=entry_task, title_suffix=list_name)

            return True, f"Saved '{only_name}' to list '{list_name}'"
        except OSError as e:
            return False, f"Failed to save image to list '{list_name}': {e}"
