"""Pure-logic helpers for the wildcard editing UI.

This module intentionally does not import modules.config so it can be
unit tested in isolation and reused by any caller that injects its own
wildcard directory / filename list (see FWDF-186 for the FastAPI callers).
"""
import os
import re
from dataclasses import dataclass

# Must mirror the expansion regex used at runtime in apply_wildcards
# (modules/util.py:470: re.findall(r'__([\w-]+)__', wildcard_text)).
# Keep these two patterns in sync.
WILDCARD_PATTERN = re.compile(r'__([\w-]+)__')

# Validates a bare wildcard name for write_wildcard: no path separators,
# no ".." traversal, and no file extension.
WILDCARD_NAME_PATTERN = re.compile(r'^[\w-]+$')


class InvalidWildcardNameError(ValueError):
    """Raised by write_wildcard when a supplied name is unsafe or invalid."""


@dataclass(frozen=True, slots=True)
class WildcardScan:
    top_level: tuple[str, ...]
    missing: tuple[str, ...]
    nested: tuple[str, ...]


def _dedupe_preserve_order(names):
    seen = set()
    result = []
    for name in names:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _wildcard_exists(name: str, filenames: list[str]) -> bool:
    # Mirrors the existence/matching rule used at runtime in apply_wildcards
    # (modules/util.py:477): a name exists if any filename's basename
    # (without extension) equals the wildcard name. Entries may live in
    # subfolders relative to the wildcard dir.
    return any(os.path.splitext(os.path.basename(f))[0] == name for f in filenames)


def scan_prompt(prompt: str, wildcard_dir: str, filenames: list[str]) -> WildcardScan:
    names = _dedupe_preserve_order(WILDCARD_PATTERN.findall(prompt))

    top_level = []
    missing = []
    nested = []
    nested_seen = set()

    for name in names:
        if _wildcard_exists(name, filenames):
            top_level.append(name)
        else:
            missing.append(name)

    # Nested detection is one level deep only: runtime expansion still
    # resolves deep nesting via util.py's BFS up to wildcards_max_bfs_depth.
    for name in top_level:
        content = read_wildcard(name, wildcard_dir, filenames)
        nested_names = _dedupe_preserve_order(WILDCARD_PATTERN.findall(content))
        for nested_name in nested_names:
            if nested_name in nested_seen:
                continue
            nested_seen.add(nested_name)
            if _wildcard_exists(nested_name, filenames):
                nested.append(nested_name)
            elif nested_name not in missing:
                missing.append(nested_name)

    return WildcardScan(
        top_level=tuple(top_level),
        missing=tuple(missing),
        nested=tuple(nested),
    )


def _resolve_confined(target: str, wildcard_dir: str) -> str | None:
    """Resolve target and wildcard_dir through symlinks and verify true
    path containment via os.path.commonpath.

    This is the authoritative safety check: os.path.commonpath on
    os.path.realpath'd inputs is what actually catches a symlink inside
    wildcard_dir pointing outside it -- a plain os.path.normpath (which
    does not follow symlinks) or a bare string-prefix comparison would not.
    Returns the resolved path when confined, else None.
    """
    real_dir = os.path.realpath(wildcard_dir)
    real_target = os.path.realpath(target)
    if os.path.commonpath([real_dir, real_target]) != real_dir:
        return None
    return real_target


def _first_match_by_basename(name: str, filenames: list[str]) -> str | None:
    """Resolve name to the first filenames entry whose basename (without
    extension) equals it -- first match wins, mirroring apply_wildcards'
    own matches[0] (modules/util.py:477), so the editor and runtime
    expansion always agree on which file a duplicate basename resolves to
    (e.g. both 'colors.txt' and 'styles/colors.txt' present)."""
    by_basename = {}
    for f in filenames:
        key = os.path.splitext(os.path.basename(f))[0]
        by_basename.setdefault(key, f)
    return by_basename.get(name)


def resolve_wildcard_path(name: str, wildcard_dir: str, filenames: list[str] | None = None) -> str | None:
    """Resolve name to an existing, confined wildcard file's absolute path,
    or None if it doesn't resolve to a real, in-bounds file.

    Shares target-resolution and containment logic with read_wildcard, but
    returns the path itself rather than file content -- callers that need
    to check existence/size (e.g. to detect a read that degraded to '' due
    to a decoding error, as opposed to a genuinely empty file) can do so
    without going through read_wildcard's tolerant fallback.
    """
    # FWDF-186 wires prompt-derived (attacker-influenced) names into this
    # function via the Gradio UI.
    if filenames is None:
        if not WILDCARD_NAME_PATTERN.match(name):
            return None
        target = os.path.join(wildcard_dir, f'{name}.txt')
    else:
        # First-match-wins lookup in an allowlist built only from the
        # trusted filenames list -- the result can only ever be a value
        # already present in filenames, never something built from name.
        match = _first_match_by_basename(name, filenames)
        if match is None:
            return None
        target = os.path.join(wildcard_dir, match)

    real_target = _resolve_confined(target, wildcard_dir)
    if real_target is None:
        return None

    # CodeQL's py/path-injection recognized containment pattern (normalize,
    # then verify the startswith-prefix), applied to the already
    # symlink-resolved real_target so the value used below is both
    # genuinely confined (via _resolve_confined's commonpath check above)
    # and guarded by the exact shape path-injection static analysis
    # expects immediately before the filesystem call.
    base_path = os.path.normpath(os.path.realpath(wildcard_dir))
    fullpath = os.path.normpath(real_target)
    if not fullpath.startswith(base_path + os.sep):
        return None

    return fullpath if os.path.isfile(fullpath) else None


def read_wildcard(name: str, wildcard_dir: str, filenames: list[str] | None = None) -> str:
    fullpath = resolve_wildcard_path(name, wildcard_dir, filenames)
    if fullpath is None:
        return ''

    # Mirrors apply_wildcards' tolerant handling of unreadable/undecodable
    # wildcard files (modules/util.py's try/except around the file read):
    # a bad file degrades to empty content rather than crashing the scan.
    try:
        with open(fullpath, encoding='utf-8') as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return ''


def write_wildcard(name: str, content: str, wildcard_dir: str, filenames: list[str] | None = None) -> str:
    if not WILDCARD_NAME_PATTERN.match(name):
        raise InvalidWildcardNameError(f'Invalid wildcard name: {name!r}')

    # When filenames is given and name already resolves to an existing
    # entry (possibly in a subfolder), overwrite that same file -- the one
    # read_wildcard would have returned -- rather than always creating a
    # new top-level '<name>.txt' that forks silently from it.
    relative = f'{name}.txt'
    if filenames:
        existing = _first_match_by_basename(name, filenames)
        if existing is not None:
            relative = existing

    target = os.path.join(wildcard_dir, relative)

    real_target = _resolve_confined(target, wildcard_dir)
    if real_target is None:
        raise InvalidWildcardNameError(f'Invalid wildcard name: {name!r}')

    # See read_wildcard's comment: the CodeQL-recognized normalize +
    # startswith guard, applied to the already symlink-resolved real_target.
    base_path = os.path.normpath(os.path.realpath(wildcard_dir))
    fullpath = os.path.normpath(real_target)
    if not fullpath.startswith(base_path + os.sep):
        raise InvalidWildcardNameError(f'Invalid wildcard name: {name!r}')

    os.makedirs(wildcard_dir, exist_ok=True)

    # _resolve_confined and the containment check above rule out an unsafe
    # fullpath, but wildcard_dir's own components -- or fullpath's final
    # component -- could still be (re)symlinked between those checks and
    # this write (TOCTOU). O_NOFOLLOW makes symlink-rejection atomic with
    # the open itself instead of just advisory.
    open_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_NOFOLLOW', 0)
    try:
        fd = os.open(fullpath, open_flags, 0o644)
    except OSError as error:
        raise InvalidWildcardNameError(f'Invalid wildcard name: {name!r}') from error

    with open(fd, 'w', encoding='utf-8') as f:
        f.write(content)

    return fullpath
