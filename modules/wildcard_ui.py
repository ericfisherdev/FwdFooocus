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


def read_wildcard(name: str, wildcard_dir: str, filenames: list[str] | None = None) -> str:
    # FWDF-186 wires prompt-derived (attacker-influenced) names into this
    # function via the Gradio UI.
    if filenames is None:
        if not WILDCARD_NAME_PATTERN.match(name):
            return ''
        target = os.path.join(wildcard_dir, f'{name}.txt')
    else:
        # Map basenames (derived only from the trusted filenames list) to
        # their original entries, then look the (attacker-influenced) name
        # up in that allowlist -- the result can only ever be a value
        # already present in filenames, never something built from name.
        by_basename = {os.path.splitext(os.path.basename(f))[0]: f for f in filenames}
        match = by_basename.get(name)
        if match is None:
            return ''
        target = os.path.join(wildcard_dir, match)

    # CodeQL's py/path-injection canonical containment check: normalize,
    # then verify the result is confined to base_path before any
    # filesystem access, immediately guarding the exact value used below
    # (github.com/github/codeql query help for py/path-injection).
    base_path = os.path.normpath(wildcard_dir)
    fullpath = os.path.normpath(target)
    if not fullpath.startswith(base_path + os.sep):
        return ''

    if not os.path.isfile(fullpath):
        return ''

    # Mirrors apply_wildcards' tolerant handling of unreadable/undecodable
    # wildcard files (modules/util.py's try/except around the file read):
    # a bad file degrades to empty content rather than crashing the scan.
    try:
        with open(fullpath, encoding='utf-8') as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return ''


def write_wildcard(name: str, content: str, wildcard_dir: str) -> str:
    if not WILDCARD_NAME_PATTERN.match(name):
        raise InvalidWildcardNameError(f'Invalid wildcard name: {name!r}')

    target = os.path.join(wildcard_dir, f'{name}.txt')

    # See read_wildcard's comment: the canonical normalize-then-verify
    # containment check, immediately guarding fullpath before it is used.
    base_path = os.path.normpath(wildcard_dir)
    fullpath = os.path.normpath(target)
    if not fullpath.startswith(base_path + os.sep):
        raise InvalidWildcardNameError(f'Invalid wildcard name: {name!r}')

    os.makedirs(wildcard_dir, exist_ok=True)

    # WILDCARD_NAME_PATTERN and the containment check above rule out an
    # unsafe fullpath, but wildcard_dir's own components could still be
    # symlinked to somewhere unexpected by the time this runs (TOCTOU).
    # O_NOFOLLOW makes symlink-rejection atomic with the open itself
    # instead of just advisory.
    open_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_NOFOLLOW', 0)
    try:
        fd = os.open(fullpath, open_flags, 0o644)
    except OSError as error:
        raise InvalidWildcardNameError(f'Invalid wildcard name: {name!r}') from error

    with open(fd, 'w', encoding='utf-8') as f:
        f.write(content)

    return fullpath
