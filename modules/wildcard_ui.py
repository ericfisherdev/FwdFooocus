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


def _confine_to_dir(target: str, wildcard_dir: str) -> str | None:
    """Resolve target and verify it stays inside wildcard_dir.

    Returns the resolved (symlink-following) path when target is confined,
    else None. FWDF-186 wires prompt-derived (attacker-influenced) names
    into read_wildcard/write_wildcard via the Gradio UI, so both callers
    route through this single check rather than trusting name-pattern
    validation alone -- name-pattern checks a few lines away from the
    eventual os.open/open call aren't reliably recognized as a traversal
    barrier by path-injection static analysis, while a normpath+startswith
    containment check immediately guarding the resolved path used at the
    sink is the standard, directly-verifiable pattern.
    """
    real_dir = os.path.realpath(wildcard_dir)
    real_target = os.path.realpath(target)
    if real_target == real_dir or real_target.startswith(real_dir + os.sep):
        return real_target
    return None


def read_wildcard(name: str, wildcard_dir: str, filenames: list[str] | None = None) -> str:
    if filenames is None:
        # name may be arbitrary caller input here (no filenames list to match
        # against), so gate the join the same way write_wildcard gates its
        # own join -- otherwise a name like '../../secrets' escapes the
        # wildcard dir. When filenames is provided, target is chosen from
        # that injected, already-trusted list by basename match instead.
        if not WILDCARD_NAME_PATTERN.match(name):
            return ''
        target = os.path.join(wildcard_dir, f'{name}.txt')
    else:
        matches = [f for f in filenames if os.path.splitext(os.path.basename(f))[0] == name]
        if not matches:
            return ''
        target = os.path.join(wildcard_dir, matches[0])

    real_target = _confine_to_dir(target, wildcard_dir)
    if real_target is None or not os.path.isfile(real_target):
        return ''

    # Mirrors apply_wildcards' tolerant handling of unreadable/undecodable
    # wildcard files (modules/util.py's try/except around the file read):
    # a bad file degrades to empty content rather than crashing the scan.
    try:
        with open(real_target, encoding='utf-8') as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return ''


def write_wildcard(name: str, content: str, wildcard_dir: str) -> str:
    if not WILDCARD_NAME_PATTERN.match(name):
        raise InvalidWildcardNameError(f'Invalid wildcard name: {name!r}')

    target = os.path.join(wildcard_dir, f'{name}.txt')

    # Defense in depth: reject any resolved path that escapes the wildcard
    # directory (covers symlink tricks the name-pattern check can't catch).
    real_target = _confine_to_dir(target, wildcard_dir)
    if real_target is None:
        raise InvalidWildcardNameError(f'Invalid wildcard name: {name!r}')

    os.makedirs(wildcard_dir, exist_ok=True)

    # The realpath check above and this write are two separate syscalls, so
    # a symlink dropped at target in between would otherwise be followed
    # (TOCTOU). O_NOFOLLOW makes symlink-rejection atomic with the open
    # itself instead of just advisory. Opening real_target (not the
    # original, unresolved target) means a symlink swapped into the
    # unresolved path after the check above has no effect either way.
    open_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_NOFOLLOW', 0)
    try:
        fd = os.open(real_target, open_flags, 0o644)
    except OSError as error:
        raise InvalidWildcardNameError(f'Invalid wildcard name: {name!r}') from error

    with open(fd, 'w', encoding='utf-8') as f:
        f.write(content)

    return target
