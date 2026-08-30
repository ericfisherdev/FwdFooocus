"""
UI Preferences Persistence Module

Stores and retrieves persistent user preferences for how metadata-loaded
fields (checkpoint, sampler, scheduler, VAE) are applied, backed by
ui_prefs.json in the repo root next to config.txt. This is the foundation
for the "remember my decision" feature consumed by the metadata-load
confirmation flow (FWDF-181/182) and the Config tab (FWDF-183).
"""

import json
import logging
import os
import threading
from enum import Enum

logger = logging.getLogger(__name__)

_PREFS_GROUP = 'metadata_load'

_path_locks: dict[str, threading.Lock] = {}
_path_locks_registry_lock = threading.Lock()


def _get_path_lock(prefs_path: str) -> threading.Lock:
    """
    Get (creating if needed) the process-wide lock for a prefs file path.

    Keyed by the normalized path so any two UIPrefs instances pointed at
    the same file, not just the same instance, serialize their
    reload-update-save sequences against each other.

    Args:
        prefs_path: Path to the JSON file used for persistence.

    Returns:
        The shared lock for that path.
    """
    normalized = os.path.realpath(prefs_path)
    with _path_locks_registry_lock:
        if normalized not in _path_locks:
            _path_locks[normalized] = threading.Lock()
        return _path_locks[normalized]


class PrefField(Enum):
    CHECKPOINT = 'checkpoint'
    SAMPLER = 'sampler'
    SCHEDULER = 'scheduler'
    VAE = 'vae'


class RememberDecision(Enum):
    ASK = 'ask'
    USE_METADATA = 'use_metadata'
    KEEP_CURRENT = 'keep_current'


DECISION_LABELS: dict[RememberDecision, str] = {
    RememberDecision.ASK: 'Always ask',
    RememberDecision.USE_METADATA: 'Always use loaded value',
    RememberDecision.KEEP_CURRENT: 'Always keep current',
}


def decision_to_label(decision: RememberDecision) -> str:
    """
    Map a RememberDecision to its UI display label.

    Args:
        decision: The decision to look up.

    Returns:
        The display label for the decision.
    """
    return DECISION_LABELS[decision]


def label_to_decision(label: str) -> RememberDecision:
    """
    Map a UI display label back to its RememberDecision.

    Args:
        label: The display label shown in the UI.

    Returns:
        The matching RememberDecision, or RememberDecision.ASK if the
        label is unrecognized.
    """
    for decision, decision_label in DECISION_LABELS.items():
        if decision_label == label:
            return decision
    logger.warning(f"Unknown remember-decision label '{label}', falling back to ASK")
    return RememberDecision.ASK


class UIPrefs:
    """
    Persistent store for remember-my-decision preferences, one decision
    per metadata field, backed by a JSON file on disk.
    """

    def __init__(self, prefs_path: str):
        """
        Args:
            prefs_path: Path to the JSON file used for persistence. Injected
                        via the constructor so tests can point at a tmp path.
        """
        self._prefs_path = prefs_path
        self._lock = _get_path_lock(prefs_path)
        self._cache: dict[PrefField, RememberDecision] | None = None

    def get(self, field: PrefField) -> RememberDecision:
        """
        Get the remembered decision for a metadata field.

        Args:
            field: The metadata field to look up.

        Returns:
            The remembered decision, or RememberDecision.ASK if unset.
        """
        with self._lock:
            self._ensure_loaded()
            return self._cache[field]

    def set(self, field: PrefField, decision: RememberDecision) -> None:
        """
        Set and persist the remembered decision for a metadata field.

        Reloads from disk immediately before applying the change so a
        second UIPrefs instance pointed at the same path (which last
        loaded before this one's update) does not clobber that update
        with its own stale cache.

        Args:
            field: The metadata field to update.
            decision: The decision to remember for that field.
        """
        with self._lock:
            self._cache = self._load()
            self._cache[field] = decision
            self._save()

    def reset_all(self) -> None:
        """Restore RememberDecision.ASK for every field and persist."""
        with self._lock:
            self._cache = {field: RememberDecision.ASK for field in PrefField}
            self._save()

    def as_dict(self) -> dict[str, str]:
        """
        Get all preferences as a plain string-keyed dict, for the Config tab.

        Returns:
            {field.value: decision.value} for all fields.
        """
        with self._lock:
            self._ensure_loaded()
            return {field.value: decision.value for field, decision in self._cache.items()}

    def _ensure_loaded(self) -> None:
        """Lazily load preferences from disk on first access. Caller holds the lock."""
        if self._cache is None:
            self._cache = self._load()

    def _read_raw_payload(self) -> dict:
        """
        Read the raw JSON root object from disk, tolerating any failure.

        Used both to derive this ticket's preferences and, on save, to
        preserve any other top-level groups already on disk.

        Returns:
            The parsed root object, or {} on a missing file, malformed
            JSON, or a payload that is not a JSON object.
        """
        try:
            with open(self._prefs_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            logger.warning(f"Failed to load UI prefs from {self._prefs_path}: {e}")
            return {}

        if not isinstance(payload, dict):
            logger.warning(f"UI prefs file {self._prefs_path} did not contain a JSON object; using defaults")
            return {}

        return payload

    def _load(self) -> dict[PrefField, RememberDecision]:
        """
        Read preferences from the JSON file.

        Falls back to defaults on any missing file, malformed JSON, or
        malformed payload rather than raising to the UI. Unknown enum
        strings degrade that field to ASK individually.

        Returns:
            The loaded (or default) preferences for every field.
        """
        defaults = {field: RememberDecision.ASK for field in PrefField}

        payload = self._read_raw_payload()
        group = payload.get(_PREFS_GROUP, {})
        if not isinstance(group, dict):
            logger.warning(f"UI prefs group '{_PREFS_GROUP}' was not a JSON object; using defaults")
            return defaults

        prefs = dict(defaults)
        for key, value in group.items():
            try:
                field = PrefField(key)
            except ValueError:
                continue
            try:
                prefs[field] = RememberDecision(value)
            except ValueError:
                logger.warning(f"Unknown remember-decision value '{value}' for field '{key}', falling back to ASK")
                prefs[field] = RememberDecision.ASK

        return prefs

    def _save(self) -> None:
        """
        Atomically persist the in-memory preferences to disk.

        Preserves any other top-level groups already on disk (only the
        metadata_load group is owned by this ticket) and writes to a temp
        file in the same directory then renames it into place, which is
        atomic on the same filesystem. On failure, logs a warning and
        leaves the in-memory state untouched.
        """
        payload = self._read_raw_payload()
        payload[_PREFS_GROUP] = {field.value: decision.value for field, decision in self._cache.items()}

        directory = os.path.dirname(self._prefs_path) or '.'
        tmp_path = self._prefs_path + '.tmp'

        try:
            os.makedirs(directory, exist_ok=True)
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, self._prefs_path)
        except OSError as e:
            logger.warning(f"Failed to save UI prefs to {self._prefs_path}: {e}")


default_prefs = UIPrefs(os.path.abspath('./ui_prefs.json'))
