"""
Metadata Load Confirmation Diff/Merge Helper

Pure-logic module (no Gradio import) that computes which metadata fields
differ from the current UI selections and merges user/remembered decisions
into the metadata dict before it is passed to the existing loader
(modules.meta_parser.load_parameter_button_click).

This is the foundation for the metadata-load confirmation modal
(FWDF-182) and reuses the remember-my-decision preferences from
FWDF-180 (modules.ui_prefs).
"""

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from modules.ui_prefs import PrefField, RememberDecision

# Mirrors the get_str calls in modules/meta_parser.py:44-49, which read
# source_dict.get(key, source_dict.get(fallback, default)). All metadata
# parsers emit the primary key (see A1111MetadataParser.to_json), but the
# fallback must still be checked and deleted since raw JSON pasted into
# the prompt box can carry either form.
METADATA_KEYS: dict[PrefField, tuple[str, str]] = {
    PrefField.CHECKPOINT: ('base_model', 'Base Model'),
    PrefField.SAMPLER: ('sampler', 'Sampler'),
    PrefField.SCHEDULER: ('scheduler', 'Scheduler'),
    PrefField.VAE: ('vae', 'VAE'),
}


@dataclass(frozen=True, slots=True)
class FieldDiff:
    field: PrefField
    metadata_value: str
    current_value: str


class PrefsLike(Protocol):
    """Structural contract for the prefs source consumed by resolve().

    Any object with this shape (e.g. modules.ui_prefs.UIPrefs) satisfies
    it without a concrete dependency.
    """

    def get(self, field: PrefField) -> RememberDecision:
        ...


def _get_metadata_value(metadata: Mapping[str, object], field: PrefField) -> str | None:
    """
    Look up a field's value in the metadata dict, honoring key-then-fallback
    order, matching the isinstance assert in modules.meta_parser.get_str.

    Args:
        metadata: The loaded metadata dict.
        field: The pref field to look up.

    Returns:
        The string value if present under either key variant and it is a
        str, otherwise None.
    """
    key, fallback_key = METADATA_KEYS[field]
    value = metadata.get(key, metadata.get(fallback_key))
    if not isinstance(value, str):
        return None
    return value


def compute_diffs(metadata: Mapping[str, object], current: Mapping[PrefField, str]) -> list[FieldDiff]:
    """
    Compute the fields whose metadata value differs from the current UI
    selection.

    Args:
        metadata: The loaded metadata dict.
        current: The current UI selection for each field.

    Returns:
        A FieldDiff for every field present in both metadata and current
        whose values differ (exact string comparison — values are
        pre-normalized filenames/sampler keys).
    """
    diffs = []
    for field in METADATA_KEYS:
        metadata_value = _get_metadata_value(metadata, field)
        if metadata_value is None or field not in current:
            continue
        current_value = current[field]
        if metadata_value != current_value:
            diffs.append(FieldDiff(field=field, metadata_value=metadata_value, current_value=current_value))
    return diffs


def _delete_field_keys(metadata: dict, field: PrefField) -> None:
    """Delete both key variants for a field from a metadata dict copy."""
    key, fallback_key = METADATA_KEYS[field]
    metadata.pop(key, None)
    metadata.pop(fallback_key, None)


def resolve(
    metadata: Mapping[str, object],
    diffs: Sequence[FieldDiff],
    prefs: PrefsLike,
) -> tuple[dict, list[FieldDiff]]:
    """
    Apply remembered decisions to the diffed fields, producing a resolved
    copy of the metadata dict plus the diffs that still need to be asked
    about.

    A bare gr.update() is Gradio 3.41.2's no-change sentinel, and
    load_parameter_button_click appends gr.update() for any key absent
    from the metadata dict (meta_parser.py:75-83). Deleting a key
    therefore implements keep-current with zero loader changes.

    Args:
        metadata: The loaded metadata dict. Never mutated.
        diffs: The diffs to resolve, typically from compute_diffs.
        prefs: Any object with get(field) -> RememberDecision.

    Returns:
        (resolved_metadata_copy, remaining_ask_diffs) where remaining_ask_diffs
        are the diffs whose remembered decision is ASK, to drive the
        FWDF-182 modal.
    """
    resolved = dict(metadata)
    ask_diffs = []

    for diff in diffs:
        decision = prefs.get(diff.field)
        if decision == RememberDecision.KEEP_CURRENT:
            _delete_field_keys(resolved, diff.field)
        elif decision == RememberDecision.USE_METADATA:
            pass
        else:
            ask_diffs.append(diff)

    return resolved, ask_diffs


def apply_decisions(metadata: Mapping[str, object], decisions: Mapping[PrefField, RememberDecision]) -> dict:
    """
    Apply a set of explicit decisions (e.g. from the FWDF-182 modal) to a
    metadata dict, producing a resolved copy.

    Args:
        metadata: The loaded metadata dict. Never mutated.
        decisions: The decision to apply for each field.

    Returns:
        A copy of metadata with both key variants deleted for every field
        whose decision is KEEP_CURRENT. All other decisions leave keys
        untouched.
    """
    resolved = dict(metadata)
    for field, decision in decisions.items():
        if decision == RememberDecision.KEEP_CURRENT:
            _delete_field_keys(resolved, field)
    return resolved
