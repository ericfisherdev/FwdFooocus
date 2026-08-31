"""Tests for modules.wildcard_ui_helpers (FWDF-186): the pure slot-mapping
and label-formatting logic behind webui.py's wildcard button pool.

Pure logic like modules.wildcard_ui (FWDF-185), so exercised directly here
without any Gradio wiring. The Gradio wiring and the JS debounce in
javascript/wildcards.js are exercised manually per this ticket's acceptance
criteria.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.wildcard_ui import WildcardScan
from modules.wildcard_ui_helpers import (
    WILDCARD_BUTTON_POOL_SIZE,
    build_button_updates,
    build_wildcard_slots,
    format_wildcard_label,
    nested_section_visible,
    top_level_section_visible,
)


def _scan(top_level=(), missing=(), nested=()):
    return WildcardScan(top_level=tuple(top_level), missing=tuple(missing), nested=tuple(nested))


class TestFormatWildcardLabel:
    def test_existing_wildcard_gets_edit_label(self):
        assert format_wildcard_label('animal', True) == '✏️ animal'

    def test_missing_wildcard_gets_create_label(self):
        assert format_wildcard_label('animal', False) == '➕ animal (create)'


class TestBuildWildcardSlots:
    def test_existing_before_missing_in_top_level_pool(self):
        scan = _scan(top_level=['cat'], missing=['dog'])
        slots = build_wildcard_slots(scan)

        assert slots[0] == {'name': 'cat', 'exists': True}
        assert slots[1] == {'name': 'dog', 'exists': False}
        assert slots[2] is None

    def test_nested_populates_second_pool(self):
        scan = _scan(top_level=['animal'], nested=['dog'])
        slots = build_wildcard_slots(scan)

        assert slots[0] == {'name': 'animal', 'exists': True}
        assert slots[WILDCARD_BUTTON_POOL_SIZE] == {'name': 'dog', 'exists': True}

    def test_empty_scan_is_all_none(self):
        slots = build_wildcard_slots(_scan())

        assert slots == [None] * (2 * WILDCARD_BUTTON_POOL_SIZE)

    def test_more_than_pool_size_top_level_entries_are_capped(self):
        scan = _scan(top_level=[f'w{i}' for i in range(WILDCARD_BUTTON_POOL_SIZE + 5)])
        slots = build_wildcard_slots(scan)

        top_half = slots[:WILDCARD_BUTTON_POOL_SIZE]
        assert len(top_half) == WILDCARD_BUTTON_POOL_SIZE
        assert all(slot is not None for slot in top_half)
        assert top_half[-1] == {'name': f'w{WILDCARD_BUTTON_POOL_SIZE - 1}', 'exists': True}

    def test_more_than_pool_size_nested_entries_are_capped(self):
        scan = _scan(top_level=['animal'], nested=[f'n{i}' for i in range(WILDCARD_BUTTON_POOL_SIZE + 3)])
        slots = build_wildcard_slots(scan)

        nested_half = slots[WILDCARD_BUTTON_POOL_SIZE:]
        assert len(nested_half) == WILDCARD_BUTTON_POOL_SIZE
        assert all(slot is not None for slot in nested_half)


class TestBuildButtonUpdates:
    def test_none_slot_is_hidden_with_empty_label(self):
        updates = build_button_updates([None])

        assert updates == [{'label': '', 'visible': False}]

    def test_populated_slot_is_visible_with_formatted_label(self):
        updates = build_button_updates([{'name': 'cat', 'exists': True}, {'name': 'dog', 'exists': False}])

        assert updates == [
            {'label': '✏️ cat', 'visible': True},
            {'label': '➕ dog (create)', 'visible': True},
        ]


class TestSectionVisibility:
    def test_empty_prompt_hides_both_sections(self):
        slots = build_wildcard_slots(_scan())

        assert top_level_section_visible(slots) is False
        assert nested_section_visible(slots) is False

    def test_top_level_only_shows_top_section_but_not_nested(self):
        slots = build_wildcard_slots(_scan(top_level=['cat']))

        assert top_level_section_visible(slots) is True
        assert nested_section_visible(slots) is False

    def test_nested_entries_show_nested_section(self):
        slots = build_wildcard_slots(_scan(top_level=['animal'], nested=['dog']))

        assert top_level_section_visible(slots) is True
        assert nested_section_visible(slots) is True
