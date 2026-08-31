"""Pure slot/label helpers for the wildcard button pool (FWDF-186).

Kept separate from modules.wildcard_ui (FWDF-185) so that pool/label
formatting can change independently of the scan/read/write logic, and so
this module stays gradio-free and unit testable the same way wildcard_ui is
(see FWDF-185's docstring). webui.py wraps this module's plain dicts in
gr.update(...) itself -- gradio isn't imported here.
"""
from modules.wildcard_ui import WildcardScan

# Matches the fixed 8-button pool declared for each half (top-level and
# nested) in webui.py.
WILDCARD_BUTTON_POOL_SIZE = 8


def format_wildcard_label(name: str, exists: bool) -> str:
    return f'✏️ {name}' if exists else f'➕ {name} (create)'


def build_wildcard_slots(scan: WildcardScan, pool_size: int = WILDCARD_BUTTON_POOL_SIZE) -> list:
    """Map a WildcardScan onto the fixed button pool.

    Returns a list of length 2 * pool_size: slots[0:pool_size] are the
    top-level pool (existing wildcards first, then missing/create entries,
    capped at pool_size), slots[pool_size:2*pool_size] are the nested pool
    (capped at pool_size). Unused slots are None. Each populated slot is
    {'name': str, 'exists': bool}.
    """
    top_entries = [{'name': name, 'exists': True} for name in scan.top_level]
    top_entries += [{'name': name, 'exists': False} for name in scan.missing]
    top_entries = top_entries[:pool_size]

    nested_entries = [{'name': name, 'exists': True} for name in scan.nested][:pool_size]

    slots = list(top_entries) + [None] * (pool_size - len(top_entries))
    slots += list(nested_entries) + [None] * (pool_size - len(nested_entries))
    return slots


def build_button_updates(slots: list) -> list:
    """Turn a slots list (see build_wildcard_slots) into per-button
    {'label': str, 'visible': bool} dicts, one per slot, in slot order."""
    updates = []
    for slot in slots:
        if slot is None:
            updates.append({'label': '', 'visible': False})
        else:
            updates.append({'label': format_wildcard_label(slot['name'], slot['exists']), 'visible': True})
    return updates


def top_level_section_visible(slots: list, pool_size: int = WILDCARD_BUTTON_POOL_SIZE) -> bool:
    return any(slot is not None for slot in slots[:pool_size])


def nested_section_visible(slots: list, pool_size: int = WILDCARD_BUTTON_POOL_SIZE) -> bool:
    return any(slot is not None for slot in slots[pool_size:2 * pool_size])
