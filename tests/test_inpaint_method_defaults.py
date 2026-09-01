"""Tests for the per-method inpaint engine, denoising strength and respective
field config keys added to modules.config (FWDF-193)."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# args_manager calls parse_args() at import time, which chokes on pytest's
# argv. Patch sys.argv before any project modules are imported.
_original_argv = sys.argv
sys.argv = [sys.argv[0]]

import modules.config  # noqa: E402
import modules.flags  # noqa: E402

sys.argv = _original_argv


SHIPPED_DEFAULTS = {
    modules.flags.inpaint_option_default: {
        'engine': modules.config.default_inpaint_engine_version,
        'strength': 1.0,
        'respective_field': 0.618,
    },
    modules.flags.inpaint_option_detail: {
        'engine': 'None',
        'strength': 0.5,
        'respective_field': 0.0,
    },
    modules.flags.inpaint_option_modify: {
        'engine': modules.config.default_inpaint_engine_version,
        'strength': 1.0,
        'respective_field': 0.0,
    },
}


def _fraction_validator(x):
    """Mirrors the production validator used for the strength and
    respective-field keys in modules.config."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 1


def _engine_validator(x):
    """Mirrors the production validator used for the engine keys in
    modules.config."""
    return x in modules.flags.inpaint_engine_versions


def _reload_config_key(key, raw_value, default_value, validator, expected_type):
    """Sets config_dict[key] to raw_value, re-runs get_config_item_or_set_default,
    then restores config_dict/visited_keys so other tests are unaffected."""
    was_visited = key in modules.config.visited_keys
    modules.config.config_dict[key] = raw_value
    try:
        return modules.config.get_config_item_or_set_default(
            key=key, default_value=default_value, validator=validator,
            expected_type=expected_type)
    finally:
        modules.config.config_dict.pop(key, None)
        if not was_visited and key in modules.config.visited_keys:
            modules.config.visited_keys.remove(key)


class TestShippedDefaults:
    def test_all_three_methods_present_in_each_dict(self):
        for method in modules.flags.inpaint_options:
            assert method in modules.config.default_inpaint_engines
            assert method in modules.config.default_inpaint_strengths
            assert method in modules.config.default_inpaint_respective_fields

    def test_shipped_values_match_description_when_no_keys_set(self):
        for method, expected in SHIPPED_DEFAULTS.items():
            assert modules.config.default_inpaint_engines[method] == expected['engine']
            assert modules.config.default_inpaint_strengths[method] == expected['strength']
            assert modules.config.default_inpaint_respective_fields[method] == expected['respective_field']


class TestNineKeysVisited:
    def test_all_nine_keys_are_visited(self):
        expected_keys = [
            f'default_inpaint_{field}_{slug}'
            for field in ('engine', 'strength', 'respective_field')
            for slug in ('default', 'detail', 'modify')
        ]
        for key in expected_keys:
            assert key in modules.config.visited_keys


class TestValidationFallback:
    def test_invalid_engine_falls_back_to_shipped_default(self):
        result = _reload_config_key(
            'default_inpaint_engine_default', 'not-a-real-engine',
            modules.config.default_inpaint_engine_version, _engine_validator, str)
        assert result == modules.config.default_inpaint_engine_version

    def test_strength_above_range_falls_back_to_shipped_default(self):
        result = _reload_config_key(
            'default_inpaint_strength_default', 1.5, 1.0, _fraction_validator, float)
        assert result == 1.0

    def test_respective_field_below_range_falls_back_to_shipped_default(self):
        result = _reload_config_key(
            'default_inpaint_respective_field_default', -0.1, 0.618, _fraction_validator, float)
        assert result == 0.618

    def test_wrong_type_string_falls_back_to_shipped_default(self):
        result = _reload_config_key(
            'default_inpaint_strength_detail', 'not-a-number', 0.5, _fraction_validator, float)
        assert result == 0.5

    def test_bool_falls_back_to_shipped_default(self):
        result = _reload_config_key(
            'default_inpaint_strength_modify', True, 1.0, _fraction_validator, float)
        assert result == 1.0


class TestAcceptedEdgeValues:
    def test_integer_zero_passes_fraction_validator(self):
        result = _reload_config_key(
            'default_inpaint_respective_field_detail', 0, 0.0, _fraction_validator, float)
        assert result == 0

    def test_integer_one_passes_fraction_validator(self):
        result = _reload_config_key(
            'default_inpaint_strength_default', 1, 1.0, _fraction_validator, float)
        assert result == 1

    def test_string_none_passes_engine_validator(self):
        result = _reload_config_key(
            'default_inpaint_engine_modify', 'None',
            modules.config.default_inpaint_engine_version, _engine_validator, str)
        assert result == 'None'
