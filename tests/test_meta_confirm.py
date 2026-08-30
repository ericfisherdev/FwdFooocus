import sys
import unittest

# modules.meta_parser transitively imports args_manager, which calls
# args_parser.parser.parse_args() at import time against sys.argv. Reset it
# before the import so pytest's own CLI args (test paths, -v, ...) aren't
# parsed as unrecognized Fooocus flags, mirroring the pattern in
# tests/test_async_worker_inpaint.py.
_original_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    from modules.meta_parser import load_parameter_button_click
    _load_parameter_button_click_import_error = None
except ImportError as e:  # pragma: no cover - exercised only without gradio installed
    load_parameter_button_click = None
    _load_parameter_button_click_import_error = e
finally:
    sys.argv = _original_argv


class TestComputeDiffs(unittest.TestCase):
    def test_empty_result_for_equal_values(self):
        from modules.meta_confirm import compute_diffs
        from modules.ui_prefs import PrefField

        metadata = {'base_model': 'sd_xl_base_1.0.safetensors'}
        current = {PrefField.CHECKPOINT: 'sd_xl_base_1.0.safetensors'}

        self.assertEqual(compute_diffs(metadata, current), [])

    def test_skips_keys_absent_from_metadata(self):
        from modules.meta_confirm import compute_diffs
        from modules.ui_prefs import PrefField

        metadata = {}
        current = {PrefField.CHECKPOINT: 'sd_xl_base_1.0.safetensors'}

        self.assertEqual(compute_diffs(metadata, current), [])

    def test_skips_keys_absent_from_current(self):
        from modules.meta_confirm import compute_diffs

        metadata = {'base_model': 'other_model.safetensors'}
        current = {}

        self.assertEqual(compute_diffs(metadata, current), [])

    def test_detects_diffs_on_all_four_fields(self):
        from modules.meta_confirm import FieldDiff, compute_diffs
        from modules.ui_prefs import PrefField

        metadata = {
            'base_model': 'other_model.safetensors',
            'sampler': 'euler_ancestral',
            'scheduler': 'karras',
            'vae': 'other_vae.safetensors',
        }
        current = {
            PrefField.CHECKPOINT: 'sd_xl_base_1.0.safetensors',
            PrefField.SAMPLER: 'dpmpp_2m',
            PrefField.SCHEDULER: 'normal',
            PrefField.VAE: 'sd_xl_vae.safetensors',
        }

        diffs = compute_diffs(metadata, current)

        self.assertEqual(len(diffs), 4)
        self.assertIn(
            FieldDiff(field=PrefField.CHECKPOINT, metadata_value='other_model.safetensors',
                      current_value='sd_xl_base_1.0.safetensors'),
            diffs,
        )
        self.assertIn(
            FieldDiff(field=PrefField.SAMPLER, metadata_value='euler_ancestral', current_value='dpmpp_2m'),
            diffs,
        )
        self.assertIn(
            FieldDiff(field=PrefField.SCHEDULER, metadata_value='karras', current_value='normal'),
            diffs,
        )
        self.assertIn(
            FieldDiff(field=PrefField.VAE, metadata_value='other_vae.safetensors',
                      current_value='sd_xl_vae.safetensors'),
            diffs,
        )

    def test_detects_diff_supplied_via_fallback_key(self):
        from modules.meta_confirm import FieldDiff, compute_diffs
        from modules.ui_prefs import PrefField

        metadata = {'Base Model': 'other_model.safetensors'}
        current = {PrefField.CHECKPOINT: 'sd_xl_base_1.0.safetensors'}

        diffs = compute_diffs(metadata, current)

        self.assertEqual(
            diffs,
            [FieldDiff(field=PrefField.CHECKPOINT, metadata_value='other_model.safetensors',
                       current_value='sd_xl_base_1.0.safetensors')],
        )


class _FakePrefs:
    """Minimal PrefsLike stand-in so resolve() tests don't need modules.ui_prefs.UIPrefs."""

    def __init__(self, decisions):
        self._decisions = decisions

    def get(self, field):
        from modules.ui_prefs import RememberDecision
        return self._decisions.get(field, RememberDecision.ASK)


class TestResolve(unittest.TestCase):
    def test_use_metadata_keeps_the_key(self):
        from modules.meta_confirm import FieldDiff, resolve
        from modules.ui_prefs import PrefField, RememberDecision

        metadata = {'base_model': 'other_model.safetensors'}
        diffs = [FieldDiff(field=PrefField.CHECKPOINT, metadata_value='other_model.safetensors',
                            current_value='sd_xl_base_1.0.safetensors')]
        prefs = _FakePrefs({PrefField.CHECKPOINT: RememberDecision.USE_METADATA})

        resolved, ask_diffs = resolve(metadata, diffs, prefs)

        self.assertEqual(resolved['base_model'], 'other_model.safetensors')
        self.assertEqual(ask_diffs, [])

    def test_keep_current_deletes_both_key_variants(self):
        from modules.meta_confirm import FieldDiff, resolve
        from modules.ui_prefs import PrefField, RememberDecision

        metadata = {'base_model': 'other_model.safetensors', 'Base Model': 'other_model.safetensors'}
        diffs = [FieldDiff(field=PrefField.CHECKPOINT, metadata_value='other_model.safetensors',
                            current_value='sd_xl_base_1.0.safetensors')]
        prefs = _FakePrefs({PrefField.CHECKPOINT: RememberDecision.KEEP_CURRENT})

        resolved, ask_diffs = resolve(metadata, diffs, prefs)

        self.assertNotIn('base_model', resolved)
        self.assertNotIn('Base Model', resolved)
        self.assertEqual(ask_diffs, [])

    def test_ask_diffs_are_returned_for_the_modal(self):
        from modules.meta_confirm import FieldDiff, resolve
        from modules.ui_prefs import PrefField, RememberDecision

        metadata = {'base_model': 'other_model.safetensors'}
        diff = FieldDiff(field=PrefField.CHECKPOINT, metadata_value='other_model.safetensors',
                          current_value='sd_xl_base_1.0.safetensors')
        prefs = _FakePrefs({PrefField.CHECKPOINT: RememberDecision.ASK})

        resolved, ask_diffs = resolve(metadata, [diff], prefs)

        self.assertEqual(resolved['base_model'], 'other_model.safetensors')
        self.assertEqual(ask_diffs, [diff])

    def test_input_dict_is_not_mutated(self):
        from modules.meta_confirm import FieldDiff, resolve
        from modules.ui_prefs import PrefField, RememberDecision

        metadata = {'base_model': 'other_model.safetensors'}
        original = dict(metadata)
        diffs = [FieldDiff(field=PrefField.CHECKPOINT, metadata_value='other_model.safetensors',
                            current_value='sd_xl_base_1.0.safetensors')]
        prefs = _FakePrefs({PrefField.CHECKPOINT: RememberDecision.KEEP_CURRENT})

        resolve(metadata, diffs, prefs)

        self.assertEqual(metadata, original)


class TestApplyDecisions(unittest.TestCase):
    def test_removes_both_key_variants_for_keep_current(self):
        from modules.meta_confirm import apply_decisions
        from modules.ui_prefs import PrefField, RememberDecision

        metadata = {'base_model': 'other_model.safetensors', 'Base Model': 'other_model.safetensors',
                    'sampler': 'euler_ancestral'}
        decisions = {PrefField.CHECKPOINT: RememberDecision.KEEP_CURRENT}

        resolved = apply_decisions(metadata, decisions)

        self.assertNotIn('base_model', resolved)
        self.assertNotIn('Base Model', resolved)
        self.assertEqual(resolved['sampler'], 'euler_ancestral')

    def test_does_not_mutate_input(self):
        from modules.meta_confirm import apply_decisions
        from modules.ui_prefs import PrefField, RememberDecision

        metadata = {'base_model': 'other_model.safetensors'}
        original = dict(metadata)
        decisions = {PrefField.CHECKPOINT: RememberDecision.KEEP_CURRENT}

        apply_decisions(metadata, decisions)

        self.assertEqual(metadata, original)


@unittest.skipIf(
    load_parameter_button_click is None,
    f'modules.meta_parser unavailable: {_load_parameter_button_click_import_error}',
)
class TestLoadParameterButtonClickIntegration(unittest.TestCase):
    def test_keep_current_key_deletion_yields_noop_update(self):
        from modules.meta_confirm import apply_decisions
        from modules.ui_prefs import PrefField, RememberDecision

        metadata = {
            'prompt': 'a cat',
            'negative_prompt': '',
            'base_model': 'sd_xl_base_1.0.safetensors',
            'sampler': 'dpmpp_2m',
            'scheduler': 'karras',
        }
        decisions = {PrefField.CHECKPOINT: RememberDecision.KEEP_CURRENT}

        resolved = apply_decisions(metadata, decisions)
        results = load_parameter_button_click(resolved, is_generating=False, inpaint_mode='inpaint')

        # Index 19 matches load_data_outputs in webui.py:1415-1419, where
        # base_model is the 20th output (0-based index 19) after
        # advanced_checkbox, image_number, prompt, negative_prompt,
        # style_selections, performance_selection, overwrite_step,
        # overwrite_switch, aspect_ratios_selection, overwrite_width,
        # overwrite_height, guidance_scale, sharpness, adm_scaler_positive,
        # adm_scaler_negative, adm_scaler_end, refiner_swap_method,
        # adaptive_cfg, clip_skip.
        base_model_result = results[19]
        self.assertIsInstance(base_model_result, dict)
        self.assertEqual(set(base_model_result.keys()), {'__type__'})
        self.assertEqual(base_model_result['__type__'], 'generic_update')


if __name__ == '__main__':
    unittest.main()
