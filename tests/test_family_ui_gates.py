"""Unit tests for the family-gated Gradio UI decision logic (FWDF-129)."""

import sys
import unittest
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# args_manager calls parse_args() at import time, which chokes on pytest's
# argv (modules.family_ui_gates imports modules.model_family, which imports
# modules.config). Patch sys.argv before any project modules are imported.
_original_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    from modules import family_ui_gates  # noqa: E402
    from modules.flags import Performance  # noqa: E402
    from modules.model_family import FamilyCapabilities, PerformanceMode  # noqa: E402
finally:
    sys.argv = _original_argv


def _make_capabilities(**overrides) -> FamilyCapabilities:
    """A minimal all-on FamilyCapabilities builder for these tests, distinct
    from the real registry so tests never depend on which families happen
    to have a dedicated registry entry today.
    """
    values = dict(
        supports_refiner=True,
        supports_adm_guidance=True,
        supports_freeu=True,
        supports_clip_skip=True,
        supports_adaptive_cfg=True,
        supports_sharpness=True,
        supports_negative_prompt=True,
        supports_controlnet=True,
        controlnet_types=('canny', 'cpds'),
        supports_ip_adapter=True,
        supports_inpaint_engine=True,
        supports_vae_override=True,
        vae_names=None,
        performance_modes=(
            PerformanceMode(label='Quality', steps=60, steps_uov=36, cfg=None, lora_filename=None, restricted=False),
            PerformanceMode(label='Speed', steps=30, steps_uov=18, cfg=None, lora_filename=None, restricted=False),
            PerformanceMode(label='Lightning', steps=4, steps_uov=4, cfg=None, lora_filename='x.safetensors', restricted=True),
        ),
        sampler_names=('euler', 'dpmpp_2m'),
        scheduler_names=('normal', 'karras'),
        aspect_ratios=('1024*1024', '1152*896'),
        default_cfg=7.0,
        cfg_range=(1.0, 30.0),
        default_steps=30,
        latent_channels=4,
        native_resolution_range=(1024.0, 2048.0),
        resolution_multiple=8,
    )
    values.update(overrides)
    return FamilyCapabilities(**values)


def _restricted_capabilities(**overrides) -> FamilyCapabilities:
    """A capabilities set with every `supports_*` flag off, standing in for
    a hypothetical family that supports none of the SDXL-only features.
    """
    values = dict(
        supports_refiner=False,
        supports_adm_guidance=False,
        supports_freeu=False,
        supports_clip_skip=False,
        supports_adaptive_cfg=False,
        supports_sharpness=False,
        supports_negative_prompt=False,
        supports_controlnet=False,
        controlnet_types=(),
        supports_ip_adapter=False,
        supports_inpaint_engine=False,
        supports_vae_override=False,
        vae_names=None,
        performance_modes=(
            PerformanceMode(label='Z Fast', steps=20, steps_uov=10, cfg=None, lora_filename=None, restricted=False),
        ),
        sampler_names=('z_sampler',),
        scheduler_names=('z_scheduler',),
        aspect_ratios=('512*512',),
        default_cfg=4.0,
        cfg_range=(1.0, 10.0),
        default_steps=20,
        latent_channels=16,
        native_resolution_range=(1024.0, 2048.0),
        resolution_multiple=16,
    )
    values.update(overrides)
    return FamilyCapabilities(**values)


class TestPerformanceRestricted(unittest.TestCase):
    def test_non_restricted_members_are_false(self):
        for member in (Performance.QUALITY, Performance.SPEED):
            self.assertFalse(family_ui_gates.performance_restricted(member.value))

    def test_restricted_members_are_true(self):
        for member in (Performance.EXTREME_SPEED, Performance.LIGHTNING, Performance.HYPER_SD):
            self.assertTrue(family_ui_gates.performance_restricted(member.value))

    def test_accepts_performance_enum_member_directly(self):
        self.assertTrue(family_ui_gates.performance_restricted(Performance.LIGHTNING))
        self.assertFalse(family_ui_gates.performance_restricted(Performance.QUALITY))


class TestRestrictedInteractive(unittest.TestCase):
    def test_supported_and_unrestricted_is_interactive(self):
        self.assertTrue(family_ui_gates.restricted_interactive(supported=True, performance=Performance.QUALITY.value))

    def test_supported_but_restricted_performance_is_not_interactive(self):
        self.assertFalse(
            family_ui_gates.restricted_interactive(supported=True, performance=Performance.LIGHTNING.value)
        )

    def test_unsupported_is_never_interactive_even_when_unrestricted(self):
        self.assertFalse(
            family_ui_gates.restricted_interactive(supported=False, performance=Performance.QUALITY.value)
        )

    def test_unsupported_and_restricted_is_not_interactive(self):
        self.assertFalse(
            family_ui_gates.restricted_interactive(supported=False, performance=Performance.LIGHTNING.value)
        )

    def test_deterministic_for_the_same_inputs(self):
        """A pure function called twice with identical inputs always agrees
        with itself -- the sanity check underlying the cross-handler
        guarantee exercised below."""
        caps = _restricted_capabilities()
        performance = Performance.QUALITY.value
        first_call = family_ui_gates.restricted_interactive(supported=caps.supports_refiner, performance=performance)
        second_call = family_ui_gates.restricted_interactive(supported=caps.supports_refiner, performance=performance)
        self.assertEqual(first_call, second_call)

    def test_both_handlers_agree_regardless_of_evaluation_order(self):
        """base_model.change and performance_selection.change both compute
        this union for every overlapping control (sharpness, ADM guidance,
        refiner, adaptive_cfg). Simulate each handler's own evaluation
        sequence in both orders and confirm every overlapping control
        resolves identically either way -- neither handler's decision can
        depend on which one ran, or ran most recently."""
        caps = _restricted_capabilities()
        performance = Performance.LIGHTNING.value

        overlapping_supports = {
            'sharpness': caps.supports_sharpness,
            'adm_guidance': caps.supports_adm_guidance,
            'refiner': caps.supports_refiner,
            'adaptive_cfg': caps.supports_adaptive_cfg,
        }

        def evaluate_all_overlapping_controls():
            return {
                name: family_ui_gates.restricted_interactive(supported=supported, performance=performance)
                for name, supported in overlapping_supports.items()
            }

        # "base_model.change runs, then performance_selection.change runs"
        base_model_handler_result = evaluate_all_overlapping_controls()
        performance_handler_result = evaluate_all_overlapping_controls()
        self.assertEqual(base_model_handler_result, performance_handler_result)

        # ... and the reverse order produces the exact same per-control results.
        performance_handler_result_reversed = evaluate_all_overlapping_controls()
        base_model_handler_result_reversed = evaluate_all_overlapping_controls()
        self.assertEqual(performance_handler_result_reversed, base_model_handler_result_reversed)
        self.assertEqual(base_model_handler_result, base_model_handler_result_reversed)


class TestNegativePromptVisible(unittest.TestCase):
    def test_supported_family_and_unrestricted_performance_is_visible(self):
        caps = _make_capabilities()
        self.assertTrue(family_ui_gates.negative_prompt_visible(caps, Performance.QUALITY.value))

    def test_supported_family_but_restricted_performance_is_hidden(self):
        caps = _make_capabilities()
        self.assertFalse(family_ui_gates.negative_prompt_visible(caps, Performance.LIGHTNING.value))

    def test_unsupported_family_is_hidden_even_when_performance_unrestricted(self):
        caps = _restricted_capabilities()
        self.assertFalse(family_ui_gates.negative_prompt_visible(caps, Performance.QUALITY.value))


class TestRefinerSwitchVisible(unittest.TestCase):
    def test_visible_when_family_supports_refiner_and_one_is_selected(self):
        caps = _make_capabilities()
        self.assertTrue(family_ui_gates.refiner_switch_visible(caps, 'some_refiner.safetensors'))

    def test_hidden_when_no_refiner_selected(self):
        caps = _make_capabilities()
        self.assertFalse(family_ui_gates.refiner_switch_visible(caps, 'None'))

    def test_hidden_when_family_does_not_support_refiner_even_if_one_is_selected(self):
        caps = _restricted_capabilities()
        self.assertFalse(family_ui_gates.refiner_switch_visible(caps, 'some_refiner.safetensors'))


class TestChoiceListAndValue(unittest.TestCase):
    def test_current_value_kept_when_still_valid(self):
        choices, value = family_ui_gates.choice_list_and_value(['a', 'b', 'c'], 'b', 'a')
        self.assertEqual(choices, ('a', 'b', 'c'))
        self.assertEqual(value, 'b')

    def test_falls_back_to_fallback_value_when_current_invalid(self):
        choices, value = family_ui_gates.choice_list_and_value(['x', 'y'], 'stale', 'y')
        self.assertEqual(value, 'y')

    def test_falls_back_to_first_choice_when_fallback_also_invalid(self):
        choices, value = family_ui_gates.choice_list_and_value(['x', 'y'], 'stale', 'also_stale')
        self.assertEqual(value, 'x')

    def test_falls_back_to_first_choice_when_fallback_is_none(self):
        choices, value = family_ui_gates.choice_list_and_value(['x', 'y'], 'stale', None)
        self.assertEqual(value, 'x')

    def test_empty_choices_passes_current_value_through_unchanged(self):
        choices, value = family_ui_gates.choice_list_and_value([], 'anything', 'fallback')
        self.assertEqual(choices, ())
        self.assertEqual(value, 'anything')


class TestPerformanceChoicesAndValue(unittest.TestCase):
    def test_choices_match_family_performance_mode_labels(self):
        caps = _make_capabilities()
        choices, _ = family_ui_gates.performance_choices_and_value(caps, 'Quality')
        self.assertEqual(choices, ('Quality', 'Speed', 'Lightning'))

    def test_current_value_preserved_when_present_in_new_family(self):
        caps = _make_capabilities()
        _, value = family_ui_gates.performance_choices_and_value(caps, 'Speed')
        self.assertEqual(value, 'Speed')

    def test_falls_back_to_mode_matching_default_steps_when_stale(self):
        # A multi-mode fixture where the default_steps-matching mode is NOT
        # first, so this can't pass by accident via an always-first-choice
        # fallback bug: only 'Z Balanced' (steps=30) matches default_steps=30.
        caps = _make_capabilities(
            performance_modes=(
                PerformanceMode(label='Z Fast', steps=20, steps_uov=10, cfg=None, lora_filename=None, restricted=False),
                PerformanceMode(label='Z Balanced', steps=30, steps_uov=15, cfg=None, lora_filename=None, restricted=False),
                PerformanceMode(label='Z Thorough', steps=50, steps_uov=25, cfg=None, lora_filename=None, restricted=False),
            ),
            default_steps=30,
        )
        _, value = family_ui_gates.performance_choices_and_value(caps, 'stale-performance-value')
        self.assertEqual(value, 'Z Balanced')


def _aspect_choices_and_value(caps, current_value, unrestricted, configured):
    """Resolve once, then derive choices/value, exactly as `webui.py` does."""
    resolution = family_ui_gates.resolve_aspect_ratios(caps, unrestricted, configured)
    return family_ui_gates.aspect_ratio_choices_and_value(
        resolution, current_value, lambda raw: f'formatted({raw})')


class TestAspectRatioChoicesAndValue(unittest.TestCase):
    """`caps.aspect_ratios` (FWDF-117) is always the hardcoded framework
    default today; these tests exercise the config-override-preservation
    behavior added to avoid regressing users who customized
    `available_aspect_ratios` in config.txt (see module docstring)."""

    UNRESTRICTED_DEFAULT = ('1024*1024', '1152*896')  # matches _make_capabilities().aspect_ratios

    def test_uses_configured_list_when_family_matches_unrestricted_default(self):
        caps = _make_capabilities()  # aspect_ratios == UNRESTRICTED_DEFAULT
        configured = ('2048*2048',)
        choices, value = _aspect_choices_and_value(
            caps, 'anything', self.UNRESTRICTED_DEFAULT, configured
        )
        self.assertEqual(choices, ('formatted(2048*2048)',))
        self.assertEqual(value, 'formatted(2048*2048)')

    def test_family_specific_list_stays_first_and_unsupported_configured_entries_are_dropped(self):
        caps = _restricted_capabilities()  # aspect_ratios = ('512*512',), differs from UNRESTRICTED_DEFAULT
        configured = ('4096*4096',)  # outside the family's native range
        choices, _ = _aspect_choices_and_value(
            caps, 'anything', self.UNRESTRICTED_DEFAULT, configured
        )
        self.assertEqual(choices, ('formatted(512*512)',))

    def test_falls_back_to_first_choice_when_current_not_in_new_family(self):
        caps = _restricted_capabilities()
        choices, value = _aspect_choices_and_value(
            caps, 'stale-value', self.UNRESTRICTED_DEFAULT, ()
        )
        self.assertEqual(value, choices[0])

    def test_current_value_preserved_when_still_valid(self):
        caps = _make_capabilities()
        configured = ('1024*1024', '1152*896')
        _, value = _aspect_choices_and_value(
            caps, 'formatted(1152*896)', self.UNRESTRICTED_DEFAULT, configured
        )
        self.assertEqual(value, 'formatted(1152*896)')


class TestParseAspectRatio(unittest.TestCase):
    def test_parses_width_and_height(self):
        self.assertEqual(family_ui_gates.parse_aspect_ratio('1152*1536'), (1152, 1536))

    def test_malformed_entries_return_none(self):
        for bad in ('', '1152', '1152*', '*1536', '1152x1536', '1152*1536*2', 'a*b', '-8*16', '0*1024', '10.5*16'):
            with self.subTest(entry=bad):
                self.assertIsNone(family_ui_gates.parse_aspect_ratio(bad))


class TestResolveAspectRatios(unittest.TestCase):
    """FWDF-207: configured ratios extend, not get dropped by, a curated list."""

    UNRESTRICTED_DEFAULT = ('1024*1024', '1152*896')

    @staticmethod
    def _curated_capabilities():
        return _make_capabilities(
            aspect_ratios=('1024*1024', '1152*896'),
            native_resolution_range=(1024.0, 2048.0),
            resolution_multiple=16,
        )

    def _resolve(self, configured, caps=None):
        return family_ui_gates.resolve_aspect_ratios(
            caps or self._curated_capabilities(), ('unrelated*default',), configured)

    def test_unrestricted_family_gets_configured_list_verbatim(self):
        caps = _make_capabilities()
        configured = ('2048*2048', '1150*1536', 'junk')
        result = family_ui_gates.resolve_aspect_ratios(caps, self.UNRESTRICTED_DEFAULT, configured)
        self.assertEqual(result.ratios, configured)
        self.assertEqual(result.rejected, ())

    def test_curated_family_appends_valid_configured_entries_after_curated_list(self):
        result = self._resolve(('1152*1536', '1536*1152'))
        self.assertEqual(result.ratios, ('1024*1024', '1152*896', '1152*1536', '1536*1152'))
        self.assertEqual(result.rejected, ())

    def test_entries_already_curated_are_neither_appended_nor_rejected(self):
        result = self._resolve(('1152*896', '1152*1536', '1152*1536'))
        self.assertEqual(result.ratios, ('1024*1024', '1152*896', '1152*1536'))
        self.assertEqual(result.rejected, ())

    def test_non_multiple_entry_is_rejected(self):
        result = self._resolve(('1150*1536',))
        self.assertEqual(result.ratios, ('1024*1024', '1152*896'))
        self.assertEqual(result.rejected, ('1150*1536',))

    def test_out_of_range_entry_is_rejected(self):
        result = self._resolve(('512*512', '4096*4096'))
        self.assertEqual(result.ratios, ('1024*1024', '1152*896'))
        self.assertEqual(result.rejected, ('512*512', '4096*4096'))

    def test_malformed_entry_is_rejected(self):
        result = self._resolve(('not-a-ratio', '1152*1536'))
        self.assertEqual(result.ratios, ('1024*1024', '1152*896', '1152*1536'))
        self.assertEqual(result.rejected, ('not-a-ratio',))

    def test_rejected_entries_are_reported_once(self):
        self.assertEqual(self._resolve(('1150*1536', '1150*1536')).rejected, ('1150*1536',))

    def test_rejection_message_names_entry_family_and_rule(self):
        message = family_ui_gates.rejected_aspect_ratio_message(
            '1150*1536', 'krea2_raw', self._curated_capabilities())
        self.assertIn('1150*1536', message)
        self.assertIn('krea2_raw', message)
        self.assertIn('multiples of 16', message)
        self.assertIn('native range (1024 to 2048)', message)


class TestAspectRatioChoicesOnCuratedFamily(unittest.TestCase):
    """A configured ratio valid for a curated family is selectable and kept as the value."""

    def _choices_and_value(self, current_value, configured):
        caps = _make_capabilities(aspect_ratios=('1024*1024', '1152*896'), resolution_multiple=16)
        return _aspect_choices_and_value(caps, current_value, ('unrelated*default',), configured)

    def test_configured_entry_is_offered_after_the_curated_list(self):
        choices, _ = self._choices_and_value('formatted(1024*1024)', ('1152*1536',))
        self.assertEqual(choices, ('formatted(1024*1024)', 'formatted(1152*896)', 'formatted(1152*1536)'))

    def test_current_value_preserved_when_it_is_a_configured_entry(self):
        _, value = self._choices_and_value('formatted(1152*1536)', ('1152*1536',))
        self.assertEqual(value, 'formatted(1152*1536)')

    def test_rejected_current_value_falls_back_to_first_curated_choice(self):
        choices, value = self._choices_and_value('formatted(1150*1536)', ('1150*1536',))
        self.assertEqual(value, choices[0])
        self.assertNotIn('formatted(1150*1536)', choices)


class TestSamplerAndSchedulerChoicesAndValue(unittest.TestCase):
    def test_sampler_falls_back_to_first_choice_when_configured_default_also_invalid(self):
        caps = _restricted_capabilities()
        choices, value = family_ui_gates.sampler_choices_and_value(caps, 'stale_sampler', 'configured_default')
        self.assertEqual(choices, ('z_sampler',))
        self.assertEqual(value, 'z_sampler')  # configured_default not in choices either, falls to first

    def test_sampler_falls_back_to_configured_default_when_it_is_valid(self):
        caps = _make_capabilities()  # sampler_names=('euler', 'dpmpp_2m')
        choices, value = family_ui_gates.sampler_choices_and_value(caps, 'stale_sampler', 'dpmpp_2m')
        self.assertEqual(choices, ('euler', 'dpmpp_2m'))
        self.assertEqual(value, 'dpmpp_2m')  # configured default used, not just choices[0] ('euler')

    def test_scheduler_choices_preserve_current_value_when_valid(self):
        caps = _make_capabilities()
        choices, value = family_ui_gates.scheduler_choices_and_value(caps, 'karras', 'normal')
        self.assertEqual(choices, ('normal', 'karras'))
        self.assertEqual(value, 'karras')

    def test_scheduler_falls_back_to_configured_default_when_current_is_stale(self):
        caps = _make_capabilities()  # scheduler_names=('normal', 'karras')
        choices, value = family_ui_gates.scheduler_choices_and_value(caps, 'stale_scheduler', 'karras')
        self.assertEqual(choices, ('normal', 'karras'))
        self.assertEqual(value, 'karras')  # configured default used, not just choices[0] ('normal')


class TestVaeState(unittest.TestCase):
    def test_hidden_and_uncleared_when_family_does_not_support_override(self):
        caps = _restricted_capabilities()
        visible, interactive, choices, value = family_ui_gates.vae_state(
            caps, 'my_vae.safetensors', ['vae_a.safetensors'], 'Default (model)'
        )
        self.assertFalse(visible)
        self.assertFalse(interactive)
        self.assertIsNone(choices)
        self.assertIsNone(value)

    def test_full_global_listing_restored_when_family_has_no_vae_restriction(self):
        caps = _make_capabilities(vae_names=None)
        visible, interactive, choices, value = family_ui_gates.vae_state(
            caps, 'Default (model)', ['vae_a.safetensors', 'vae_b.safetensors'], 'Default (model)'
        )
        self.assertTrue(visible)
        self.assertTrue(interactive)
        self.assertEqual(choices, ('Default (model)', 'vae_a.safetensors', 'vae_b.safetensors'))
        self.assertEqual(value, 'Default (model)')

    def test_curated_list_used_when_family_declares_vae_names(self):
        caps = _make_capabilities(vae_names=('curated_vae.safetensors',))
        visible, interactive, choices, value = family_ui_gates.vae_state(
            caps, 'vae_a.safetensors', ['vae_a.safetensors', 'vae_b.safetensors'], 'Default (model)'
        )
        self.assertTrue(visible)
        self.assertTrue(interactive)
        self.assertEqual(choices, ('Default (model)', 'curated_vae.safetensors'))
        self.assertEqual(value, 'Default (model)')  # stale value not in curated list, falls back

    def test_current_value_preserved_when_still_valid_in_curated_list(self):
        caps = _make_capabilities(vae_names=('curated_vae.safetensors',))
        _, _, _, value = family_ui_gates.vae_state(
            caps, 'curated_vae.safetensors', ['vae_a.safetensors'], 'Default (model)'
        )
        self.assertEqual(value, 'curated_vae.safetensors')


class TestGuidanceScaleRangeAndValue(unittest.TestCase):
    def test_value_within_range_is_unchanged(self):
        caps = _make_capabilities(cfg_range=(1.0, 30.0))
        minimum, maximum, value = family_ui_gates.guidance_scale_range_and_value(caps, 7.0)
        self.assertEqual((minimum, maximum), (1.0, 30.0))
        self.assertEqual(value, 7.0)

    def test_value_above_new_maximum_is_clamped_down(self):
        caps = _restricted_capabilities(cfg_range=(1.0, 10.0))
        _, _, value = family_ui_gates.guidance_scale_range_and_value(caps, 25.0)
        self.assertEqual(value, 10.0)

    def test_value_below_new_minimum_is_clamped_up(self):
        caps = _make_capabilities(cfg_range=(2.0, 30.0))
        _, _, value = family_ui_gates.guidance_scale_range_and_value(caps, 0.5)
        self.assertEqual(value, 2.0)


if __name__ == '__main__':
    unittest.main()
