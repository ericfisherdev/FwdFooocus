"""Unit tests for the model family capability registry."""

import dataclasses
import sys
import unittest

import pytest
from enum import Enum
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# args_manager calls parse_args() at import time, which chokes on pytest's
# argv (modules.model_family imports modules.config, which imports
# args_manager). Patch sys.argv before any project modules are imported.
_original_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    from modules import model_family  # noqa: E402
    from modules.flags import Performance, guidance_scale_range, sampler_list, scheduler_list, sdxl_aspect_ratios  # noqa: E402
finally:
    sys.argv = _original_argv


class TestSdxlMatchesFlags(unittest.TestCase):
    """The SDXL registry entry must replicate flags.py, not re-type it."""

    def setUp(self):
        self.sdxl = model_family.get_capabilities(model_family.ModelFamily.SDXL)

    def test_performance_modes_cover_every_performance_member(self):
        self.assertEqual(len(self.sdxl.performance_modes), len(list(Performance)))

    def test_performance_modes_match_flags_per_member(self):
        modes_by_label = {mode.label: mode for mode in self.sdxl.performance_modes}
        for member in Performance:
            mode = modes_by_label[member.value]
            self.assertEqual(mode.steps, member.steps())
            self.assertEqual(mode.steps_uov, member.steps_uov())
            self.assertEqual(mode.lora_filename, member.lora_filename())
            self.assertIsNone(mode.cfg)
            self.assertEqual(mode.restricted, Performance.has_restricted_features(member))

    def test_aspect_ratios_match_flags(self):
        self.assertEqual(self.sdxl.aspect_ratios, tuple(sdxl_aspect_ratios))

    def test_sampler_names_match_flags(self):
        self.assertEqual(self.sdxl.sampler_names, tuple(sampler_list))

    def test_scheduler_names_match_flags(self):
        self.assertEqual(self.sdxl.scheduler_names, tuple(scheduler_list))

    def test_cfg_range_matches_guidance_scale_slider_bounds(self):
        self.assertEqual(self.sdxl.cfg_range, guidance_scale_range)

    def test_latent_channels(self):
        self.assertEqual(self.sdxl.latent_channels, 4)

    def test_all_capability_flags_true_except_documented(self):
        self.assertTrue(self.sdxl.supports_refiner)
        self.assertTrue(self.sdxl.supports_adm_guidance)
        self.assertTrue(self.sdxl.supports_freeu)
        self.assertTrue(self.sdxl.supports_clip_skip)
        self.assertTrue(self.sdxl.supports_adaptive_cfg)
        self.assertTrue(self.sdxl.supports_sharpness)
        self.assertTrue(self.sdxl.supports_negative_prompt)
        self.assertTrue(self.sdxl.supports_controlnet)
        self.assertTrue(self.sdxl.supports_ip_adapter)
        self.assertTrue(self.sdxl.supports_inpaint_engine)

    def test_vae_override_unrestricted_by_default(self):
        self.assertTrue(self.sdxl.supports_vae_override)
        self.assertIsNone(self.sdxl.vae_names)


class TestSd15Entry(unittest.TestCase):
    """SD15 shares SDXL's values today (no SD1.5-specific behavior exists yet)."""

    def test_sd15_equals_sdxl_by_value(self):
        sdxl = model_family.get_capabilities(model_family.ModelFamily.SDXL)
        sd15 = model_family.get_capabilities(model_family.ModelFamily.SD15)
        self.assertEqual(sd15, sdxl)

    def test_sd15_is_a_distinct_instance_from_sdxl(self):
        # Unlike UNKNOWN (required to be identical to SDXL), SD15 is a
        # separate object so it can diverge independently in the future.
        sdxl = model_family.get_capabilities(model_family.ModelFamily.SDXL)
        sd15 = model_family.get_capabilities(model_family.ModelFamily.SD15)
        self.assertIsNot(sd15, sdxl)


class TestZImageEntry(unittest.TestCase):
    """Z-Image-Turbo's registry entry, added by FWDF-127."""

    def setUp(self):
        self.z_image = model_family.get_capabilities(model_family.ModelFamily.Z_IMAGE)

    def test_distinct_instance_from_sdxl(self):
        sdxl = model_family.get_capabilities(model_family.ModelFamily.SDXL)
        self.assertIsNot(self.z_image, sdxl)

    def test_no_refiner_no_adm_no_freeu_no_clip_skip(self):
        self.assertFalse(self.z_image.supports_refiner)
        self.assertFalse(self.z_image.supports_adm_guidance)
        self.assertFalse(self.z_image.supports_freeu)
        self.assertFalse(self.z_image.supports_clip_skip)

    def test_adaptive_cfg_and_sharpness_disabled(self):
        self.assertFalse(self.z_image.supports_adaptive_cfg)
        self.assertFalse(self.z_image.supports_sharpness)

    def test_controlnet_ip_adapter_inpaint_engine_unsupported(self):
        self.assertFalse(self.z_image.supports_controlnet)
        self.assertFalse(self.z_image.supports_ip_adapter)
        self.assertFalse(self.z_image.supports_inpaint_engine)

    def test_negative_prompt_supported_and_cfg_non_zero(self):
        # cfg=0 would silently make the negative-prompt field a no-op.
        self.assertTrue(self.z_image.supports_negative_prompt)
        self.assertGreater(self.z_image.default_cfg, 0.0)
        self.assertGreaterEqual(self.z_image.default_cfg, self.z_image.cfg_range[0])
        self.assertLessEqual(self.z_image.default_cfg, self.z_image.cfg_range[1])

    def test_vae_not_overridable(self):
        self.assertFalse(self.z_image.supports_vae_override)

    def test_latent_channels_is_sixteen(self):
        self.assertEqual(self.z_image.latent_channels, 16)

    def test_turbo_performance_mode(self):
        self.assertEqual(len(self.z_image.performance_modes), 1)
        turbo = self.z_image.performance_modes[0]
        self.assertEqual(turbo.label, 'Turbo')
        self.assertTrue(1 <= turbo.steps <= 20)

    def test_sampler_names_are_euler_family_only(self):
        self.assertTrue(all('euler' in name for name in self.z_image.sampler_names))

    def test_scheduler_names_exclude_hardcoded_architecture_specific_ones(self):
        # modules/sample_hijack.py hardcodes 'turbo'/'align_your_steps' to
        # SDXL/SD1 today (see modules/model_family.py's module docstring).
        self.assertNotIn('turbo', self.z_image.scheduler_names)
        self.assertNotIn('align_your_steps', self.z_image.scheduler_names)


class TestUnknownFallback(unittest.TestCase):
    def test_unknown_is_identical_to_sdxl(self):
        sdxl = model_family.FAMILY_CAPABILITIES[model_family.ModelFamily.SDXL]
        unknown = model_family.FAMILY_CAPABILITIES[model_family.ModelFamily.UNKNOWN]
        self.assertIs(unknown, sdxl)

    def test_get_capabilities_falls_back_to_unknown_for_unpopulated_family(self):
        # KREA2 is a Krea 2 backlog placeholder with no registry entry yet.
        # Z_IMAGE has a real entry as of FWDF-127 (see TestZImageEntry).
        sdxl = model_family.get_capabilities(model_family.ModelFamily.SDXL)
        self.assertIs(model_family.get_capabilities(model_family.ModelFamily.KREA2), sdxl)

    def test_fallback_routes_through_the_unknown_entry_not_a_hardcoded_default(self):
        # Swap in a distinct UNKNOWN descriptor: unregistered families must
        # resolve to it, proving get_capabilities() reads the UNKNOWN entry
        # rather than defaulting to SDXL directly. KREA2 stands in for an
        # unregistered family (Z_IMAGE has a real entry as of FWDF-127).
        original = model_family.FAMILY_CAPABILITIES[model_family.ModelFamily.UNKNOWN]
        distinct = dataclasses.replace(original, supports_freeu=not original.supports_freeu)
        model_family.FAMILY_CAPABILITIES[model_family.ModelFamily.UNKNOWN] = distinct
        try:
            self.assertIs(model_family.get_capabilities(model_family.ModelFamily.KREA2), distinct)
        finally:
            model_family.FAMILY_CAPABILITIES[model_family.ModelFamily.UNKNOWN] = original


def _make_blank_capabilities(**overrides):
    """A minimal all-off FamilyCapabilities for extensibility tests."""
    values = dict(
        supports_refiner=False,
        supports_adm_guidance=False,
        supports_freeu=False,
        supports_clip_skip=False,
        supports_adaptive_cfg=False,
        supports_sharpness=False,
        supports_negative_prompt=False,
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_inpaint_engine=False,
        supports_vae_override=False,
        vae_names=None,
        performance_modes=(),
        sampler_names=(),
        scheduler_names=(),
        aspect_ratios=(),
        default_cfg=1.0,
        cfg_range=(1.0, 1.0),
        default_steps=1,
        latent_channels=4,
    )
    values.update(overrides)
    return model_family.FamilyCapabilities(**values)


class TestRegistryExtensibility(unittest.TestCase):
    """Adding a family should require only an enum member and a registry entry."""

    def setUp(self):
        self._original_entries = dict(model_family.FAMILY_CAPABILITIES)

    def tearDown(self):
        model_family.FAMILY_CAPABILITIES.clear()
        model_family.FAMILY_CAPABILITIES.update(self._original_entries)

    def test_synthetic_family_works_through_get_capabilities(self):
        class SyntheticFamily(Enum):
            WIDGET = 'widget'

        synthetic_capabilities = _make_blank_capabilities()

        model_family.FAMILY_CAPABILITIES[SyntheticFamily.WIDGET] = synthetic_capabilities

        result = model_family.get_capabilities(SyntheticFamily.WIDGET)

        self.assertIs(result, synthetic_capabilities)
        self.assertFalse(result.supports_vae_override)

    def test_existing_entries_are_unaffected_by_extension(self):
        class SyntheticFamily(Enum):
            WIDGET = 'widget'

        model_family.FAMILY_CAPABILITIES[SyntheticFamily.WIDGET] = _make_blank_capabilities()

        sdxl = model_family.get_capabilities(model_family.ModelFamily.SDXL)
        self.assertIs(sdxl, self._original_entries[model_family.ModelFamily.SDXL])


class TestImmutability(unittest.TestCase):
    def test_family_capabilities_is_frozen(self):
        capabilities = model_family.get_capabilities(model_family.ModelFamily.SDXL)
        with pytest.raises(dataclasses.FrozenInstanceError):
            capabilities.supports_refiner = False

    def test_performance_mode_is_frozen(self):
        capabilities = model_family.get_capabilities(model_family.ModelFamily.SDXL)
        mode = capabilities.performance_modes[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            mode.steps = 1


class TestPerformanceModeBuildValidation(unittest.TestCase):
    def test_missing_steps_entry_fails_fast_at_build_time(self):
        """A Performance member without Steps/StepsUOV must break the registry
        build with a clear error, not store None in an int-typed field."""
        from unittest.mock import patch

        broken_member = next(iter(Performance))
        with patch.object(type(broken_member), 'steps', return_value=None):
            with pytest.raises(ValueError, match=broken_member.name):
                model_family._build_sdxl_performance_modes()


if __name__ == '__main__':
    unittest.main()
