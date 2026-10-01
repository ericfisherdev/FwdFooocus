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
    from modules.flags import (  # noqa: E402
        Performance,
        guidance_scale_range,
        krea2_aspect_ratios,
        sampler_list,
        scheduler_list,
        sdxl_aspect_ratios,
    )
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

    def test_native_resolution_range_matches_hardcoded_vary_upscale_literals(self):
        # Golden-path regression: apply_vary/apply_upscale (modules/async_worker.py)
        # used to hardcode 1024/2048 directly; the registry-derived value must
        # stay numerically identical so SDXL's Vary/Upscale behavior is unchanged.
        self.assertEqual(self.sdxl.native_resolution_range, (1024.0, 2048.0))

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

    def test_supports_both_canny_and_cpds_controlnet_types(self):
        self.assertEqual(self.sdxl.controlnet_types, ('canny', 'cpds'))

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

    def test_ip_adapter_inpaint_engine_unsupported(self):
        self.assertFalse(self.z_image.supports_ip_adapter)
        self.assertFalse(self.z_image.supports_inpaint_engine)

    def test_controlnet_supported_as_of_fwdf_156(self):
        # Scoped to PyraCanny only -- see modules/model_family.py's
        # _build_z_image_capabilities docstring (FWDF-156).
        self.assertTrue(self.z_image.supports_controlnet)

    def test_controlnet_types_scoped_to_canny_only(self):
        # CPDS has no published DiT equivalent (FWDF-156 follow-up fix):
        # unlike SDXL, Z-Image's controlnet_types omits it.
        self.assertEqual(self.z_image.controlnet_types, ('canny',))

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

    def test_native_resolution_range_is_a_valid_floor_ceiling_pair(self):
        floor, ceiling = self.z_image.native_resolution_range
        self.assertGreater(floor, 0.0)
        self.assertEqual(ceiling, floor * 2.0)

    def test_native_resolution_range_matches_sdxl_today(self):
        # Z-Image reuses SDXL's aspect_ratios list (no Z-Image-specific
        # entries exist yet -- see modules/model_family.py's
        # _build_z_image_capabilities), so the registry-derived resolution
        # bucket is currently identical to SDXL's. This documents that
        # parity is intentional (driven by the shared aspect_ratios data,
        # not a hardcoded duplicate) rather than a bug.
        sdxl = model_family.get_capabilities(model_family.ModelFamily.SDXL)
        self.assertEqual(self.z_image.native_resolution_range, sdxl.native_resolution_range)

    def test_sampler_names_are_euler_family_only(self):
        self.assertTrue(all('euler' in name for name in self.z_image.sampler_names))

    def test_scheduler_names_exclude_hardcoded_architecture_specific_ones(self):
        # modules/sample_hijack.py hardcodes 'turbo'/'align_your_steps' to
        # SDXL/SD1 today (see modules/model_family.py's module docstring).
        self.assertNotIn('turbo', self.z_image.scheduler_names)
        self.assertNotIn('align_your_steps', self.z_image.scheduler_names)


class _Krea2EntryContract:
    """What Krea 2's Raw and Turbo registry entries share (FWDF-152).

    Not collected itself (no `Test` prefix, no TestCase base): the concrete
    subclasses below bind `family` and add the values that differ.
    """

    family: model_family.ModelFamily

    def setUp(self):
        self.caps = model_family.get_capabilities(self.family)

    def test_distinct_from_sdxl_and_z_image(self):
        self.assertIsNot(self.caps, model_family.get_capabilities(model_family.ModelFamily.SDXL))
        self.assertIsNot(self.caps, model_family.get_capabilities(model_family.ModelFamily.Z_IMAGE))

    def test_sdxl_only_features_are_off(self):
        for flag in ('supports_refiner', 'supports_adm_guidance', 'supports_freeu', 'supports_clip_skip',
                     'supports_adaptive_cfg', 'supports_sharpness', 'supports_ip_adapter',
                     'supports_inpaint_engine', 'supports_vae_override'):
            with self.subTest(flag=flag):
                self.assertFalse(getattr(self.caps, flag))

    def test_controlnet_unsupported_because_hooks_are_unet_block_patches(self):
        self.assertFalse(self.caps.supports_controlnet)
        self.assertEqual(self.caps.controlnet_types, ())

    def test_sixteen_channel_latent_matching_the_model_config(self):
        from ldm_patched.modules import supported_models
        self.assertEqual(self.caps.latent_channels, 16)
        self.assertEqual(self.caps.latent_channels, supported_models.Krea2.latent_format.latent_channels)

    def test_exactly_one_performance_mode_and_default_steps_follow_it(self):
        self.assertEqual(len(self.caps.performance_modes), 1)
        self.assertEqual(self.caps.default_steps, self.caps.performance_modes[0].steps)

    def test_performance_mode_is_unrestricted_and_loraless(self):
        mode = self.caps.performance_modes[0]
        self.assertFalse(mode.restricted)
        self.assertIsNone(mode.lora_filename)
        self.assertEqual(mode.steps_uov, mode.steps)

    def test_samplers_and_schedulers_start_with_the_effective_defaults(self):
        # The UIs fall back to the first entry when the configured SDXL
        # default is not valid for the family.
        self.assertEqual(self.caps.sampler_names, ('euler', 'euler_ancestral'))
        self.assertEqual(self.caps.scheduler_names, ('simple', 'normal'))

    def test_scheduler_names_exclude_hardcoded_architecture_specific_ones(self):
        self.assertNotIn('turbo', self.caps.scheduler_names)
        self.assertNotIn('align_your_steps', self.caps.scheduler_names)

    def test_samplers_and_schedulers_are_known_to_flags(self):
        self.assertTrue(set(self.caps.sampler_names) <= set(sampler_list))
        self.assertTrue(set(self.caps.scheduler_names) <= set(scheduler_list))

    def test_aspect_ratios_reach_2048_and_are_patch_aligned(self):
        self.assertEqual(self.caps.aspect_ratios, tuple(krea2_aspect_ratios))
        self.assertIn('2048*2048', self.caps.aspect_ratios)
        for entry in self.caps.aspect_ratios:
            width, height = (int(side) for side in entry.split('*'))
            with self.subTest(entry=entry):
                # DiT patch size 2 x VAE downscale 8.
                self.assertEqual(width % 16, 0)
                self.assertEqual(height % 16, 0)

    def test_native_resolution_range_is_derived_from_the_aspect_ratios(self):
        self.assertEqual(self.caps.native_resolution_range,
                         model_family._native_resolution_range(self.caps.aspect_ratios))
        self.assertEqual(self.caps.native_resolution_range, (1024.0, 2048.0))

    def test_default_cfg_lies_inside_the_cfg_range(self):
        low, high = self.caps.cfg_range
        self.assertGreaterEqual(self.caps.default_cfg, low)
        self.assertLessEqual(self.caps.default_cfg, high)

    def test_resolves_its_own_mode_label(self):
        label = self.caps.performance_modes[0].label
        self.assertIs(model_family.resolve_performance_mode(label, self.caps), self.caps.performance_modes[0])


class TestKrea2RawEntry(_Krea2EntryContract, unittest.TestCase):
    family = model_family.ModelFamily.KREA2_RAW

    def test_single_52_step_raw_mode(self):
        mode = self.caps.performance_modes[0]
        self.assertEqual(mode.label, 'Raw')
        self.assertEqual(mode.steps, 52)
        self.assertEqual(self.caps.default_steps, 52)

    def test_negative_prompt_works_with_ordinary_cfg(self):
        self.assertTrue(self.caps.supports_negative_prompt)
        self.assertEqual(self.caps.default_cfg, 3.5)
        # The unconditional pass is only skipped at exactly 1.0, so Raw must
        # allow values above it.
        self.assertGreater(self.caps.cfg_range[1], 1.0)


class TestKrea2TurboEntry(_Krea2EntryContract, unittest.TestCase):
    family = model_family.ModelFamily.KREA2_TURBO

    def test_single_8_step_turbo_mode(self):
        mode = self.caps.performance_modes[0]
        self.assertEqual(mode.label, 'Turbo')
        self.assertEqual(mode.steps, 8)
        self.assertEqual(self.caps.default_steps, 8)

    def test_cfg_is_pinned_at_exactly_one(self):
        # samplers.py / patch.py skip the unconditional pass only when
        # math.isclose(cond_scale, 1.0); below 1 would interpolate toward the
        # negative prompt instead of being CFG-free.
        self.assertEqual(self.caps.default_cfg, 1.0)
        self.assertEqual(self.caps.cfg_range, (1.0, 1.0))

    def test_negative_prompt_hidden(self):
        self.assertFalse(self.caps.supports_negative_prompt)


class TestKrea2VariantsDiffer(unittest.TestCase):
    def test_variants_are_separate_registry_entries(self):
        raw = model_family.get_capabilities(model_family.ModelFamily.KREA2_RAW)
        turbo = model_family.get_capabilities(model_family.ModelFamily.KREA2_TURBO)
        self.assertIsNot(raw, turbo)
        self.assertNotEqual(model_family.ModelFamily.KREA2_RAW.value, model_family.ModelFamily.KREA2_TURBO.value)

    def test_variants_agree_on_everything_but_the_variant_values(self):
        raw = model_family.get_capabilities(model_family.ModelFamily.KREA2_RAW)
        turbo = model_family.get_capabilities(model_family.ModelFamily.KREA2_TURBO)
        variant_fields = {'supports_negative_prompt', 'performance_modes', 'default_cfg', 'cfg_range', 'default_steps'}
        for field in dataclasses.fields(raw):
            if field.name in variant_fields:
                continue
            with self.subTest(field=field.name):
                self.assertEqual(getattr(raw, field.name), getattr(turbo, field.name))


class TestResolvePerformanceMode(unittest.TestCase):
    def setUp(self):
        self.z_image = model_family.get_capabilities(model_family.ModelFamily.Z_IMAGE)
        self.sdxl = model_family.get_capabilities(model_family.ModelFamily.SDXL)

    def test_family_label_resolves_to_the_family_mode(self):
        mode = model_family.resolve_performance_mode('Turbo', self.z_image)
        self.assertIs(mode, self.z_image.performance_modes[0])

    def test_every_sdxl_label_resolves_to_its_own_mode(self):
        for expected in self.sdxl.performance_modes:
            with self.subTest(label=expected.label):
                self.assertIs(model_family.resolve_performance_mode(expected.label, self.sdxl), expected)

    def test_legacy_label_resolves_for_a_family_that_does_not_declare_it(self):
        # The Gradio radio can still carry a legacy label when a non-SDXL
        # checkpoint is selected; tasks have always accepted those.
        mode = model_family.resolve_performance_mode('Speed', self.z_image)
        self.assertEqual(mode.steps, Performance.SPEED.steps())

    def test_restricted_legacy_labels_are_rejected_for_a_family_that_omits_them(self):
        # Extreme Speed / Lightning / Hyper-SD would apply an SDXL accelerator
        # LoRA and sampler defaults to a DiT.
        for label in ('Extreme Speed', 'Lightning', 'Hyper-SD'):
            for family in (model_family.ModelFamily.Z_IMAGE, model_family.ModelFamily.KREA2_RAW,
                           model_family.ModelFamily.KREA2_TURBO):
                with self.subTest(label=label, family=family):
                    with self.assertRaises(ValueError):
                        model_family.resolve_performance_mode(label, model_family.get_capabilities(family))

    def test_restricted_legacy_labels_still_resolve_for_sdxl(self):
        for label in ('Extreme Speed', 'Lightning', 'Hyper-SD'):
            with self.subTest(label=label):
                self.assertTrue(model_family.resolve_performance_mode(label, self.sdxl).restricted)

    def test_family_label_wins_over_a_same_named_legacy_label(self):
        synthetic = _make_blank_capabilities(performance_modes=(
            model_family.PerformanceMode(label='Speed', steps=7, steps_uov=7, cfg=None,
                                         lora_filename=None, restricted=False),))
        self.assertEqual(model_family.resolve_performance_mode('Speed', synthetic).steps, 7)

    def test_unknown_label_raises_value_error(self):
        with self.assertRaises(ValueError):
            model_family.resolve_performance_mode('Warp', self.z_image)

    def test_a_family_label_is_not_valid_for_another_family(self):
        with self.assertRaises(ValueError):
            model_family.resolve_performance_mode('Raw', self.z_image)


class TestUnknownFallback(unittest.TestCase):
    def test_unknown_is_identical_to_sdxl(self):
        sdxl = model_family.FAMILY_CAPABILITIES[model_family.ModelFamily.SDXL]
        unknown = model_family.FAMILY_CAPABILITIES[model_family.ModelFamily.UNKNOWN]
        self.assertIs(unknown, sdxl)

    def test_get_capabilities_falls_back_to_unknown_for_unpopulated_family(self):
        # A family that was never registered, as a local Enum so the test
        # does not depend on which real families happen to be registered.
        class UnregisteredFamily(Enum):
            WIDGET = 'widget'

        sdxl = model_family.get_capabilities(model_family.ModelFamily.SDXL)
        self.assertIs(model_family.get_capabilities(UnregisteredFamily.WIDGET), sdxl)

    def test_fallback_routes_through_the_unknown_entry_not_a_hardcoded_default(self):
        # Swap in a distinct UNKNOWN descriptor: unregistered families must
        # resolve to it, proving get_capabilities() reads the UNKNOWN entry
        # rather than defaulting to SDXL directly.
        class UnregisteredFamily(Enum):
            WIDGET = 'widget'

        original = model_family.FAMILY_CAPABILITIES[model_family.ModelFamily.UNKNOWN]
        distinct = dataclasses.replace(original, supports_freeu=not original.supports_freeu)
        model_family.FAMILY_CAPABILITIES[model_family.ModelFamily.UNKNOWN] = distinct
        try:
            self.assertIs(model_family.get_capabilities(UnregisteredFamily.WIDGET), distinct)
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
        controlnet_types=(),
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
        native_resolution_range=(1.0, 1.0),
        resolution_multiple=1,
    )
    values.update(overrides)
    return model_family.FamilyCapabilities(**values)


class TestControlnetTypesConsistency(unittest.TestCase):
    """FWDF-156 follow-up: controlnet_types is the single source of truth
    for per-type ControlNet support; supports_controlnet must always agree
    with it, enforced at construction time."""

    def test_supports_controlnet_true_with_empty_types_raises(self):
        with self.assertRaises(ValueError):
            _make_blank_capabilities(supports_controlnet=True, controlnet_types=())

    def test_supports_controlnet_false_with_nonempty_types_raises(self):
        with self.assertRaises(ValueError):
            _make_blank_capabilities(supports_controlnet=False, controlnet_types=('canny',))

    def test_consistent_values_construct_successfully(self):
        caps = _make_blank_capabilities(supports_controlnet=True, controlnet_types=('canny', 'cpds'))
        self.assertTrue(caps.supports_controlnet)
        self.assertEqual(caps.controlnet_types, ('canny', 'cpds'))


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


class TestNativeResolutionRangeDerivation(unittest.TestCase):
    """`_native_resolution_range()` (FWDF-154) drives the Vary/Upscale
    resolution floor/ceiling that used to be hardcoded 1024/2048 literals
    in modules/async_worker.py.
    """

    def test_uniform_aspect_ratio_list_yields_floor_equal_to_every_entry(self):
        # Every SDXL aspect ratio resolves to the same 1024.0 shape_ceil, so
        # the derived floor must equal that shared value, not some other entry.
        floor, ceiling = model_family._native_resolution_range(tuple(sdxl_aspect_ratios))
        self.assertEqual(floor, 1024.0)
        self.assertEqual(ceiling, 2048.0)

    def test_ceiling_is_always_double_the_floor(self):
        floor, ceiling = model_family._native_resolution_range(('512*512', '1024*1024'))
        self.assertEqual(ceiling, floor * 2.0)

    def test_floor_is_the_minimum_shape_ceil_across_entries(self):
        # '512*512' -> shape_ceil 512.0 (smaller than '1024*1024' -> 1024.0);
        # the floor must track the smallest bucket, not the largest or an
        # average, so a family with one small aspect ratio isn't force-upsized
        # past that entry's own native bucket.
        floor, _ = model_family._native_resolution_range(('512*512', '1024*1024'))
        self.assertEqual(floor, 512.0)

    def test_matches_modules_util_get_shape_ceil_formula(self):
        from modules.util import get_shape_ceil
        floor, _ = model_family._native_resolution_range(('768*1344',))
        self.assertEqual(floor, get_shape_ceil(768, 1344))


class TestAcceptsResolution(unittest.TestCase):
    """FWDF-207: the rule for admitting a user-configured W*H into a curated list."""

    def setUp(self):
        self.krea2 = model_family.get_capabilities(model_family.ModelFamily.KREA2_RAW)
        self.sdxl = model_family.get_capabilities(model_family.ModelFamily.SDXL)

    def test_krea2_accepts_aligned_in_range_resolution(self):
        # 1152*1536 buckets to 1344, inside Krea 2's (1024, 2048).
        self.assertTrue(self.krea2.accepts_resolution(1152, 1536))

    def test_krea2_rejects_side_not_a_multiple_of_sixteen(self):
        self.assertFalse(self.krea2.accepts_resolution(1150, 1536))
        self.assertFalse(self.krea2.accepts_resolution(1152, 1544))

    def test_krea2_rejects_resolution_below_native_range(self):
        self.assertFalse(self.krea2.accepts_resolution(512, 512))

    def test_krea2_rejects_resolution_above_native_range(self):
        self.assertFalse(self.krea2.accepts_resolution(4096, 4096))

    def test_range_bounds_are_inclusive(self):
        floor, ceiling = self.krea2.native_resolution_range
        self.assertTrue(self.krea2.accepts_resolution(int(floor), int(floor)))
        self.assertTrue(self.krea2.accepts_resolution(int(ceiling), int(ceiling)))

    def test_non_positive_sides_are_rejected(self):
        self.assertFalse(self.krea2.accepts_resolution(0, 1024))
        self.assertFalse(self.krea2.accepts_resolution(-1024, 1024))

    def test_sdxl_only_requires_a_multiple_of_eight(self):
        self.assertTrue(self.sdxl.accepts_resolution(1160, 904))
        self.assertFalse(self.sdxl.accepts_resolution(1161, 904))

    def test_every_family_accepts_its_own_aspect_ratios(self):
        for family, caps in model_family.FAMILY_CAPABILITIES.items():
            for entry in caps.aspect_ratios:
                width, height = (int(side) for side in entry.split('*'))
                with self.subTest(family=family.name, entry=entry):
                    self.assertTrue(caps.accepts_resolution(width, height))


class TestResolutionMultipleValidation(unittest.TestCase):
    def test_non_positive_resolution_multiple_raises(self):
        for bad in (0, -8):
            with self.subTest(resolution_multiple=bad):
                with self.assertRaises(ValueError):
                    _make_blank_capabilities(resolution_multiple=bad)


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
