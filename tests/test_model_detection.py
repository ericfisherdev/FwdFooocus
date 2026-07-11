"""
Unit tests for the pluggable architecture detection registry (FWDF-116).

Covers: UNet routing stays unaffected by the refactor, a detector registered ahead of
the terminal UNet fallback wins even when the UNet detector would also match, an
architecture that is recognized but has no supported_models entry fails gracefully
instead of crashing, and an unrecognized architecture returns None instead of raising
a KeyError. All state dicts are synthetic (torch tensors of the minimal shapes
detection reads) -- no real model files are used.
"""

import unittest
from unittest.mock import patch

import pytest
import torch

from ldm_patched.modules import model_detection
from ldm_patched.modules import sd as sd_module
from ldm_patched.modules import supported_models
from ldm_patched.modules import supported_models_base


class _Unmatchable:
    """Sentinel that compares unequal to everything, including itself."""

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True


def _unmatchable_unet_config():
    """A unet_config dict that cannot match any registered supported_models class.

    BASE.matches() indexes unet_config[k] directly (no .get()) for every key its
    subclass declares, so this derives the union of every key any registered class
    checks (keeping it in sync automatically as new architectures are added) and fills
    them with sentinel values that compare unequal to anything -- otherwise
    model_config_from_unet_config() raises KeyError instead of returning None. Returns
    a fresh dict each call since BASE.__init__() mutates whatever dict it is given
    (it merges in unet_extra_config).
    """
    keys = set()
    for model_config_class in supported_models.models:
        keys.update(model_config_class.unet_config.keys())
    return {key: _Unmatchable() for key in keys}


class TestUnetDetectorRouting(unittest.TestCase):
    """The registry-based model_config_from_unet() must route a real UNet-shaped
    checkpoint through the unmodified detect_unet_config() exactly as before this
    refactor -- no detection regression for existing architectures."""

    def test_sd15_shaped_checkpoint_matches_sd15(self):
        prefix = "model.diffusion_model."
        state_dict = {
            prefix + "input_blocks.0.0.weight": torch.zeros(320, 4, 3, 3),
            prefix + "input_blocks.1.0.in_layers.0.weight": torch.zeros(1),
            prefix + "input_blocks.1.0.out_layers.3.weight": torch.zeros(320),
            prefix + "input_blocks.1.1.proj_in.weight": torch.zeros(320, 320, 1, 1),
            prefix + "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": torch.zeros(320, 768),
        }

        model_config = model_detection.model_config_from_unet(state_dict, prefix, torch.float32)

        self.assertIsInstance(model_config, supported_models.SD15)
        self.assertEqual(model_config.unet_config["model_channels"], 320)
        self.assertEqual(model_config.unet_config["in_channels"], 4)
        self.assertEqual(model_config.unet_config["context_dim"], 768)
        self.assertIsNone(model_config.unet_config["adm_in_channels"])


class TestDetectorTableOrdering(unittest.TestCase):
    """A detector registered via register_detector() must be tried before the
    terminal UNet fallback, even when the state dict also satisfies the UNet
    detector's matches()."""

    def setUp(self):
        self._original_table = list(model_detection._DETECTOR_TABLE)

    def tearDown(self):
        model_detection._DETECTOR_TABLE[:] = self._original_table

    def test_earlier_registered_detector_wins_over_unet_fallback(self):
        prefix = "model.diffusion_model."
        state_dict = {
            prefix + "x_embedder.weight": torch.zeros(1),
            # Also present, so the terminal UNet detector's matches() would fire too.
            prefix + "input_blocks.0.0.weight": torch.zeros(320, 4, 3, 3),
        }
        detected_config = _unmatchable_unet_config()
        model_detection.register_detector(
            name="fake-dit",
            matches=lambda keys, key_prefix: (key_prefix + "x_embedder.weight") in keys,
            detect_config=lambda _state_dict, _key_prefix, _dtype: detected_config,
        )

        with pytest.raises(model_detection.UnsupportedArchitectureError) as exc_info:
            model_detection.model_config_from_unet(state_dict, prefix, torch.float32)

        # Raising UnsupportedArchitectureError (rather than routing through
        # detect_unet_config, which would KeyError on this synthetic dict) proves the
        # fake-dit detector -- not the UNet fallback -- handled this state dict.
        self.assertEqual(exc_info.value.architecture_name, "fake-dit")
        self.assertIs(exc_info.value.unet_config, detected_config)

    def test_recognized_unsupported_architecture_uses_base_when_flag_set(self):
        """controlnet.py calls model_config_from_unet(..., True) and expects a usable
        BASE config back, never an exception, when a detector recognizes the
        architecture but no supported_models entry exists for it."""
        prefix = ""
        state_dict = {
            prefix + "x_embedder.weight": torch.zeros(1),
        }
        detected_config = _unmatchable_unet_config()

        model_detection.register_detector(
            name="fake-dit",
            matches=lambda keys, key_prefix: (key_prefix + "x_embedder.weight") in keys,
            detect_config=lambda _state_dict, _key_prefix, _dtype: detected_config,
        )

        model_config = model_detection.model_config_from_unet(state_dict, prefix, torch.float32, use_base_if_no_match=True)

        self.assertIsInstance(model_config, supported_models_base.BASE)
        self.assertIs(model_config.unet_config, detected_config)

    def test_duplicate_detector_name_is_rejected(self):
        model_detection.register_detector(
            name="fake-dit",
            matches=lambda _keys, _key_prefix: False,
            detect_config=lambda _state_dict, _key_prefix, _dtype: {},
        )

        with pytest.raises(ValueError, match="fake-dit"):
            model_detection.register_detector(
                name="fake-dit",
                matches=lambda _keys, _key_prefix: False,
                detect_config=lambda _state_dict, _key_prefix, _dtype: {},
            )


class TestNoDetectorMatched(unittest.TestCase):
    """An architecture no detector recognizes must return None (the existing
    'unknown architecture' contract), not raise."""

    def test_returns_none_when_no_detector_recognizes_the_state_dict(self):
        state_dict = {"totally.unrelated.key": torch.zeros(1)}

        result = model_detection.model_config_from_unet(state_dict, "model.diffusion_model.", torch.float32)

        self.assertIsNone(result)

    def test_unet_shaped_but_unresolvable_checkpoint_still_returns_none(self):
        """A checkpoint the terminal UNet fallback detects but that matches no
        supported_models entry must keep the pre-registry contract of returning None
        (so sd.py raises 'Could not detect model type'), not raise
        UnsupportedArchitectureError naming 'unet' as an unregistered architecture."""
        prefix = "model.diffusion_model."
        state_dict = {
            prefix + "input_blocks.0.0.weight": torch.zeros(320, 4, 3, 3),
            prefix + "input_blocks.1.0.in_layers.0.weight": torch.zeros(1),
            prefix + "input_blocks.1.0.out_layers.3.weight": torch.zeros(320),
            prefix + "input_blocks.1.1.proj_in.weight": torch.zeros(320, 320, 1, 1),
            # context_dim 999 matches no registered supported_models class.
            prefix + "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": torch.zeros(320, 999),
        }

        result = model_detection.model_config_from_unet(state_dict, prefix, torch.float32)

        self.assertIsNone(result)


class TestLoadCheckpointGuessConfigOrderingFix(unittest.TestCase):
    """Regression tests for ldm_patched.modules.sd.load_checkpoint_guess_config():
    the None check must run before set_manual_cast() is called on model_config, and
    UnsupportedArchitectureError must be converted into a descriptive RuntimeError."""

    @patch("ldm_patched.modules.utils.load_torch_file")
    @patch("ldm_patched.modules.model_detection.model_config_from_unet")
    def test_none_model_config_raises_runtime_error_not_attribute_error(self, mock_detect, mock_load):
        mock_load.return_value = {
            "model.diffusion_model.input_blocks.0.0.weight": torch.zeros(4, 4, 3, 3),
        }
        mock_detect.return_value = None

        with pytest.raises(RuntimeError, match="Could not detect model type"):
            sd_module.load_checkpoint_guess_config(
                "fake-checkpoint.safetensors",
                output_vae=False,
                output_clip=False,
                output_model=False,
            )

    @patch("ldm_patched.modules.utils.load_torch_file")
    @patch("ldm_patched.modules.model_detection.model_config_from_unet")
    def test_unsupported_architecture_is_wrapped_in_descriptive_runtime_error(self, mock_detect, mock_load):
        mock_load.return_value = {
            "model.diffusion_model.x_embedder.weight": torch.zeros(4, 4),
        }
        mock_detect.side_effect = model_detection.UnsupportedArchitectureError(
            "z-image", {"architecture": "z-image"}
        )

        with pytest.raises(RuntimeError, match=r"z-image.*fake-checkpoint\.safetensors|fake-checkpoint\.safetensors.*z-image"):
            sd_module.load_checkpoint_guess_config(
                "fake-checkpoint.safetensors",
                output_vae=False,
                output_clip=False,
                output_model=False,
            )


class TestLoadControlnetGuard(unittest.TestCase):
    """Regression test for ldm_patched.modules.controlnet.load_controlnet(): a
    controlnet state dict whose architecture resolves to no model config must raise a
    descriptive RuntimeError, not AttributeError on None.unet_config."""

    @patch("ldm_patched.modules.utils.load_torch_file")
    @patch("ldm_patched.modules.model_detection.model_config_from_unet")
    def test_unresolvable_controlnet_raises_descriptive_runtime_error(self, mock_detect, mock_load):
        from ldm_patched.modules import controlnet as controlnet_module

        mock_load.return_value = {
            "zero_convs.0.0.weight": torch.zeros(1),
        }
        mock_detect.return_value = None

        with pytest.raises(RuntimeError, match=r"controlnet model type.*fake-controlnet\.safetensors"):
            controlnet_module.load_controlnet("fake-controlnet.safetensors")

    @patch("ldm_patched.modules.utils.load_torch_file")
    @patch("ldm_patched.modules.model_detection.unet_config_from_diffusers_unet")
    def test_unresolvable_diffusers_controlnet_raises_descriptive_runtime_error(self, mock_diffusers_detect, mock_load):
        from ldm_patched.modules import controlnet as controlnet_module

        mock_load.return_value = {
            "controlnet_cond_embedding.conv_in.weight": torch.zeros(1),
        }
        mock_diffusers_detect.return_value = None

        with pytest.raises(RuntimeError, match=r"controlnet model type.*fake-diffusers-controlnet\.safetensors"):
            controlnet_module.load_controlnet("fake-diffusers-controlnet.safetensors")

    @patch("ldm_patched.modules.utils.load_torch_file")
    @patch("ldm_patched.modules.model_detection.model_config_from_unet")
    def test_loader_requests_base_fallback_for_unsupported_architectures(self, mock_detect, mock_load):
        """The controlnet loader must ask detection for the BASE fallback
        (use_base_if_no_match=True): controlnet state dicts are generally not in
        supported_models, and a recognized-but-unregistered architecture is expected
        to yield a generic BASE(unet_config) rather than UnsupportedArchitectureError
        (that BASE contract itself is covered in TestDetectorTableOrdering)."""
        from ldm_patched.modules import controlnet as controlnet_module

        class _StopLoader(Exception):
            pass

        mock_load.return_value = {
            "zero_convs.0.0.weight": torch.zeros(1),
        }
        mock_detect.side_effect = _StopLoader()

        with pytest.raises(_StopLoader):
            controlnet_module.load_controlnet("fake-controlnet.safetensors")

        self.assertTrue(mock_detect.call_args.kwargs.get("use_base_if_no_match"))
