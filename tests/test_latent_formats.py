import unittest

import torch

from ldm_patched.modules.latent_formats import (
    LatentFormat,
    QwenImage,
    SD15,
    SDXL,
    SDXL_Playground_2_5,
    SD_X4,
    SC_Prior,
    SC_B,
    Flux,
)


class TestLatentChannels(unittest.TestCase):
    def test_base_class_defaults_to_four_channels(self):
        self.assertEqual(LatentFormat.latent_channels, 4)

    def test_sd_family_inherits_four_channels(self):
        for cls in (SD15, SDXL, SDXL_Playground_2_5, SD_X4, SC_Prior, SC_B):
            self.assertEqual(cls.latent_channels, 4, msg=f"{cls.__name__} should stay 4-channel")

    def test_flux_is_sixteen_channels(self):
        self.assertEqual(Flux.latent_channels, 16)

    def test_qwen_image_is_sixteen_channels(self):
        self.assertEqual(QwenImage.latent_channels, 16)


class TestProcessRoundtrip(unittest.TestCase):
    """process_out(process_in(x)) should recover x within floating-point tolerance."""

    def _assert_roundtrip(self, latent_format, channels):
        torch.manual_seed(0)
        x = torch.randn(2, channels, 4, 4)
        y = latent_format.process_out(latent_format.process_in(x))
        self.assertTrue(torch.allclose(x, y, atol=1e-4),
                         msg=f"{type(latent_format).__name__} roundtrip mismatch")

    def test_sd15_roundtrip(self):
        self._assert_roundtrip(SD15(), 4)

    def test_sdxl_roundtrip(self):
        self._assert_roundtrip(SDXL(), 4)

    def test_flux_roundtrip(self):
        self._assert_roundtrip(Flux(), 16)

    def test_qwen_image_roundtrip(self):
        self._assert_roundtrip(QwenImage(), 16)


class TestFluxFormat(unittest.TestCase):
    def test_scale_and_shift_factors(self):
        flux = Flux()
        self.assertAlmostEqual(flux.scale_factor, 0.3611)
        self.assertAlmostEqual(flux.shift_factor, 0.1159)

    def test_latent_rgb_factors_shape(self):
        flux = Flux()
        self.assertEqual(len(flux.latent_rgb_factors), 16)
        self.assertTrue(all(len(row) == 3 for row in flux.latent_rgb_factors))


class TestQwenImageFormat(unittest.TestCase):
    def test_per_channel_stats_shape(self):
        qwen = QwenImage()
        self.assertEqual(qwen.latents_mean.shape, (1, 16, 1, 1))
        self.assertEqual(qwen.latents_std.shape, (1, 16, 1, 1))

    def test_latent_rgb_factors_shape(self):
        qwen = QwenImage()
        self.assertEqual(len(qwen.latent_rgb_factors), 16)
        self.assertTrue(all(len(row) == 3 for row in qwen.latent_rgb_factors))


if __name__ == "__main__":
    unittest.main()
