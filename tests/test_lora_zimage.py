"""Tests for FWDF-153's P0 crash fix and LoRA loading for Z-Image (NextDiT):

- ldm_patched.modules.lora.model_lora_keys_unet(): selecting a Z-Image
  checkpoint used to raise `KeyError: 'num_res_blocks'` because this function
  unconditionally called unet_to_diffusers() against Z-Image's NextDiT
  unet_config, which never described a UNetModel and lacks the keys that
  function indexes. The fix guards that call behind
  unet_config["disable_unet_model_creation"] (set by model_base.ZImage);
  the SDXL/SD15 diffusers-key branch this function shares must keep running
  unchanged for real UNetModel configs.
- modules.core.StableDiffusionModel.__init__: the first test in this
  codebase to construct the real (not _FakeStableDiffusionModel) class
  around a real, non-SDXL backbone (model_base.ZImage).
- A synthetic LoRA targeting a real NextDiT module name, run through
  model_lora_keys_unet -> modules.lora.match_lora -> ModelPatcher.add_patches,
  measurably changes apply_model()'s output.
- modules.core.StableDiffusionModel.refresh_loras(): a LoRA whose keys
  belong to the wrong model family (SDXL-shaped keys on a Z-Image model, or
  vice versa) must be skipped safely (no exception, no weights patched) and
  now must also log a visible warning naming the LoRA file, the model file,
  and the unmatched-key count -- previously this branch `continue`d with no
  log output at all.

All state dicts/weights below are synthetic (tiny, random-initialized torch
tensors) -- no real checkpoint files are used.
"""

import sys
from pathlib import Path

import safetensors.torch
import re
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# args_manager calls parse_args() at import time, which chokes on pytest's
# argv. Patch sys.argv before any project modules are imported (mirrors
# tests/test_core_empty_latent.py).
_original_argv = sys.argv
sys.argv = [sys.argv[0]]

import modules.core as core  # noqa: E402
from ldm_patched.modules import latent_formats  # noqa: E402
from ldm_patched.modules import model_base  # noqa: E402
from ldm_patched.modules import model_patcher  # noqa: E402
import ldm_patched.modules.utils  # noqa: E402
from ldm_patched.modules.lora import model_lora_keys_unet  # noqa: E402
from modules.lora import match_lora  # noqa: E402

sys.argv = _original_argv


# ---------------------------------------------------------------------------
# Shared fixtures: a real, tiny model_base.ZImage instance, and a lightweight
# duck-typed stand-in for a real UNetModel-based BaseModel (SDXL-shaped
# unet_config, no real weights) -- model_lora_keys_unet() only ever reads
# .state_dict() and .model_config.unet_config off its argument.
# ---------------------------------------------------------------------------


class _TinyZImageConfig:
    def __init__(self, unet_config, shift=3.0):
        self.unet_config = unet_config
        self.latent_format = latent_formats.Flux()
        self.manual_cast_dtype = None
        self.sampling_settings = {"shift": shift}


def _tiny_z_image_unet_config():
    return dict(
        patch_size=2,
        in_channels=4,
        dim=64,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=2,
        n_kv_heads=2,
        multiple_of=8,
        ffn_dim_multiplier=2.0,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=32,
        axes_dims=(8, 12, 12),
        axes_lens=(16, 8, 8),
        rope_theta=256.0,
        z_image_modulation=True,
        time_scale=1000.0,
    )


def _build_tiny_z_image_model():
    """A real model_base.ZImage instance wrapping a real (tiny) NextDiT, with
    random weights broken out of symmetry so LoRA patches produce a
    measurable output difference."""
    config = _TinyZImageConfig(_tiny_z_image_unet_config())
    model = model_base.ZImage(config, device="cpu")
    with torch.no_grad():
        for p in model.diffusion_model.parameters():
            p.normal_(mean=0.0, std=0.02)
    model.diffusion_model.eval()
    return model


# Real SDXL unet_config (copied from supported_models.SDXL's declared shape,
# exercised the same way in tests/test_zimage_model.py's
# test_sdxl_shaped_config_still_resolves_to_sdxl): no
# disable_unet_model_creation key, so the diffusers-key branch must run.
_SDXL_UNET_CONFIG = {
    "use_checkpoint": False, "image_size": 32, "use_spatial_transformer": True, "legacy": False,
    "dtype": torch.float32, "num_classes": "sequential", "adm_in_channels": 2816,
    "in_channels": 4, "out_channels": 4, "model_channels": 320,
    "num_res_blocks": [2, 2, 2], "transformer_depth": [0, 0, 2, 2, 10, 10],
    "transformer_depth_output": [0, 0, 0, 2, 2, 2, 10, 10, 10],
    "channel_mult": [1, 2, 4], "transformer_depth_middle": 10,
    "use_linear_in_transformer": True, "context_dim": 2048,
    "use_temporal_attention": False, "use_temporal_resblock": False,
}


class _FakeModelConfig:
    def __init__(self, unet_config):
        self.unet_config = unet_config


class _FakeUnetModel:
    """Duck-types the two attributes model_lora_keys_unet() reads off its
    argument (.state_dict(), .model_config.unet_config) without needing a
    real weighted UNetModel/NextDiT instance."""

    def __init__(self, unet_config, state_dict=None):
        self.model_config = _FakeModelConfig(unet_config)
        self._state_dict = state_dict if state_dict is not None else {}

    def state_dict(self):
        return self._state_dict


class _RecordingClonableUnet:
    """Enough of modules.core.StableDiffusionModel's .unet contract
    (clone() + add_patches()) to exercise refresh_loras() without a real
    ModelPatcher, recording which patches (if any) actually got applied."""

    def __init__(self):
        self.patches = {}

    def clone(self):
        return self

    def add_patches(self, patches, weight):
        self.patches.update(patches)
        return list(patches.keys())


def _bare_stable_diffusion_model(lora_key_map_unet, filename):
    """Bypasses StableDiffusionModel.__init__ (which would need a real
    model to build the key maps from) the same way
    tests/test_zimage_pipeline.py::TestRefreshLorasCloneInvariant does, to
    exercise refresh_loras() in isolation with a pre-built key map."""
    model = core.StableDiffusionModel.__new__(core.StableDiffusionModel)
    model.unet = _RecordingClonableUnet()
    model.vae = None
    model.clip = None
    model.clip_vision = None
    model.filename = filename
    model.vae_filename = None
    model.visited_loras = ''
    model.lora_key_map_unet = lora_key_map_unet
    model.lora_key_map_clip = {}
    return model


# ---------------------------------------------------------------------------
# ldm_patched.modules.lora.model_lora_keys_unet(): the P0 guard itself
# ---------------------------------------------------------------------------


class TestModelLoraKeysUnetDisableUnetModelCreationGuard:
    def test_disable_unet_model_creation_skips_diffusers_branch_without_raising(self):
        # Before the fix: unet_to_diffusers() would unconditionally index
        # unet_config["num_res_blocks"], raising KeyError for a Z-Image-shaped
        # config that never declares it.
        unet_config = {"disable_unet_model_creation": True, "dim": 64}
        fake_model = _FakeUnetModel(unet_config, state_dict={})

        key_map = model_lora_keys_unet(fake_model, {})

        assert key_map == {}

    def test_guard_skips_diffusers_branch_but_keeps_generic_prefix_loop(self):
        # The generic `diffusion_model.`-prefix loop (unaffected by the
        # guard) must remain the sole key-mapping path for a
        # disable_unet_model_creation-flagged architecture.
        unet_config = {"disable_unet_model_creation": True}
        state_dict = {"diffusion_model.layers.0.attention.qkv.weight": torch.zeros(1)}
        fake_model = _FakeUnetModel(unet_config, state_dict=state_dict)

        key_map = model_lora_keys_unet(fake_model, {})

        assert key_map == {
            "lora_unet_layers_0_attention_qkv": "diffusion_model.layers.0.attention.qkv.weight",
        }

    def test_sdxl_shaped_config_diffusers_branch_still_populates_key_map(self):
        # Regression: the guard must not disable diffusers-format SDXL LoRA
        # support. Every diffusers-style weight key unet_to_diffusers()
        # produces for a real SDXL unet_config must still show up as a
        # lora_unet_* mapping.
        fake_model = _FakeUnetModel(_SDXL_UNET_CONFIG, state_dict={})

        key_map = model_lora_keys_unet(fake_model, {})

        expected_diffusers_keys = ldm_patched.modules.utils.unet_to_diffusers(_SDXL_UNET_CONFIG)
        weight_keys = [k for k in expected_diffusers_keys if k.endswith(".weight")]
        assert weight_keys, "test fixture produced no diffusers weight keys"

        for k in weight_keys:
            lora_key = "lora_unet_{}".format(k[:-len(".weight")].replace(".", "_"))
            assert lora_key in key_map
            assert key_map[lora_key] == "diffusion_model.{}".format(expected_diffusers_keys[k])


# ---------------------------------------------------------------------------
# modules.core.StableDiffusionModel.__init__: the P0 crash's actual site
# ---------------------------------------------------------------------------


class TestStableDiffusionModelZImageP0Fix:
    """Before the fix, constructing StableDiffusionModel around a Z-Image
    unet raised KeyError('num_res_blocks') from inside __init__, with or
    without any LoRA selected -- this is the first test in this codebase to
    build the real StableDiffusionModel (not _FakeStableDiffusionModel)
    around a non-SDXL backbone."""

    def test_init_does_not_raise_for_real_zimage_backbone(self):
        model = _build_tiny_z_image_model()
        unet_patcher = model_patcher.ModelPatcher(
            model, load_device=torch.device('cpu'), offload_device=torch.device('cpu'),
        )

        sd_model = core.StableDiffusionModel(
            unet=unet_patcher, vae=None, clip=None, clip_vision=None,
            filename='z_image_turbo.safetensors',
        )

        assert sd_model.unet is unet_patcher
        # The generic diffusion_model.-prefix loop must have produced real
        # NextDiT key mappings (fused attention.qkv, SwiGLU w1/w2/w3, etc.).
        assert 'lora_unet_layers_0_attention_qkv' in sd_model.lora_key_map_unet
        assert sd_model.lora_key_map_unet['lora_unet_layers_0_attention_qkv'] == \
            'diffusion_model.layers.0.attention.qkv.weight'


# ---------------------------------------------------------------------------
# A synthetic LoRA targeting a real NextDiT attribute measurably changes
# apply_model()'s output
# ---------------------------------------------------------------------------


class TestSyntheticLoraChangesZImageOutput:
    def test_lora_targeting_real_nextdit_attention_qkv_changes_apply_model_output(self):
        torch.manual_seed(0)
        model = _build_tiny_z_image_model()
        unet_patcher = model_patcher.ModelPatcher(
            model, load_device=torch.device('cpu'), offload_device=torch.device('cpu'),
        )

        key_map = model_lora_keys_unet(model, {})
        target_key = "lora_unet_layers_0_attention_qkv"
        assert target_key in key_map
        real_key = key_map[target_key]
        assert real_key == "diffusion_model.layers.0.attention.qkv.weight"

        qkv_weight = model.diffusion_model.layers[0].attention.qkv.weight
        out_features, in_features = qkv_weight.shape
        rank = 4
        lora_sd = {
            "{}.lora_up.weight".format(target_key): torch.randn(out_features, rank) * 0.1,
            "{}.lora_down.weight".format(target_key): torch.randn(rank, in_features) * 0.1,
        }

        lora_patches, lora_unmatch = match_lora(lora_sd, key_map)
        assert real_key in lora_patches
        assert lora_unmatch == {}

        loaded_keys = unet_patcher.add_patches(lora_patches, strength_patch=1.0)
        assert real_key in loaded_keys

        x = torch.randn(1, 4, 8, 8)
        sigma = torch.tensor([0.3])
        context = torch.randn(1, 5, 32)

        with torch.no_grad():
            out_before = model.apply_model(x, sigma, c_crossattn=context)

        unet_patcher.patch_model()
        try:
            with torch.no_grad():
                out_after = model.apply_model(x, sigma, c_crossattn=context)
        finally:
            unet_patcher.unpatch_model()

        assert out_before.shape == out_after.shape
        assert not torch.allclose(out_before, out_after)


# ---------------------------------------------------------------------------
# refresh_loras(): family mismatches are safe, and now logged
# ---------------------------------------------------------------------------


class TestRefreshLorasFamilyMismatchWarning:
    def _write_lora_file(self, tmp_path, sd, name):
        lora_path = str(tmp_path / name)
        safetensors.torch.save_file(sd, lora_path)
        return lora_path

    def test_sdxl_shaped_lora_on_z_image_model_is_safely_skipped_and_logged(self, tmp_path, capsys):
        zimage_model = _build_tiny_z_image_model()
        key_map = model_lora_keys_unet(zimage_model, {})

        # 13 valid up/down LoRA pairs (26 tensors) targeting SDXL block names that
        # do not exist in the Z-Image key map — exercises genuine cross-family
        # mismatch, not malformed-LoRA rejection. 26 unmatched > the 12 threshold.
        sdxl_style_lora = {}
        for i in range(13):
            base = "lora_unet_input_blocks_{}_1_proj_in".format(i)
            sdxl_style_lora[base + ".lora_up.weight"] = torch.randn(4, 2)
            sdxl_style_lora[base + ".lora_down.weight"] = torch.randn(2, 4)
        lora_path = self._write_lora_file(tmp_path, sdxl_style_lora, 'sdxl_style.safetensors')

        model = _bare_stable_diffusion_model(key_map, 'z_image_turbo.safetensors')
        model.refresh_loras([(lora_path, 1.0)])  # must not raise

        captured = capsys.readouterr()
        assert lora_path in captured.out
        assert 'z_image_turbo.safetensors' in captured.out
        # Assert the skip path fired with the unmatched count, not exact wording.
        assert 'unmatched' in captured.out.lower()
        assert re.search(r'\d+ unmatched', captured.out), captured.out
        assert model.unet_with_lora.patches == {}

    def test_z_image_shaped_lora_on_sdxl_model_is_safely_skipped_and_logged(self, tmp_path, capsys):
        fake_sdxl_model = _FakeUnetModel(_SDXL_UNET_CONFIG, state_dict={})
        key_map = model_lora_keys_unet(fake_sdxl_model, {})

        # 13 valid up/down LoRA pairs targeting Z-Image (NextDiT) attention keys
        # that do not exist in the SDXL key map.
        z_image_style_lora = {}
        for i in range(13):
            base = "lora_unet_layers_{}_attention_qkv".format(i)
            z_image_style_lora[base + ".lora_up.weight"] = torch.randn(4, 2)
            z_image_style_lora[base + ".lora_down.weight"] = torch.randn(2, 4)
        lora_path = self._write_lora_file(tmp_path, z_image_style_lora, 'z_image_style.safetensors')

        model = _bare_stable_diffusion_model(key_map, 'sdxl_base.safetensors')
        model.refresh_loras([(lora_path, 1.0)])  # must not raise

        captured = capsys.readouterr()
        assert lora_path in captured.out
        assert 'sdxl_base.safetensors' in captured.out
        assert 'unmatched' in captured.out.lower()
        assert re.search(r'\d+ unmatched', captured.out), captured.out
        assert model.unet_with_lora.patches == {}
