import math

import torch
from . import model_base
from . import utils

from . import sd1_clip
from . import sd2_clip
from . import sdxl_clip

from . import supported_models_base
from . import latent_formats

from . import diffusers_convert

class SD15(supported_models_base.BASE):
    unet_config = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
        "use_temporal_attention": False,
    }

    unet_extra_config = {
        "num_heads": 8,
        "num_head_channels": -1,
    }

    latent_format = latent_formats.SD15

    def process_clip_state_dict(self, state_dict):
        k = list(state_dict.keys())
        for x in k:
            if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):
                y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")
                state_dict[y] = state_dict.pop(x)

        if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in state_dict:
            ids = state_dict['cond_stage_model.transformer.text_model.embeddings.position_ids']
            if ids.dtype == torch.float32:
                state_dict['cond_stage_model.transformer.text_model.embeddings.position_ids'] = ids.round()

        replace_prefix = {}
        replace_prefix["cond_stage_model."] = "cond_stage_model.clip_l."
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix)
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {"clip_l.": "cond_stage_model."}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def clip_target(self):
        return supported_models_base.ClipTarget(sd1_clip.SD1Tokenizer, sd1_clip.SD1ClipModel)

class SD20(supported_models_base.BASE):
    unet_config = {
        "context_dim": 1024,
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "adm_in_channels": None,
        "use_temporal_attention": False,
    }

    latent_format = latent_formats.SD15

    def model_type(self, state_dict, prefix=""):
        if self.unet_config["in_channels"] == 4: #SD2.0 inpainting models are not v prediction
            k = "{}output_blocks.11.1.transformer_blocks.0.norm1.bias".format(prefix)
            out = state_dict[k]
            if torch.std(out, unbiased=False) > 0.09: # not sure how well this will actually work. I guess we will find out.
                return model_base.ModelType.V_PREDICTION
        return model_base.ModelType.EPS

    def process_clip_state_dict(self, state_dict):
        replace_prefix = {}
        replace_prefix["conditioner.embedders.0.model."] = "cond_stage_model.model." #SD2 in sgm format
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix)

        state_dict = utils.transformers_convert(state_dict, "cond_stage_model.model.", "cond_stage_model.clip_h.transformer.text_model.", 24)
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        replace_prefix["clip_h"] = "cond_stage_model.model"
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix)
        state_dict = diffusers_convert.convert_text_enc_state_dict_v20(state_dict)
        return state_dict

    def clip_target(self):
        return supported_models_base.ClipTarget(sd2_clip.SD2Tokenizer, sd2_clip.SD2ClipModel)

class SD21UnclipL(SD20):
    unet_config = {
        "context_dim": 1024,
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "adm_in_channels": 1536,
        "use_temporal_attention": False,
    }

    clip_vision_prefix = "embedder.model.visual."
    noise_aug_config = {"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 768}


class SD21UnclipH(SD20):
    unet_config = {
        "context_dim": 1024,
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "adm_in_channels": 2048,
        "use_temporal_attention": False,
    }

    clip_vision_prefix = "embedder.model.visual."
    noise_aug_config = {"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 1024}

class SDXLRefiner(supported_models_base.BASE):
    unet_config = {
        "model_channels": 384,
        "use_linear_in_transformer": True,
        "context_dim": 1280,
        "adm_in_channels": 2560,
        "transformer_depth": [0, 0, 4, 4, 4, 4, 0, 0],
        "use_temporal_attention": False,
    }

    latent_format = latent_formats.SDXL

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.SDXLRefiner(self, device=device)

    def process_clip_state_dict(self, state_dict):
        keys_to_replace = {}
        replace_prefix = {}

        state_dict = utils.transformers_convert(state_dict, "conditioner.embedders.0.model.", "cond_stage_model.clip_g.transformer.text_model.", 32)
        keys_to_replace["conditioner.embedders.0.model.text_projection"] = "cond_stage_model.clip_g.text_projection"
        keys_to_replace["conditioner.embedders.0.model.logit_scale"] = "cond_stage_model.clip_g.logit_scale"

        state_dict = utils.state_dict_key_replace(state_dict, keys_to_replace)
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        state_dict_g = diffusers_convert.convert_text_enc_state_dict_v20(state_dict, "clip_g")
        if "clip_g.transformer.text_model.embeddings.position_ids" in state_dict_g:
            state_dict_g.pop("clip_g.transformer.text_model.embeddings.position_ids")
        replace_prefix["clip_g"] = "conditioner.embedders.0.model"
        state_dict_g = utils.state_dict_prefix_replace(state_dict_g, replace_prefix)
        return state_dict_g

    def clip_target(self):
        return supported_models_base.ClipTarget(sdxl_clip.SDXLTokenizer, sdxl_clip.SDXLRefinerClipModel)

class SDXL(supported_models_base.BASE):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }

    latent_format = latent_formats.SDXL

    def model_type(self, state_dict, prefix=""):
        if "v_pred" in state_dict:
            return model_base.ModelType.V_PREDICTION
        else:
            return model_base.ModelType.EPS

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.SDXL(self, model_type=self.model_type(state_dict, prefix), device=device)
        if self.inpaint_model():
            out.set_inpaint()
        return out

    def process_clip_state_dict(self, state_dict):
        keys_to_replace = {}
        replace_prefix = {}

        replace_prefix["conditioner.embedders.0.transformer.text_model"] = "cond_stage_model.clip_l.transformer.text_model"
        state_dict = utils.transformers_convert(state_dict, "conditioner.embedders.1.model.", "cond_stage_model.clip_g.transformer.text_model.", 32)
        keys_to_replace["conditioner.embedders.1.model.text_projection"] = "cond_stage_model.clip_g.text_projection"
        keys_to_replace["conditioner.embedders.1.model.text_projection.weight"] = "cond_stage_model.clip_g.text_projection"
        keys_to_replace["conditioner.embedders.1.model.logit_scale"] = "cond_stage_model.clip_g.logit_scale"

        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix)
        state_dict = utils.state_dict_key_replace(state_dict, keys_to_replace)
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        keys_to_replace = {}
        state_dict_g = diffusers_convert.convert_text_enc_state_dict_v20(state_dict, "clip_g")
        if "clip_g.transformer.text_model.embeddings.position_ids" in state_dict_g:
            state_dict_g.pop("clip_g.transformer.text_model.embeddings.position_ids")
        for k in state_dict:
            if k.startswith("clip_l"):
                state_dict_g[k] = state_dict[k]

        replace_prefix["clip_g"] = "conditioner.embedders.1.model"
        replace_prefix["clip_l"] = "conditioner.embedders.0"
        state_dict_g = utils.state_dict_prefix_replace(state_dict_g, replace_prefix)
        return state_dict_g

    def clip_target(self):
        return supported_models_base.ClipTarget(sdxl_clip.SDXLTokenizer, sdxl_clip.SDXLClipModel)

class SSD1B(SDXL):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 4, 4],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }

class Segmind_Vega(SDXL):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 1, 1, 2, 2],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }

class SVD_img2vid(supported_models_base.BASE):
    unet_config = {
        "model_channels": 320,
        "in_channels": 8,
        "use_linear_in_transformer": True,
        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
        "context_dim": 1024,
        "adm_in_channels": 768,
        "use_temporal_attention": True,
        "use_temporal_resblock": True
    }

    clip_vision_prefix = "conditioner.embedders.0.open_clip.model.visual."

    latent_format = latent_formats.SD15

    sampling_settings = {"sigma_max": 700.0, "sigma_min": 0.002}

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.SVD_img2vid(self, device=device)
        return out

    def clip_target(self):
        return None

class Stable_Zero123(supported_models_base.BASE):
    unet_config = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
        "use_temporal_attention": False,
        "in_channels": 8,
    }

    unet_extra_config = {
        "num_heads": 8,
        "num_head_channels": -1,
    }

    clip_vision_prefix = "cond_stage_model.model.visual."

    latent_format = latent_formats.SD15

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Stable_Zero123(self, device=device, cc_projection_weight=state_dict["cc_projection.weight"], cc_projection_bias=state_dict["cc_projection.bias"])
        return out

    def clip_target(self):
        return None

class SD_X4Upscaler(SD20):
    unet_config = {
        "context_dim": 1024,
        "model_channels": 256,
        'in_channels': 7,
        "use_linear_in_transformer": True,
        "adm_in_channels": None,
        "use_temporal_attention": False,
    }

    unet_extra_config = {
        "disable_self_attentions": [True, True, True, False],
        "num_classes": 1000,
        "num_heads": 8,
        "num_head_channels": -1,
    }

    latent_format = latent_formats.SD_X4

    sampling_settings = {
        "linear_start": 0.0001,
        "linear_end": 0.02,
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.SD_X4Upscaler(self, device=device)
        return out

class ZImage(supported_models_base.BASE):
    """Tongyi-MAI's Z-Image, wrapping the NextDiT (Lumina2) backbone (FWDF-123)
    instead of UNetModel. unet_config carries NextDiT's constructor keys, detected
    off tensor shapes and fixed architectural constants by
    model_detection.detect_z_image_config().
    """
    unet_config = {
        "dim": 3840,
        "n_layers": 30,
        "n_heads": 30,
        "n_kv_heads": 30,
        "in_channels": 16,
        "cap_feat_dim": 2560,
        "axes_dims": [32, 48, 48],
        "axes_lens": [1536, 512, 512],
        "norm_eps": 1e-05,
        "qk_norm": True,
        "n_refiner_layers": 2,
        "rope_theta": 256.0,
    }

    unet_extra_config = {}

    latent_format = latent_formats.Flux

    sampling_settings = {
        "shift": 3.0,
    }

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.ZImage(self, device=device)

    def clip_target(self):
        # FWDF-125's Qwen3-4B ClipTarget has not landed on this branch yet;
        # placeholder mirrors Stable_Zero123/SVD_img2vid until it does.
        return None

# Krea 2's published Flux-style timestep shift parameter (mu). ComfyUI feeds it
# to ModelSamplingFlux, whose schedule is flux_time_shift(mu, 1.0, t) =
# e^mu / (e^mu + 1/t - 1). This fork only has ModelSamplingDiscreteFlow, whose
# time_snr_shift(alpha, t) = alpha * t / (1 + (alpha - 1) * t) is the same
# function iff alpha = e^mu, so the shift handed to it must be exp(mu), never
# mu itself.
KREA2_FLUX_SHIFT_MU = 1.15


class Krea2(supported_models_base.BASE):
    """Krea 2 (Raw / Turbo), wrapping the SingleStreamDiT backbone (FWDF-131).

    Differences from Z-Image: no guidance-distillation embedding (classic
    two-pass CFG applies, so negative prompts work), a Flux-style timestep
    shift, and the Qwen Image / Wan 2.1 16-channel latent space.

    unet_config matches on the detector's discriminant only: the Raw, Turbo,
    fp8 and int8 variants share one architecture, and BASE.matches() treats
    every listed key as a hard equality.
    """
    unet_config = {
        "image_model": "krea2",
    }

    # BASE's default injects num_heads / num_head_channels, which
    # SingleStreamDiT does not accept.
    unet_extra_config = {}

    # Qwen Image reuses the Wan 2.1 VAE statistics ComfyUI's Krea2 names
    # directly; this fork's QwenImage carries them in the layout it needs.
    latent_format = latent_formats.QwenImage

    # multiplier 1.0 makes ModelSamplingDiscreteFlow.timestep(sigma) == sigma,
    # so the DiT receives the raw [0, 1] flow time.
    sampling_settings = {
        "shift": math.exp(KREA2_FLUX_SHIFT_MU),
        "multiplier": 1.0,
    }

    memory_usage_factor = 2.2

    def get_model(self, state_dict, prefix="", device=None):
        # No self.inpaint_model() call: it indexes unet_config["in_channels"],
        # which this config does not have (the key is "channels").
        return model_base.Krea2(self, device=device)

    def clip_target(self):
        # The checkpoint carries no text encoder; the Qwen3-VL encoder is
        # attached by family in modules/default_pipeline.py::refresh_base_model()
        # (loader: FWDF-133, wiring: FWDF-152), as Z-Image does.
        return None

models = [Stable_Zero123, SD15, SD20, SD21UnclipL, SD21UnclipH, SDXLRefiner, SDXL, SSD1B, Segmind_Vega, SD_X4Upscaler]
models += [SVD_img2vid]
models += [ZImage]
models += [Krea2]
