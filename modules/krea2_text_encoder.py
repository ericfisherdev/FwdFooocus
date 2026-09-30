"""Qwen3-VL-4B text encoder wiring for Krea 2 (FWDF-133).

Assembles `modules.text_encoder.TransformerTextEncoder` (FWDF-122) with the
concrete Qwen3-VL-4B model from `ldm_patched.modules.qwen3vl_clip` and the
shared `Qwen3Tokenizer` into a ready `TextEncoder`, mirroring
`modules.qwen3_text_encoder` (Z-Image).

No `supported_models.clip_target()` wiring happens here, following the Z-Image
precedent: `Krea2.clip_target()` returns `None` and
`modules.default_pipeline.refresh_base_model` wires the standalone encoder per
family (FWDF-152).
"""

import os

import modules.config
from ldm_patched.modules.qwen3_clip import Qwen3Tokenizer
from ldm_patched.modules.qwen3vl_clip import KREA2_TAP_LAYERS, QWEN3_VL_4B_TEXT_CONFIG, Qwen3VLTextModel
from modules.text_encoder import TextEncoderStateDictMismatchError, TransformerTextEncoder, strip_chat_role_markers

# Byte-identical to ComfyUI's `KREA2_TEMPLATE` (the Qwen-Image system
# template). No `<think>` block: the reference conditions on the no-think
# template.
KREA2_CHAT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
KREA2_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

QWEN3_VL_WEIGHTS_FILENAME = "qwen3vl_4b_bf16.safetensors"

# A pipeline-level choice matching Z-Image, not an architectural ceiling: the
# reference does not truncate, and the DiT places all text tokens at RoPE
# position (0, 0, 0).
MAX_SEQUENCE_LENGTH = 512

LANGUAGE_MODEL_PREFIX = "model.language_model."
VISUAL_PREFIX = "model.visual."
TEXT_TOWER_PREFIX = "model."


def select_text_tower_state_dict(state_dict: dict) -> dict:
    """Extracts the text tower from a Qwen3-VL checkpoint's state dict.

    Drops every `model.visual.*` key and renames `model.language_model.X` to
    `model.X` (the layout `Qwen3VLTextModel` exposes). Any other key passes
    through unchanged so the strict load that follows reports it.

    Raises:
        TextEncoderStateDictMismatchError: no `model.language_model.*` key is
            present (wrong file, e.g. a plain Qwen3-4B checkpoint).
    """
    if not any(key.startswith(LANGUAGE_MODEL_PREFIX) for key in state_dict):
        raise TextEncoderStateDictMismatchError(
            "Qwen3VLTextModel",
            "no '{}*' keys found; expected a Qwen3-VL-4B checkpoint such as {}".format(
                LANGUAGE_MODEL_PREFIX, QWEN3_VL_WEIGHTS_FILENAME),
        )

    text_tower = {}
    for key, tensor in state_dict.items():
        if key.startswith(VISUAL_PREFIX):
            continue
        if key.startswith(LANGUAGE_MODEL_PREFIX):
            key = TEXT_TOWER_PREFIX + key[len(LANGUAGE_MODEL_PREFIX):]
        text_tower[key] = tensor
    return text_tower


class Krea2PromptTemplate:
    """Satisfies `modules.text_encoder.PromptTemplate` structurally. Wraps the
    prompt in `KREA2_CHAT_TEMPLATE` after stripping literal role markers. An
    empty prompt is formatted as-is (as in the reference); after prefix
    stripping the conditioning is then just the template's trailing tokens --
    a valid, deterministic, non-empty tensor."""

    def apply(self, text: str) -> str:
        return KREA2_CHAT_TEMPLATE.format(strip_chat_role_markers(text))


def load_krea2_text_encoder(tokenizer_path=None, weights_path=None, config_dict=None, hf_tokenizer=None,
                            tap_layers=KREA2_TAP_LAYERS):
    """Builds the Qwen3-VL-4B `TextEncoder` for Krea 2.

    Args:
        tokenizer_path: Directory holding the HF tokenizer assets. Defaults to
            `modules.config.path_text_encoders`. Qwen3-VL-4B uses the same
            Qwen2 BPE vocabulary as Qwen3-4B, so the Z-Image assets work.
        weights_path: Absolute path to the Qwen3-VL-4B safetensors weights.
            Defaults to `QWEN3_VL_WEIGHTS_FILENAME` inside
            `modules.config.path_text_encoders`.
        config_dict: Transformer config override (tests use a tiny config).
        hf_tokenizer: Pre-built HF tokenizer, used instead of loading assets.
        tap_layers: Hidden-state indices to condition on; tests pass a subset
            that fits a tiny config.

    Returns:
        A `modules.text_encoder.TextEncoder` whose conditioning is
        `(B, seq, 12 * 2560)` with `pooled` always `None`.

    Raises:
        text_encoder.TextEncoderNotFoundError: the weights file is missing.
        text_encoder.TextEncoderStateDictMismatchError: the file is not a
            Qwen3-VL checkpoint, or its text tower doesn't match
            `Qwen3VLTextModel`'s expected keys/shapes.
        ValueError: a tap layer is out of range for the configured depth.
    """
    resolved_tokenizer_path = tokenizer_path or modules.config.path_text_encoders
    tokenizer = Qwen3Tokenizer(tokenizer_path=None if hf_tokenizer is not None else resolved_tokenizer_path,
                               hf_tokenizer=hf_tokenizer, max_length=MAX_SEQUENCE_LENGTH,
                               template_suffix=KREA2_TEMPLATE_SUFFIX, disable_weights=True)

    resolved_weights_path = weights_path or os.path.join(modules.config.path_text_encoders, QWEN3_VL_WEIGHTS_FILENAME)

    module_kwargs = {
        "config_dict": config_dict or QWEN3_VL_4B_TEXT_CONFIG,
        "tap_layers": tap_layers,
        "pad_token_id": tokenizer.pad_token_id,
    }

    return TransformerTextEncoder(
        module_class=Qwen3VLTextModel,
        module_kwargs=module_kwargs,
        filename=resolved_weights_path,
        tokenizer=tokenizer,
        prompt_template=Krea2PromptTemplate(),
        state_dict_adapter=select_text_tower_state_dict,
    )
