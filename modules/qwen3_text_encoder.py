"""Qwen3-4B text encoder wiring for Z-Image (FWDF-125).

Assembles `modules.text_encoder.TransformerTextEncoder` (FWDF-122) with the
concrete Qwen3 model/tokenizer from `ldm_patched.modules.qwen3_clip`
(FWDF-125) into a ready `TextEncoder`. No `supported_models`/`clip_target`
wiring happens here yet -- FWDF-124's `ZImage` supported-models class isn't
present on this branch (it lives on a sibling branch stacked separately);
that wiring, and FWDF-127's smoke test against a real checkpoint, are
follow-up work once the branches converge.
"""

import os

import modules.config
from ldm_patched.modules.qwen3_clip import QWEN3_4B_CONFIG, Qwen3TextModel, Qwen3Tokenizer
from modules.text_encoder import TransformerTextEncoder

# Z-Image's reference pipeline truncates the templated prompt at this many
# tokens. The DiT's caption RoPE axis supports up to 1536 tokens
# (FWDF-123's verified config) -- 512 is a pipeline-level choice inherited
# from the reference implementation, not an architectural ceiling.
MAX_SEQUENCE_LENGTH = 512

# Community-documented (fblissjr/ComfyUI-QwenImageWanBridge) as the
# `enable_thinking=True`-equivalent chat template matching the Z-Image
# reference pipeline. NOT independently confirmed against Tongyi-MAI's own
# inference code -- verify against the official Z-Image repo's pipeline
# source before this ships to production.
CHAT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

QWEN3_WEIGHTS_FILENAME = "qwen_3_4b.safetensors"


class Qwen3ChatPromptTemplate:
    """Satisfies `modules.text_encoder.PromptTemplate` structurally (same
    convention as that module's own `IdentityPromptTemplate` -- no nominal
    inheritance needed). Wraps the raw prompt in `CHAT_TEMPLATE` before
    tokenization. See this module's docstring for the template's
    verification status."""

    def apply(self, text: str) -> str:
        return CHAT_TEMPLATE.format(text)


def load_qwen3_text_encoder(tokenizer_path=None, weights_path=None):
    """Builds the Qwen3-4B `TextEncoder` for Z-Image.

    Args:
        tokenizer_path: Directory holding the HF tokenizer assets
            (tokenizer.json/tokenizer_config.json). Defaults to
            `modules.config.path_text_encoders` -- the same directory the
            safetensors weights load from, matching how Tongyi-MAI ships
            both together under one `text_encoder/` directory.
        weights_path: Absolute path to the Qwen3-4B safetensors weights.
            Defaults to `QWEN3_WEIGHTS_FILENAME` inside
            `modules.config.path_text_encoders`.

    Returns:
        A `modules.text_encoder.TextEncoder` ready for `tokenize()` /
        `encode_from_tokens()`.

    Raises:
        text_encoder.TextEncoderNotFoundError: the weights file is missing.
        text_encoder.TextEncoderStateDictMismatchError: the weights don't
            match `Qwen3TextModel`'s expected keys/shapes (see
            qwen3_model.py's docstring for the unverified state-dict-key-
            mapping risk this would surface).
    """
    resolved_tokenizer_path = tokenizer_path or modules.config.path_text_encoders
    tokenizer = Qwen3Tokenizer(tokenizer_path=resolved_tokenizer_path, max_length=MAX_SEQUENCE_LENGTH)

    resolved_weights_path = weights_path or os.path.join(modules.config.path_text_encoders, QWEN3_WEIGHTS_FILENAME)

    module_kwargs = {
        "config_dict": QWEN3_4B_CONFIG,
        "pad_token_id": tokenizer.pad_token_id,
    }

    return TransformerTextEncoder(
        module_class=Qwen3TextModel,
        module_kwargs=module_kwargs,
        filename=resolved_weights_path,
        tokenizer=tokenizer,
        prompt_template=Qwen3ChatPromptTemplate(),
    )
