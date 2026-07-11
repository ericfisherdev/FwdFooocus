"""Encoder-agnostic text-encoder infrastructure.

`ldm_patched.modules.sd.CLIP` is the only text encoder
`modules/default_pipeline.py::clip_encode()` knows about today, and its
`encode_from_tokens()` contract already returns `(cond, pooled)` where
`pooled` can be `None` (`sd1_clip`/`sd2_clip`/`sdxl_clip` return `None` for
`pooled` whenever no projection head is configured). This module makes that
contract explicit and reusable so upcoming non-CLIP encoders -- Qwen3-4B
(FWDF-125, causal LM hidden states, never pooled) and Qwen3-VL-4B (Krea 2
backlog, hidden-state tap) -- can sit behind `final_clip` without
`clip_encode()`, `clip_encode_single()`, or `clone_cond()` caring which kind
of encoder they're talking to.

No concrete non-CLIP encoder ships here. `ldm_patched.modules.sd.CLIP` is
untouched -- it already satisfies `TextEncoder` structurally (see the
regression test in tests/test_text_encoder.py).
"""
from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable

import torch

import modules.config
from ldm_patched.modules import model_management
from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules.utils import load_torch_file


class TextEncoderError(Exception):
    """Base class for domain errors raised while locating or loading a text encoder.

    Callers should catch this (or one of the specific subclasses below)
    rather than letting a bare KeyError/RuntimeError/FileNotFoundError escape
    from encoder loading.
    """


class TextEncoderNotFoundError(TextEncoderError):
    """Raised when a text encoder's weight file does not exist on disk."""

    def __init__(self, filename: str, search_dir: str):
        self.filename = filename
        self.search_dir = search_dir
        super().__init__(
            f"Text encoder file '{filename}' was not found in '{search_dir}'. "
            f"Place the encoder weights there, or point `path_text_encoders` "
            f"in config.txt at the directory that has them."
        )


class TextEncoderStateDictMismatchError(TextEncoderError):
    """Raised when a text encoder's state dict doesn't match the target module's expected keys/shapes."""

    def __init__(self, module_name: str, detail: str):
        self.module_name = module_name
        self.detail = detail
        super().__init__(f"Text encoder state dict does not match '{module_name}': {detail}")


@runtime_checkable
class Tokenizer(Protocol):
    """Structural contract for the tokenizer side of a `TextEncoder`.

    `ldm_patched.modules.sd1_clip.SDTokenizer` (and its sd2/sdxl variants)
    already satisfy this; a future HF-backed tokenizer for Qwen3/Qwen3-VL only
    needs to implement `tokenize_with_weights` the same way.
    """

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False):
        ...


@runtime_checkable
class PromptTemplate(Protocol):
    """Per-family hook applied to raw prompt text before tokenization.

    CLIP families need no template. Chat-tuned causal LM encoders (Qwen3-4B,
    Qwen3-VL-4B) format the raw prompt into their expected chat/system
    template here, before tokenization, so `clip_encode_single()` never needs
    to know which family it's talking to.
    """

    def apply(self, text: str) -> str:
        ...


class IdentityPromptTemplate:
    """No-op `PromptTemplate`: returns the prompt unchanged.

    Default for encoders (like CLIP) that don't need per-family formatting.
    """

    def apply(self, text: str) -> str:
        return text


@runtime_checkable
class TextEncoder(Protocol):
    """Structural contract for anything that can sit behind
    `modules.default_pipeline.final_clip`.

    `ldm_patched.modules.sd.CLIP` already satisfies this without
    modification. `pooled` is intentionally `Optional`: encoders without a
    pooled projection head (Qwen3-4B, Qwen3-VL-4B) return `None`, which
    `clip_encode()`, `clip_encode_single()`, and `clone_cond()` must treat as
    a valid, non-error state rather than a bug.
    """

    def tokenize(self, text: str, return_word_ids: bool = False):
        ...

    def encode_from_tokens(
        self, tokens, return_pooled: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]] | torch.Tensor:
        ...


def load_text_encoder_state_dict(filename: str) -> dict:
    """Load a text encoder's weights from disk.

    Args:
        filename: Absolute path to the encoder's weight file (`.safetensors`
            or a torch checkpoint).

    Returns:
        The raw state dict.

    Raises:
        TextEncoderNotFoundError: `filename` does not exist.
    """
    if not os.path.isfile(filename):
        raise TextEncoderNotFoundError(filename, modules.config.path_text_encoders)
    return load_torch_file(filename, safe_load=True)


class TransformerTextEncoder:
    """Generic `TextEncoder` built from any transformer module class.

    Mirrors `ldm_patched.modules.sd.CLIP`'s device-placement and offload
    wiring (`model_management.text_encoder_device()` /
    `text_encoder_offload_device()` / `text_encoder_dtype()`, and
    `ModelPatcher` for lowvram/offload), but is parameterized over the
    concrete module class and weight file instead of being hard-wired to a
    CLIP variant. A concrete family (Qwen3-4B in FWDF-125, Qwen3-VL-4B for
    Krea 2) supplies `module_class`, `module_kwargs`, `filename`, and a
    `Tokenizer`; this class only handles loading, device placement, and the
    tokenize/encode contract -- not a parallel VRAM-management system.

    `module_class` is expected to implement `encode_token_weights(tokens) ->
    (cond, pooled)` the same way `sd1_clip`/`sd2_clip`/`sdxl_clip`'s
    `cond_stage_model` classes do, with `pooled` being `None` when the family
    has no pooled projection.

    Raises:
        TextEncoderNotFoundError: `filename` does not exist.
        TextEncoderStateDictMismatchError: the state dict doesn't match
            `module_class`'s expected keys/shapes.
    """

    def __init__(
        self,
        module_class,
        module_kwargs: dict,
        filename: str,
        tokenizer: Tokenizer,
        prompt_template: Optional[PromptTemplate] = None,
    ):
        self.prompt_template = prompt_template or IdentityPromptTemplate()

        state_dict = load_text_encoder_state_dict(filename)

        load_device = model_management.text_encoder_device()
        offload_device = model_management.text_encoder_offload_device()

        kwargs = dict(module_kwargs)
        kwargs.setdefault('device', offload_device)
        kwargs.setdefault('dtype', model_management.text_encoder_dtype(load_device))

        self.cond_stage_model = module_class(**kwargs)

        try:
            self.cond_stage_model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise TextEncoderStateDictMismatchError(module_class.__name__, str(exc)) from exc

        self.tokenizer = tokenizer
        self.patcher = ModelPatcher(self.cond_stage_model, load_device=load_device, offload_device=offload_device)

    def tokenize(self, text: str, return_word_ids: bool = False):
        text = self.prompt_template.apply(text)
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

    def load_model(self):
        model_management.load_model_gpu(self.patcher)
        return self.patcher

    def encode_from_tokens(self, tokens, return_pooled: bool = False):
        self.load_model()
        cond, pooled = self.cond_stage_model.encode_token_weights(tokens)
        if return_pooled:
            return cond, pooled
        return cond
