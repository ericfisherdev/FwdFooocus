"""Qwen3-VL-4B text encoder for Krea 2 (FWDF-133): the `module_class` that
`modules.text_encoder.TransformerTextEncoder` (FWDF-122) loads and drives.

Krea 2 conditions on twelve intermediate hidden states of Qwen3-VL-4B
(`KREA2_TAP_LAYERS`), which the DiT's in-checkpoint `txtfusion` adapter
aggregates -- not on a single final hidden state, and not on a pooled
embedding. Follows ComfyUI's `comfy/text_encoders/krea2.py`.

The text tower of `qwen3vl_4b_bf16.safetensors` is shape-identical to
Qwen3-4B (`qwen3_clip.QWEN3_4B_CONFIG`) apart from `rope_theta`, so the
existing `Qwen3Transformer_` is reused unchanged. The tower lives under the
`model.language_model.` prefix next to a `model.visual.` vision tower that is
irrelevant for text conditioning; `modules.krea2_text_encoder` picks the text
tower out of such a checkpoint before it reaches this module.
"""

from collections.abc import Sequence
from typing import Final

import torch

import ldm_patched.modules.ops
from ldm_patched.modules.qwen3_clip import QWEN3_4B_CONFIG
from ldm_patched.modules.qwen3_model import Qwen3Transformer_
from ldm_patched.modules.sd1_clip import ClipTokenWeightEncoder

# HF `Qwen/Qwen3-VL-4B-Instruct` text config. Its
# `rope_scaling={"mrope_interleaved": True, "mrope_section": [24, 20, 20]}`
# is intentionally not implemented: MRoPE only differs from plain 1-D RoPE
# when image positions are present. For text-only conditioning all three
# position axes coincide, which is exactly what ComfyUI's
# `precompute_freqs_cis` short-circuits to as well.
QWEN3_VL_4B_TEXT_CONFIG: Final = {**QWEN3_4B_CONFIG, "rope_theta": 5000000.0}

# `all_hidden_states[k]` (HF `hidden_states[k]` indexing: index 0 is the
# embeddings, index k the un-normed output of decoder layer k-1). No offset
# versus the ComfyUI reference. A named constant, asserted by tests, because
# a wrong index yields plausible-looking but wrong generations.
KREA2_TAP_LAYERS: Final[tuple[int, ...]] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

IM_START_TOKEN_ID: Final = 151644
USER_ROLE_TOKEN_ID: Final = 872
NEWLINE_TOKEN_ID: Final = 198
# `<|im_start|>`, `user`, `\n`: the tokens that open the user turn.
USER_TURN_OPENER_LENGTH: Final = 3


def krea2_conditioning_start_index(token_ids: Sequence[int]) -> int:
    """Index of the first token that belongs in Krea 2's conditioning.

    Mirrors ComfyUI's `Krea2TEModel.encode_token_weights`: the leading system
    turn and the `<|im_start|>user\\n` opener are dropped, so the start is
    the second `<|im_start|>` (+3 to also skip `<|im_start|>`, `user`, `\\n`
    when all three are present with room for content after them). The user
    prompt and the trailing `<|im_end|>\\n<|im_start|>assistant\\n` are kept.

    Returns 0 when fewer than two `<|im_start|>` markers exist (template
    absent), so nothing is dropped silently.
    """
    marker_positions = [i for i, token_id in enumerate(token_ids) if token_id == IM_START_TOKEN_ID]
    if len(marker_positions) < 2:
        return 0

    start = marker_positions[1]
    has_room = len(token_ids) > start + USER_TURN_OPENER_LENGTH
    if has_room and token_ids[start + 1] == USER_ROLE_TOKEN_ID and token_ids[start + 2] == NEWLINE_TOKEN_ID:
        start += USER_TURN_OPENER_LENGTH
    return start


class Qwen3VLTextModel(torch.nn.Module, ClipTokenWeightEncoder):
    """Qwen3-VL-4B text tower that emits Krea 2's multi-layer conditioning.

    Implements the `encode_token_weights(tokens) -> (cond, pooled)` contract
    `modules.text_encoder.TransformerTextEncoder` expects from
    `module_class`. `cond` is `(B, seq, len(tap_layers) * hidden_size)` with
    the per-token layout `[tap0 | tap1 | ... ]`, which
    `SingleStreamDiT._unpack_context` reshapes to `(B, seq, taps, hidden)`.
    `pooled` is always `None`: there is no pooled projection head.

    The attribute holding the transformer is named `model`, so its
    `state_dict()` keys are `model.embed_tokens.weight`, `model.layers.{i}...`
    and `model.norm.weight`.

    Raises:
        ValueError: a tap index lies outside `[0, num_hidden_layers]`.
    """

    def __init__(self, config_dict=None, tap_layers=KREA2_TAP_LAYERS, dtype=None, device=None,
                 operations=ldm_patched.modules.ops.manual_cast, pad_token_id=151643):
        super().__init__()
        if config_dict is None:
            config_dict = QWEN3_VL_4B_TEXT_CONFIG
        self._validate_tap_layers(tap_layers, config_dict["num_hidden_layers"])

        self.tap_layers = tuple(tap_layers)
        self.model = Qwen3Transformer_(config_dict, dtype, device, operations)
        self.special_tokens = {"pad": pad_token_id}
        self.freeze()

    @staticmethod
    def _validate_tap_layers(tap_layers, num_hidden_layers):
        if not tap_layers:
            raise ValueError("tap_layers must not be empty.")
        for tap in tap_layers:
            if not 0 <= tap <= num_hidden_layers:
                raise ValueError(
                    "tap layer {} is outside [0, {}] for a {}-layer model.".format(
                        tap, num_hidden_layers, num_hidden_layers))

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, tokens):
        device = self.model.embed_tokens.weight.device
        input_ids = torch.LongTensor(tokens).to(device)

        all_hidden_states = self.model(input_ids)
        taps = torch.stack([all_hidden_states[k] for k in self.tap_layers], dim=1)  # (B, taps, seq, hidden)
        batch, tap_count, seq_len, hidden = taps.shape
        cond = taps.permute(0, 2, 1, 3).reshape(batch, seq_len, tap_count * hidden)
        return cond.float(), None

    def encode(self, tokens):
        return self(tokens)

    def encode_token_weights(self, token_weight_pairs):
        """Encodes the full templated prompt, then drops the leading template
        tokens from the output. Slicing after the forward pass (not before) is
        required: the stripped prefix must still act as attention context,
        exactly as in the reference."""
        cond, pooled = super().encode_token_weights(token_weight_pairs)
        if token_weight_pairs:
            token_ids = [token_id for token_id, *_ in token_weight_pairs[0]]
            cond = cond[:, krea2_conditioning_start_index(token_ids):]
        return cond, pooled
