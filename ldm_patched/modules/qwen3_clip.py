"""Qwen3-4B text encoder wrapper (FWDF-125): the `module_class` that
`modules.text_encoder.TransformerTextEncoder` (FWDF-122) loads and drives.

Pairs with `ldm_patched/modules/qwen3_model.py`'s hand-rolled transformer.
Mirrors `sd1_clip.py`'s single-file model+tokenizer convention (SDClipModel +
SDTokenizer live together there; Qwen3TextModel + Qwen3Tokenizer live
together here).
"""

import torch

import ldm_patched.modules.ops
from ldm_patched.modules.qwen3_model import Qwen3Transformer_
from ldm_patched.modules.sd1_clip import (
    ClipTokenWeightEncoder,
    escape_important,
    token_weights,
    unescape_important,
)

# Verified against Tongyi-MAI's published Qwen3-4B `text_encoder/config.json`
# (see qwen3_model.py's module docstring for the full verification note).
QWEN3_4B_CONFIG = dict(
    vocab_size=151936,
    hidden_size=2560,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=9728,
    rms_norm_eps=1e-06,
    rope_theta=1000000.0,
    hidden_act="silu",
)


class Qwen3TextModel(torch.nn.Module, ClipTokenWeightEncoder):
    """Wraps `Qwen3Transformer_`, taps a hidden state for conditioning, and
    implements the `encode_token_weights(tokens) -> (cond, pooled)` contract
    `modules.text_encoder.TransformerTextEncoder` expects from `module_class`
    (the same contract `SDClipModel` satisfies for CLIP).

    The top-level attribute holding the transformer is named `model` (not,
    say, `transformer`) so that `state_dict()` keys come out as
    `model.embed_tokens.weight`, `model.layers.{i}...`, `model.norm.weight`
    -- matching the HF `Qwen3ForCausalLM` checkpoint layout with the LM head
    stripped, which is what `qwen_3_4b.safetensors` is expected to contain.

    `pooled` is always `None`: causal-LM hidden states have no pooled
    projection head, matching `TextEncoder`'s documented contract
    (modules/text_encoder.py).
    """

    # Second-to-last decoder layer, NOT the final layer and NOT an LM-head
    # projection. Source: community-documented by fblissjr/
    # ComfyUI-QwenImageWanBridge as the tap used by the Z-Image reference
    # pipeline. NOT independently confirmed against Tongyi-MAI's own
    # inference code -- this is the single highest-risk unverified detail in
    # FWDF-125. A wrong index here produces plausible-but-wrong generations
    # with no error signal, which is why it's a named constant asserted by
    # tests/test_qwen3_encoder.py rather than an inline literal.
    HIDDEN_STATE_TAP_INDEX = -2

    def __init__(self, config_dict=None, dtype=None, device=None,
                 operations=ldm_patched.modules.ops.manual_cast, pad_token_id=0):
        super().__init__()
        if config_dict is None:
            config_dict = QWEN3_4B_CONFIG

        self.model = Qwen3Transformer_(config_dict, dtype, device, operations)
        self.special_tokens = {"pad": pad_token_id}
        self.freeze()

    def freeze(self):
        # Qwen3Transformer_ has no dropout layers, so this has no numerical
        # effect on inference determinism -- kept for the same reason
        # SDClipModel.freeze() does it: an explicit, inspectable "this is a
        # frozen inference-only encoder" state rather than an implicit one.
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, tokens):
        device = self.model.embed_tokens.weight.device
        input_ids = torch.LongTensor(tokens).to(device)

        all_hidden_states = self.model(input_ids)
        cond = all_hidden_states[self.HIDDEN_STATE_TAP_INDEX]
        return cond.float(), None

    def encode(self, tokens):
        return self(tokens)


class Qwen3Tokenizer:
    """`Tokenizer`-protocol wrapper around an HF tokenizer (AutoTokenizer /
    Qwen2Tokenizer -- Qwen3 reuses Qwen2's BPE tokenizer). Only tokenization
    is needed here, not model weights, so depending on `transformers` for
    this part is fine even with the repo's `transformers==4.42.4` pin --
    Qwen2 tokenizer support predates that pin (added 4.37.0), matching how
    `sd1_clip.py` already imports `transformers.CLIPTokenizer` for the same
    reason.

    Reuses the CLIP tokenizer's `(word:weight)` emphasis-syntax parsing
    utilities from `sd1_clip.py` (`token_weights`/`escape_important`/
    `unescape_important`) since that syntax is a Fooocus/A1111-wide prompt
    convention, not CLIP-specific.

    `hf_tokenizer` is injectable (DIP) so callers/tests can supply a
    pre-built tokenizer instead of loading real Qwen3 tokenizer assets from
    disk via `tokenizer_path`.
    """

    def __init__(self, tokenizer_path=None, max_length=512, hf_tokenizer=None, tokenizer_class=None):
        if hf_tokenizer is not None:
            self.tokenizer = hf_tokenizer
        else:
            if tokenizer_path is None:
                raise ValueError("Qwen3Tokenizer requires either tokenizer_path or hf_tokenizer.")
            if tokenizer_class is None:
                from transformers import AutoTokenizer
                tokenizer_class = AutoTokenizer
            self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path)

        self.max_length = max_length

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = pad_token_id
        self.special_tokens = {"pad": self.pad_token_id}

    def tokenize_with_weights(self, text, return_word_ids=False):
        """Returns a single-section `[[(token_id, weight[, word_id]), ...]]`
        list -- matching the shape `ClipTokenWeightEncoder.encode_token_weights`
        expects. Unlike CLIP's `SDTokenizer`, this never splits one prompt
        into multiple 77-token sections: `max_length` (default 512, per the
        Z-Image reference pipeline) is applied as a single truncation instead.
        """
        text = escape_important(text)
        parsed_weights = token_weights(text, 1.0)

        tokens = []
        for weighted_segment, weight in parsed_weights:
            segment = unescape_important(weighted_segment)
            if segment == "":
                continue
            ids = self.tokenizer(segment, add_special_tokens=False)["input_ids"]
            tokens.extend((token_id, weight) for token_id in ids)

        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        if return_word_ids:
            batch = [(token_id, weight, i + 1) for i, (token_id, weight) in enumerate(tokens)]
        else:
            batch = list(tokens)

        return [batch]

    def untokenize(self, token_weight_pair):
        return list(map(lambda a: (a, self.tokenizer.decode([a[0]])), token_weight_pair))
