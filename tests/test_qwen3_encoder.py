"""Tests for FWDF-125's Qwen3-4B text encoder.

Covers:
- ldm_patched.modules.qwen3_model: the hand-rolled Qwen3 transformer forward
  pass (GQA head expansion, RoPE, the hidden_states[-2] conditioning tap).
- ldm_patched.modules.qwen3_clip: Qwen3TextModel's HF-layout state-dict key
  mapping and encode_token_weights()/pooled=None contract, and
  Qwen3Tokenizer's weight-syntax parsing, truncation, and pad-token
  resolution.
- modules.qwen3_text_encoder: the chat prompt template and the
  load_qwen3_text_encoder() factory, end-to-end through the real
  (unmodified) TransformerTextEncoder from FWDF-122.

All model tests use a tiny synthetic config, not the real Qwen3-4B config
(hidden_size=2560, 36 layers) -- no real checkpoint exists in this
environment. head_dim is deliberately NOT derived from
hidden_size // num_attention_heads (24 vs 16) to regression-test that the
model honors head_dim as an independent config value, matching real Qwen3
(hidden_size=2560, num_attention_heads=32 -> 80, but head_dim=128).
"""
import os
import sys
from pathlib import Path

import pytest
import safetensors.torch
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# args_manager calls parse_args() at import time, which chokes on pytest's
# argv. Patch sys.argv before any project modules are imported (mirrors
# tests/test_text_encoder.py).
_original_argv = sys.argv
sys.argv = [sys.argv[0]]

import ldm_patched.modules.ops as ldm_patched_ops  # noqa: E402
import modules.text_encoder as text_encoder  # noqa: E402
from modules.qwen3_text_encoder import Qwen3ChatPromptTemplate, CHAT_TEMPLATE  # noqa: E402
from ldm_patched.modules.qwen3_clip import Qwen3TextModel, Qwen3Tokenizer  # noqa: E402
from ldm_patched.modules.qwen3_model import Qwen3Transformer_, repeat_kv  # noqa: E402

sys.argv = _original_argv

# Vocab size used by the tiny synthetic config in these tests. _char_id()
# mods every character into this range so _FakeHFTokenizer never emits an
# id out of bounds for Qwen3TextModel's (tiny) embedding table.
_TINY_VOCAB_SIZE = 64


def _char_id(c):
    return ord(c) % _TINY_VOCAB_SIZE


def make_tiny_config():
    return dict(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=24,
        intermediate_size=48,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        hidden_act="silu",
    )


def init_small_weights(model):
    """`ldm_patched`'s ops.Linear/RMSNorm intentionally override
    reset_parameters to a no-op (real usage always loads a trained
    checkpoint immediately after construction), so a freshly constructed
    module holds uninitialized memory until a state dict is loaded. Tests
    that run a forward pass without loading a checkpoint must initialize
    weights themselves."""
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(mean=0.0, std=0.02)


class _FakeHFTokenizer:
    """Deterministic char-level tokenizer double satisfying the surface
    Qwen3Tokenizer needs (__call__(text, add_special_tokens=False)
    ["input_ids"], .pad_token_id, .eos_token_id) without loading any real
    HF tokenizer assets from disk."""

    EOS_TOKEN_ID = 1

    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
        self.eos_token_id = self.EOS_TOKEN_ID

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [_char_id(c) for c in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


# ---------------------------------------------------------------------------
# ldm_patched.modules.qwen3_model: hand-rolled transformer forward pass
# ---------------------------------------------------------------------------


class TestQwen3TransformerForward:
    def _make_model(self):
        config = make_tiny_config()
        model = Qwen3Transformer_(config, dtype=torch.float32, device="cpu",
                                   operations=ldm_patched_ops.manual_cast)
        init_small_weights(model)
        model.eval()
        return model, config

    def test_forward_returns_one_hidden_state_per_layer_plus_embeddings(self):
        model, config = self._make_model()
        input_ids = torch.randint(0, config["vocab_size"], (2, 5))

        with torch.no_grad():
            all_hidden_states = model(input_ids)

        assert len(all_hidden_states) == config["num_hidden_layers"] + 1
        for hidden_state in all_hidden_states:
            assert hidden_state.shape == (2, 5, config["hidden_size"])

    def test_head_dim_is_honored_independently_of_hidden_size_over_heads(self):
        """Real Qwen3-4B: hidden_size=2560, num_attention_heads=32 (ratio
        80) but head_dim=128 -- head_dim must come from config, not be
        derived. The tiny config here uses head_dim=24 vs. ratio 16 to
        catch a regression that silently ignores config head_dim."""
        model, config = self._make_model()
        layer0_attn = model.layers[0].self_attn

        assert layer0_attn.q_proj.weight.shape == (
            config["num_attention_heads"] * config["head_dim"], config["hidden_size"],
        )
        assert layer0_attn.k_proj.weight.shape == (
            config["num_key_value_heads"] * config["head_dim"], config["hidden_size"],
        )
        assert layer0_attn.o_proj.weight.shape == (
            config["hidden_size"], config["num_attention_heads"] * config["head_dim"],
        )

    def test_forward_is_deterministic(self):
        model, config = self._make_model()
        input_ids = torch.randint(0, config["vocab_size"], (1, 4))

        with torch.no_grad():
            first = model(input_ids)
            second = model(input_ids)

        for a, b in zip(first, second):
            assert torch.equal(a, b)

    def test_hidden_states_minus_two_differs_from_final_normed_output(self):
        """The single highest-risk detail in FWDF-125: conditioning taps
        hidden_states[-2] (second-to-last decoder layer, un-normed), not
        hidden_states[-1] (final layer + final norm). If a refactor
        collapses these to the same tensor, this must fail."""
        model, config = self._make_model()
        input_ids = torch.randint(0, config["vocab_size"], (1, 4))

        with torch.no_grad():
            all_hidden_states = model(input_ids)

        assert not torch.equal(all_hidden_states[-2], all_hidden_states[-1])

    def test_forward_handles_single_token_prompt(self):
        model, config = self._make_model()
        input_ids = torch.randint(0, config["vocab_size"], (1, 1))

        with torch.no_grad():
            all_hidden_states = model(input_ids)

        assert all_hidden_states[-1].shape == (1, 1, config["hidden_size"])


class TestRepeatKvGqaExpansion:
    def test_expands_each_kv_head_contiguously_n_rep_times(self):
        # batch=1, kv_heads=2, seq=3, head_dim=4
        x = torch.stack([
            torch.full((3, 4), fill_value=0.0),
            torch.full((3, 4), fill_value=1.0),
        ]).unsqueeze(0)

        out = repeat_kv(x, n_rep=2)

        assert out.shape == (1, 4, 3, 4)
        assert torch.equal(out[0, 0], x[0, 0])
        assert torch.equal(out[0, 1], x[0, 0])
        assert torch.equal(out[0, 2], x[0, 1])
        assert torch.equal(out[0, 3], x[0, 1])

    def test_n_rep_one_returns_input_unchanged(self):
        x = torch.randn(1, 2, 3, 4)
        assert repeat_kv(x, n_rep=1) is x


# ---------------------------------------------------------------------------
# ldm_patched.modules.qwen3_clip: Qwen3TextModel
# ---------------------------------------------------------------------------


class TestQwen3TextModelStateDictKeys:
    def test_state_dict_matches_expected_hf_causal_lm_layout(self):
        """Strict key mapping from the HF checkpoint layout
        (model.layers.N.self_attn.q_proj.weight etc.) is the load-bearing
        assumption for loading a real qwen_3_4b.safetensors file. This pins
        the exact key set for the tiny 2-layer config so a future rename
        anywhere in the module tree fails loudly instead of only surfacing
        as a strict-load error against real weights."""
        model = Qwen3TextModel(config_dict=make_tiny_config(), dtype=torch.float32, device="cpu")

        expected_keys = {"model.embed_tokens.weight", "model.norm.weight"}
        for i in range(2):
            prefix = "model.layers.{}.".format(i)
            expected_keys.update({
                prefix + "input_layernorm.weight",
                prefix + "post_attention_layernorm.weight",
                prefix + "self_attn.q_proj.weight",
                prefix + "self_attn.k_proj.weight",
                prefix + "self_attn.v_proj.weight",
                prefix + "self_attn.o_proj.weight",
                prefix + "self_attn.q_norm.weight",
                prefix + "self_attn.k_norm.weight",
                prefix + "mlp.gate_proj.weight",
                prefix + "mlp.up_proj.weight",
                prefix + "mlp.down_proj.weight",
            })

        assert set(model.state_dict().keys()) == expected_keys

    def test_synthetic_hf_layout_dict_loads_strict_without_missing_or_unexpected_keys(self):
        config = make_tiny_config()
        source_model = Qwen3TextModel(config_dict=config, dtype=torch.float32, device="cpu")
        init_small_weights(source_model)
        synthetic_state_dict = source_model.state_dict()

        target_model = Qwen3TextModel(config_dict=config, dtype=torch.float32, device="cpu")
        result = target_model.load_state_dict(synthetic_state_dict, strict=True)

        assert result.missing_keys == []
        assert result.unexpected_keys == []


class TestQwen3TextModelEncodeTokenWeights:
    def _make_model(self):
        model = Qwen3TextModel(config_dict=make_tiny_config(), dtype=torch.float32, device="cpu", pad_token_id=0)
        init_small_weights(model)
        return model

    def test_hidden_state_tap_index_is_pinned(self):
        """A future refactor cannot silently change which hidden state is
        used without this failing (FWDF-125 AC)."""
        assert Qwen3TextModel.HIDDEN_STATE_TAP_INDEX == -2

    def test_encode_token_weights_returns_none_pooled(self):
        model = self._make_model()
        tokens = [[(1, 1.0), (2, 1.0), (3, 1.0)]]

        cond, pooled = model.encode_token_weights(tokens)

        assert isinstance(cond, torch.Tensor)
        assert cond.shape == (1, 3, 32)
        assert pooled is None

    def test_encode_token_weights_is_deterministic(self):
        model = self._make_model()
        tokens = [[(1, 1.0), (2, 1.0), (3, 1.0)]]

        first, _ = model.encode_token_weights(tokens)
        second, _ = model.encode_token_weights(tokens)

        assert torch.equal(first, second)

    def test_satisfies_text_encoder_module_class_contract(self):
        model = self._make_model()
        assert hasattr(model, "encode_token_weights")
        cond, pooled = model.encode_token_weights([[(1, 1.0)]])
        assert pooled is None


# ---------------------------------------------------------------------------
# ldm_patched.modules.qwen3_clip: Qwen3Tokenizer
# ---------------------------------------------------------------------------


class TestQwen3Tokenizer:
    def test_requires_tokenizer_path_or_hf_tokenizer(self):
        with pytest.raises(ValueError):
            Qwen3Tokenizer()

    def test_tokenize_with_weights_roundtrips_plain_text(self):
        tokenizer = Qwen3Tokenizer(hf_tokenizer=_FakeHFTokenizer())

        sections = tokenizer.tokenize_with_weights("abc")

        assert len(sections) == 1
        assert sections[0] == [(_char_id("a"), 1.0), (_char_id("b"), 1.0), (_char_id("c"), 1.0)]

    def test_tokenize_with_weights_applies_emphasis_syntax(self):
        tokenizer = Qwen3Tokenizer(hf_tokenizer=_FakeHFTokenizer())

        sections = tokenizer.tokenize_with_weights("(a:1.5)")

        assert sections == [[(_char_id("a"), 1.5)]]

    def test_tokenize_with_weights_handles_empty_prompt(self):
        tokenizer = Qwen3Tokenizer(hf_tokenizer=_FakeHFTokenizer())

        sections = tokenizer.tokenize_with_weights("")

        assert sections == [[]]

    def test_tokenize_with_weights_truncates_at_max_length(self):
        tokenizer = Qwen3Tokenizer(hf_tokenizer=_FakeHFTokenizer(), max_length=5)

        sections = tokenizer.tokenize_with_weights("abcdefghij")

        assert len(sections[0]) == 5

    def test_tokenize_with_weights_return_word_ids(self):
        tokenizer = Qwen3Tokenizer(hf_tokenizer=_FakeHFTokenizer())

        sections = tokenizer.tokenize_with_weights("ab", return_word_ids=True)

        assert sections == [[(_char_id("a"), 1.0, 1), (_char_id("b"), 1.0, 2)]]

    def test_pad_token_id_uses_tokenizer_value_when_present(self):
        tokenizer = Qwen3Tokenizer(hf_tokenizer=_FakeHFTokenizer(pad_token_id=42))
        assert tokenizer.pad_token_id == 42
        assert tokenizer.special_tokens == {"pad": 42}

    def test_pad_token_id_falls_back_to_eos_when_tokenizer_has_none(self):
        tokenizer = Qwen3Tokenizer(hf_tokenizer=_FakeHFTokenizer(pad_token_id=None))
        assert tokenizer.pad_token_id == _FakeHFTokenizer.EOS_TOKEN_ID


# ---------------------------------------------------------------------------
# modules.qwen3_text_encoder: chat prompt template
# ---------------------------------------------------------------------------


class TestQwen3ChatPromptTemplate:
    def test_satisfies_prompt_template_protocol(self):
        template = Qwen3ChatPromptTemplate()
        assert isinstance(template, text_encoder.PromptTemplate)

    def test_apply_wraps_prompt_in_chat_markup(self):
        template = Qwen3ChatPromptTemplate()
        assert template.apply("a cat") == CHAT_TEMPLATE.format("a cat")
        assert template.apply("a cat") == "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n"

    def test_apply_handles_empty_prompt_without_error(self):
        template = Qwen3ChatPromptTemplate()
        result = template.apply("")
        assert result == "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"


# ---------------------------------------------------------------------------
# End-to-end through the real, unmodified TransformerTextEncoder (FWDF-122)
# ---------------------------------------------------------------------------


@pytest.fixture
def make_encoder(tmp_path):
    """Factory fixture: builds a TransformerTextEncoder wired around a tiny
    Qwen3TextModel, with weights saved to `tmp_path` (pytest-managed,
    cleaned up automatically)."""

    def _make(max_length=512):
        config = make_tiny_config()
        module = Qwen3TextModel(config_dict=config, dtype=torch.float32, device="cpu", pad_token_id=0)
        init_small_weights(module)

        weights_path = os.path.join(tmp_path, "qwen_3_4b_{}.safetensors".format(max_length))
        safetensors.torch.save_file(module.state_dict(), weights_path)

        return text_encoder.TransformerTextEncoder(
            module_class=Qwen3TextModel,
            module_kwargs={"config_dict": config, "pad_token_id": 0},
            filename=weights_path,
            tokenizer=Qwen3Tokenizer(hf_tokenizer=_FakeHFTokenizer(), max_length=max_length),
            prompt_template=Qwen3ChatPromptTemplate(),
        )

    return _make


class TestQwen3EndToEndThroughTransformerTextEncoder:
    def test_encoder_satisfies_text_encoder_protocol(self, make_encoder):
        encoder = make_encoder()
        assert isinstance(encoder, text_encoder.TextEncoder)

    def test_encode_from_tokens_returns_none_pooled_for_normal_prompt(self, make_encoder):
        encoder = make_encoder()
        tokens = encoder.tokenize("a photo of a cat")
        cond, pooled = encoder.encode_from_tokens(tokens, return_pooled=True)

        assert isinstance(cond, torch.Tensor)
        assert pooled is None

    def test_empty_prompt_handled_without_error(self, make_encoder):
        encoder = make_encoder()
        tokens = encoder.tokenize("")
        cond, pooled = encoder.encode_from_tokens(tokens, return_pooled=True)

        assert isinstance(cond, torch.Tensor)
        assert cond.shape[1] > 0  # chat template markup still produces tokens
        assert pooled is None

    def test_very_long_prompt_is_truncated_without_error(self, make_encoder):
        encoder = make_encoder(max_length=16)
        tokens = encoder.tokenize("x" * 1000)
        cond, pooled = encoder.encode_from_tokens(tokens, return_pooled=True)

        assert isinstance(cond, torch.Tensor)
        assert cond.shape[1] <= 16
        assert pooled is None

    def test_prompt_template_applied_before_tokenization(self, make_encoder):
        encoder = make_encoder()
        tokens = encoder.tokenize("a")
        # The chat-templated string is much longer than the bare "a"
        # prompt; if the template weren't applied, this would be length 1.
        assert len(tokens[0]) > 1
