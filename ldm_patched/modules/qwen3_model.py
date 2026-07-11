"""Hand-rolled Qwen3 causal-LM forward pass (hidden states only), used as
Z-Image's text encoder (FWDF-125).

The repo pins ``transformers==4.42.4`` (``requirements_versions.txt``), which
predates Qwen3 support (added around 4.51.0, matching the checkpoint's own
recorded ``transformers_version``). Depending on ``transformers.Qwen3Model``
directly would force a dependency bump that risks every other
``transformers``-based path in the repo (CLIP tokenizers, BLIP interrogation,
the safety checker), and would bypass the lowvram/dtype-casting machinery
(the ``operations`` convention) the rest of ``ldm_patched`` relies on for
VRAM-constrained users. This mirrors ``ldm_patched/modules/clip_model.py``,
which hand-builds ``CLIPAttention``/``CLIPMLP`` from ``operations.Linear``
instead of wrapping ``transformers.CLIPTextModel``.

Only the transformer forward returning hidden states is implemented -- no
causal-LM head, no sampling, no KV-cache (a single forward pass over the full
prompt is sufficient for conditioning).

Verified against Tongyi-MAI's published Qwen3-4B ``text_encoder/config.json``:
hidden_size=2560, num_hidden_layers=36, num_attention_heads=32,
num_key_value_heads=8 (4:1 grouped-query attention), head_dim=128,
intermediate_size=9728, vocab_size=151936, rms_norm_eps=1e-06,
rope_theta=1000000, hidden_act=silu. Note head_dim (128) is configured
independently of hidden_size / num_attention_heads (2560 / 32 = 80) -- Qwen3
does not derive head_dim from that ratio the way older architectures do, so
q_proj/k_proj/v_proj/o_proj shapes below use head_dim directly rather than
hidden_size // num_attention_heads.

RoPE here uses the standard HF Llama/Qwen "rotate_half" convention (cos/sin
tables, ``rotate_half``), which is NOT the same convention as
``ldm_patched/ldm/lumina/model.py``'s Flux-style rotation-matrix RoPE
(``rope_freqs``/``apply_rope1``) -- the two families compute RoPE
differently and must not be cross-applied.

State-dict key names are expected to match the standard HF Llama-family/
Qwen3 ``*ForCausalLM`` layout with the LM head stripped
(``model.embed_tokens.weight``, ``model.layers.{i}.self_attn.{q,k,v,o}_proj.
weight``, ``model.layers.{i}.self_attn.{q,k}_norm.weight``,
``model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight``,
``model.layers.{i}.mlp.{gate,up,down}_proj.weight``, ``model.norm.weight``).
This has NOT been verified against a real downloaded checkpoint in this
environment -- flag any strict-load key mismatch against the real
``qwen_3_4b.safetensors`` file back to this ticket (FWDF-125).
"""

import torch

from ldm_patched.ldm.modules.attention import optimized_attention_for_device


def rotate_half(x):
    """Splits the last dim in half and rotates: (-x2, x1). HF Llama/Qwen
    convention -- distinct from Lumina's rotation-matrix RoPE in this repo."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """q, k: (batch, heads, seq, head_dim). cos, sin: (seq, head_dim)."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def compute_rope_cos_sin(seq_len, head_dim, theta, dtype, device):
    """HF Qwen3RotaryEmbedding equivalent: computed in float32 for
    precision, then cast to the working dtype."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    position_ids = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(position_ids, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def repeat_kv(x, n_rep):
    """Expands grouped KV heads to match the query head count for GQA.
    x: (batch, num_kv_heads, seq, head_dim) -> (batch, num_kv_heads * n_rep, seq, head_dim)."""
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class Qwen3Attention(torch.nn.Module):
    """GQA self-attention with per-head RMSNorm (qk-norm) on Q/K before RoPE,
    matching Qwen3's attention block. No qkv/out bias (Qwen3 dropped the
    Qwen2-style QKV bias in favor of qk-norm)."""

    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads, head_dim,
                 rms_norm_eps, dtype, device, operations):
        super().__init__()
        assert num_attention_heads % num_key_value_heads == 0, (
            "num_attention_heads ({}) must be divisible by num_key_value_heads ({})".format(
                num_attention_heads, num_key_value_heads
            )
        )
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.head_dim = head_dim

        self.q_proj = operations.Linear(hidden_size, num_attention_heads * head_dim, bias=False, dtype=dtype, device=device)
        self.k_proj = operations.Linear(hidden_size, num_key_value_heads * head_dim, bias=False, dtype=dtype, device=device)
        self.v_proj = operations.Linear(hidden_size, num_key_value_heads * head_dim, bias=False, dtype=dtype, device=device)
        self.o_proj = operations.Linear(num_attention_heads * head_dim, hidden_size, bias=False, dtype=dtype, device=device)

        self.q_norm = operations.RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True, dtype=dtype, device=device)

    def forward(self, x, cos, sin, mask, optimized_attention):
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_key_value_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        q = q.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)
        k = k.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)
        v = v.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)

        out = optimized_attention(q, k, v, self.num_heads, mask)
        return self.o_proj(out)


class Qwen3MLP(torch.nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x)), no bias on any projection."""

    def __init__(self, hidden_size, intermediate_size, dtype, device, operations):
        super().__init__()
        self.gate_proj = operations.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.up_proj = operations.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.down_proj = operations.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        hidden_size = config_dict["hidden_size"]
        rms_norm_eps = config_dict["rms_norm_eps"]

        self.input_layernorm = operations.RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True, dtype=dtype, device=device)
        self.self_attn = Qwen3Attention(
            hidden_size, config_dict["num_attention_heads"], config_dict["num_key_value_heads"],
            config_dict["head_dim"], rms_norm_eps, dtype, device, operations,
        )
        self.post_attention_layernorm = operations.RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True, dtype=dtype, device=device)
        self.mlp = Qwen3MLP(hidden_size, config_dict["intermediate_size"], dtype, device, operations)

    def forward(self, x, cos, sin, mask, optimized_attention):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin, mask, optimized_attention)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class Qwen3Transformer_(torch.nn.Module):
    """Embeddings + decoder stack + final norm. No LM head -- only hidden
    states are needed for conditioning."""

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        assert config_dict["hidden_act"] == "silu", (
            "Qwen3Transformer_ only implements the SwiGLU/silu MLP Qwen3 ships with, got '{}'".format(
                config_dict["hidden_act"]
            )
        )
        self.hidden_size = config_dict["hidden_size"]
        self.head_dim = config_dict["head_dim"]
        self.rope_theta = config_dict["rope_theta"]

        # Plain nn.Embedding, not wrapped in the `operations` cast convention:
        # ldm_patched.modules.clip_model.CLIPEmbeddings does the same (its
        # token_embedding is a raw torch.nn.Embedding), since none of the
        # `operations` classes (Linear/Conv*/GroupNorm/LayerNorm/RMSNorm)
        # cover embedding lookups.
        self.embed_tokens = torch.nn.Embedding(config_dict["vocab_size"], self.hidden_size, dtype=dtype, device=device)
        self.layers = torch.nn.ModuleList([
            Qwen3DecoderLayer(config_dict, dtype, device, operations)
            for _ in range(config_dict["num_hidden_layers"])
        ])
        self.norm = operations.RMSNorm(self.hidden_size, eps=config_dict["rms_norm_eps"], elementwise_affine=True, dtype=dtype, device=device)

    def forward(self, input_ids, attention_mask=None):
        """Returns a list of hidden states: [embeddings, after-layer-0, ...,
        after-layer-(N-2), final-normed-output]. Length == num_hidden_layers
        + 1, matching HF's `output_hidden_states=True` semantics -- index -1
        is the final normed output, index -2 is the un-normed output of the
        second-to-last decoder layer.
        """
        x = self.embed_tokens(input_ids)
        seq_len = x.shape[1]

        cos, sin = compute_rope_cos_sin(seq_len, self.head_dim, self.rope_theta, x.dtype, x.device)

        mask = torch.empty(seq_len, seq_len, dtype=x.dtype, device=x.device).fill_(float("-inf")).triu_(1)
        if attention_mask is not None:
            padding_mask = 1.0 - attention_mask.to(x.dtype).unsqueeze(1).unsqueeze(1).expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )
            padding_mask = padding_mask.masked_fill(padding_mask.to(torch.bool), float("-inf"))
            mask = mask + padding_mask

        optimized_attention = optimized_attention_for_device(x.device, mask=True, small_input=True)

        all_hidden_states = [x]
        for layer in self.layers:
            x = layer(x, cos, sin, mask, optimized_attention)
            all_hidden_states.append(x)

        all_hidden_states[-1] = self.norm(all_hidden_states[-1])
        return all_hidden_states
