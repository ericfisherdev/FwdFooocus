"""Tests for ldm_patched.modules.utils.dequantize_comfy_scaled_fp8 -- folding
ComfyUI per-tensor scaled-fp8 weights (float8_e4m3fn `.weight` + scalar
`.weight_scale` + `.comfy_quant` marker) back into ordinary weights at load
time, so such checkpoints load into this fork's unquantized models correctly.
"""

import json
import unittest

import torch

from ldm_patched.modules.utils import dequantize_comfy_scaled_fp8


def _marker(**metadata):
    """Build a `.comfy_quant` marker the way ComfyUI stores it: JSON bytes in a uint8 tensor."""
    return torch.tensor(list(json.dumps(metadata).encode()), dtype=torch.uint8)


class TestDequantizeComfyScaledFp8(unittest.TestCase):
    def test_dequantizes_fp8_weight_and_drops_scale_and_marker(self):
        base = torch.randn(8, 8)
        fp8_weight = base.to(torch.float8_e4m3fn)
        scale = torch.tensor(0.5)
        state_dict = {
            "layers.1.attn.weight": fp8_weight,
            "layers.1.attn.weight_scale": scale,
            "layers.1.attn.comfy_quant": _marker(format="float8_e4m3fn"),
            # a genuine bf16 layer fixes the target compute dtype
            "layers.0.attn.weight": base.to(torch.bfloat16),
        }

        out = dequantize_comfy_scaled_fp8(state_dict)

        self.assertNotIn("layers.1.attn.weight_scale", out)
        self.assertNotIn("layers.1.attn.comfy_quant", out)
        self.assertEqual(out["layers.1.attn.weight"].dtype, torch.bfloat16)
        expected = fp8_weight.to(torch.bfloat16) * scale.to(torch.bfloat16)
        torch.testing.assert_close(out["layers.1.attn.weight"], expected)

    def test_leaves_non_fp8_weights_untouched(self):
        weight = torch.randn(4, 4, dtype=torch.bfloat16)
        state_dict = {"layers.0.attn.weight": weight.clone()}
        out = dequantize_comfy_scaled_fp8(state_dict)
        torch.testing.assert_close(out["layers.0.attn.weight"], weight)

    def test_no_markers_is_a_noop(self):
        state_dict = {"a.weight": torch.randn(2, 2), "b.bias": torch.randn(2)}
        keys_before = set(state_dict.keys())
        out = dequantize_comfy_scaled_fp8(state_dict)
        self.assertEqual(set(out.keys()), keys_before)

    def test_all_fp8_dict_without_markers_is_returned_unchanged(self):
        # Lustify-style flat Krea 2 checkpoint: every tensor is plain float8_e4m3fn
        # and there is no weight_scale / comfy_quant, so nothing may be rescaled.
        state_dict = {
            "blocks.0.attn.wq.weight": torch.randn(4, 4).to(torch.float8_e4m3fn),
            "blocks.0.attn.qknorm.qnorm.scale": torch.randn(4).to(torch.float8_e4m3fn),
        }
        dtypes_before = {k: v.dtype for k, v in state_dict.items()}
        out = dequantize_comfy_scaled_fp8(state_dict)
        self.assertIs(out, state_dict)
        self.assertEqual({k: v.dtype for k, v in out.items()}, dtypes_before)

    def test_scale_without_marker_is_accepted_for_fp8_weight(self):
        # Older scaled-fp8 files carry only weight_scale, no comfy_quant marker.
        base = torch.randn(4, 4)
        state_dict = {
            "x.weight": base.to(torch.float8_e4m3fn),
            "x.weight_scale": torch.tensor(2.0),
            "ref.weight": base.to(torch.bfloat16),
        }
        out = dequantize_comfy_scaled_fp8(state_dict)
        self.assertNotIn("x.weight_scale", out)
        self.assertEqual(out["x.weight"].dtype, torch.bfloat16)

    def test_int8_marker_with_scalar_scale_is_rejected(self):
        state_dict = {
            "x.weight": torch.ones(4, 4, dtype=torch.int8),
            "x.weight_scale": torch.tensor(3.0),
            "x.comfy_quant": _marker(format="int8_tensorwise"),
        }
        with self.assertRaisesRegex(ValueError, r"'x'.*int8_tensorwise.*torch\.int8"):
            dequantize_comfy_scaled_fp8(state_dict)
        self.assertIn("x.weight_scale", state_dict)
        self.assertIn("x.comfy_quant", state_dict)

    def test_mxfp8_block_scale_is_rejected(self):
        state_dict = {
            "x.weight": torch.randn(4, 64).to(torch.float8_e4m3fn),
            "x.weight_scale": torch.ones(4, 2),
            "x.comfy_quant": _marker(format="mxfp8"),
        }
        with self.assertRaisesRegex(ValueError, r"'x'.*mxfp8.*scale shape=\(4, 2\)"):
            dequantize_comfy_scaled_fp8(state_dict)

    def test_non_scalar_scale_is_rejected_even_without_marker(self):
        state_dict = {
            "x.weight": torch.randn(4, 4).to(torch.float8_e4m3fn),
            "x.weight_scale": torch.ones(4, 1),
        }
        with self.assertRaisesRegex(ValueError, "scale shape=\\(4, 1\\)"):
            dequantize_comfy_scaled_fp8(state_dict)

    def test_scale_on_non_fp8_weight_without_marker_is_rejected(self):
        state_dict = {"x.weight": torch.ones(4, 4, dtype=torch.bfloat16),
                      "x.weight_scale": torch.tensor(3.0)}
        with self.assertRaisesRegex(ValueError, "'x'"):
            dequantize_comfy_scaled_fp8(state_dict)

    def test_marker_without_scale_is_rejected(self):
        state_dict = {
            "x.weight": torch.randn(4, 4).to(torch.float8_e4m3fn),
            "x.comfy_quant": _marker(format="nvfp4"),
        }
        with self.assertRaisesRegex(ValueError, "'x'.*nvfp4"):
            dequantize_comfy_scaled_fp8(state_dict)

    def test_unreadable_marker_is_rejected(self):
        state_dict = {
            "x.weight": torch.randn(4, 4).to(torch.float8_e4m3fn),
            "x.weight_scale": torch.tensor(1.0),
            "x.comfy_quant": torch.zeros(4, dtype=torch.uint8),
        }
        with self.assertRaisesRegex(ValueError, "Unreadable ComfyUI quantization marker 'x.comfy_quant'"):
            dequantize_comfy_scaled_fp8(state_dict)

    def test_one_unsupported_layer_leaves_valid_layers_unmodified(self):
        good = torch.randn(4, 4).to(torch.float8_e4m3fn)
        state_dict = {
            "a.weight": good, "a.weight_scale": torch.tensor(2.0),
            "b.weight": torch.ones(4, 4, dtype=torch.int8), "b.weight_scale": torch.tensor(1.0),
            "b.comfy_quant": _marker(format="int8_tensorwise"),
        }
        with self.assertRaises(ValueError):
            dequantize_comfy_scaled_fp8(state_dict)
        self.assertEqual(state_dict["a.weight"].dtype, torch.float8_e4m3fn)
        self.assertIn("a.weight_scale", state_dict)

    def test_plain_fp8_without_scale_is_cast_when_a_scaled_tensor_is_present(self):
        # A scaled tensor triggers the path; an unscaled plain-fp8 tensor (e.g.
        # cap_pad_token) in the same checkpoint must be cast to the compute
        # dtype so the result carries no fp8 tensors an unquantized model
        # could not receive.
        base = torch.randn(4, 4)
        state_dict = {
            "scaled.weight": base.to(torch.float8_e4m3fn),
            "scaled.weight_scale": torch.tensor(0.25),
            "pad_token": torch.randn(1, 4).to(torch.float8_e4m3fn),
            "ref.weight": base.to(torch.bfloat16),  # fixes compute dtype
        }
        plain_before = state_dict["pad_token"].clone()

        out = dequantize_comfy_scaled_fp8(state_dict)

        self.assertFalse(any(v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                             for v in out.values()))
        self.assertEqual(out["pad_token"].dtype, torch.bfloat16)
        torch.testing.assert_close(out["pad_token"], plain_before.to(torch.bfloat16))

    def test_fp8_is_left_untouched_when_no_scaled_markers_are_present(self):
        # A pure fp8-storage checkpoint (no comfy_quant/weight_scale markers) is
        # not this function's concern -- leave its fp8 tensors for the existing
        # fp8 load path rather than silently upcasting them.
        w = torch.randn(4, 4).to(torch.float8_e4m3fn)
        state_dict = {"a.weight": w}
        out = dequantize_comfy_scaled_fp8(state_dict)
        self.assertEqual(out["a.weight"].dtype, torch.float8_e4m3fn)

    def test_float8_e5m2_is_also_dequantized_to_the_checkpoint_dtype(self):
        base = torch.randn(4, 4)
        scale = torch.tensor(2.0)
        state_dict = {
            "y.weight": base.to(torch.float8_e5m2),
            "y.weight_scale": scale,
            "z.weight": base.to(torch.float16),  # fixes compute dtype to fp16
        }
        out = dequantize_comfy_scaled_fp8(state_dict)
        self.assertEqual(out["y.weight"].dtype, torch.float16)
        expected = base.to(torch.float8_e5m2).to(torch.float16) * scale.to(torch.float16)
        torch.testing.assert_close(out["y.weight"], expected)


if __name__ == "__main__":
    unittest.main()
