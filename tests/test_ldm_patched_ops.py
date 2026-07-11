import unittest

import torch

import ldm_patched.modules.ops as ops


class TestRMSNormCastPath(unittest.TestCase):
    """Regression tests for the RMSNorm class added to ldm_patched's ops
    module (needed by ldm_patched/ldm/lumina/model.py), and for the
    cast_bias_weight() weight=None fix that RMSNorm with
    elementwise_affine=False required.
    """

    def test_manual_cast_rmsnorm_matches_functional_reference_affine(self):
        dim = 16
        norm = ops.manual_cast.RMSNorm(dim, eps=1e-5, elementwise_affine=True)
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, dim))
        x = torch.randn(2, 5, dim, dtype=torch.float16)

        out = norm(x)
        expected = torch.nn.functional.rms_norm(
            x, norm.normalized_shape, norm.weight.to(dtype=x.dtype), norm.eps,
        )
        self.assertEqual(out.dtype, x.dtype)
        self.assertTrue(torch.equal(out, expected))

    def test_manual_cast_rmsnorm_matches_functional_reference_non_affine(self):
        # elementwise_affine=False -> weight is None; this is exactly the
        # ops.cast_bias_weight() code path that used to raise
        # AttributeError: 'NoneType' object has no attribute 'to'.
        dim = 16
        norm = ops.manual_cast.RMSNorm(dim, eps=1e-6, elementwise_affine=False)
        self.assertIsNone(norm.weight)
        x = torch.randn(2, 5, dim, dtype=torch.bfloat16)

        out = norm(x)
        expected = torch.nn.functional.rms_norm(x, norm.normalized_shape, None, norm.eps)
        self.assertEqual(out.dtype, x.dtype)
        self.assertTrue(torch.equal(out, expected))

    def test_disable_weight_init_rmsnorm_cast_path_directly(self):
        # disable_weight_init.RMSNorm defaults ldm_patched_cast_weights=False,
        # so its forward() calls torch.nn.RMSNorm.forward() directly rather
        # than the cast path -- but forward_ldm_patched_cast_weights is the
        # exact method manual_cast.RMSNorm uses at runtime, so it must be
        # correct and independently callable too.
        dim = 8
        norm = ops.disable_weight_init.RMSNorm(dim, eps=1e-5, elementwise_affine=True)
        with torch.no_grad():
            norm.weight.copy_(torch.arange(1, dim + 1, dtype=torch.float32))
        x = torch.randn(3, dim, dtype=torch.float32)

        out = norm.forward_ldm_patched_cast_weights(x)
        expected = torch.nn.functional.rms_norm(x, norm.normalized_shape, norm.weight, norm.eps)
        self.assertTrue(torch.equal(out, expected))

    def test_disable_weight_init_rmsnorm_cast_path_non_affine_directly(self):
        dim = 8
        norm = ops.disable_weight_init.RMSNorm(dim, eps=1e-5, elementwise_affine=False)
        x = torch.randn(3, dim, dtype=torch.float32)

        out = norm.forward_ldm_patched_cast_weights(x)
        expected = torch.nn.functional.rms_norm(x, norm.normalized_shape, None, norm.eps)
        self.assertTrue(torch.equal(out, expected))


if __name__ == "__main__":
    unittest.main()
