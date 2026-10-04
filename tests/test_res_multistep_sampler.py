import math
import unittest

import torch

from ldm_patched.k_diffusion.sampling import sample_res_multistep


class TestResMultistepSampler(unittest.TestCase):
    """`sample_res_multistep` integrates dx/dsigma = (x - D(x, sigma)) / sigma.

    For the linear denoiser D = a*x + c*sigma the exact solution is
    x(sigma) = A * sigma**(1 - a) - (c / a) * sigma, so a correct second-order
    step cuts the error ~4x per step halving; Euler (or swapped b1/b2, or a
    sign-flipped c2) only manages ~2x.
    """
    A, C, SIGMA_START, SIGMA_END = 0.5, 0.3, 1.0, 0.05

    def _error_at(self, steps):
        a, c, s0, s1 = self.A, self.C, self.SIGMA_START, self.SIGMA_END
        sigmas = torch.logspace(math.log10(s0), math.log10(s1), steps + 1, dtype=torch.float64)
        x = torch.ones(1, dtype=torch.float64)
        out = sample_res_multistep(lambda x, sigma: a * x + c * sigma, x, sigmas, disable=True)
        amplitude = (1 + c * s0 / a) / s0 ** (1 - a)
        exact = amplitude * s1 ** (1 - a) - (c / a) * s1
        return abs(float(out) - exact)

    def test_converges_at_second_order(self):
        self.assertGreater(self._error_at(16) / self._error_at(32), 3.5)

    def test_perfect_denoiser_lands_on_target_at_sigma_zero(self):
        target = torch.tensor([0.25, -1.5], dtype=torch.float64)
        sigmas = torch.linspace(1.0, 0.0, 10, dtype=torch.float64)
        x = target + sigmas[0] * torch.randn(2, dtype=torch.float64)
        out = sample_res_multistep(lambda x, sigma: target, x, sigmas, disable=True)
        self.assertTrue(torch.allclose(out, target, atol=1e-9))


if __name__ == '__main__':
    unittest.main()
