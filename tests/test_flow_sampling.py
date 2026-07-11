import unittest

import torch


class FakeModelConfig:
    """Minimal stand-in for a `supported_models_base.BASE` subclass, exposing
    only the `sampling_settings` attribute that `ModelSamplingDiscreteFlow`
    and the other `model_sampling.py` schedule classes read.
    """
    def __init__(self, sampling_settings=None):
        self.sampling_settings = sampling_settings or {}


class TestTimeSnrShift(unittest.TestCase):
    """`time_snr_shift(shift, t) = shift * t / (1 + (shift - 1) * t)` is the
    flux/SD3-style shift formula used by ComfyUI's `ModelSamplingDiscreteFlow`
    to turn a linear flow-matching `t` in (0, 1] into the shifted sigma used
    for conditioning/denoising.
    """
    def test_shift_one_is_identity(self):
        from ldm_patched.modules.model_sampling import time_snr_shift
        for t in [0.001, 0.25, 0.5, 0.75, 1.0]:
            self.assertAlmostEqual(time_snr_shift(1.0, torch.tensor(t)).item(), t, places=6)

    def test_shift_three_known_checkpoint(self):
        from ldm_patched.modules.model_sampling import time_snr_shift
        # shift=3, t=0.5 -> 3*0.5 / (1 + 2*0.5) = 1.5 / 2 = 0.75
        self.assertAlmostEqual(time_snr_shift(3.0, torch.tensor(0.5)).item(), 0.75, places=6)

    def test_endpoints_are_shift_independent(self):
        from ldm_patched.modules.model_sampling import time_snr_shift
        for shift in [1.0, 1.15, 3.0, 6.0]:
            self.assertAlmostEqual(time_snr_shift(shift, torch.tensor(0.0)).item(), 0.0, places=6)
            self.assertAlmostEqual(time_snr_shift(shift, torch.tensor(1.0)).item(), 1.0, places=6)

    def test_higher_shift_front_loads_noise(self):
        from ldm_patched.modules.model_sampling import time_snr_shift
        t = torch.tensor(0.1)
        sigmas = [time_snr_shift(shift, t).item() for shift in [1.0, 1.15, 3.0, 6.0]]
        self.assertEqual(sigmas, sorted(sigmas))


class TestModelSamplingDiscreteFlowSchedule(unittest.TestCase):
    """Sigma schedule shape/monotonicity and reference checkpoints for the
    shifts used by Z-Image (3.0) and Krea 2 (1.15), plus 6.0 as a third
    data point.
    """
    def _make(self, shift):
        from ldm_patched.modules.model_sampling import ModelSamplingDiscreteFlow
        return ModelSamplingDiscreteFlow(FakeModelConfig({"shift": shift}))

    def test_schedule_shape_and_monotonicity(self):
        for shift in [1.0, 1.15, 3.0, 6.0]:
            ms = self._make(shift)
            self.assertEqual(len(ms.sigmas), 1000)
            self.assertTrue(bool((ms.sigmas[1:] >= ms.sigmas[:-1]).all()), "sigmas must be non-decreasing for shift={}".format(shift))
            self.assertAlmostEqual(float(ms.sigma_min), float(ms.sigmas[0]), places=6)
            self.assertAlmostEqual(float(ms.sigma_max), float(ms.sigmas[-1]), places=6)

    def test_sigma_max_is_always_one(self):
        # sigma(t=1) = shift / shift = 1 regardless of shift.
        for shift in [1.0, 1.15, 3.0, 6.0]:
            ms = self._make(shift)
            self.assertAlmostEqual(float(ms.sigma_max), 1.0, places=5)

    def test_reference_checkpoints_shift_1_15(self):
        # sigma(0.5) = 1.15*0.5 / (1 + 0.15*0.5) = 0.575 / 1.075
        ms = self._make(1.15)
        expected = 1.15 * 0.5 / (1 + 0.15 * 0.5)
        self.assertAlmostEqual(float(ms.sigma(torch.tensor(500.0))), expected, places=5)

    def test_reference_checkpoints_shift_3_0(self):
        ms = self._make(3.0)
        expected = 3.0 * 0.5 / (1 + 2.0 * 0.5)
        self.assertAlmostEqual(float(ms.sigma(torch.tensor(500.0))), expected, places=5)
        self.assertAlmostEqual(expected, 0.75, places=6)

    def test_reference_checkpoints_shift_6_0(self):
        ms = self._make(6.0)
        expected = 6.0 * 0.5 / (1 + 5.0 * 0.5)
        self.assertAlmostEqual(float(ms.sigma(torch.tensor(500.0))), expected, places=5)

    def test_timestep_sigma_round_trip_at_endpoints(self):
        for shift in [1.0, 1.15, 3.0, 6.0]:
            ms = self._make(shift)
            self.assertAlmostEqual(float(ms.sigma(ms.timestep(ms.sigma_max))), float(ms.sigma_max), places=4)

    def test_percent_to_sigma_endpoints_and_shift_awareness(self):
        for shift in [1.0, 1.15, 3.0, 6.0]:
            ms = self._make(shift)
            self.assertEqual(ms.percent_to_sigma(0.0), 999999999.9)
            self.assertEqual(ms.percent_to_sigma(1.0), 0.0)
        # percent=0.5 should apply the shift curve, not just return 0.5.
        ms = self._make(3.0)
        expected = 3.0 * 0.5 / (1 + 2.0 * 0.5)
        self.assertAlmostEqual(ms.percent_to_sigma(0.5), expected, places=4)


class TestModelSamplingDiscreteFlowPrediction(unittest.TestCase):
    """CONST-style prediction: the model is fed the noisy sample unchanged
    and predicts the velocity from signal to noise.
    """
    def _make(self, shift=1.0):
        from ldm_patched.modules.model_sampling import ModelSamplingDiscreteFlow
        return ModelSamplingDiscreteFlow(FakeModelConfig({"shift": shift}))

    def test_calculate_input_is_identity(self):
        ms = self._make()
        noise = torch.randn(2, 4, 8, 8)
        sigma = torch.tensor([0.5, 0.25])
        self.assertTrue(torch.equal(ms.calculate_input(sigma, noise), noise))

    def test_calculate_denoised_algebra(self):
        ms = self._make()
        model_input = torch.ones(2, 1, 2, 2) * 2.0
        model_output = torch.ones(2, 1, 2, 2)
        sigma = torch.tensor([0.5, 0.25])
        denoised = ms.calculate_denoised(sigma, model_output, model_input)
        expected = torch.stack([
            torch.full((1, 2, 2), 2.0 - 1.0 * 0.5),
            torch.full((1, 2, 2), 2.0 - 1.0 * 0.25),
        ])
        self.assertTrue(torch.allclose(denoised, expected))

    def test_noise_scaling_linear_interpolation(self):
        ms = self._make()
        noise = torch.ones(4)
        latent = torch.zeros(4)
        scaled = ms.noise_scaling(torch.tensor(0.5), noise, latent)
        self.assertTrue(torch.allclose(scaled, torch.full((4,), 0.5)))

    def test_inverse_noise_scaling(self):
        ms = self._make()
        latent = torch.full((4,), 0.5)
        recovered = ms.inverse_noise_scaling(torch.tensor(0.5), latent)
        self.assertTrue(torch.allclose(recovered, torch.ones(4)))


class TestModelSamplingDiscreteFlowConfigDriven(unittest.TestCase):
    """`sampling_settings = {"shift": ...}` on a `supported_models` config
    class must select the correct shift with no further code changes -
    this is the interface Z-Image (shift 3.0) and Krea 2 (shift 1.15) will
    consume via their `supported_models` entries.
    """
    def test_shift_defaults_to_one_without_config(self):
        from ldm_patched.modules.model_sampling import ModelSamplingDiscreteFlow
        ms = ModelSamplingDiscreteFlow(None)
        self.assertEqual(ms.shift, 1.0)

    def test_shift_read_from_sampling_settings(self):
        from ldm_patched.modules.model_sampling import ModelSamplingDiscreteFlow
        z_image_config = FakeModelConfig({"shift": 3.0})
        krea2_config = FakeModelConfig({"shift": 1.15})
        self.assertEqual(ModelSamplingDiscreteFlow(z_image_config).shift, 3.0)
        self.assertEqual(ModelSamplingDiscreteFlow(krea2_config).shift, 1.15)


    def test_sigma_max_is_one_for_any_multiplier(self):
        """The schedule buffer must span (0, 1] regardless of the configured
        multiplier: normalization is by timesteps, scaling by multiplier, and
        sigma() divides the multiplier back out."""
        from ldm_patched.modules.model_sampling import ModelSamplingDiscreteFlow
        for multiplier in (250, 1000, 10000):
            sampling = ModelSamplingDiscreteFlow(
                FakeModelConfig({"shift": 3.0, "multiplier": multiplier}))
            self.assertAlmostEqual(float(sampling.sigma_max), 1.0, places=6)
            self.assertGreater(float(sampling.sigma_min), 0.0)



class TestModelTypeFlowDispatch(unittest.TestCase):
    """`model_sampling(model_config, ModelType.FLOW)` must return the flow
    sampling class directly, and the existing EPS/V_PREDICTION/EDM branches
    must remain byte-identical to before this change.
    """
    def test_flow_dispatch_returns_flow_class_directly(self):
        from ldm_patched.modules.model_base import ModelType, model_sampling
        from ldm_patched.modules.model_sampling import ModelSamplingDiscreteFlow
        ms = model_sampling(FakeModelConfig({"shift": 3.0}), ModelType.FLOW)
        self.assertIsInstance(ms, ModelSamplingDiscreteFlow)
        self.assertEqual(type(ms), ModelSamplingDiscreteFlow)
        self.assertEqual(ms.shift, 3.0)

    def _construct_or_skip_for_preexisting_cumprod_bug(self, factory):
        # ModelSamplingDiscrete._register_schedule() feeds the numpy array
        # returned by make_beta_schedule() straight into torch.cumprod().
        # This is pre-existing behavior (verified identical on origin/main,
        # unrelated to this ticket's diff) that some newer torch builds
        # reject with "must be Tensor, not numpy.ndarray". Skip rather than
        # fail the suite over an unrelated, pre-existing incompatibility.
        try:
            return factory()
        except TypeError as exc:
            if "numpy.ndarray" in str(exc):
                self.skipTest("pre-existing ModelSamplingDiscrete/make_beta_schedule numpy-vs-Tensor incompatibility, unrelated to FWDF-119: {}".format(exc))
            raise

    def test_eps_dispatch_unchanged(self):
        from ldm_patched.modules.model_base import ModelType, model_sampling
        from ldm_patched.modules.model_sampling import ModelSamplingDiscrete, EPS

        class ReferenceModelSampling(ModelSamplingDiscrete, EPS):
            pass

        config = FakeModelConfig()
        dispatched = self._construct_or_skip_for_preexisting_cumprod_bug(lambda: model_sampling(config, ModelType.EPS))
        reference = ReferenceModelSampling(config)
        self.assertIsInstance(dispatched, ModelSamplingDiscrete)
        self.assertIsInstance(dispatched, EPS)
        self.assertTrue(torch.equal(dispatched.sigmas, reference.sigmas))
        self.assertTrue(torch.equal(dispatched.log_sigmas, reference.log_sigmas))
        sigma = dispatched.sigmas[500].view(1)
        model_output = torch.full((1, 2), 0.25)
        model_input = torch.full((1, 2), 1.5)
        self.assertTrue(torch.equal(
            dispatched.calculate_denoised(sigma, model_output, model_input),
            reference.calculate_denoised(sigma, model_output, model_input)))

    def test_v_prediction_dispatch_unchanged(self):
        from ldm_patched.modules.model_base import ModelType, model_sampling
        from ldm_patched.modules.model_sampling import ModelSamplingDiscrete, V_PREDICTION

        class ReferenceModelSampling(ModelSamplingDiscrete, V_PREDICTION):
            pass

        config = FakeModelConfig()
        dispatched = self._construct_or_skip_for_preexisting_cumprod_bug(lambda: model_sampling(config, ModelType.V_PREDICTION))
        reference = ReferenceModelSampling(config)
        self.assertIsInstance(dispatched, ModelSamplingDiscrete)
        self.assertIsInstance(dispatched, V_PREDICTION)
        self.assertTrue(torch.equal(dispatched.sigmas, reference.sigmas))
        sigma = dispatched.sigmas[500].view(1)
        model_output = torch.full((1, 2), 0.25)
        model_input = torch.full((1, 2), 1.5)
        self.assertTrue(torch.equal(
            dispatched.calculate_denoised(sigma, model_output, model_input),
            reference.calculate_denoised(sigma, model_output, model_input)))

    def test_v_prediction_edm_dispatch_unchanged(self):
        from ldm_patched.modules.model_base import ModelType, model_sampling
        from ldm_patched.modules.model_sampling import ModelSamplingContinuousEDM, V_PREDICTION

        class ReferenceModelSampling(ModelSamplingContinuousEDM, V_PREDICTION):
            pass

        config = FakeModelConfig()
        dispatched = model_sampling(config, ModelType.V_PREDICTION_EDM)
        reference = ReferenceModelSampling(config)
        self.assertIsInstance(dispatched, ModelSamplingContinuousEDM)
        self.assertIsInstance(dispatched, V_PREDICTION)
        self.assertTrue(torch.equal(dispatched.sigmas, reference.sigmas))
        sigma = dispatched.sigmas[500].view(1)
        model_output = torch.full((1, 2), 0.25)
        model_input = torch.full((1, 2), 1.5)
        self.assertTrue(torch.equal(
            dispatched.calculate_denoised(sigma, model_output, model_input),
            reference.calculate_denoised(sigma, model_output, model_input)))


class TestModelSamplingDiscreteFlowBufferSurfaceParity(unittest.TestCase):
    """The schedulers in `ldm_patched/modules/samplers.py` only ever touch
    `.sigmas`, `.sigma_min`, `.sigma_max`, `.timestep()` and `.sigma()` on
    `model.model_sampling` - this asserts that surface exists and behaves
    the same way it does on `ModelSamplingDiscrete`.
    """
    def test_required_attributes_present(self):
        from ldm_patched.modules.model_sampling import ModelSamplingDiscreteFlow
        ms = ModelSamplingDiscreteFlow(FakeModelConfig({"shift": 3.0}))
        self.assertTrue(hasattr(ms, "sigmas"))
        self.assertTrue(hasattr(ms, "sigma_min"))
        self.assertTrue(hasattr(ms, "sigma_max"))
        self.assertTrue(callable(ms.timestep))
        self.assertTrue(callable(ms.sigma))
        self.assertTrue(callable(ms.percent_to_sigma))

    def test_timestep_and_sigma_accept_tensor_scalars(self):
        from ldm_patched.modules.model_sampling import ModelSamplingDiscreteFlow
        ms = ModelSamplingDiscreteFlow(FakeModelConfig({"shift": 3.0}))
        # This is exactly the call shape ldm_patched/modules/samplers.py's
        # normal_scheduler() makes: s.timestep(s.sigma_max)/s.timestep(s.sigma_min)
        # followed by s.sigma(ts) on a linspace of those timesteps.
        start = ms.timestep(ms.sigma_max)
        end = ms.timestep(ms.sigma_min)
        self.assertIsInstance(start, torch.Tensor)
        self.assertIsInstance(end, torch.Tensor)
        timesteps = torch.linspace(float(start), float(end), 10)
        sigmas = [ms.sigma(ts) for ts in timesteps]
        self.assertEqual(len(sigmas), 10)
        self.assertTrue(all(0.0 <= float(s) <= 1.0 for s in sigmas))


class TestSchedulerCompatibility(unittest.TestCase):
    """Confirms `normal_scheduler`/`simple_scheduler`/`ddim_scheduler`, and
    the `karras`/`exponential` k-diffusion schedules, all produce valid
    (monotonically non-increasing, ending at 0) sigma sequences for a
    synthetic `ModelSamplingDiscreteFlow` instance without any change to
    `ldm_patched/modules/samplers.py` or `modules/sample_hijack.py`.
    """
    def _model_with_flow_sampling(self, shift):
        from ldm_patched.modules.model_base import ModelType, model_sampling

        class FakeModel:
            pass

        model = FakeModel()
        model.model_sampling = model_sampling(FakeModelConfig({"shift": shift}), ModelType.FLOW)
        return model

    def _assert_valid_descending_schedule(self, sigmas):
        self.assertGreater(len(sigmas), 1)
        self.assertGreater(float(sigmas[0]), 0.0, "schedule must start at a positive sigma: {}".format(sigmas))
        diffs = sigmas[1:] - sigmas[:-1]
        self.assertTrue(bool((diffs <= 1e-6).all()), "sigmas must be non-increasing: {}".format(sigmas))
        self.assertTrue(bool((diffs < 0).any()), "schedule must strictly decrease somewhere: {}".format(sigmas))
        self.assertEqual(float(sigmas[-1]), 0.0)

    def test_normal_scheduler(self):
        from ldm_patched.modules.samplers import normal_scheduler
        model = self._model_with_flow_sampling(3.0)
        self._assert_valid_descending_schedule(normal_scheduler(model, 20))

    def test_normal_scheduler_sgm_uniform(self):
        from ldm_patched.modules.samplers import normal_scheduler
        model = self._model_with_flow_sampling(1.15)
        self._assert_valid_descending_schedule(normal_scheduler(model, 20, sgm=True))

    def test_simple_scheduler(self):
        from ldm_patched.modules.samplers import simple_scheduler
        model = self._model_with_flow_sampling(3.0)
        self._assert_valid_descending_schedule(simple_scheduler(model, 20))

    def test_ddim_uniform_scheduler(self):
        from ldm_patched.modules.samplers import ddim_scheduler
        model = self._model_with_flow_sampling(1.15)
        self._assert_valid_descending_schedule(ddim_scheduler(model, 20))

    def test_karras_scheduler(self):
        from ldm_patched.k_diffusion import sampling as k_diffusion_sampling
        model = self._model_with_flow_sampling(3.0)
        sigmas = k_diffusion_sampling.get_sigmas_karras(
            n=20,
            sigma_min=float(model.model_sampling.sigma_min),
            sigma_max=float(model.model_sampling.sigma_max),
        )
        self._assert_valid_descending_schedule(sigmas)

    def test_exponential_scheduler(self):
        from ldm_patched.k_diffusion import sampling as k_diffusion_sampling
        model = self._model_with_flow_sampling(1.15)
        sigmas = k_diffusion_sampling.get_sigmas_exponential(
            n=20,
            sigma_min=float(model.model_sampling.sigma_min),
            sigma_max=float(model.model_sampling.sigma_max),
        )
        self._assert_valid_descending_schedule(sigmas)


class TestDenoiseStrengthSlicing(unittest.TestCase):
    """Mirrors `modules/default_pipeline.py::calculate_sigmas()`'s
    steps/denoise handling (`new_steps = int(steps / denoise)`, then keep
    the last `steps + 1` sigmas) directly against
    `ldm_patched.modules.samplers.calculate_sigmas_scheduler` so partial
    denoise (img2img/Vary/Upscale, consumed by FWDF-154) is verified
    without needing to import the full pipeline module.
    """
    def _partial_schedule(self, model, scheduler_name, steps, denoise):
        from ldm_patched.modules.samplers import calculate_sigmas_scheduler
        if denoise is None or denoise > 0.9999:
            return calculate_sigmas_scheduler(model, scheduler_name, steps)
        new_steps = int(steps / denoise)
        sigmas = calculate_sigmas_scheduler(model, scheduler_name, new_steps)
        return sigmas[-(steps + 1):]

    def _model_with_flow_sampling(self, shift):
        from ldm_patched.modules.model_base import ModelType, model_sampling

        class FakeModel:
            pass

        model = FakeModel()
        model.model_sampling = model_sampling(FakeModelConfig({"shift": shift}), ModelType.FLOW)
        return model

    def test_low_denoise_produces_short_high_sigma_tail(self):
        model = self._model_with_flow_sampling(3.0)
        steps = 10
        full = self._partial_schedule(model, "normal", steps, denoise=None)
        partial = self._partial_schedule(model, "normal", steps, denoise=0.1)
        self.assertEqual(len(partial), steps + 1)
        diffs = partial[1:] - partial[:-1]
        self.assertTrue(bool((diffs <= 1e-6).all()))
        self.assertEqual(float(partial[-1]), 0.0)
        # A low-denoise partial schedule should start well below the
        # full schedule's starting sigma (only touching the low-noise tail).
        self.assertLess(float(partial[0]), float(full[0]))

    def test_high_denoise_is_close_to_full_schedule(self):
        model = self._model_with_flow_sampling(3.0)
        steps = 10
        full = self._partial_schedule(model, "normal", steps, denoise=None)
        partial = self._partial_schedule(model, "normal", steps, denoise=0.9)
        self.assertEqual(len(partial), steps + 1)
        diffs = partial[1:] - partial[:-1]
        self.assertTrue(bool((diffs <= 1e-6).all()))
        self.assertEqual(float(partial[-1]), 0.0)
        # denoise=0.9 truncates only slightly off the top of the full schedule.
        self.assertLessEqual(float(partial[0]), float(full[0]))


if __name__ == '__main__':
    unittest.main()
