"""Regression tests for FWDF-157's ip-adapter capability gate.

modules.async_worker's apply_image_input()/apply_control_nets() gate
ImagePrompt/FaceSwap ip-adapter downloads and UNet patching on
modules.model_family.get_capabilities(family).supports_ip_adapter, resolved
from modules.model_family_detection.get_family(base_model_name) -- the same
two-call idiom modules.default_pipeline.refresh_everything() uses for its
supports_refiner gate.

modules.async_worker cannot be imported directly in this environment: its
module-level `from extras.inpaint_mask import ...` pulls in
extras.GroundingDINO, which requires the 'supervision' package -- a
pre-existing, unrelated test-environment gap (not introduced by this
ticket). These tests instead exercise the exact family-detection ->
capability-lookup resolution chain the gate calls, using the same synthetic
checkpoint fixtures as tests/test_model_family_detection.py, so a regression
in either module is still caught.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import safetensors.torch  # noqa: E402
import torch  # noqa: E402

# args_manager calls parse_args() at import time, which chokes on pytest's
# argv (modules.model_family_detection imports modules.config, which imports
# args_manager). Mirrors the convention in tests/test_model_family_detection.py.
_original_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    from modules import model_family, model_family_detection  # noqa: E402
finally:
    sys.argv = _original_argv


_SDXL_KEYS = [
    'model.diffusion_model.input_blocks.0.0.weight',
    'model.diffusion_model.label_emb.0.0.weight',
]
_Z_IMAGE_KEYS = [
    'model.diffusion_model.x_embedder.weight',
    'model.diffusion_model.cap_embedder.mlp.weight',
]


def _write_checkpoint(path, tensor_names, tensor_shape=(2, 2)):
    """Write a minimal synthetic safetensors file with the given tensor names."""
    state_dict = {name: torch.zeros(*tensor_shape) for name in tensor_names}
    safetensors.torch.save_file(state_dict, str(path))


def _base_model_supports_ip_adapter(base_model_name):
    """Mirrors modules.async_worker._base_model_supports_ip_adapter()'s
    resolution chain without importing that module (see file docstring)."""
    base_family = model_family_detection.get_family(base_model_name)
    return model_family.get_capabilities(base_family).supports_ip_adapter


class TestIpAdapterCapabilityGate(unittest.TestCase):
    """The resolution chain modules.async_worker's ip-adapter gate relies on."""

    def setUp(self):
        self.checkpoint_dir = tempfile.mkdtemp()
        self._original_paths_checkpoints = model_family_detection.modules.config.paths_checkpoints
        self._original_path_fast_checkpoints = model_family_detection.modules.config.path_fast_checkpoints
        model_family_detection.modules.config.paths_checkpoints = [self.checkpoint_dir]
        model_family_detection.modules.config.path_fast_checkpoints = None
        model_family_detection._family_cache.clear()

    def tearDown(self):
        model_family_detection.modules.config.paths_checkpoints = self._original_paths_checkpoints
        model_family_detection.modules.config.path_fast_checkpoints = self._original_path_fast_checkpoints
        model_family_detection._family_cache.clear()
        shutil.rmtree(self.checkpoint_dir, ignore_errors=True)

    def _checkpoint_path(self, filename):
        return os.path.join(self.checkpoint_dir, filename)

    def test_z_image_checkpoint_does_not_support_ip_adapter(self):
        # Z-Image task with ip args: the gate must resolve False so
        # apply_image_input() skips downloading_ip_adapters() and
        # apply_control_nets() skips ip_adapter.preprocess()/patch_model().
        _write_checkpoint(self._checkpoint_path('z_image.safetensors'), _Z_IMAGE_KEYS)
        self.assertFalse(_base_model_supports_ip_adapter('z_image.safetensors'))

    def test_sdxl_checkpoint_supports_ip_adapter(self):
        # SDXL regression: the gate must remain a no-op so the existing
        # ImagePrompt/FaceSwap download and UNet-patch path is unchanged.
        _write_checkpoint(self._checkpoint_path('sdxl.safetensors'), _SDXL_KEYS)
        self.assertTrue(_base_model_supports_ip_adapter('sdxl.safetensors'))

    def test_unknown_checkpoint_falls_back_to_sdxl_behavior(self):
        # UNKNOWN resolves to the same capability object as SDXL by
        # construction (modules/model_family.py), so an undetectable
        # checkpoint's ip-adapter behavior is unchanged from today.
        _write_checkpoint(self._checkpoint_path('mystery.safetensors'), ['some.unrelated.tensor.weight'])
        self.assertTrue(_base_model_supports_ip_adapter('mystery.safetensors'))


if __name__ == '__main__':
    unittest.main()
