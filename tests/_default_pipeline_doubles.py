"""Shared test doubles for importing the real modules.default_pipeline in a
torch/torchvision-optional sandbox.

Not a test module itself (pytest's default collection only picks up
test_*.py/*_test.py), so it can be imported by any test file that needs the
real, unmodified pipeline functions without duplicating this setup.
"""
import sys
import types


class FakeStableDiffusionModel:
    """Stands in for modules.core.StableDiffusionModel: a lightweight
    surface for direct per-test construction of model_base/model_refiner
    fixtures, decoupled from real model loading."""

    def __init__(self, unet=None, vae=None, clip=None, clip_vision=None,
                 filename=None, vae_filename=None):
        self.unet = unet
        self.vae = vae
        self.clip = clip
        self.clip_vision = clip_vision
        self.filename = filename
        self.vae_filename = vae_filename
        self.unet_with_lora = unet
        self.clip_with_lora = clip

    def refresh_loras(self, loras):
        pass


class FakeUnet:
    def __init__(self, model=None):
        self.model = model if model is not None else object()


def install_default_pipeline_test_doubles():
    """Install a torchvision stand-in (when torchvision isn't installed) so
    the real modules.default_pipeline can be imported. Returns a zero-arg
    callable that restores the prior state.
    """
    import transformers  # noqa: F401  (forces the real torchvision-unavailable check first)

    restore_actions = []

    torchvision_available = True
    try:
        import torchvision  # noqa: F401
    except ImportError:
        torchvision_available = False

    if not torchvision_available:
        stub_names = ('torchvision', 'torchvision.transforms', 'torchvision.transforms.functional')
        for name in stub_names:
            assert name not in sys.modules, f'unexpected pre-existing stub conflict for {name}'
        functional_stub = types.ModuleType('torchvision.transforms.functional')
        functional_stub.InterpolationMode = object
        functional_stub.rotate = lambda *a, **k: None
        transforms_stub = types.ModuleType('torchvision.transforms')
        transforms_stub.functional = functional_stub
        torchvision_stub = types.ModuleType('torchvision')
        torchvision_stub.transforms = transforms_stub

        sys.modules['torchvision'] = torchvision_stub
        sys.modules['torchvision.transforms'] = transforms_stub
        sys.modules['torchvision.transforms.functional'] = functional_stub
        restore_actions.append(lambda: [sys.modules.pop(n, None) for n in stub_names])

    def _restore():
        for action in reversed(restore_actions):
            action()

    return _restore
