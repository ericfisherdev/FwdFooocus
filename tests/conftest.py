"""Shared pytest hooks: the `requires_models` marker (FWDF-160).

Tests that load real model weights carry `@pytest.mark.requires_models` and are
skipped unless `FWDF_RUN_MODEL_TESTS=1` is set and the Krea 2 files exist; see
`tests/_model_gate.py` for the decision logic. Everything else is untouched, so
a plain `python -m pytest tests/` stays model-free.
"""
import os
import sys

import pytest

from tests._model_gate import REQUIRES_MODELS_MARKER, Krea2ModelLocations, skip_reason


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        f'{REQUIRES_MODELS_MARKER}: loads real Krea 2 weights; skipped unless FWDF_RUN_MODEL_TESTS=1 '
        'and the checkpoint, text encoder and VAE are present',
    )


def _configured_krea2_locations() -> Krea2ModelLocations:
    # args_manager parses argv when modules.config is first imported, which
    # chokes on pytest's own arguments; every test module guards the same way.
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        import modules.config
    finally:
        sys.argv = original_argv
    return Krea2ModelLocations(
        checkpoint_dirs=modules.config.paths_checkpoints,
        text_encoder_path=modules.config.krea2_text_encoder_path(),
        vae_path=modules.config.qwen_image_vae_path(),
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    if next(item.iter_markers(name=REQUIRES_MODELS_MARKER), None) is None:
        return
    reason = skip_reason(os.environ, _configured_krea2_locations())
    if reason is not None:
        pytest.skip(reason)
