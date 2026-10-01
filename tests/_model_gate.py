"""Decides whether a `requires_models` test may run (FWDF-160).

Kept free of `modules.config` (which parses `sys.argv` at import) so it can be
unit-tested with injected locations; `tests/conftest.py` builds the real
`Krea2ModelLocations` from `modules.config`.

A test marked `requires_models` runs only when BOTH hold:
  1. the developer opted in with `FWDF_RUN_MODEL_TESTS=1` -- a 12B bf16 model
     generating 52 steps must never start just because files sit on disk;
  2. every Krea 2 file the pipeline needs is present.
"""
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

REQUIRES_MODELS_MARKER = 'requires_models'
RUN_MODEL_TESTS_ENV = 'FWDF_RUN_MODEL_TESTS'
_OPT_IN_VALUE = '1'
_CHECKPOINT_EXTENSION = '.safetensors'
_KREA2_CHECKPOINT_NAME_PART = 'krea2'


@dataclass(frozen=True, slots=True)
class Krea2ModelLocations:
    """Where the three Krea 2 files live: checkpoint search directories, and
    the exact paths of the text encoder and VAE companions."""

    checkpoint_dirs: Sequence[str]
    text_encoder_path: str
    vae_path: str


def find_krea2_checkpoints(checkpoint_dirs: Sequence[str]) -> list[str]:
    """Checkpoint names, relative to the directory that holds them (the form
    `refresh_everything` takes), whose file name contains `krea2`.

    The variant (Raw or Turbo) is not decided here; that is
    `modules.model_family_detection.get_family`'s job.
    """
    names = []
    for directory in checkpoint_dirs:
        for folder, _, files in os.walk(directory):
            for file_name in sorted(files):
                if (file_name.lower().endswith(_CHECKPOINT_EXTENSION)
                        and _KREA2_CHECKPOINT_NAME_PART in file_name.lower()):
                    names.append(os.path.relpath(os.path.join(folder, file_name), directory))
    return names


def skip_reason(environ: Mapping[str, str], locations: Krea2ModelLocations) -> str | None:
    """Why a `requires_models` test must be skipped, or None when it may run."""
    if environ.get(RUN_MODEL_TESTS_ENV) != _OPT_IN_VALUE:
        return f'set {RUN_MODEL_TESTS_ENV}=1 to run tests that load real Krea 2 weights'
    if not find_krea2_checkpoints(locations.checkpoint_dirs):
        return f'no krea2*{_CHECKPOINT_EXTENSION} checkpoint under {list(locations.checkpoint_dirs)}'
    for description, path in (('text encoder', locations.text_encoder_path), ('VAE', locations.vae_path)):
        if not os.path.isfile(path):
            return f'Krea 2 {description} not found at {path}'
    return None
