"""Model family detection from checkpoint headers.

Maps a checkpoint file to its `ModelFamily` (see `modules.model_family`) by
reading only the safetensors header -- the tensor name/shape/dtype/offset
metadata -- never the tensor data itself. This keeps detection fast enough
to call from a UI change handler on every checkpoint-dropdown selection.

Discriminant keys mirror the architecture detection registry added in
FWDF-116 (`ldm_patched/modules/model_detection.py`) so the two detectors
stay in sync by construction rather than by convention:
  - `{prefix}x_embedder.weight` + `{prefix}cap_embedder.*` -> `Z_IMAGE`.
  - `{prefix}txtfusion.projector.weight` -> a Krea 2 checkpoint. Raw and
    Turbo share one state-dict layout (distillation does not rename or
    reshape tensors), so the header cannot tell them apart: the header check
    yields `KREA2_RAW` as the architecture default and `get_family()` then
    resolves the variant from the `krea2_variant_overrides` config entry or
    the checkpoint file name (see `_resolve_krea2_variant`).
  - `{prefix}input_blocks.0.0.weight` -> a UNet checkpoint, disambiguated
    into `SDXL` vs `SD15` via `{prefix}label_emb.0.0.weight`, the same
    ADM/conditioning signal `detect_unet_config` reads at load time
    (`ldm_patched/modules/model_detection.py`).
  - anything else -> `UNKNOWN`.
`{prefix}` is `model.diffusion_model.`, matching the `unet_key_prefix`
`ldm_patched.modules.sd` uses when loading a checkpoint's state dict.
"""

import logging
import os
import re

from safetensors import SafetensorError, safe_open

import modules.config
from modules.fast_checkpoint import resolve_checkpoint_path
from modules.model_family import ModelFamily

logger = logging.getLogger(__name__)

_KEY_PREFIX = 'model.diffusion_model.'
_UNET_KEY = f'{_KEY_PREFIX}input_blocks.0.0.weight'
_SDXL_ADM_KEY = f'{_KEY_PREFIX}label_emb.0.0.weight'
_Z_IMAGE_X_EMBEDDER_KEY = f'{_KEY_PREFIX}x_embedder.weight'
_Z_IMAGE_CAP_EMBEDDER_PREFIX = f'{_KEY_PREFIX}cap_embedder.'
_KREA2_PROJECTOR_KEY = f'{_KEY_PREFIX}txtfusion.projector.weight'

_KREA2_FAMILIES = frozenset({ModelFamily.KREA2_RAW, ModelFamily.KREA2_TURBO})
# One family per value in `modules.config.KREA2_VARIANTS`, the values the
# `krea2_variant_overrides` config item may map a checkpoint to.
_KREA2_VARIANT_FAMILIES = {'raw': ModelFamily.KREA2_RAW, 'turbo': ModelFamily.KREA2_TURBO}
# A variant name counts only as a standalone word (no adjacent letter), so
# 'drawing' / 'straw' never read as 'raw'; a digit or separator is a boundary.
_KREA2_VARIANT_PATTERNS = {
    variant: re.compile(rf'(?<![a-z]){variant}(?![a-z])') for variant in _KREA2_VARIANT_FAMILIES
}


class CorruptCheckpointError(Exception):
    """Raised when a checkpoint's safetensors header cannot be parsed."""


# Keyed by absolute path, storing the (mtime, size) fingerprint alongside
# the family rather than in the key: a checkpoint can be replaced in place
# (re-download, LoRA merge in place) without its name changing, so the
# fingerprint must invalidate the entry -- but keeping the fingerprint in
# the key would leave every superseded entry behind forever, growing the
# cache without bound in a long-lived UI process. One entry per path,
# latest fingerprint wins.
_family_cache: dict[str, tuple[tuple[float, int], ModelFamily]] = {}


def _read_state_dict_keys(path: str) -> frozenset[str]:
    """Read the tensor name set from a safetensors header, no tensor data.

    Uses `framework='numpy'` rather than `'pt'` so this never touches torch
    device state -- `keys()` only needs the header, and numpy has no device
    concept to accidentally initialize.
    """
    try:
        with safe_open(path, framework='numpy') as f:
            return frozenset(f.keys())
    except (SafetensorError, OSError) as e:
        # OSError covers the file vanishing or losing read permission between
        # the caller's os.stat() and this open (get_family() must never raise).
        raise CorruptCheckpointError(f"cannot read safetensors header of '{path}': {e}") from e


def _detect_family_from_keys(keys: frozenset[str]) -> ModelFamily:
    """Pure discriminant logic over a checkpoint's tensor name set."""
    if _Z_IMAGE_X_EMBEDDER_KEY in keys and any(k.startswith(_Z_IMAGE_CAP_EMBEDDER_PREFIX) for k in keys):
        return ModelFamily.Z_IMAGE
    if _KREA2_PROJECTOR_KEY in keys:
        return ModelFamily.KREA2_RAW
    if _UNET_KEY in keys:
        return ModelFamily.SDXL if _SDXL_ADM_KEY in keys else ModelFamily.SD15
    return ModelFamily.UNKNOWN


def _resolve_krea2_variant(checkpoint_filename: str, family: ModelFamily) -> ModelFamily:
    """Pick Krea 2's Raw or Turbo family for a checkpoint the header check
    classified as Krea 2. Any other family is returned unchanged.

    Precedence: a `modules.config.krea2_variant_overrides` entry (looked up by
    the filename exactly as given, then by its basename), else the
    case-insensitive whole words `turbo` / `raw` in the basename (a word has no
    adjacent letter, so `drawing` does not name Raw). A name that contains both
    or neither is ambiguous: it resolves to `KREA2_RAW` and
    logs a warning naming the config key that fixes it.
    """
    if family not in _KREA2_FAMILIES:
        return family

    basename = os.path.basename(checkpoint_filename)
    overrides = modules.config.krea2_variant_overrides
    override = overrides.get(checkpoint_filename, overrides.get(basename))
    if override is not None:
        return _KREA2_VARIANT_FAMILIES[override]

    lowered = basename.lower()
    named_variants = [variant for variant, pattern in _KREA2_VARIANT_PATTERNS.items() if pattern.search(lowered)]
    if len(named_variants) == 1:
        return _KREA2_VARIANT_FAMILIES[named_variants[0]]

    # The message is deliberately static: CodeQL's clear-text-logging query
    # treats checkpoint names derived from config values as secrets, so it
    # names the config key instead of the file.
    logger.warning(
        "Cannot tell whether a Krea 2 checkpoint is Raw or Turbo from its file name; assuming Raw. "
        "Map the file to 'turbo' or 'raw' in the 'krea2_variant_overrides' entry of config.txt."
    )
    return ModelFamily.KREA2_RAW


def get_family(checkpoint_filename: str) -> ModelFamily:
    """Detect the `ModelFamily` of a checkpoint by filename.

    Resolves `checkpoint_filename` the same way the pipeline resolves it
    for loading (`modules.fast_checkpoint.resolve_checkpoint_path`), then
    reads only the safetensors header. Results are cached per path with an
    `(mtime, size)` fingerprint; a checkpoint that changes on disk replaces
    its own cache entry rather than serving a stale family or accumulating
    superseded entries.

    Never raises: a checkpoint that cannot be found or whose header cannot
    be parsed resolves to `ModelFamily.UNKNOWN`, so this can be called
    unconditionally from a UI change handler.
    """
    resolved_path = resolve_checkpoint_path(
        checkpoint_filename, modules.config.paths_checkpoints, modules.config.path_fast_checkpoints
    )

    try:
        file_stat = os.stat(resolved_path)
    except OSError as e:
        logger.warning(f"Cannot stat checkpoint '{checkpoint_filename}' for family detection: {e}")
        return ModelFamily.UNKNOWN

    fingerprint = (file_stat.st_mtime, file_stat.st_size)
    cached = _family_cache.get(resolved_path)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]

    try:
        keys = _read_state_dict_keys(resolved_path)
        family = _resolve_krea2_variant(checkpoint_filename, _detect_family_from_keys(keys))
    except CorruptCheckpointError as e:
        logger.warning(f"Could not detect model family for '{checkpoint_filename}': {e}")
        family = ModelFamily.UNKNOWN

    _family_cache[resolved_path] = (fingerprint, family)
    return family
