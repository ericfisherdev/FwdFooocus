"""Model family detection from checkpoint headers.

Maps a checkpoint file to its `ModelFamily` (see `modules.model_family`) by
reading only the safetensors header -- the tensor name/shape/dtype/offset
metadata -- never the tensor data itself. This keeps detection fast enough
to call from a UI change handler on every checkpoint-dropdown selection.

Discriminant keys mirror the architecture detection registry added in
FWDF-116 (`ldm_patched/modules/model_detection.py`) so the two detectors
stay in sync by construction rather than by convention:
  - `{prefix}x_embedder.weight` + `{prefix}cap_embedder.*` -> `Z_IMAGE`.
  - `{prefix}txtfusion.projector.weight` -> `KREA2` (backlog family; this
    detector is inert until Krea 2 lands but costs nothing to include now).
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
        return ModelFamily.KREA2
    if _UNET_KEY in keys:
        return ModelFamily.SDXL if _SDXL_ADM_KEY in keys else ModelFamily.SD15
    return ModelFamily.UNKNOWN


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
        family = _detect_family_from_keys(keys)
    except CorruptCheckpointError as e:
        logger.warning(f"Could not detect model family for '{checkpoint_filename}': {e}")
        family = ModelFamily.UNKNOWN

    _family_cache[resolved_path] = (fingerprint, family)
    return family


def session_state_id(checkpoint_filename: str) -> str:
    """Resolve the session-state persistence key for a checkpoint.

    Uses the detected `ModelFamily` when recognized. Falls back to the
    hand-authored `modules.config.default_base_model` string only when
    detection returns `UNKNOWN`, so existing session rows for users who
    already configured `default_base_model` in `config.txt` keep resolving
    to the same key as before. Shared by both the load site (`webui.py`)
    and the save site (`modules/async_worker.py`) so this fallback rule
    lives in exactly one place.
    """
    family = get_family(checkpoint_filename)
    if family is ModelFamily.UNKNOWN and modules.config.default_base_model is not None:
        return modules.config.default_base_model
    return family.value
