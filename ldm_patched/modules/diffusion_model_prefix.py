"""Where a checkpoint keeps its diffusion model weights.

Kept free of torch and `supported_models` imports so that header-only callers
(`modules.model_family_detection`, which runs in a UI change handler) and the
full loader (`ldm_patched.modules.sd`) can share one decision without either
paying for the other's dependencies.
"""

from collections.abc import Iterable
from typing import Final

DIFFUSION_MODEL_PREFIX: Final = "model.diffusion_model."


def diffusion_model_prefix(state_dict_keys: Iterable[str]) -> str:
    """Return the key prefix under which a checkpoint stores its diffusion model.

    All-in-one SD1.x/SDXL checkpoints nest the UNet under
    `DIFFUSION_MODEL_PREFIX` next to the VAE and text encoders; a published
    single-file Krea 2 or Z-Image checkpoint is the diffusion model alone and
    stores its keys flat. Any key carrying the prefix selects it, otherwise
    the file is flat and the prefix is empty.

    A flat file that is not a diffusion model still yields `""` and then fails
    architecture detection exactly as it did before this helper existed.
    """
    if any(key.startswith(DIFFUSION_MODEL_PREFIX) for key in state_dict_keys):
        return DIFFUSION_MODEL_PREFIX
    return ""
