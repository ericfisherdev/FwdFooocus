"""Model family capability registry.

This module is the central Open/Closed extension point for architecture
support: adding a new model family (e.g. a future Krea 2 backlog entry)
means adding one `ModelFamily` member and one `FAMILY_CAPABILITIES` entry.
Consumers (pipeline, Gradio UI, new-UI API) look up capabilities through
`get_capabilities()` instead of hardcoding SDXL assumptions.

This module ships only the registry; no consumer reads it yet (that is
FWDF-127 for the pipeline, FWDF-128 for the new-UI capabilities API, and
FWDF-129/FWDF-130 for Gradio/new-UI show-hide).

Scheduler caveat for future family entries: `"turbo"` and
`"align_your_steps"` in `scheduler_names` are architecture-specific today.
`modules/sample_hijack.py` hardcodes `SDTurboScheduler` and switches on
`isinstance(model.latent_format, latent_formats.SDXL)` to pick
`AlignYourStepsScheduler`'s `model_type` (`'SDXL'` or `'SD1'` only). A
family entry must not list these two scheduler names until that hijack is
extended to support it.
"""

from dataclasses import dataclass
from enum import Enum

import modules.config
from modules.flags import Performance, Steps, guidance_scale_range, sampler_list, scheduler_list, sdxl_aspect_ratios


class ModelFamily(Enum):
    """Architectures recognized by the capability registry."""

    SDXL = 'sdxl'
    SD15 = 'sd15'
    Z_IMAGE = 'z_image'
    KREA2 = 'krea2'
    UNKNOWN = 'unknown'


@dataclass(frozen=True, slots=True)
class PerformanceMode:
    """One performance preset (e.g. Quality, Speed, Lightning) for a family."""

    label: str
    steps: int
    steps_uov: int
    cfg: float | None
    lora_filename: str | None
    restricted: bool


@dataclass(frozen=True, slots=True)
class FamilyCapabilities:
    """Everything a consumer needs to know about a model family.

    `vae_names=None` means "no per-family restriction, list whatever
    `modules.config.path_vae` contains" (today's global behavior). A
    populated tuple lets a family declare a curated/compatible VAE subset
    instead of the full global listing.
    """

    supports_refiner: bool
    supports_adm_guidance: bool
    supports_freeu: bool
    supports_clip_skip: bool
    supports_adaptive_cfg: bool
    supports_sharpness: bool
    supports_negative_prompt: bool
    supports_controlnet: bool
    supports_ip_adapter: bool
    supports_inpaint_engine: bool
    supports_vae_override: bool
    vae_names: tuple[str, ...] | None
    performance_modes: tuple[PerformanceMode, ...]
    sampler_names: tuple[str, ...]
    scheduler_names: tuple[str, ...]
    aspect_ratios: tuple[str, ...]
    default_cfg: float
    cfg_range: tuple[float, float]
    default_steps: int
    latent_channels: int


def _build_sdxl_performance_modes() -> tuple[PerformanceMode, ...]:
    """Replicate today's Performance/Steps/StepsUOV/PerformanceLoRA behavior.

    SDXL performance modes do not override CFG today -- the Gradio handler
    only toggles `guidance_scale.interactive` (webui.py:1269-1277) -- so
    `cfg` is always `None` here, meaning "use the family's `default_cfg`".

    Fails fast if a `Performance` member has no `Steps`/`StepsUOV` entry
    (their lookups return `None`): `PerformanceMode.steps`/`steps_uov` are
    typed `int` and consumers do arithmetic on them, so a `None` must
    surface at registry build time, not deep inside a consumer.
    """
    modes = []
    for member in Performance:
        steps = member.steps()
        steps_uov = member.steps_uov()
        if steps is None or steps_uov is None:
            raise ValueError(
                f"Performance member {member.name!r} has no Steps/StepsUOV entry; "
                f"cannot build a PerformanceMode with steps={steps!r}, steps_uov={steps_uov!r}"
            )
        modes.append(
            PerformanceMode(
                label=member.value,
                steps=steps,
                steps_uov=steps_uov,
                cfg=None,
                lora_filename=member.lora_filename(),
                restricted=Performance.has_restricted_features(member),
            )
        )
    return tuple(modes)


def _build_sdxl_capabilities() -> FamilyCapabilities:
    return FamilyCapabilities(
        supports_refiner=True,
        supports_adm_guidance=True,
        supports_freeu=True,
        supports_clip_skip=True,
        supports_adaptive_cfg=True,
        supports_sharpness=True,
        supports_negative_prompt=True,
        supports_controlnet=True,
        supports_ip_adapter=True,
        supports_inpaint_engine=True,
        supports_vae_override=True,
        vae_names=None,
        performance_modes=_build_sdxl_performance_modes(),
        sampler_names=tuple(sampler_list),
        scheduler_names=tuple(scheduler_list),
        aspect_ratios=tuple(sdxl_aspect_ratios),
        default_cfg=modules.config.default_cfg_scale,
        cfg_range=guidance_scale_range,
        default_steps=Steps.SPEED.value,
        latent_channels=4,
    )


_SDXL_CAPABILITIES = _build_sdxl_capabilities()

# SD15 shares every SDXL value in this codebase today: aspect ratios and
# sampler/scheduler lists are global in modules/flags.py, and no SD1.5
# performance-mode divergence exists yet. This is a placeholder for future
# SD1.5-specific behavior, not an assertion that SD15 == SDXL forever.
_SD15_CAPABILITIES = _build_sdxl_capabilities()

FAMILY_CAPABILITIES: dict[ModelFamily, FamilyCapabilities] = {
    ModelFamily.SDXL: _SDXL_CAPABILITIES,
    ModelFamily.SD15: _SD15_CAPABILITIES,
    # UNKNOWN must resolve to the exact same object as SDXL (identity, not
    # a duplicate literal) so unrecognized checkpoints keep today's
    # behavior and the two stay in lockstep by construction.
    ModelFamily.UNKNOWN: _SDXL_CAPABILITIES,
}


def get_capabilities(family: ModelFamily) -> FamilyCapabilities:
    """Look up the capability descriptor for a model family.

    Unrecognized families (not yet present in `FAMILY_CAPABILITIES`) fall
    back to `ModelFamily.UNKNOWN`'s descriptor, which is today's SDXL
    behavior. This is the single public entry point for capability
    lookups -- consumers should not index `FAMILY_CAPABILITIES` directly.
    """
    return FAMILY_CAPABILITIES.get(family, FAMILY_CAPABILITIES[ModelFamily.UNKNOWN])
