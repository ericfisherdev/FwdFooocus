"""Model family capability registry.

This module is the central Open/Closed extension point for architecture
support: adding a new model family (e.g. a future Krea 2 backlog entry)
means adding one `ModelFamily` member and one `FAMILY_CAPABILITIES` entry.
Consumers (pipeline, Gradio UI, new-UI API) look up capabilities through
`get_capabilities()` instead of hardcoding SDXL assumptions.

FWDF-127 is the first consumer, reading `get_capabilities()` from
`modules/default_pipeline.py` to gate refiner assembly. FWDF-128 (new-UI
capabilities API) and FWDF-129/FWDF-130 (Gradio/new-UI show-hide) are the
remaining planned consumers.

Scheduler caveat for future family entries: `"turbo"` and
`"align_your_steps"` in `scheduler_names` are architecture-specific today.
`modules/sample_hijack.py` hardcodes `SDTurboScheduler` and switches on
`isinstance(model.latent_format, latent_formats.SDXL)` to pick
`AlignYourStepsScheduler`'s `model_type` (`'SDXL'` or `'SD1'` only). A
family entry must not list these two scheduler names until that hijack is
extended to support it.
"""

import math
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

    `native_resolution_range` is the (floor, ceiling) `modules.util.get_shape_ceil()`
    bucket that `apply_vary`/`apply_upscale` (`modules/async_worker.py`) clamp
    an input image into before VAE-encoding it -- see `_native_resolution_range()`
    below for how it is derived from `aspect_ratios`.

    `controlnet_types` (FWDF-156) is the coarse-grained `supports_controlnet`
    flag's per-type breakdown: which `modules.flags` ControlNet annotators
    (`'canny'`, `'cpds'`) this family's ControlNet implementation actually
    supports. SDXL/SD15/UNKNOWN support both; Z-Image's DiT ControlNet
    backport supports Canny only (CPDS has no published DiT equivalent).
    `supports_controlnet` must always equal `bool(controlnet_types)` --
    enforced in `__post_init__` -- so the two can never silently disagree.
    """

    supports_refiner: bool
    supports_adm_guidance: bool
    supports_freeu: bool
    supports_clip_skip: bool
    supports_adaptive_cfg: bool
    supports_sharpness: bool
    supports_negative_prompt: bool
    supports_controlnet: bool
    controlnet_types: tuple[str, ...]
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
    native_resolution_range: tuple[float, float]

    def __post_init__(self):
        if self.supports_controlnet != bool(self.controlnet_types):
            raise ValueError(
                f"supports_controlnet ({self.supports_controlnet!r}) must equal "
                f"bool(controlnet_types) ({bool(self.controlnet_types)!r}); "
                f"got controlnet_types={self.controlnet_types!r}"
            )


def _native_resolution_range(aspect_ratios: tuple[str, ...]) -> tuple[float, float]:
    """Derive a family's (floor, ceiling) native resolution bucket from its
    aspect-ratio list.

    `modules.util.get_shape_ceil(h, w)` computes `ceil(sqrt(h*w) / 64) * 64`,
    an image's total-pixel-count bucket rounded up to a multiple of 64. Every
    entry in a well-formed `aspect_ratios` list resolves to the same bucket
    (the family's native megapixel resolution, e.g. SDXL's ~1-megapixel
    presets all resolve to 1024.0) -- the floor here is the minimum across
    the list, and the ceiling is double that: the point past which
    `apply_vary`/`apply_upscale` (`modules/async_worker.py`) stop upsizing an
    input image further. The formula is duplicated rather than imported from
    `modules.util` because that module pulls in cv2/PIL/numpy, dependencies
    this lightweight capability-registry module should not need at import
    time.

    For SDXL's `aspect_ratios` this yields exactly `(1024.0, 2048.0)`,
    matching the literals this replaces in `apply_vary`/`apply_upscale`.
    """
    def shape_ceil(h: int, w: int) -> float:
        return math.ceil(((h * w) ** 0.5) / 64.0) * 64.0

    floor = min(shape_ceil(*(int(v) for v in entry.split('*'))) for entry in aspect_ratios)
    return floor, floor * 2.0


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
        controlnet_types=('canny', 'cpds'),
        supports_ip_adapter=True,
        supports_inpaint_engine=True,
        supports_vae_override=True,
        vae_names=None,
        performance_modes=_build_sdxl_performance_modes(),
        sampler_names=tuple(sampler_list),
        scheduler_names=tuple(scheduler_list),
        aspect_ratios=tuple(sdxl_aspect_ratios),
        native_resolution_range=_native_resolution_range(tuple(sdxl_aspect_ratios)),
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


def _build_z_image_capabilities() -> FamilyCapabilities:
    """Z-Image-Turbo: a CFG-distilled flow-matching DiT (FWDF-123/124) with a
    hand-assembled Qwen3-4B text encoder (FWDF-122/125) and a standalone
    Flux-format VAE (FWDF-121/126) -- none of which are wired through
    modules/config.py's global checkpoint/CLIP/VAE machinery the way SDXL's
    single-file checkpoint is. It has no refiner, ADM guidance, FreeU, or
    CLIP-skip concept: those are all UNet-block or CLIP-specific tricks this
    DiT+Qwen3 stack doesn't expose. adaptive_cfg and sharpness are also
    disabled: both are eps-space post-processing heuristics tuned for
    SDXL's noise parameterization, unverified against this model's
    flow-matching/velocity output.

    The ~8-9 step, low-CFG performance mode mirrors the community-documented
    ComfyUI Z-Image-Turbo workflow -- see modules/qwen3_text_encoder.py's
    docstring for the same "not independently verified against Tongyi-MAI's
    own pipeline" caveat. cfg deliberately stays non-zero (unlike some
    community workflows that use cfg=0) so supports_negative_prompt=True
    remains meaningful.

    supports_controlnet=True as of FWDF-156, scoped to PyraCanny only via
    controlnet_types=('canny',): the DiT ControlNet backport
    (`ldm_patched/ldm/lumina/controlnet.py`,
    `ldm_patched/modules/controlnet.py:ZImageControlNetPatch`) only covers
    Alibaba/Tongyi-MAI's Z-Image-Turbo-Fun-Controlnet-Union-2.1 Canny path;
    CPDS has no published DiT equivalent and is omitted from
    controlnet_types, gating both its download and its per-task processing
    in modules/async_worker.py (`_controlnet_type_supported()`). The
    per-control-type list itself has no dedicated UI consumer yet
    (FWDF-128/129/130, still open as of this ticket) -- whichever of those
    wires it into the new-UI/Gradio ImagePrompt tab must read
    controlnet_types (not just the coarse supports_controlnet flag) so
    CPDS/ImagePrompt/FaceSwap stay hidden for Z_IMAGE until they have their
    own DiT-native implementations.
    """
    turbo = PerformanceMode(
        label='Turbo',
        steps=9,
        steps_uov=9,
        cfg=None,
        lora_filename=None,
        restricted=False,
    )
    return FamilyCapabilities(
        supports_refiner=False,
        supports_adm_guidance=False,
        supports_freeu=False,
        supports_clip_skip=False,
        supports_adaptive_cfg=False,
        supports_sharpness=False,
        supports_negative_prompt=True,
        supports_controlnet=True,
        controlnet_types=('canny',),
        # FWDF-157 (parked, investigated 2026-07-11): no official IP-Adapter
        # exists for Z-Image-Turbo. extras/ip_adapter.py's patch_model()
        # (extras/ip_adapter.py:202-284) hooks a UNet's attn2 cross-attention
        # layers via literal (input/output/middle, block_id, index) triples
        # -- a convention with no DiT equivalent. Revisit when either
        # Tongyi-MAI/Alibaba PAI ships an official ip-adapter-style
        # checkpoint, or a community adapter reaches the same maturity bar
        # used for FWDF-156's ControlNet "implement" decision (first-party
        # framework documentation, not a discussion-thread workaround).
        supports_ip_adapter=False,
        supports_inpaint_engine=False,
        supports_vae_override=False,
        vae_names=None,
        performance_modes=(turbo,),
        # Euler-family samplers are documented as best-behaved for this
        # Turbo model; SDXL's dpmpp_2m_sde_gpu/karras default is untested
        # against the flow schedule and deliberately excluded.
        sampler_names=('euler', 'euler_ancestral'),
        # 'turbo' and 'align_your_steps' are excluded: both are hardcoded to
        # specific architectures in modules/sample_hijack.py (see this
        # module's docstring) and are not valid for Z-Image yet.
        scheduler_names=('normal', 'simple'),
        aspect_ratios=tuple(sdxl_aspect_ratios),
        # Derived from the same aspect_ratios list as SDXL (see the
        # aspect_ratios= line above -- Z-Image has no aspect-ratio list of
        # its own yet), so this numerically equals SDXL's (1024.0, 2048.0)
        # today. That is not a bug: Z-Image's real native resolution is a
        # community-documented ~1-megapixel bucket, same ballpark as SDXL's.
        # Once Z-Image gets its own aspect_ratios entries this recomputes
        # automatically with zero changes needed at the Vary/Upscale call
        # sites in modules/async_worker.py.
        native_resolution_range=_native_resolution_range(tuple(sdxl_aspect_ratios)),
        default_cfg=1.5,
        cfg_range=(1.0, 4.0),
        default_steps=9,
        latent_channels=16,
    )


_Z_IMAGE_CAPABILITIES = _build_z_image_capabilities()

FAMILY_CAPABILITIES: dict[ModelFamily, FamilyCapabilities] = {
    ModelFamily.SDXL: _SDXL_CAPABILITIES,
    ModelFamily.SD15: _SD15_CAPABILITIES,
    ModelFamily.Z_IMAGE: _Z_IMAGE_CAPABILITIES,
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
