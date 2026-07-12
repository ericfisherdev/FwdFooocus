"""Pure decision logic for family-gated Gradio controls (FWDF-129).

`webui.py` is import-heavy Gradio and cannot be imported in a gradio-less
test environment, so every show/hide/interactive/choice-swap rule for the
`base_model.change` handler (and the parts of `performance_selection.change`
that now overlap with it) lives here instead, where it can be unit tested
directly. Every function in this module returns plain data (`bool` / `tuple`
/ `str` / `float`) -- `webui.py` is the only place that wraps these return
values in `gr.update(...)`.

Two independent Gradio event handlers drive some of the same controls
(`base_model.change` for family support, `performance_selection.change` for
LoRA-accelerated performance modes). Whichever fires last must not clobber
the other's decision, so both handlers call the *same* function here with
the *same* two source-of-truth values (the current performance value and
the current base model's `FamilyCapabilities`) -- the result is therefore
identical regardless of firing order, by construction rather than by care
taken at each call site.
"""

from collections.abc import Callable, Iterable, Sequence

from modules.flags import Performance
from modules.model_family import FamilyCapabilities


def performance_restricted(performance: str) -> bool:
    """True for the LoRA-accelerated performance modes with fewer supported features."""
    return Performance.has_restricted_features(performance)


def restricted_interactive(*, supported: bool, performance: str) -> bool:
    """Most-restrictive-wins union for controls both handlers drive.

    A control is interactive only when the current family supports it AND
    the current performance mode does not restrict it. Keyword-only so a
    `bool`/`str` pair can never be transposed at a call site.
    """
    return supported and not performance_restricted(performance)


def negative_prompt_visible(caps: FamilyCapabilities, performance: str) -> bool:
    """Union rule for `negative_prompt`, the one control both handlers show/hide.

    Delegates to `restricted_interactive` so the union rule lives in exactly
    one place — by construction, not by call-site care.
    """
    return restricted_interactive(supported=caps.supports_negative_prompt, performance=performance)


def refiner_switch_visible(caps: FamilyCapabilities, refiner_model_value: str) -> bool:
    """`refiner_switch` is shown only when the family supports a refiner AND a
    refiner model is actually selected (the pre-existing `refiner_model.change`
    rule) -- family support is combined with, not overridden by, that rule.
    """
    return caps.supports_refiner and refiner_model_value != 'None'


def choice_list_and_value(
    choices: Iterable[str], current_value: str, fallback_value: str | None
) -> tuple[tuple[str, ...], str]:
    """Resolve the `(choices, value)` pair for a swapped Dropdown/Radio.

    `current_value` is kept when still present in `choices`; otherwise this
    falls back to `fallback_value` if that is itself valid, else the first
    entry of `choices`. An empty `choices` has nothing valid to fall back
    to, so `current_value` passes through unchanged rather than raising.
    """
    choices = tuple(choices)
    if current_value in choices:
        return choices, current_value
    if fallback_value is not None and fallback_value in choices:
        return choices, fallback_value
    if choices:
        return choices, choices[0]
    return choices, current_value


def performance_choices_and_value(caps: FamilyCapabilities, current_value: str) -> tuple[tuple[str, ...], str]:
    """Performance-radio choices/value for the given family.

    Falls back to the mode whose `steps` matches `caps.default_steps` (the
    family's own notion of "default"), not a hardcoded label, so this stays
    correct for any family added to the registry later.
    """
    choices = tuple(mode.label for mode in caps.performance_modes)
    default_label = next((mode.label for mode in caps.performance_modes if mode.steps == caps.default_steps), None)
    return choice_list_and_value(choices, current_value, default_label)


def aspect_ratio_choices_and_value(
    caps: FamilyCapabilities,
    current_value: str,
    add_ratio: Callable[[str], str],
    unrestricted_aspect_ratios: Sequence[str],
    configured_aspect_ratios: Sequence[str],
) -> tuple[tuple[str, ...], str]:
    """Aspect-ratio radio choices/value for the given family.

    `FamilyCapabilities.aspect_ratios` (FWDF-117) is always the hardcoded
    framework default (`modules.flags.sdxl_aspect_ratios`) today -- unlike
    `vae_names`, it has no `None`-means-"no family restriction" sentinel.
    When a family's declared list is value-equal to that hardcoded default,
    this prefers the user's actually-configured `available_aspect_ratios`
    (which may have been customized in `config.txt`) instead of silently
    discarding that customization on every `base_model` change; a family
    that declares a genuinely different/curated list is still honored as
    an intentional restriction.

    `add_ratio` is `modules.config.add_ratio`, injected rather than imported
    here to keep this module's only dependency direction explicit (and the
    function trivially testable with a stub formatter).
    """
    if tuple(caps.aspect_ratios) == tuple(unrestricted_aspect_ratios):
        raw_ratios = configured_aspect_ratios
    else:
        raw_ratios = caps.aspect_ratios
    choices = tuple(add_ratio(ratio) for ratio in raw_ratios)
    return choice_list_and_value(choices, current_value, choices[0] if choices else None)


def sampler_choices_and_value(
    caps: FamilyCapabilities, current_value: str, configured_default: str
) -> tuple[tuple[str, ...], str]:
    """Sampler dropdown choices/value for the given family."""
    return choice_list_and_value(caps.sampler_names, current_value, configured_default)


def scheduler_choices_and_value(
    caps: FamilyCapabilities, current_value: str, configured_default: str
) -> tuple[tuple[str, ...], str]:
    """Scheduler dropdown choices/value for the given family."""
    return choice_list_and_value(caps.scheduler_names, current_value, configured_default)


def vae_state(
    caps: FamilyCapabilities,
    current_value: str,
    all_vae_names: Sequence[str],
    default_vae_label: str,
) -> tuple[bool, bool, tuple[str, ...] | None, str | None]:
    """`(visible, interactive, choices, value)` for the VAE dropdown.

    `supports_vae_override=False` hides the control entirely without
    clearing its value (`choices`/`value` are returned as `None`, which
    `webui.py` must not forward to `gr.update()`, so re-selecting an SDXL
    checkpoint restores the prior selection as-is). `vae_names=None` means
    "no per-family restriction", restoring the full global `all_vae_names`
    listing; a populated tuple curates the choice list to that family's
    compatible VAEs instead.
    """
    if not caps.supports_vae_override:
        return False, False, None, None
    choices = (default_vae_label,) + (caps.vae_names if caps.vae_names is not None else tuple(all_vae_names))
    _, value = choice_list_and_value(choices, current_value, default_vae_label)
    return True, True, choices, value


def guidance_scale_range_and_value(caps: FamilyCapabilities, current_value: float) -> tuple[float, float, float]:
    """`(minimum, maximum, value)` for the guidance-scale slider.

    Clamps `current_value` into the family's range instead of resetting it
    to `caps.default_cfg`, preserving the user's choice whenever it is
    still valid for the new family.
    """
    minimum, maximum = caps.cfg_range
    value = min(max(current_value, minimum), maximum)
    return minimum, maximum, value
