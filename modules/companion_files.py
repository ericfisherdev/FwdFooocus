"""Registry of companion files (text encoder + VAE) for multi-file model families.

Z-Image and Krea 2 checkpoints carry only diffusion weights; their text
encoder and VAE ship as separate downloads. This module is the Open/Closed
extension point for that: supporting another multi-file family means adding one
`FamilyCompanions` entry to `COMPANIONS`, not another `if family == ...:` chain
in `modules/default_pipeline.py`. `get_companions()` returns `None` for
single-file families (SDXL / SD15 / UNKNOWN).

Nothing here performs I/O at import time: the entries only hold callables.
"""

import urllib.error
from dataclasses import dataclass
from typing import Callable

import modules.config
from modules.model_family import ModelFamily
from modules.model_loader import resolve_hf_mirror


class MissingCompanionFileError(RuntimeError):
    """A companion file is absent and could not be downloaded.

    Carries the file name, the directory it belongs in and the source URL so
    the user can either place the file by hand or fix the network / mirror.
    """

    def __init__(self, file_name: str, directory: str, url: str):
        self.file_name = file_name
        self.directory = directory
        self.url = url
        super().__init__(
            f'Companion file "{file_name}" is missing from "{directory}" and could not be '
            f'downloaded from "{url}". Place the file in that directory, or fix the network '
            f'connection / HF_MIRROR setting and retry.'
        )


@dataclass(frozen=True, slots=True)
class FamilyCompanions:
    """Path lookups (cheap, no I/O) and verifying downloads for one family's
    companion files.

    `ensure_*` raises `MissingCompanionFileError` when the file cannot be
    obtained.
    """

    text_encoder_path: Callable[[], str]
    ensure_text_encoder: Callable[[], str]
    vae_path: Callable[[], str]
    ensure_vae: Callable[[], str]


def _ensure_present(file_name: str, directory: str, url: str, download: Callable[[], str]) -> Callable[[], str]:
    """Wrap a download helper so every acquisition failure surfaces as
    `MissingCompanionFileError`.

    Catches `RuntimeError` (hash/size failure from `load_file_from_url`),
    `urllib.error.URLError` / `HTTPError` and `OSError` (from
    `torch.hub.download_url_to_file` or the filesystem). `URLError` is an
    `OSError` subclass; it is listed for clarity. The error names the URL
    actually requested, i.e. after the `HF_MIRROR` rewrite.
    """

    def ensure() -> str:
        try:
            return download()
        except MissingCompanionFileError:
            raise
        except (RuntimeError, urllib.error.URLError, OSError) as e:
            raise MissingCompanionFileError(file_name, directory, resolve_hf_mirror(url)) from e

    return ensure


def _build_companions(
        *,
        text_encoder_file: str,
        text_encoder_url: str,
        text_encoder_path: Callable[[], str],
        download_text_encoder: Callable[[], str],
        vae_file: str,
        vae_url: str,
        vae_path: Callable[[], str],
        download_vae: Callable[[], str],
) -> FamilyCompanions:
    return FamilyCompanions(
        text_encoder_path=text_encoder_path,
        ensure_text_encoder=_ensure_present(
            text_encoder_file, modules.config.path_text_encoders, text_encoder_url, download_text_encoder),
        vae_path=vae_path,
        ensure_vae=_ensure_present(vae_file, modules.config.path_vae, vae_url, download_vae),
    )


# The callables delegate late-bound through `modules.config` (lambdas) so they
# resolve the config helpers at call time, as the pipeline's direct calls do.
# Krea 2's Raw and Turbo variants are separate model families (see
# modules.model_family) that load the same text encoder and VAE, so both
# families share one entry.
_KREA2_COMPANIONS = _build_companions(
    text_encoder_file=modules.config.KREA2_TEXT_ENCODER_FILENAME,
    text_encoder_url=modules.config.KREA2_TEXT_ENCODER_URL,
    text_encoder_path=lambda: modules.config.krea2_text_encoder_path(),
    download_text_encoder=lambda: modules.config.downloading_krea2_text_encoder(),
    vae_file=modules.config.QWEN_IMAGE_VAE_FILENAME,
    vae_url=modules.config.QWEN_IMAGE_VAE_URL,
    vae_path=lambda: modules.config.qwen_image_vae_path(),
    download_vae=lambda: modules.config.downloading_qwen_image_vae(),
)

COMPANIONS: dict[ModelFamily, FamilyCompanions] = {
    ModelFamily.Z_IMAGE: _build_companions(
        text_encoder_file=modules.config.Z_IMAGE_TEXT_ENCODER_FILENAME,
        text_encoder_url=modules.config.Z_IMAGE_TEXT_ENCODER_URL,
        text_encoder_path=lambda: modules.config.z_image_text_encoder_path(),
        download_text_encoder=lambda: modules.config.downloading_z_image_text_encoder(),
        vae_file=modules.config.Z_IMAGE_VAE_FILENAME,
        vae_url=modules.config.Z_IMAGE_VAE_URL,
        vae_path=lambda: modules.config.z_image_vae_path(),
        download_vae=lambda: modules.config.downloading_z_image_vae(),
    ),
    ModelFamily.KREA2_RAW: _KREA2_COMPANIONS,
    ModelFamily.KREA2_TURBO: _KREA2_COMPANIONS,
}


def get_companions(family: ModelFamily) -> FamilyCompanions | None:
    """Return the companion-file descriptor for `family`, or `None` for
    single-file families (SDXL, SD15, UNKNOWN) whose checkpoint carries its own
    text encoder and VAE."""
    return COMPANIONS.get(family)
