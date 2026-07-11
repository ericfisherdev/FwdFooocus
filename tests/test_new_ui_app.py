"""Tests for new_ui.app FastAPI endpoints."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# args_manager calls parse_args() at import time, which chokes on pytest's
# argv.  Patch sys.argv before any project modules are imported.
_original_argv = sys.argv
sys.argv = [sys.argv[0]]

from fastapi.testclient import TestClient  # noqa: E402
from new_ui.app import app, _build_generate_args  # noqa: E402
import modules.config as config  # noqa: E402
from modules.model_family import FamilyCapabilities, ModelFamily, PerformanceMode  # noqa: E402

sys.argv = _original_argv

client = TestClient(app)


def _make_z_image_like_capabilities() -> FamilyCapabilities:
    """A synthetic capabilities descriptor with every SDXL-only flag off.

    Stands in for a real Z-Image registry entry (which does not exist yet --
    `ModelFamily.Z_IMAGE` currently falls back to the SDXL descriptor, see
    `tests/test_model_family.py::TestUnknownFallback`). Used to exercise
    `_build_generate_args`'s family-aware gating without depending on
    FWDF-123..127 landing first.
    """
    return FamilyCapabilities(
        supports_refiner=False,
        supports_adm_guidance=False,
        supports_freeu=False,
        supports_clip_skip=False,
        supports_adaptive_cfg=False,
        supports_sharpness=False,
        supports_negative_prompt=False,
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_inpaint_engine=False,
        supports_vae_override=False,
        vae_names=None,
        performance_modes=(
            PerformanceMode(label="Fast", steps=20, steps_uov=10, cfg=None, lora_filename=None, restricted=False),
        ),
        sampler_names=("euler",),
        scheduler_names=("simple",),
        aspect_ratios=("512*512",),
        default_cfg=4.0,
        cfg_range=(1.0, 10.0),
        default_steps=20,
        latent_channels=16,
    )


class TestIndexPage:
    def test_returns_html(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_contains_title(self):
        r = client.get("/")
        assert "FwdFooocus" in r.text


class TestFamilyAwareUIBindings:
    """Renders '/' and asserts the FWDF-130 capability-driven Alpine
    bindings are present -- there is no JS test runner in this repo, so
    this is the template-level check that the $store.model wiring and
    gates actually made it into the served markup."""

    def test_base_model_select_binds_to_shared_model_store(self):
        r = client.get("/")
        assert r.status_code == 200
        assert 'x-model="$store.model.baseModel"' in r.text

    def test_refiner_fields_gated_on_supports_refiner(self):
        r = client.get("/")
        assert r.text.count('x-show="$store.model.capabilities?.supports_refiner ?? true"') == 2

    def test_sharpness_gated_on_supports_sharpness(self):
        r = client.get("/")
        assert 'x-show="$store.model.capabilities?.supports_sharpness ?? true"' in r.text

    def test_clip_skip_gated_on_supports_clip_skip(self):
        r = client.get("/")
        assert 'x-show="$store.model.capabilities?.supports_clip_skip ?? true"' in r.text

    def test_vae_field_gated_on_family(self):
        r = client.get("/")
        assert 'supports_vae_override ?? true' in r.text

    def test_negative_prompt_gated_on_supports_negative_prompt(self):
        r = client.get("/")
        assert 'x-show="$store.model.capabilities?.supports_negative_prompt ?? true"' in r.text

    def test_performance_radio_driven_by_capabilities(self):
        r = client.get("/")
        assert 'x-for="mode in performanceModes"' in r.text
        assert "setPerformance(mode.label)" in r.text

    def test_aspect_ratio_select_driven_by_capabilities(self):
        r = client.get("/")
        assert 'x-for="ar in aspectRatioOptions"' in r.text


class TestConfigAPI:
    def test_returns_config(self):
        r = client.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        assert "default_model" in data
        assert "default_performance" in data
        assert "available_aspect_ratios" in data
        assert isinstance(data["available_aspect_ratios"], list)

    def test_config_has_lora_settings(self):
        r = client.get("/api/config")
        data = r.json()
        assert "default_loras_min_weight" in data
        assert "default_loras_max_weight" in data
        assert "default_max_lora_number" in data


class TestModelsAPI:
    def test_returns_model_lists(self):
        r = client.get("/api/models")
        assert r.status_code == 200
        data = r.json()
        assert "checkpoints" in data
        assert "loras" in data
        assert isinstance(data["checkpoints"], list)
        assert isinstance(data["loras"], list)

    def test_checkpoints_field_stays_a_flat_string_list(self):
        # checkpoint_families must be purely additive -- checkpoints itself
        # must not be reshaped into {name, family} objects (stores.js and
        # settings-drawer.html both treat each entry as a plain string).
        r = client.get("/api/models")
        data = r.json()
        for entry in data["checkpoints"]:
            assert isinstance(entry, str)

    def test_checkpoint_families_covers_every_checkpoint(self):
        with patch.object(config, "model_filenames", ["sdxl.safetensors", "z_image.safetensors"]), \
             patch(
                 "modules.model_family_detection.get_family",
                 side_effect=lambda filename: {
                     "sdxl.safetensors": ModelFamily.SDXL,
                     "z_image.safetensors": ModelFamily.Z_IMAGE,
                 }[filename],
             ):
            r = client.get("/api/models")
        data = r.json()
        assert data["checkpoint_families"] == {
            "sdxl.safetensors": "sdxl",
            "z_image.safetensors": "z_image",
        }


class TestCapabilitiesAPI:
    def test_defaults_to_configured_base_model(self):
        with patch("modules.model_family_detection.get_family", return_value=ModelFamily.SDXL) as mock_get_family:
            r = client.get("/api/capabilities")
        assert r.status_code == 200
        mock_get_family.assert_called_once_with(config.default_base_model_name)

    def test_returns_capabilities_for_explicit_sdxl_checkpoint(self):
        with patch("modules.model_family_detection.get_family", return_value=ModelFamily.SDXL), \
             patch.object(config, "model_filenames", ["sdxl_base.safetensors"]):
            r = client.get("/api/capabilities", params={"checkpoint": "sdxl_base.safetensors"})
        assert r.status_code == 200
        data = r.json()
        assert data["family"] == "sdxl"
        assert data["supports_refiner"] is True
        assert data["supports_adm_guidance"] is True
        assert "dpmpp_2m_sde_gpu" in data["sampler_names"]
        assert isinstance(data["performance_modes"], list)
        assert data["performance_modes"][0]["label"]

    def test_returns_capabilities_for_z_image_checkpoint(self):
        synthetic = _make_z_image_like_capabilities()
        with patch("modules.model_family_detection.get_family", return_value=ModelFamily.Z_IMAGE), \
             patch("modules.model_family.get_capabilities", return_value=synthetic), \
             patch.object(config, "model_filenames", ["z_image.safetensors"]):
            r = client.get("/api/capabilities", params={"checkpoint": "z_image.safetensors"})
        assert r.status_code == 200
        data = r.json()
        assert data["family"] == "z_image"
        assert data["supports_refiner"] is False
        assert data["supports_adm_guidance"] is False
        assert data["sampler_names"] == ["euler"]

    def test_family_is_a_plain_string_not_an_enum_repr(self):
        with patch("modules.model_family_detection.get_family", return_value=ModelFamily.UNKNOWN), \
             patch.object(config, "model_filenames", ["unknown.safetensors"]):
            r = client.get("/api/capabilities", params={"checkpoint": "unknown.safetensors"})
        data = r.json()
        assert data["family"] == "unknown"


class TestStylesAPI:
    def test_returns_styles(self):
        r = client.get("/api/styles")
        assert r.status_code == 200
        data = r.json()
        assert "styles" in data
        assert isinstance(data["styles"], list)
        assert len(data["styles"]) > 0


class TestLoRALibraryAPI:
    def test_library_data_returns_list(self):
        r = client.get("/api/lora-library-data")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_scan_status_returns_dict(self):
        r = client.get("/api/lora-library-scan-status")
        assert r.status_code == 200
        data = r.json()
        assert "is_scanning" in data
        assert "total_indexed" in data

    def test_trigger_words_requires_filename(self):
        r = client.get("/api/lora-trigger-words")
        assert r.status_code == 422  # FastAPI validation error

    def test_trigger_words_with_filename(self):
        r = client.get("/api/lora-trigger-words", params={"filename": "nonexistent.safetensors"})
        assert r.status_code == 200
        data = r.json()
        assert data["filename"] == "nonexistent.safetensors"
        assert isinstance(data["trigger_words"], list)


class TestHeartbeatAPI:
    def test_heartbeat_returns_ok(self):
        r = client.post("/api/heartbeat")
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_heartbeat_rejects_get(self):
        r = client.get("/api/heartbeat")
        assert r.status_code == 405


class TestBuildGenerateArgsFamilyAware:
    """Exercises `_build_generate_args`'s family-conditional gating directly.

    Zeroes out the three variable-length padding sections (LoRA/ControlNet/
    enhance-tab slots) so every fixed field lands at a stable, hand-countable
    index -- the indices below assume `default_max_lora_number ==
    default_controlnet_image_count == default_enhance_tabs == 0` and must be
    recounted from `_build_generate_args`'s `args = [...]` literal (new_ui/
    app.py) if a field is inserted, removed, or reordered there.
    """

    IDX_NEGATIVE_PROMPT = 2
    IDX_PERFORMANCE_SELECTION = 4
    IDX_ASPECT_RATIOS_SELECTION = 5
    IDX_SHARPNESS = 10
    IDX_CFG_SCALE = 11
    IDX_REFINER_MODEL_NAME = 13
    IDX_REFINER_SWITCH = 14
    IDX_ADM_SCALER_POSITIVE = 27
    IDX_ADM_SCALER_NEGATIVE = 28
    IDX_ADM_SCALER_END = 29
    IDX_ADAPTIVE_CFG = 30
    IDX_CLIP_SKIP = 31
    IDX_SAMPLER_NAME = 32
    IDX_SCHEDULER_NAME = 33
    IDX_VAE_NAME = 34
    IDX_OVERWRITE_STEP = 35
    IDX_FREEU_ENABLED = 49

    def _zero_length_padding_patches(self) -> list:
        return [
            patch.object(config, "default_max_lora_number", 0),
            patch.object(config, "default_controlnet_image_count", 0),
            patch.object(config, "default_enhance_tabs", 0),
        ]

    def _build(self, body, family=ModelFamily.SDXL, capabilities=None) -> list:
        patches = self._zero_length_padding_patches()
        patches.append(patch("modules.model_family_detection.get_family", return_value=family))
        if capabilities is not None:
            patches.append(patch("modules.model_family.get_capabilities", return_value=capabilities))
        for p in patches:
            p.start()
        try:
            return _build_generate_args(body)
        finally:
            for p in patches:
                p.stop()

    def test_sdxl_family_matches_historic_hardcoded_defaults(self):
        # Regression fixture: an empty body's resolved values for an SDXL
        # checkpoint must equal today's pre-FWDF-128 hardcoded defaults.
        args = self._build({}, family=ModelFamily.SDXL)

        assert args[self.IDX_NEGATIVE_PROMPT] == ""
        assert args[self.IDX_PERFORMANCE_SELECTION] == config.default_performance
        assert args[self.IDX_ASPECT_RATIOS_SELECTION] == config.default_aspect_ratio
        assert args[self.IDX_SHARPNESS] == config.default_sample_sharpness
        assert args[self.IDX_CFG_SCALE] == config.default_cfg_scale
        assert args[self.IDX_REFINER_MODEL_NAME] == config.default_refiner_model_name
        assert args[self.IDX_REFINER_SWITCH] == config.default_refiner_switch
        assert args[self.IDX_ADM_SCALER_POSITIVE] == 1.5
        assert args[self.IDX_ADM_SCALER_NEGATIVE] == 0.8
        assert args[self.IDX_ADM_SCALER_END] == 0.3
        assert args[self.IDX_ADAPTIVE_CFG] == 7.0
        assert args[self.IDX_CLIP_SKIP] == 2
        assert args[self.IDX_SAMPLER_NAME] == config.default_sampler
        assert args[self.IDX_SCHEDULER_NAME] == config.default_scheduler
        assert args[self.IDX_VAE_NAME] == "Default (model)"
        assert args[self.IDX_FREEU_ENABLED] is False

    def test_unknown_family_behaves_identically_to_sdxl(self):
        # UNKNOWN's FamilyCapabilities is the exact same object as SDXL's
        # (modules/model_family.py), so an unrecognized checkpoint must
        # produce byte-for-byte identical args to an explicit SDXL family.
        body = {"cfg_scale": 5.5, "sharpness": 3.3, "refiner_switch": 0.4}
        sdxl_args = self._build(body, family=ModelFamily.SDXL)
        unknown_args = self._build(body, family=ModelFamily.UNKNOWN)
        assert sdxl_args == unknown_args

    def test_sdxl_family_honors_explicit_request_values(self, monkeypatch):
        monkeypatch.setattr(config, "model_filenames", ["refiner.safetensors"])
        monkeypatch.setattr(config, "vae_filenames", ["custom.vae.safetensors"])
        body = {
            "negative_prompt": "blurry",
            "refiner_model_name": "refiner.safetensors",
            "refiner_switch": 0.4,
            "adm_scaler_positive": 2.0,
            "clip_skip": 4,
            "freeu_enabled": True,
            "vae_name": "custom.vae.safetensors",
        }
        args = self._build(body, family=ModelFamily.SDXL)

        assert args[self.IDX_NEGATIVE_PROMPT] == "blurry"
        assert args[self.IDX_REFINER_MODEL_NAME] == "refiner.safetensors"
        assert args[self.IDX_REFINER_SWITCH] == 0.4
        assert args[self.IDX_ADM_SCALER_POSITIVE] == 2.0
        assert args[self.IDX_CLIP_SKIP] == 4
        assert args[self.IDX_FREEU_ENABLED] is True
        assert args[self.IDX_VAE_NAME] == "custom.vae.safetensors"

    def test_z_image_family_forces_sdxl_only_fields_to_disable_values(self):
        synthetic = _make_z_image_like_capabilities()
        body = {
            "negative_prompt": "ignored",
            "refiner_model_name": "some_refiner.safetensors",
            "refiner_switch": 0.5,
            "adm_scaler_positive": 2.0,
            "adm_scaler_negative": 1.5,
            "adm_scaler_end": 0.9,
            "adaptive_cfg": 3.3,
            "clip_skip": 5,
            "sharpness": 9.9,
            "freeu_enabled": True,
            "vae_name": "custom_vae.safetensors",
            "cfg_scale": 6.0,
        }
        args = self._build(body, family=ModelFamily.Z_IMAGE, capabilities=synthetic)

        assert args[self.IDX_NEGATIVE_PROMPT] == ""
        assert args[self.IDX_REFINER_MODEL_NAME] == "None"
        assert args[self.IDX_REFINER_SWITCH] == 0.0
        assert args[self.IDX_ADM_SCALER_POSITIVE] == 1.0
        assert args[self.IDX_ADM_SCALER_NEGATIVE] == 1.0
        assert args[self.IDX_ADM_SCALER_END] == 0.0
        assert args[self.IDX_CFG_SCALE] == 6.0
        assert args[self.IDX_ADAPTIVE_CFG] == 6.0  # forced equal to the resolved cfg_scale
        assert args[self.IDX_CLIP_SKIP] == 1
        assert args[self.IDX_SHARPNESS] == 0.0
        assert args[self.IDX_FREEU_ENABLED] is False
        assert args[self.IDX_VAE_NAME] == "Default (model)"

    def test_omitted_cfg_scale_uses_family_default(self):
        synthetic = _make_z_image_like_capabilities()
        args = self._build({}, family=ModelFamily.Z_IMAGE, capabilities=synthetic)
        assert args[self.IDX_CFG_SCALE] == synthetic.default_cfg

    def test_z_image_family_falls_back_to_its_own_choice_lists(self):
        synthetic = _make_z_image_like_capabilities()
        body = {
            "sampler_name": "dpmpp_2m_sde_gpu",  # SDXL-only, not in the synthetic family's list
            "scheduler_name": "karras",          # SDXL-only, not in the synthetic family's list
            "performance_selection": "Quality",  # not in the synthetic family's performance modes
            "aspect_ratios_selection": "1024*1024",  # not in the synthetic family's aspect ratios
        }
        args = self._build(body, family=ModelFamily.Z_IMAGE, capabilities=synthetic)

        assert args[self.IDX_SAMPLER_NAME] == "euler"
        assert args[self.IDX_SCHEDULER_NAME] == "simple"
        # 'Fast' is not a legacy flags.Performance member: AsyncTask would
        # raise ValueError on it, so the builder maps it to 'Speed' and
        # carries the mode's step count via overwrite_step.
        assert args[self.IDX_PERFORMANCE_SELECTION] == "Speed"
        assert args[self.IDX_OVERWRITE_STEP] == 20
        assert args[self.IDX_ASPECT_RATIOS_SELECTION] == "512*512"


class TestCheckpointNameBoundary:
    """Checkpoint names from requests flow into filesystem path resolution;
    only configured checkpoints may pass (CodeQL py/path-injection)."""

    def test_capabilities_rejects_unknown_checkpoint(self):
        r = client.get("/api/capabilities", params={"checkpoint": "../../../etc/passwd"})
        assert r.status_code == 404

    def test_capabilities_accepts_configured_checkpoint(self, monkeypatch):
        monkeypatch.setattr(config, "model_filenames", ["known.safetensors"])
        from unittest.mock import MagicMock
        import modules.model_family_detection as mfd
        from modules.model_family import ModelFamily
        monkeypatch.setattr(mfd, "get_family", MagicMock(return_value=ModelFamily.SDXL))
        r = client.get("/api/capabilities", params={"checkpoint": "known.safetensors"})
        assert r.status_code == 200

    def test_generate_args_falls_back_for_unknown_base_model(self, monkeypatch):
        from unittest.mock import MagicMock
        import new_ui.app as app_module
        import modules.model_family_detection as mfd
        spy = MagicMock(side_effect=mfd.get_family)
        monkeypatch.setattr(app_module, "config", config)
        monkeypatch.setattr(mfd, "get_family", spy)
        args = app_module._build_generate_args({"base_model_name": "../../evil"})
        assert spy.call_args[0][0] == config.default_base_model_name
