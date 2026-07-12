"""
FwdFooocus New UI - FastAPI Application

Runs alongside the existing Gradio server on a separate port.
Serves the Alpine.js/HTMX/GSAP frontend via Jinja2 templates.
Shares the same backend modules (async_worker, config, lora_metadata).
"""

import asyncio
import base64
import logging
import os
from pathlib import Path

from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import modules.config as config
import modules.lora_metadata as lora_metadata
from modules.heartbeat import update_heartbeat

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

app = FastAPI(title="FwdFooocus", docs_url=None, redoc_url=None)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "base.html")


# ---------------------------------------------------------------------------
# LoRA Library APIs (migrated from Gradio routes in webui.py)
# ---------------------------------------------------------------------------

@app.post("/api/lora-library-rescan")
async def lora_library_rescan():
    """Trigger a rescan of the LoRA library."""
    scanner = lora_metadata.get_scanner()
    if scanner.is_scanning:
        return {"success": False, "error": "Scan already in progress"}
    scanner.start_scan(blocking=False)
    return {"success": True}


@app.get("/api/lora-library-scan-status")
async def lora_library_scan_status():
    """Get the current scan status."""
    scanner = lora_metadata.get_scanner()
    stats = scanner.scan_stats
    return {
        "is_scanning": stats["is_scanning"],
        "scan_complete": stats["scan_complete"],
        "files_scanned": stats["files_scanned"],
        "files_failed": stats["files_failed"],
        "total_indexed": stats["total_indexed"],
        "elapsed_time": stats["elapsed_time"],
    }


@app.get("/api/lora-library-data")
async def lora_library_data():
    """Get all LoRA metadata for the library/picker."""
    return lora_metadata.get_all_library_data()


@app.get("/api/lora-trigger-words")
async def lora_trigger_words(filename: str = Query(..., description="LoRA filename or relative path")):
    """Get trigger words for a specific LoRA."""
    trigger_words = lora_metadata.get_trigger_words_for_filename(filename)
    return {"filename": filename, "trigger_words": trigger_words}


# ---------------------------------------------------------------------------
# Heartbeat (migrated from Gradio route in webui.py)
# ---------------------------------------------------------------------------

@app.post("/api/heartbeat")
async def heartbeat_ping():
    """Receive a heartbeat ping from the browser client."""
    update_heartbeat()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Config & Model Data APIs
# ---------------------------------------------------------------------------

@app.get("/api/config")
async def get_config():
    """Return UI-relevant config values."""
    return {
        "default_model": config.default_base_model_name,
        "default_refiner": config.default_refiner_model_name,
        "default_refiner_switch": config.default_refiner_switch,
        "default_performance": config.default_performance,
        "default_aspect_ratio": config.default_aspect_ratio,
        "available_aspect_ratios": config.available_aspect_ratios,
        "default_image_number": config.default_image_number,
        "max_image_number": config.default_max_image_number,
        "default_output_format": config.default_output_format,
        "default_prompt": config.default_prompt,
        "default_prompt_negative": config.default_prompt_negative,
        "default_styles": config.default_styles,
        "default_cfg_scale": config.default_cfg_scale,
        "default_sample_sharpness": config.default_sample_sharpness,
        "default_sampler": config.default_sampler,
        "default_scheduler": config.default_scheduler,
        "default_loras": config.default_loras,
        "default_loras_min_weight": config.default_loras_min_weight,
        "default_loras_max_weight": config.default_loras_max_weight,
        "default_max_lora_number": config.default_max_lora_number,
    }


@app.get("/api/models")
def get_models():
    # Plain def (not async): FastAPI runs sync routes in a threadpool, so the
    # blocking os.stat/safetensors header reads in get_family() cannot stall
    # the event loop on a cache-cold call.
    """Return available checkpoints, refiners, and VAEs.

    `checkpoint_families` is an additive sibling map (filename -> ModelFamily
    value) alongside `checkpoints`; the `checkpoints` list itself stays a
    flat list of filename strings since the new-UI frontend (stores.js,
    settings-drawer.html) already treats each entry as a plain string.
    """
    from modules.model_family_detection import get_family

    checkpoint_families = {
        filename: get_family(filename).value for filename in config.model_filenames
    }
    return {
        "checkpoints": config.model_filenames,
        "loras": config.lora_filenames,
        "checkpoint_families": checkpoint_families,
    }


@app.get("/api/styles")
async def get_styles():
    """Return available style names."""
    from modules.sdxl_styles import legal_style_names
    return {"styles": legal_style_names}


@app.get("/api/samplers")
async def get_samplers():
    """Return available sampler and scheduler names.

    Predates the model-family capability registry and reflects only the
    global SDXL/default list. `GET /api/capabilities` is the per-checkpoint
    source of truth for sampler/scheduler/aspect-ratio/performance-mode
    choices going forward; this endpoint is kept for backward compatibility.
    """
    from modules.flags import sampler_list, scheduler_list
    return {"samplers": sampler_list, "schedulers": scheduler_list}


@app.get("/api/capabilities")
def get_capabilities_endpoint(
    checkpoint: str | None = Query(
        default=None,
        description="Checkpoint filename; defaults to the configured base model",
    )
):
    """Return the FamilyCapabilities registry entry for a checkpoint's model family.

    Serializes the frozen `FamilyCapabilities` dataclass verbatim (no field
    re-declaration here -- the registry in `modules.model_family` is the
    single source of truth) plus a `family` key with the resolved
    `ModelFamily`'s string value.
    """
    import dataclasses

    from modules.model_family import get_capabilities
    from modules.model_family_detection import get_family

    resolved_checkpoint = checkpoint or config.default_base_model_name
    # The filename flows into filesystem path resolution (get_family reads
    # the safetensors header) — only known, configured checkpoints may pass,
    # so a crafted query string cannot probe arbitrary paths.
    if resolved_checkpoint != config.default_base_model_name \
            and resolved_checkpoint not in config.model_filenames:
        raise HTTPException(status_code=404, detail="unknown checkpoint")
    family = get_family(resolved_checkpoint)
    capabilities = get_capabilities(family)
    return {"family": family.value, **dataclasses.asdict(capabilities)}


# ---------------------------------------------------------------------------
# Generation — POST /api/generate
# ---------------------------------------------------------------------------

def _validated_choice(value, allowed: tuple, fallback) -> object:
    """Resolve a request value against a family's allowed-choices tuple.

    Falls back to `fallback` (the global `config.default_*` value) when
    `value` isn't in `allowed`; falls back further to the family's own
    first allowed entry when even `fallback` isn't valid for this family
    (only reachable for a family whose registry entry restricts a list
    below the global default -- none do yet, but this keeps the resolver
    correct once one does).
    """
    if value in allowed:
        return value
    if fallback in allowed:
        return fallback
    return allowed[0] if allowed else fallback


def _validated_or_default(value, allowed: tuple, default) -> object:
    """Resolve a request value, honoring an absent value's `default` verbatim.

    Unlike `_validated_choice`, this never re-validates `default` against
    `allowed`: `config.default_aspect_ratio` is reassigned to a decorated
    display label by `modules.config.add_ratio()` at config-load time (e.g.
    `'1152×896 <span ...> 9:7</span>'`), which never literally appears
    in `FamilyCapabilities.aspect_ratios` (the plain `'WIDTH*HEIGHT'` list
    from `modules/flags.py`). Re-validating it would silently replace a
    correctly configured default with the family's first list entry on
    every request that omits the field. An explicitly *requested* value
    that fails family validation still falls back to the family's own
    first allowed entry.
    """
    if value is None:
        return default
    return value if value in allowed else (allowed[0] if allowed else default)


def _resolve_performance_selection(value, caps, fallback) -> str:
    """Resolve `performance_selection` against a family's performance modes.

    Same precedence as `_validated_choice`, but the final fallback (when
    neither the request nor the global default is valid for this family)
    is the mode whose `steps` matches `caps.default_steps` -- the family's
    own documented default -- rather than an arbitrary first entry.
    """
    labels = tuple(mode.label for mode in caps.performance_modes)
    if value in labels:
        return value
    if fallback in labels:
        return fallback
    for mode in caps.performance_modes:
        if mode.steps == caps.default_steps:
            return mode.label
    return labels[0] if labels else fallback


def _build_generate_args(body: dict) -> list:
    """
    Build the positional args list that AsyncTask.__init__ expects.

    The args list is consumed via reverse() + pop() so we build it
    in the same order that webui.py's generate_clicked() does.
    Params not exposed by the new UI yet get sensible defaults.

    Resolves the checkpoint's `ModelFamily` once and gates every SDXL-only
    field through `FamilyCapabilities` (`modules.model_family`) so families
    lacking a capability (refiner, ADM guidance, CLIP skip, adaptive CFG,
    sharpness, FreeU, negative prompt, VAE override) get that field's
    documented disable value instead of the request body's value. An
    unrecognized checkpoint resolves to `ModelFamily.UNKNOWN`, whose
    capabilities are identical (by object identity) to SDXL's, so behavior
    is unchanged for both SDXL and UNKNOWN checkpoints.
    """
    from modules.config import (
        default_max_lora_number, default_controlnet_image_count,
        default_enhance_tabs,
    )
    from modules.flags import disabled
    from modules.model_family import get_capabilities
    from modules.model_family_detection import get_family

    requested_base_model = body.get("base_model_name", config.default_base_model_name)
    # Same path-injection boundary as /api/capabilities: unknown names fall
    # back to the configured default instead of reaching path resolution.
    if requested_base_model != config.default_base_model_name \
            and requested_base_model not in config.model_filenames:
        requested_base_model = config.default_base_model_name
    family = get_family(requested_base_model)
    caps = get_capabilities(family)

    negative_prompt = body.get("negative_prompt", "") if caps.supports_negative_prompt else ""
    performance_selection = _resolve_performance_selection(
        body.get("performance_selection", body.get("performance")), caps, config.default_performance
    )
    aspect_ratios_selection = _validated_or_default(
        body.get("aspect_ratios_selection"), caps.aspect_ratios, config.default_aspect_ratio
    )
    sharpness = float(body.get("sharpness", config.default_sample_sharpness)) if caps.supports_sharpness else 0.0
    cfg_scale = float(body.get("cfg_scale", caps.default_cfg))
    refiner_model_name = (
        body.get("refiner_model_name", config.default_refiner_model_name) if caps.supports_refiner else "None"
    )
    refiner_switch = (
        float(body.get("refiner_switch", config.default_refiner_switch)) if caps.supports_refiner else 0.0
    )
    adm_scaler_positive = float(body.get("adm_scaler_positive", 1.5)) if caps.supports_adm_guidance else 1.0
    adm_scaler_negative = float(body.get("adm_scaler_negative", 0.8)) if caps.supports_adm_guidance else 1.0
    adm_scaler_end = float(body.get("adm_scaler_end", 0.3)) if caps.supports_adm_guidance else 0.0
    adaptive_cfg = float(body.get("adaptive_cfg", 7.0)) if caps.supports_adaptive_cfg else cfg_scale
    clip_skip = int(body.get("clip_skip", 2)) if caps.supports_clip_skip else 1
    sampler_name = _validated_choice(body.get("sampler_name"), caps.sampler_names, config.default_sampler)
    scheduler_name = _validated_choice(body.get("scheduler_name"), caps.scheduler_names, config.default_scheduler)
    vae_name = body.get("vae_name", "Default (model)") if caps.supports_vae_override else "Default (model)"
    freeu_enabled = body.get("freeu_enabled", False) if caps.supports_freeu else False

    loras_input = body.get("loras", [])
    # Pad to default_max_lora_number slots: (enabled, filename, weight)
    lora_args = []
    for i in range(default_max_lora_number):
        if i < len(loras_input):
            entry = loras_input[i]
            lora_args.extend([True, entry.get("filename", "None"), float(entry.get("weight", 1.0))])
        else:
            lora_args.extend([False, "None", 1.0])

    # ControlNet image slots (all empty for now)
    cn_args = []
    for _ in range(default_controlnet_image_count):
        cn_args.extend([None, 0.5, 1.0, disabled])  # img, stop, weight, type

    # Enhance tabs (all disabled for now)
    enhance_args = []
    for _ in range(default_enhance_tabs):
        enhance_args.extend([
            False,   # enhance_enabled
            '',      # enhance_mask_dino_prompt_text
            '',      # enhance_prompt
            '',      # enhance_negative_prompt
            'u2net', # enhance_mask_model
            'full',  # enhance_mask_cloth_category
            'sam_vit_b_01ec64', # enhance_mask_sam_model
            0.25,    # enhance_mask_text_threshold
            0.3,     # enhance_mask_box_threshold
            0,       # enhance_mask_sam_max_detections
            False,   # enhance_inpaint_disable_initial_latent
            'None',  # enhance_inpaint_engine
            1.0,     # enhance_inpaint_strength
            0.618,   # enhance_inpaint_respective_field
            0,       # enhance_inpaint_erode_or_dilate
            False,   # enhance_mask_invert
        ])

    args = [
        body.get("generate_image_grid", False),
        body.get("prompt", ""),
        negative_prompt,
        body.get("style_selections", []),
        performance_selection,
        aspect_ratios_selection,
        max(1, min(int(body.get("image_number", config.default_image_number)), config.default_max_image_number)),
        body.get("output_format", "png"),
        int(body.get("seed", -1)),
        body.get("read_wildcards_in_order", False),
        sharpness,
        cfg_scale,
        requested_base_model,
        refiner_model_name,
        refiner_switch,
        *lora_args,
        body.get("input_image_checkbox", False),
        body.get("current_tab", "uov"),
        body.get("uov_method", disabled),
        None,  # uov_input_image
        [],    # outpaint_selections
        None,  # inpaint_input_image (dict with image+mask)
        "",    # inpaint_additional_prompt
        None,  # inpaint_mask_image_upload
        # Developer/debug settings
        body.get("disable_preview", False),
        body.get("disable_intermediate_results", False),
        body.get("disable_seed_increment", False),
        body.get("black_out_nsfw", False),
        adm_scaler_positive,
        adm_scaler_negative,
        adm_scaler_end,
        adaptive_cfg,
        clip_skip,
        sampler_name,
        scheduler_name,
        vae_name,
        int(body.get("overwrite_step", -1)),
        int(body.get("overwrite_switch", -1)),
        int(body.get("overwrite_width", -1)),
        int(body.get("overwrite_height", -1)),
        float(body.get("overwrite_vary_strength", -1)),
        float(body.get("overwrite_upscale_strength", -1)),
        body.get("mixing_image_prompt_and_vary_upscale", False),
        body.get("mixing_image_prompt_and_inpaint", False),
        body.get("debugging_cn_preprocessor", False),
        body.get("skipping_cn_preprocessor", False),
        int(body.get("canny_low_threshold", 64)),
        int(body.get("canny_high_threshold", 128)),
        body.get("refiner_swap_method", "joint"),
        float(body.get("controlnet_softness", 0.25)),
        freeu_enabled,
        float(body.get("freeu_b1", 1.01)),
        float(body.get("freeu_b2", 1.02)),
        float(body.get("freeu_s1", 0.99)),
        float(body.get("freeu_s2", 0.95)),
        body.get("debugging_inpaint_preprocessor", False),
        body.get("inpaint_disable_initial_latent", False),
        body.get("inpaint_engine", "None"),
        float(body.get("inpaint_strength", 1.0)),
        float(body.get("inpaint_respective_field", 0.618)),
        body.get("inpaint_advanced_masking_checkbox", False),
        body.get("invert_mask_checkbox", False),
        int(body.get("inpaint_erode_or_dilate", 0)),
        body.get("save_final_enhanced_image_only", False),
        body.get("save_metadata_to_images", True),
        body.get("metadata_scheme", "fooocus"),
        *cn_args,
        # DINO / enhance
        body.get("debugging_dino", False),
        int(body.get("dino_erode_or_dilate", 0)),
        body.get("debugging_enhance_masks_checkbox", False),
        None,  # enhance_input_image
        body.get("enhance_checkbox", False),
        body.get("enhance_uov_method", disabled),
        body.get("enhance_uov_processing_order", "Before First Enhancement"),
        body.get("enhance_uov_prompt_type", "original"),
        *enhance_args,
    ]
    return args


@app.post("/api/generate")
async def generate(request: Request):
    """Submit a generation job to the async task queue."""
    from modules.async_worker import AsyncTask, async_tasks

    body = await request.json()
    # _build_generate_args reads a safetensors header via get_family(); keep
    # that blocking I/O off the event loop.
    from fastapi.concurrency import run_in_threadpool
    args = await run_in_threadpool(_build_generate_args, body)
    task = AsyncTask(args)
    async_tasks.append(task)
    return {"queued": True, "task_id": id(task)}


@app.post("/api/generate/stop")
async def generate_stop():
    """Stop the current generation.

    The worker pops the active task from async_tasks before processing,
    so we must also check the module-level current_task reference.
    Mirrors the stop logic used in the Gradio UI (webui.py stop_clicked).
    """
    import ldm_patched.modules.model_management as model_management
    from modules.async_worker import async_tasks, current_task

    # Check the currently running task first (already popped from queue)
    if current_task is not None and current_task.processing:
        current_task.last_stop = 'stop'
        model_management.interrupt_current_processing()
        return {"stopped": True}

    # Fall back to queued tasks that may have started processing
    for task in async_tasks:
        if task.processing:
            task.last_stop = 'stop'
            model_management.interrupt_current_processing()
            return {"stopped": True}

    # Clear queued tasks so the next job doesn't start immediately
    async_tasks.clear()

    return {"stopped": False}


# ---------------------------------------------------------------------------
# WebSocket — generation progress streaming
# ---------------------------------------------------------------------------

def _encode_preview_image(img) -> str | None:
    """Encode a preview image (numpy array or PIL Image) to base64 JPEG."""
    if img is None:
        return None
    try:
        import io
        from PIL import Image
        import numpy as np

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if not isinstance(img, Image.Image):
            return None

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        logger.debug("Failed to encode preview image", exc_info=True)
        return None


def _build_yield_message(flag: str, product) -> dict | None:
    """Build a JSON-serialisable message dict from a task yield entry."""
    if flag == "preview":
        percentage, text, img = product
        return {
            "type": "preview",
            "percentage": percentage,
            "text": text,
            "image": _encode_preview_image(img),
        }
    if flag == "results":
        return {
            "type": "results",
            "images": [
                str(p) if not isinstance(p, str) else p for p in product
            ],
        }
    if flag == "finish":
        return {
            "type": "finish",
            "images": [
                str(p) if not isinstance(p, str) else p for p in product
            ],
        }
    return None


@app.websocket("/ws/generation")
async def ws_generation(websocket: WebSocket):
    """
    Stream generation progress to the client.

    Polls the active task's yields list and forwards them as JSON messages.
    Message types: preview, results, finish, heartbeat.
    """
    origin = websocket.headers.get("origin")
    host = websocket.headers.get("host", "")
    if origin:
        origin_netloc = urlsplit(origin).netloc
        if origin_netloc != host:
            logger.warning(
                "Rejected WebSocket from mismatched origin: %s (host: %s)",
                origin,
                host,
            )
            await websocket.close(code=1008, reason="Origin not allowed")
            return

    await websocket.accept()

    async def _send_and_heartbeat(message: dict) -> None:
        await websocket.send_json(message)
        update_heartbeat()

    from modules.async_worker import async_tasks, current_task

    try:
        yield_index = 0
        active_task = None
        idle_count = 0

        while True:
            # Find an active (processing) task
            if active_task is None or not active_task.processing:
                # Drain any remaining yields before discarding the task
                if active_task is not None:
                    remaining = active_task.yields[yield_index:]
                    for flag, product in remaining:
                        msg = _build_yield_message(flag, product)
                        if msg is not None:
                            await _send_and_heartbeat(msg)
                active_task = None
                yield_index = 0
                for task in list(async_tasks):
                    if task.processing:
                        active_task = task
                        break
                # Fallback: the worker pops the task from async_tasks
                # before processing, so check current_task as well.
                if active_task is None and current_task is not None and current_task.processing:
                    active_task = current_task

            # Always refresh heartbeat so the backend knows a client
            # is connected — even when the pipeline is between yields
            # (model loading, long sampling steps).  The 15-second
            # timeout in is_browser_connected() would otherwise fire
            # during silent gaps between yield messages.
            update_heartbeat()

            current_yields = active_task.yields if active_task is not None else []
            if yield_index < len(current_yields):
                idle_count = 0
                flag, product = current_yields[yield_index]
                yield_index += 1

                msg = _build_yield_message(flag, product)
                if msg is not None:
                    await websocket.send_json(msg)

                if flag == "finish":
                    active_task = None
                    yield_index = 0
            else:
                # No new yields — send heartbeat message to client every ~5s
                idle_count += 1
                if idle_count >= 50:  # 50 * 100ms = 5s
                    await websocket.send_json({"type": "heartbeat"})
                    idle_count = 0

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)


# ---------------------------------------------------------------------------
# Generated image file serving
# ---------------------------------------------------------------------------

if os.path.isdir(config.path_outputs):
    app.mount(
        "/outputs",
        StaticFiles(directory=config.path_outputs),
        name="outputs",
    )
