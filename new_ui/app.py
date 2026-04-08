"""
FwdFooocus New UI - FastAPI Application

Runs alongside the existing Gradio server on a separate port.
Serves the Alpine.js/HTMX/GSAP frontend via Jinja2 templates.
Shares the same backend modules (async_worker, config, lora_metadata).
"""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Query, Request
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
async def get_models():
    """Return available checkpoints, refiners, and VAEs."""
    return {
        "checkpoints": config.model_filenames,
        "loras": config.lora_filenames,
    }


@app.get("/api/styles")
async def get_styles():
    """Return available style names."""
    from modules.sdxl_styles import legal_style_names
    return {"styles": legal_style_names}


# ---------------------------------------------------------------------------
# Generated image file serving
# ---------------------------------------------------------------------------

if os.path.isdir(config.path_outputs):
    app.mount(
        "/outputs",
        StaticFiles(directory=config.path_outputs),
        name="outputs",
    )
