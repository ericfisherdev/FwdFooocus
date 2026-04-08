"""
FwdFooocus New UI - Server Launcher

Starts the new FastAPI UI server in a daemon thread alongside Gradio.
"""

import logging
import threading

import uvicorn

logger = logging.getLogger(__name__)

DEFAULT_PORT = 7866


def start(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
    """
    Launch the new UI server in a background daemon thread.

    Args:
        host: Bind address. Matches Gradio's --listen flag when set to 0.0.0.0.
        port: Port number. Defaults to 7866 (one above Gradio's default 7865).
    """
    thread = threading.Thread(
        target=_run_uvicorn,
        args=(host, port),
        daemon=True,
        name="new-ui-server",
    )
    thread.start()
    logger.info(f"New UI server starting on http://{host}:{port}")


def _run_uvicorn(host: str, port: int) -> None:
    from new_ui.app import app

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
