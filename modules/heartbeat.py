"""
Browser Heartbeat Module

Tracks whether the browser client is still connected via periodic heartbeat pings.
Used by the generation loop to cancel batches when the browser disconnects.
"""

import time

_last_heartbeat_time: float = time.time()


def update_heartbeat() -> None:
    """Record a heartbeat from the browser client."""
    global _last_heartbeat_time
    _last_heartbeat_time = time.time()


def is_browser_connected(timeout_seconds: float = 15.0) -> bool:
    """
    Check if the browser client is still connected.

    Args:
        timeout_seconds: Maximum seconds since last heartbeat before
                         considering the browser disconnected.

    Returns:
        True if a heartbeat was received within the timeout window.
    """
    return (time.time() - _last_heartbeat_time) < timeout_seconds
