"""Runtime plumbing shared by the COMPASS engine and the dashboard service."""

from .event_bus import (  # noqa: F401
    EventStore,
    RunEventEmitter,
    STAGE_NAMES,
    attach_sink,
    detach_sink,
    get_event_store,
    get_ui,
    json_safe,
    reset_ui,
)

__all__ = [
    "EventStore",
    "RunEventEmitter",
    "STAGE_NAMES",
    "attach_sink",
    "detach_sink",
    "get_event_store",
    "get_ui",
    "json_safe",
    "reset_ui",
]
