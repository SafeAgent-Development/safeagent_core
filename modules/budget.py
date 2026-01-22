from __future__ import annotations

import time
from typing import Any, Dict, List

from source.states import SafeAgentWorldState


async def call_budget_control(state: SafeAgentWorldState) -> Dict[str, Any]:
    """Frequency-only call budget control (no gating, no action decision)."""
    hook = state.get("hook")
    if hook != "tool_wrapper":
        return {}

    observation = state.get("observation") or {}
    tool_name = str(observation.get("name", "")).strip()
    if not tool_name:
        return {}

    args = observation.get("args", {})

    cfg: Dict[str, Any] = state.get("config") or {}
    budget_cfg: Dict[str, Any] = cfg.get("call_budget_profiles") or {}

    default_cfg: Dict[str, Any] = budget_cfg.get("default") or {}
    tools_cfg: Dict[str, Any] = budget_cfg.get("tools") or {}
    tool_cfg: Dict[str, Any] = tools_cfg.get(tool_name) or {}

    window_seconds = int(tool_cfg.get("window_seconds", default_cfg.get("window_seconds", 0)))
    max_calls = int(tool_cfg.get("max_calls", default_cfg.get("max_calls", 0)))

    now_ts = time.time()

    call_history: List[Dict[str, Any]] = state.get("call_history") or []

    # Not configuring window or max_calls considered as having no frequency limit.
    if window_seconds <= 0 or max_calls <= 0:
        allowed_at = now_ts
        used_calls = 0
    else:
        window_start = now_ts - float(window_seconds)

        # Count the number of requests made by this tool within the window (by requested_at)
        inside_window: List[Dict[str, Any]] = []
        for item in call_history:
            if item.get("name") != tool_name:
                continue
            ts = item.get("requested_at")
            if isinstance(ts, (int, float)) and ts >= window_start:
                inside_window.append(item)

        used_calls = len(inside_window)

        if used_calls < max_calls:
            allowed_at = now_ts
        else:
            # Exceeded: Shift the request one window forward based on the earliest request time.
            earliest_ts = min(float(item["requested_at"]) for item in inside_window)
            allowed_at = earliest_ts + float(window_seconds)

    new_entry = [{
        "name": tool_name,
        "args": args,
        "requested_at": now_ts,
        "allowed_at": allowed_at,
    }]

    return {
        "call_history": new_entry,
        "error": None,
    }
