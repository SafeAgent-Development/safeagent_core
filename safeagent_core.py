from __future__ import annotations

import time
import asyncio
from typing import Any, Dict
from fastmcp import FastMCP
from source.core import safeagent_app
from source.states import SafeAgentWorldState
from modules.validation import validate_cfg, validate_core_request

mcp = FastMCP("safeagent-core")


@mcp.tool(name="safeagent_register_session")
async def safeagent_register_session(
    session_id: str,
    runtime_cfg: Dict[str, Any],
    dev_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    MCP tool: register a SafeAgent session and persist its initial world state
    into LangGraph's checkpointer.

    Behavior:
    - Validate and merge server / developer / runtime configs via `validate_cfg`.
    - On validation failure, return {ok: False, session_id, error}.
    - On success:
      * Build an empty SafeAgentWorldState with the merged config.
      * Store it as the initial checkpoint under thread_id = session_id.
      * Return {ok: True, session_id} to the caller.

    Parameters
    ----------
    session_id:
        Opaque session identifier from the controller. It is used as
        `configurable.thread_id` for the LangGraph app so that all subsequent
        steps for this session share the same world state.

    runtime_cfg:
        Per-session runtime configuration provided by the client (e.g. user /
        app level tweaks).

    dev_cfg:
        Developer configuration provided by the integrator (e.g. policy, costs,
        update profiles overrides).

    Returns
    -------
    Dict[str, Any]:
        - On success:
          {"ok": True, "session_id": <str>}
        - On validation failure:
          {"ok": False, "session_id": <str>, "error": <str>}
    """

    # Validate and merge configs
    validation = validate_cfg(runtime_cfg=runtime_cfg, dev_cfg=dev_cfg)
    error = validation.get("error")

    if error is not None:
        return {
            "ok": False,
            "session_id": session_id,
            "error": error,
        }

    merged_cfg: Dict[str, Any] = validation.get("config") or {}

    # Build an empty baseline world state for this session
    base_state = SafeAgentWorldState(
        hook=None, action=None, allow_long_term_memory=False,
        observation={}, user_intent=None, call_history=[],
        ltm_scores={}, stm_scores={}, obs_scores={},
        ltm_evidence=[], stm_evidence=[], obs_evidence=[],
        override={}, candidates={}, consequence={}, violations=[],
        runtime_replan_count=0, runtime_rollback_count=0,
        runtime_trajectory_length=0, runtime_last_user_stm_scores={},
        config=merged_cfg, error=[],
    )

    # Persist as initial checkpoint (no graph execution)
    safeagent_app.update_state(
        config={"configurable": {"thread_id": session_id}},
        values=base_state,
    )

    # MCP response
    return {
        "ok": True,
        "session_id": session_id,
    }


@mcp.tool(name="safeagent_step")
async def safeagent_step(session_id: str, core_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run one SafeAgent core step for a given session.

    This MCP tool:
      1) Validates the incoming request shape and semantics via `validate_core_request`.
      2) Merges the request into the stored world state (using LangGraph checkpointer)
         and runs the main `safeagent_app` graph once.
      3) Extracts the resulting action / override / LTM decision / violations / error.
      4) For `hook == "tool_wrapper"`, enforces the call budget:
         - If `policy.wait_until_available == true`:
             * If the tool call is not yet allowed (allowed_at > now),
               it waits (async sleep) until `allowed_at` and then returns.
         - If `policy.wait_until_available == false`:
             * If the tool call is not yet allowed, the action is forcibly changed
               to `CALL_BLOCK` and an explanatory error suffix is appended.

    Parameters
    ----------
    session_id : str
        Logical session / thread identifier. Used as LangGraph `thread_id`.
    core_request : Dict[str, Any]
        A single core request, expected to contain at least:
          - "hook": str, e.g. "before_agent", "after_model", "tool_wrapper", ...
          - "observation": dict, hook-specific payload.

    Returns
    -------
    Dict[str, Any]
        {
          "session_id": str,
          "action": Optional[str],
          "override": Optional[Any],
          "allow_long_term_memory": Optional[bool],
          "violations": List[str],
          "error": Optional[str],
        }
    """

    # -------- 1) Validate request at the core-request layer --------
    validation = validate_core_request(core_request)
    error = validation.get("error")

    if error is not None:
        return {
            "session_id": session_id,
            "action": None,
            "override": None,
            "allow_long_term_memory": None,
            "violations": [],
            "error": f"validate_core_request failed: {error}",
        }

    # -------- 2) Build state delta and invoke the main graph --------
    hook = core_request.get("hook")
    observation = core_request.get("observation")

    # Only patch hook/observation/error; the rest of the state comes from checkpointer.
    state_delta = {
        "hook": hook,
        "observation": observation,
        "error": [],
    }

    try:
        out_state: SafeAgentWorldState = await safeagent_app.ainvoke(
            state_delta,
            config={"configurable": {"thread_id": session_id}},
        )
    except Exception as e:
        return {
            "session_id": session_id,
            "action": None,
            "override": None,
            "allow_long_term_memory": None,
            "violations": [],
            "error": f"safeagent_app.ainvoke failed: {type(e).__name__}: {e}",
        }

    # -------- 3) Extract graph outputs --------
    action = out_state.get("action")  # type: ignore[assignment]
    override = out_state.get("override")
    allow_ltm = out_state.get("allow_long_term_memory")  # type: ignore[assignment]
    violations = out_state.get("violations") or []
    graph_error = out_state.get("error")

    # -------- 4) Enforce call budget for tool_wrapper hook --------
    if hook == "tool_wrapper":
        cfg: Dict[str, Any] = out_state.get("config") or {}
        policy_cfg: Dict[str, Any] = cfg.get("policy") or {}

        # If true: wait until tool call becomes available.
        # If false: immediately force CALL_BLOCK when over budget.
        wait_until_available = bool(policy_cfg.get("wait_until_available", False))

        call_history = out_state.get("call_history") or []
        obs = observation or {}
        tool_name = str(obs.get("name", "")).strip()

        if tool_name and isinstance(call_history, list):
            latest_entry = None

            # Find the most recent call_history entry for this tool.
            for item in call_history:
                if item.get("name") != tool_name:
                    continue
                if latest_entry is None:
                    latest_entry = item
                else:
                    prev_ts = float(latest_entry.get("requested_at", 0.0))
                    cur_ts = float(item.get("requested_at", 0.0))
                    if cur_ts > prev_ts:
                        latest_entry = item

            if latest_entry is not None:
                allowed_at = float(latest_entry.get("allowed_at", 0.0))
                now = time.time()

                if allowed_at > now:
                    if wait_until_available:
                        delay = allowed_at - now
                        if delay > 0:
                            # Sleep until this tool call becomes available according to budget.
                            await asyncio.sleep(delay)
                    else:
                        # Do not wait: force a hard block and attach an error message.
                        extra_err = (
                            f"Call budget exceeded for tool '{tool_name}', "
                            f"forced CALL_BLOCK instead of {action!r}"
                        )
                        if graph_error:
                            graph_error = f"{graph_error}; {extra_err}"
                        else:
                            graph_error = extra_err
                        action = "CALL_BLOCK"

    return {
        "session_id": session_id,
        "action": action,
        "override": override,
        "allow_long_term_memory": allow_ltm,
        "violations": violations,
        "error": graph_error,
    }

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
