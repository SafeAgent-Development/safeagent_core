from __future__ import annotations

from typing import Any, Dict
from source.utils import get_scope_params, get_actions_for_hook
from source.states import SafeAgentWorldState


def _get_stm_baseline(
    stm_scores: Dict[str, float],
    obs_scores: Dict[str, float],
    score_profiles: Dict[str, Any],
    update_profiles: Dict[str, Any],
) -> Dict[str, float]:
    stm_baseline: Dict[str, float] = {}
    keys = set(stm_scores.keys()) | set(obs_scores.keys())
    for k in keys:
        params = get_scope_params(k, score_profiles, update_profiles)
        gamma = params["stm_decay_gamma"]
        s = float(stm_scores.get(k, 0.0))
        o = float(obs_scores.get(k, 0.0))
        stm_baseline[k] = max(gamma * s, o)
    return stm_baseline


def _hook_action_advantage_cost(
    action: str, stm_scores: Dict[str, float], obs_scores: Dict[str, float],
    score_profiles: Dict[str, Any], update_profiles: Dict[str, Any],
    cost_profiles: Dict[str, Any], runtime_replan_count: int, runtime_rollback_count: int,
    runtime_trajectory_length: int, runtime_last_user_stm_scores: Dict[str, float]
) -> Dict[str, Any]:
    """
    Compute per-dimension advantage (risk reduction vs APPROVE baseline) and a 3D cost vector.

    - baseline = max(gamma_k * stm[k], obs[k])  (decay + max injection)
    - advantage[k] = baseline[k] - next[k]      (must be >= 0 to preserve "risk reduction" semantics)
    - cost = {latency, utility, ux} from cost_profiles (REPLAN/ROLLBACK dynamic by runtime counters)
    """
    stm_baseline = _get_stm_baseline(stm_scores, obs_scores, score_profiles, update_profiles)
    action_costs = (cost_profiles.get("actions") or {})
    keys = stm_baseline.keys()

    # APPROVE: do nothing; baseline action.
    if action == "APPROVE":
        return {
            "advantage": {k: 0.0 for k in keys},
            "cost": action_costs.get("APPROVE", {"latency": 0.0, "utility": 0.0, "ux": 0.0}),
        }

    # REJECT: block OBS injection; only keep decay of existing STM.
    if action == "REJECT":
        stm_next: Dict[str, float] = {}
        for k in keys:
            gamma = get_scope_params(k, score_profiles, update_profiles)["stm_decay_gamma"]
            stm_next[k] = gamma * float(stm_scores.get(k, 0.0))

        advantage = {k: stm_baseline[k] - stm_next[k] for k in keys}
        return {
            "advantage": advantage,
            "cost": action_costs.get("REJECT", {"latency": 0.0, "utility": 0.8, "ux": 0.8}),
        }

    # OVERRIDE: suppress OBS injection by (1 - override_confidence).
    if action == "OVERRIDE":
        stm_next: Dict[str, float] = {}
        for k in keys:
            params = get_scope_params(k, score_profiles, update_profiles)
            gamma, conf = params["stm_decay_gamma"], params["override_confidence"]
            s, o = float(stm_scores.get(k, 0.0)), float(obs_scores.get(k, 0.0))
            stm_next[k] = max(gamma * s, (1.0 - conf) * o)

        advantage = {k: stm_baseline[k] - stm_next[k] for k in keys}
        return {
            "advantage": advantage,
            "cost": action_costs.get("OVERRIDE", {"latency": 0.2, "utility": 0.3, "ux": 0.3}),
        }

    # REPLAN: "time travel" to pre-model state (drop the last plan); cost grows with replan count.
    if action == "REPLAN":
        cost_cfg = action_costs.get(action, {"utility": 0.25, "base_cost": 0.25, "max_counts": 3})

        # baseline - current (clamped to keep "risk reduction" semantics)
        advantage = {k: stm_baseline.get(k, 0.0) - stm_scores.get(k, 0.0) for k in keys}

        base = float(cost_cfg.get("base_cost", 0.25))
        max_counts = int(cost_cfg.get("max_counts", 3))
        step = (1.0 - base) / max(1.0, max_counts)

        ramp = min(base + step * runtime_replan_count, 1.0)
        cost = {"latency": ramp, "utility": float(cost_cfg.get("utility", 0.25)), "ux": ramp}

        return {"advantage": advantage, "cost": cost}

    # ROLLBACK: revert to last-user STM snapshot; allow at most once; cost grows with trajectory length.
    if action == "ROLLBACK":
        cost_cfg = action_costs.get(action, {"utility": 0.5, "base_cost": 0.5, "max_steps": 10})
        last_user_stm = runtime_last_user_stm_scores or {}
        keys2 = set(keys) | set(last_user_stm.keys())

        stm_next = {k: float(last_user_stm.get(k, 0.0)) for k in keys2}
        advantage = {k: stm_baseline.get(k, 0.0) - stm_next[k] for k in keys2}

        if int(runtime_rollback_count) >= 1:
            cost = {"latency": 1.0, "utility": 1.0, "ux": 1.0}
        else:
            base = float(cost_cfg.get("base_cost", 0.5))
            max_steps = int(cost_cfg.get("max_steps", 10))
            step = (1.0 - base) / max(1.0, max_steps)
            ramp = min(base + step * runtime_trajectory_length, 1.0)
            cost = {"latency": ramp, "utility": float(cost_cfg.get("utility", 0.5)), "ux": ramp}

        return {"advantage": advantage, "cost": cost}

    # TERMINATE: stop the task immediately; revert to last-user snapshot; maximum utility/ux cost.
    if action == "TERMINATE":
        last_user_stm = runtime_last_user_stm_scores or {}
        keys2 = set(keys) | set(last_user_stm.keys())

        stm_next = {k: float(last_user_stm.get(k, 0.0)) for k in keys2}
        advantage = {k: stm_baseline.get(k, 0.0) - stm_next[k] for k in keys2}

        return {
            "advantage": advantage,
            "cost": action_costs.get("TERMINATE", {"latency": 0.0, "utility": 1.0, "ux": 1.0}),
        }

    raise ValueError(f"Unknown action: {action}")


def _warpper_action_advantage_cost(
    action: str, obs_scores: Dict[str, float], score_profiles: Dict[str, Any],
    update_profiles: Dict[str, Any], cost_profiles: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute advantage and cost for tool_wrapper actions.

    - Baseline action: CALL_ALLOW
    - Advantage is defined over tool-execution risk dimensions only (from obs_scores)
    - Tool wrapper actions DO NOT update STM/LTM (no context transition)
    """
    action_costs = (cost_profiles.get("actions") or {})

    # tool execution risk baseline: directly from wrapper encoders
    baseline_risk: Dict[str, float] = {k: float(v) for k, v in (obs_scores or {}).items()}
    keys = baseline_risk.keys()

    # CALL_ALLOW: baseline execution, no risk reduction
    if action == "CALL_ALLOW":
        return {
            "advantage": {k: 0.0 for k in keys},
            "cost": action_costs.get("CALL_ALLOW", {"latency": 0.0, "utility": 0.0, "ux": 0.0}),
        }

    # CALL_BLOCK: do not execute tool; execution risk eliminated
    if action == "CALL_BLOCK":
        next_risk = {k: 0.0 for k in keys}
        advantage = {k: baseline_risk[k] - next_risk[k] for k in keys}
        return {
            "advantage": advantage,
            "cost": action_costs.get("CALL_BLOCK", {"latency": 0.0, "utility": 0.8, "ux": 0.8}),
        }

    # CALL_JIT_APPROVAL: pause execution; risk eliminated for now but high latency/ux cost
    if action == "CALL_JIT_APPROVAL":
        next_risk = {k: 0.0 for k in keys}
        advantage = {k: baseline_risk[k] - next_risk[k] for k in keys}
        return {
            "advantage": advantage,
            "cost": action_costs.get("CALL_JIT_APPROVAL", {"latency": 0.8, "utility": 0.2, "ux": 0.8}),
        }

    # CALL_REWRITE: rewrite args, then execute; risk reduction proportional to confidence
    if action == "CALL_REWRITE":
        next_risk: Dict[str, float] = {}
        for k in keys:
            params = get_scope_params(k, score_profiles, update_profiles)
            conf = params["override_confidence"]
            next_risk[k] = (1.0 - conf) * baseline_risk[k]

        advantage = {k: baseline_risk[k] - next_risk[k] for k in keys}
        return {
            "advantage": advantage,
            "cost": action_costs.get("CALL_REWRITE", {"latency": 0.2, "utility": 0.3, "ux": 0.3}),
        }

    raise ValueError(f"Unknown tool_wrapper action: {action}")


async def compute_advantage_cost(state: SafeAgentWorldState) -> Dict[str, Any]:
    """
    LangGraph node: compute per-action advantage vectors and 3D cost vectors.

    - If hook in regular hooks: call hook_action_advantage_cost
    - If hook == tool_wrapper: call warpper_action_advantage_cost
    - Writes results into state["candidates"][action]["advantage"|"cost"] (overwrite MVP)
    """
    hook = state.get("hook")
    cfg = state.get("config") or {}
    candidates = {}

    stm_scores = state.get("stm_scores") or {}
    obs_scores = state.get("obs_scores") or {}

    score_profiles = (cfg.get("score_profiles") or {})
    update_profiles = (cfg.get("update_profiles") or {})
    cost_profiles = (cfg.get("cost_profiles") or {})

    # runtime counters (for REPLAN/ROLLBACK in hook action space)
    runtime_replan_count = int(state.get("runtime_replan_count") or 0)
    runtime_rollback_count = int(state.get("runtime_rollback_count") or 0)
    runtime_trajectory_length = int(state.get("runtime_trajectory_length") or 0)
    runtime_last_user_stm_scores = state.get("runtime_last_user_stm_scores") or {}

    # action list: prefer enumerator output, fallback to default sets
    actions = get_actions_for_hook(hook)
    if not actions:
        return {"error": [f"compute_advantage_cost: no actions for hook={hook}"]}
    candidates["actions"] = actions

    # choose which advantage/cost function to call
    if hook in ["before_agent", "after_agent", "before_model", "after_model"]:
        for a in actions:
            candidates[a] = _hook_action_advantage_cost(
                action=a,
                stm_scores=stm_scores,
                obs_scores=obs_scores,
                score_profiles=score_profiles,
                update_profiles=update_profiles,
                cost_profiles=cost_profiles,
                runtime_replan_count=runtime_replan_count,
                runtime_rollback_count=runtime_rollback_count,
                runtime_trajectory_length=runtime_trajectory_length,
                runtime_last_user_stm_scores=runtime_last_user_stm_scores,
            )

    elif hook in ["tool_wrapper"]:
        for a in actions:
            candidates[a] = _warpper_action_advantage_cost(
                action=a,
                obs_scores=obs_scores,
                score_profiles=score_profiles,
                update_profiles=update_profiles,
                cost_profiles=cost_profiles,
            )

    else:
        return {
            "error": [f"compute_advantage_cost: unsupported hook={hook}"],
        }

    return {
        "candidates": candidates,
    }
