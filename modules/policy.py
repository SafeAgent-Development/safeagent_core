from __future__ import annotations

from typing import Any, Dict, List, Optional
from source.utils import get_actions_for_hook
from source.states import SafeAgentWorldState


def _check_hard_threshold(
    gate: str,
    ltm_scores: Dict[str, float],
    stm_scores: Dict[str, float],
    obs_scores: Dict[str, float],
    hard_thresholds_cfg: Dict[str, Any],
) -> List[str]:
    """
    Check hard-threshold violations for a given gate.

    Args:
        gate: name of the gate, e.g. "action_default", "ltm_write", "run_command".
        ltm_scores: current long-term risk scores (dim -> float in [0,1]).
        stm_scores: current short-term risk scores (dim -> float in [0,1]).
        obs_scores: current observation risk scores (dim -> float in [0,1]).
        hard_thresholds_cfg: policy["hard_thresholds"] dict.

    Returns:
        A list of strings describing all violations, e.g.:
          ["obs_scores.secret_leak", "stm_scores.backdoor_trigger"]
        Empty list means no threshold is exceeded for this gate.
    """
    violations: List[str] = []

    if not isinstance(hard_thresholds_cfg, dict):
        return violations

    gate_cfg = hard_thresholds_cfg.get(gate, {})
    if not isinstance(gate_cfg, dict):
        # no thresholds defined for this gate
        return violations

    # Helper to check one scope
    def _check_scope(
        scope_name: str,
        scores: Dict[str, float],
        scope_cfg: Dict[str, Any],
    ) -> None:
        if not isinstance(scope_cfg, dict):
            return
        for dim, thr in scope_cfg.items():
            try:
                threshold = float(thr)
            except (TypeError, ValueError):
                continue  # ignore invalid threshold config
            value = float(scores.get(dim, 0.0))
            if value > threshold:
                violations.append(f"{scope_name}.{dim}")

    _check_scope("ltm_scores", ltm_scores, gate_cfg.get("ltm_scores") or {})
    _check_scope("stm_scores", stm_scores, gate_cfg.get("stm_scores") or {})
    _check_scope("obs_scores", obs_scores, gate_cfg.get("obs_scores") or {})

    return violations


def _check_hook_threshold(
    ltm_scores: Dict[str, float],
    stm_scores: Dict[str, float],
    obs_scores: Dict[str, float],
    consequence: Dict[str, Any],
    hard_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    action_violations = _check_hard_threshold(
        gate="action_default",
        ltm_scores=ltm_scores,
        stm_scores=stm_scores,
        obs_scores=obs_scores,
        hard_thresholds_cfg=hard_cfg,
    )

    ltm_violations = _check_hard_threshold(
        gate="ltm_write",
        ltm_scores=ltm_scores,
        stm_scores=stm_scores,
        obs_scores=obs_scores,
        hard_thresholds_cfg=hard_cfg,
    )

    if action_violations:
        filtered = {k: v for k, v in consequence.items() if k not in ("APPROVE", "OVERRIDE")}
    else:
        filtered = consequence

    violations: List[str] = []
    violations.extend(action_violations)
    violations.extend(ltm_violations)

    return {
        "actions": filtered,
        "allow_long_term_memory": not bool(ltm_violations),
        "violations": violations,
    }


def _check_tool_threshold(
    tool_name: str,
    ltm_scores: Dict[str, float],
    stm_scores: Dict[str, float],
    obs_scores: Dict[str, float],
    consequence: Dict[str, Any],
    hard_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    action_violations = _check_hard_threshold(
        gate="action_default",
        ltm_scores=ltm_scores,
        stm_scores=stm_scores,
        obs_scores=obs_scores,
        hard_thresholds_cfg=hard_cfg,
    )

    tool_violations = _check_hard_threshold(
        gate=tool_name,
        ltm_scores=ltm_scores,
        stm_scores=stm_scores,
        obs_scores=obs_scores,
        hard_thresholds_cfg=hard_cfg,
    )

    violations: List[str] = []
    violations.extend(action_violations)
    violations.extend(tool_violations)

    if violations:
        filtered = {k: v for k, v in consequence.items() if k not in ("CALL_ALLOW", "CALL_REWRITE")}
    else:
        filtered = consequence

    return {
        "actions": filtered,
        "violations": violations,
    }


async def safeagent_policy(state: SafeAgentWorldState) -> Dict[str, Any]:
    """
    Select a single action for the current hook.

    Decision steps:
      1) Hard gates:
         - Non-tool hooks: apply "action_default" and "ltm_write" gates.
         - tool_wrapper: apply "action_default" and tool-specific gates (e.g. "run_command").
         These gates may:
           * remove high-risk actions from the candidate set
           * forbid LTM write for the current message.
      2) Preference-based ranking:
         - For remaining actions, use world-model 5-level ratings
           (risk_control / task_completion / user_experience)
           and policy.preference_weights to compute a linear score.
         - Choose the action with the highest score.
    """
    hook = state.get("hook")
    actions_for_hook = get_actions_for_hook(hook) if isinstance(hook, str) else None
    if not isinstance(hook, str) or not actions_for_hook:
        return {"error": f"safeagent_policy: unsupported hook {hook!r}"}

    cfg: Dict[str, Any] = state.get("config") or {}
    policy_cfg: Dict[str, Any] = cfg.get("policy") or {}
    hard_cfg: Dict[str, Any] = policy_cfg.get("hard_thresholds") or {}

    ltm_scores: Dict[str, float] = state.get("ltm_scores") or {}
    stm_scores: Dict[str, float] = state.get("stm_scores") or {}
    obs_scores: Dict[str, float] = state.get("obs_scores") or {}

    # World-model consequence: action -> ratings dict
    consequence: Dict[str, Any] = state.get("consequence") or {}

    # Initial candidate set: intersection of WM actions and hook-allowed actions
    allowed_actions = set(actions_for_hook)
    consequence = {a: v for a, v in consequence.items() if a in allowed_actions}

    if not consequence:
        # No usable world-model consequence → hard fallback
        if hook == "tool_wrapper":
            return {
                "action": "CALL_BLOCK",
                "allow_long_term_memory": False,
                "violations": ["no_consequence_for_hook"],
            }
        else:
            return {
                "action": "REJECT",
                "allow_long_term_memory": False,
                "violations": ["no_consequence_for_hook"],
            }

    # -------------------------------------------------------------------------
    # Phase 1: apply hard thresholds
    # -------------------------------------------------------------------------
    violations: List[str] = []
    allow_ltm: Optional[bool] = None
    filtered_consequence: Dict[str, Any]

    if hook == "tool_wrapper":
        # Tool wrapper: combine global action_default gate + tool-specific gate
        obs = state.get("observation") or {}
        tool_name = str(obs.get("name", "")).strip() or "default"

        tool_res = _check_tool_threshold(
            tool_name=tool_name,
            ltm_scores=ltm_scores,
            stm_scores=stm_scores,
            obs_scores=obs_scores,
            consequence=consequence,
            hard_cfg=hard_cfg,
        )
        filtered_consequence = tool_res["actions"]
        violations.extend(tool_res.get("violations") or [])
        allow_ltm = False  # tool calls themselves are not written to LTM
    else:
        # Non-tool hooks: action_default + ltm_write gates
        hook_res = _check_hook_threshold(
            ltm_scores=ltm_scores,
            stm_scores=stm_scores,
            obs_scores=obs_scores,
            consequence=consequence,
            hard_cfg=hard_cfg,
        )
        filtered_consequence = hook_res["actions"]
        allow_ltm = bool(hook_res.get("allow_long_term_memory"))
        violations.extend(hook_res.get("violations") or [])

    if not filtered_consequence:
        # All actions removed by hard gates → fallback
        if hook == "tool_wrapper":
            return {
                "action": "CALL_BLOCK",
                "allow_long_term_memory": False,
                "violations": violations or ["all_actions_blocked"],
            }
        else:
            return {
                "action": "REJECT",
                "allow_long_term_memory": False,
                "violations": violations or ["all_actions_blocked"],
            }

    # -------------------------------------------------------------------------
    # Phase 2: preference-based ranking over remaining actions
    # -------------------------------------------------------------------------
    pref_mode = str(policy_cfg.get("preference", "safety_first"))
    pref_weights_cfg: Dict[str, Any] = policy_cfg.get("preference_weights") or {}
    w = pref_weights_cfg.get(pref_mode) or {
        "risk_control": 0.6,
        "task_completion": 0.25,
        "user_experience": 0.15,
    }

    w_risk = float(w.get("risk_control", 0.6))
    w_task = float(w.get("task_completion", 0.25))
    w_ux = float(w.get("user_experience", 0.15))

    def _score_action(action: str) -> float:
        wm = filtered_consequence.get(action) or {}
        # Default to neutral ratings (3) if the field is missing
        rc = int(wm.get("risk_control", 3))
        tc = int(wm.get("task_completion", 3))
        ux = int(wm.get("user_experience", 3))
        return w_risk * rc + w_task * tc + w_ux * ux

    best_action: Optional[str] = None
    best_score: float = float("-inf")

    for action in filtered_consequence.keys():
        score = _score_action(action)
        if score > best_score:
            best_score = score
            best_action = action

    result: Dict[str, Any] = {
        "action": best_action,
        "violations": violations,
    }

    # Only non-tool hooks carry LTM write decision
    if hook != "tool_wrapper":
        result["allow_long_term_memory"] = bool(allow_ltm)
    else:
        result["allow_long_term_memory"] = False

    return result
