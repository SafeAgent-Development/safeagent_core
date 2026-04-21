from __future__ import annotations
import json
import numpy as np
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from source.utils import get_scope_params
from source.utils import get_runnable_llm
from source.prompt import EVIDENCE_SUMMARY_SYSTEM_PROMPT
from source.states import SafeAgentWorldState


_summary_prompt = ChatPromptTemplate.from_messages([
    ("system", EVIDENCE_SUMMARY_SYSTEM_PROMPT),
    ("user", "{payload}"),
])
_summary_parser = JsonOutputParser()


async def stm_scores_synchronize(state: SafeAgentWorldState) -> Dict[str, Any]:
    """
    Synchronize STM risk scores after policy has chosen an action.

    Semantics:
      - APPROVE: standard OBS→STM update (decay + max injection).
      - REJECT: decay STM only, block OBS injection.
      - OVERRIDE: decay STM + weakened OBS injection using (1 - override_confidence).
      - REPLAN: keep STM as-is (no additional risk transition).
      - ROLLBACK: revert STM to last-user snapshot.
      - TERMINATE: same STM transition as ROLLBACK (controller will end the task).

    Reads:
      - state["action"]: chosen action name by policy
      - state["stm_scores"], state["obs_scores"]
      - state["runtime_last_user_stm_scores"]
      - state["config"]["scores"], state["config"]["update_profiles"]

    Writes:
      - "stm_scores": updated STM scores
    """
    action = state.get("action")
    if not action:
        return {}
    if action in ("CALL_ALLOW", "CALL_BLOCK", "CALL_REWRITE", "CALL_JIT_APPROVAL"):
        return {}

    cfg: Dict[str, Any] = state.get("config") or {}
    score_profiles: Dict[str, Any] = (cfg.get("score_profiles") or {})
    update_profiles: Dict[str, Any] = (cfg.get("update_profiles") or {})

    stm_scores: Dict[str, float] = state.get("stm_scores") or {}
    obs_scores: Dict[str, float] = state.get("obs_scores") or {}
    last_user_stm: Dict[str, float] = state.get("runtime_last_user_stm_scores") or {}

    keys = set(stm_scores.keys()) | set(obs_scores.keys())

    stm_next: Dict[str, float] = {}

    # --- APPROVE: standard OBS→STM update: decay + max injection ---
    if action == "APPROVE":
        for k in keys:
            params = get_scope_params(k, score_profiles, update_profiles)
            gamma = params["stm_decay_gamma"]
            s = float(stm_scores.get(k, 0.0))
            o = float(obs_scores.get(k, 0.0))
            stm_next[k] = max(gamma * s, o)

        return {"obs_scores": {}, "stm_scores": stm_next}

    # --- REJECT: block OBS injection; only decay STM ---
    if action == "REJECT":
        for k in keys:
            params = get_scope_params(k, score_profiles, update_profiles)
            gamma = params["stm_decay_gamma"]
            s = float(stm_scores.get(k, 0.0))
            stm_next[k] = gamma * s

        return {"obs_scores": {}, "stm_scores": stm_next}

    # --- OVERRIDE: suppress OBS injection by (1 - override_confidence) ---
    if action == "OVERRIDE":
        for k in keys:
            params = get_scope_params(k, score_profiles, update_profiles)
            gamma = params["stm_decay_gamma"]
            conf = params["override_confidence"]
            s = float(stm_scores.get(k, 0.0))
            o = float(obs_scores.get(k, 0.0))
            # override weakens risk injection from this OBS
            stm_next[k] = max(gamma * s, (1.0 - conf) * o)

        return {"obs_scores": {}, "stm_scores": stm_next}

    # --- REPLAN: do not change STM; we only discard current plan and re-run planning ---
    if action == "REPLAN":
        return {"obs_scores": {}}

    # --- ROLLBACK: revert to last-user STM snapshot ---
    if action == "ROLLBACK":
        return {"obs_scores": {}, "stm_scores": last_user_stm}

    # --- TERMINATE: same STM transition as ROLLBACK; controller will terminate the task ---
    if action == "TERMINATE":
        return {"obs_scores": {}, "stm_scores": last_user_stm}

    # Unknown action: fail-closed by not updating STM
    raise ValueError(f"stm_scores_synchronize: unknown action '{action}'")


def _ltm_softmax_blend(old_s: float, old_l: float, temperature: float) -> float:
    if old_s <= 0.0 and old_l <= 0.0:
        return 0.0

    old_s = np.clip(old_s, 0.0, 1.0)
    old_l = np.clip(old_l, 0.0, 1.0)

    tau = max(1e-3, float(temperature))
    logits = np.array([old_s, old_l], dtype=float) / tau

    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    denom = float(exp_vals.sum())
    if denom <= 0.0:
        return old_l

    probs = exp_vals / denom

    new_l = np.clip(probs[0] * old_s + probs[1] * old_l, 0.0, 1.0)
    return float(new_l)


async def ltm_scores_synchronize(state: SafeAgentWorldState) -> Dict[str, Any]:
    """
    Synchronize LTM risk scores from STM using an asymmetric softmax-style update.

    Semantics:
      - New high STM risk is injected into LTM quickly.
      - Decreasing STM risk leads to slow decay in LTM.
      - Temperature is taken from update_profiles[scope].ltm_ema_beta
        (interpreted as a "softmax temperature" parameter).

    Reads:
      - state["stm_scores"], state["ltm_scores"]
      - state["config"]["scores"], state["config"]["update_profiles"]

    Writes:
      - "ltm_scores": updated LTM scores
    """
    hook = state.get("hook")
    if hook != "after_agent":
        return {}
    cfg: Dict[str, Any] = state.get("config") or {}
    score_profiles: Dict[str, Any] = (cfg.get("score_profiles") or {})
    update_profiles: Dict[str, Any] = (cfg.get("update_profiles") or {})

    stm_scores: Dict[str, float] = state.get("stm_scores") or {}
    ltm_scores: Dict[str, float] = state.get("ltm_scores") or {}

    keys = set(stm_scores.keys()) | set(ltm_scores.keys())
    ltm_next: Dict[str, float] = {}

    for k in keys:
        params = get_scope_params(k, score_profiles, update_profiles)
        temperature = params["ltm_softmax_temperature"]

        stm = float(stm_scores.get(k, 0.0))
        ltm = float(ltm_scores.get(k, 0.0))

        ltm_next[k] = _ltm_softmax_blend(stm, ltm, temperature)

    return {"ltm_scores": ltm_next}


async def stm_evidence_synchronize(state: SafeAgentWorldState) -> Dict[str, Any]:
    obs_evidence = state.get("obs_evidence") or []
    payload = {"evidence": [str(e) for e in obs_evidence if isinstance(e, str) and e.strip()]}

    if not payload["evidence"]:
        return {}

    llm = get_runnable_llm("aggregator")

    try:
        chain = _summary_prompt | llm | _summary_parser
        out: Dict[str, Any] = await chain.ainvoke({"payload": json.dumps(payload, ensure_ascii=False)})

        summary = out.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(f"summary missing or empty: {summary!r}")

        obs_evidence.clear()
        return {
            "stm_evidence": [summary.strip()],
        }

    except Exception as e:
        return {
            "error": [f"stm_evidence_synchronize failed: {type(e).__name__}: {e}"],
        }


async def ltm_evidence_synchronize(state: SafeAgentWorldState) -> Dict[str, Any]:
    hook = state.get("hook")
    if hook != "after_agent":
        return {}
    stm_evidence = state.get("stm_evidence") or []
    payload = {"evidence": [str(e) for e in stm_evidence if isinstance(e, str) and e.strip()]}

    if not payload["evidence"]:
        return {}

    llm = get_runnable_llm("aggregator")

    try:
        chain = _summary_prompt | llm | _summary_parser
        out: Dict[str, Any] = await chain.ainvoke({"payload": json.dumps(payload, ensure_ascii=False)})

        summary = out.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(f"summary missing or empty: {summary!r}")

        stm_evidence.clear()
        return {
            "ltm_evidence": [summary.strip()],
        }

    except Exception as e:
        return {
            "error": [f"ltm_evidence_synchronize failed: {type(e).__name__}: {e}"],
        }
