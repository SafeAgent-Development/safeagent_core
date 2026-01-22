from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from source.prompt import WORLD_MODEL_SYSTEM_PROMPT
from source.utils import get_runnable_llm
from source.states import SafeAgentWorldState


def _build_worldmodel_payload(state: SafeAgentWorldState) -> Dict[str, Any]:
    hook = state.get("hook")
    user_intent = state.get("user_intent") or ""

    ltm_scores = state.get("ltm_scores") or {}
    stm_scores = state.get("stm_scores") or {}
    obs_scores = state.get("obs_scores") or {}

    ltm_evidence = state.get("ltm_evidence") or []
    stm_evidence = state.get("stm_evidence") or []
    obs_evidence = state.get("obs_evidence") or []

    candidates = state.get("candidates") or {}

    return {
        "hook": hook,
        "user_intent": user_intent,
        "ltm": {
            "scores": ltm_scores,
            "evidence": ltm_evidence,
        },
        "stm": {
            "scores": stm_scores,
            "evidence": stm_evidence,
        },
        "obs": {
            "scores": obs_scores,
            "evidence": obs_evidence,
        },
        "candidates": candidates,
    }


async def world_model(state: SafeAgentWorldState) -> Dict[str, Any]:
    """
    Call the WORLD MODEL LLM with structured world state + candidates,
    get composite attack evidence + per-action consequences, and
    merge composite evidence into obs_evidence.
    """
    try:
        payload = _build_worldmodel_payload(state)

        llm = get_runnable_llm("worldmodel")
        worldmodel_prompt = ChatPromptTemplate.from_messages([
            ("system", WORLD_MODEL_SYSTEM_PROMPT),
        ])

        chain = worldmodel_prompt | llm | JsonOutputParser()
        out: Dict[str, Any] = await chain.ainvoke({"payload": json.dumps(payload, ensure_ascii=False)})

        composite = out.get("composite_attack_evidence") or []
        if not isinstance(composite, list):
            raise ValueError("composite_attack_evidence must be a list")

        composite = [item[:512] for item in composite if isinstance(item, str)]

        consequence = out.get("consequence") or {}
        if not isinstance(consequence, dict):
            raise ValueError("consequence must be an object mapping action -> consequence")

        # validate each action consequence: 1–5 integer ratings
        for action_name, c in consequence.items():
            if not isinstance(c, dict):
                raise ValueError(f"consequence[{action_name}] must be an object")
            for key in ("risk_control", "task_completion", "user_experience"):
                v = c.get(key)
                if not isinstance(v, int) or not (1 <= v <= 5):
                    raise ValueError(
                        f"consequence[{action_name}].{key} must be int in [1,5], got {v!r}"
                    )
            reason = c.get("reason")
            if not isinstance(reason, str):
                raise ValueError(f"consequence[{action_name}].reason must be a string")

        return {
            "consequence": consequence,
            "obs_evidence": composite,
            "error": None,
        }

    except Exception as e:
        return {
            "error": f"world_model failed: {type(e).__name__}: {e}",
        }
