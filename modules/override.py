from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from source.states import SafeAgentWorldState
from source.utils import get_runnable_llm
from source.prompt import CONTENT_OVERRIDE_PROMPT, FUNCTION_CALL_OVERRIDE_PROMPT


def _build_override_payload(
    hook: str, observation: Dict[str, Any],
    obs_scores: Dict[str, Any], obs_evidence: Any
) -> Dict[str, Any]:
    mode_map = {
        "before_agent": "user_input",
        "after_agent": "agent_output",
        "before_model": "before_model",
        "tool_wrapper": "function_call",
    }

    if hook == "tool_wrapper":
        return {
            "mode": "function_call",
            "plan": str(observation.get("plan", "")),
            "name": str(observation.get("name", "")),
            "args": observation.get("args", {}) or {},
            "description": str(observation.get("description", "")),
            "obs_scores": obs_scores,
            "obs_evidence": obs_evidence,
        }
    else:
        return {
            "mode": mode_map[hook],
            "obs_scores": obs_scores,
            "obs_evidence": obs_evidence,
            "content": str(observation.get("content", "")),
        }


async def override_generate(state: SafeAgentWorldState) -> Dict[str, Any]:
    """
    Generate override candidates using obs + obs_scores + obs_evidence only.
    Writes result into state["candidates"]["overrides"].
    """
    content_prompt = ChatPromptTemplate.from_messages([("system", CONTENT_OVERRIDE_PROMPT)])
    func_prompt = ChatPromptTemplate.from_messages([("system", FUNCTION_CALL_OVERRIDE_PROMPT)])

    hook = state.get("hook")
    if hook not in ("before_agent", "after_agent", "before_model", "tool_wrapper"):
        return {}

    observation = state.get("observation") or {}
    obs_scores = state.get("obs_scores") or {}
    obs_evidence = state.get("obs_evidence") or []
    override = state.get("override") or {}

    llm = get_runnable_llm("override")

    try:
        payload = _build_override_payload(hook, observation, obs_scores, obs_evidence)

        # content override: expects {"override": "<string>"}
        if hook in ("before_agent", "after_agent", "before_model"):
            if not str(payload.get("content", "")).strip():
                return {}

            chain = content_prompt | llm | JsonOutputParser()
            out = await chain.ainvoke({"payload": json.dumps(payload, ensure_ascii=False)})

            override_val = out.get("override")
            if not isinstance(override_val, str) or not override_val.strip():
                raise ValueError("override missing or empty")

            override = {"override": override_val}

        # function-call override: expects {"override": <object>}
        else:
            if not str(payload.get("name", "")).strip():
                return {}

            chain = func_prompt | llm | JsonOutputParser()
            out = await chain.ainvoke({"payload": json.dumps(payload, ensure_ascii=False)})

            override_val = out.get("override")
            if not isinstance(override_val, dict):
                raise ValueError("override missing or not an object")

            override = {"override": override_val}

        return {"override": override, "error": None}

    except Exception as e:
        return {"error": f"override_generate failed: {type(e).__name__}: {e}"}
