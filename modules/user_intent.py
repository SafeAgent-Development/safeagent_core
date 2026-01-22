from __future__ import annotations

from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from source.states import SafeAgentWorldState
from source.utils import get_runnable_llm
from source.prompt import USER_INTENT_SYSTEM_PROMPT


_user_intent_prompt = ChatPromptTemplate.from_messages([
    ("system", USER_INTENT_SYSTEM_PROMPT),
    ("user", "{content}"),
])
_user_intent_parser = JsonOutputParser()


async def user_intent(state: SafeAgentWorldState) -> Dict[str, Any]:
    """
    Infer user_intent from the current observation.content.

    - Only runs for hook == "before_agent".
    - For other hooks, returns {} (no-op).
    - Writes the inferred intent into state["user_intent"].
    """
    hook = state.get("hook")
    if hook != "before_agent":
        return {}

    observation = state.get("observation") or {}
    content = str(observation.get("content", "")).strip()
    if not content:
        return {}

    llm = get_runnable_llm("override")

    chain = _user_intent_prompt | llm | _user_intent_parser

    try:
        out: Dict[str, Any] = await chain.ainvoke({"content": content})
        intent = out.get("user_intent")

        if not isinstance(intent, str) or not intent.strip():
            raise ValueError(f"user_intent missing or empty: {intent!r}")

        return {
            "user_intent": intent.strip(),
            "error": None,
        }

    except Exception as e:
        return {
            "error": f"user_intent failed: {type(e).__name__}: {e}",
        }
