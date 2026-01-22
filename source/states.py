from __future__ import annotations

import operator
from typing import TypedDict, List, Dict, Any, Optional, Annotated


class SafeAgentWorldState(TypedDict):
    hook: Optional[str]
    action: Optional[str]
    allow_long_term_memory: bool
    observation: Dict[str, Any]
    user_intent: Optional[str]

    # Function Call history
    call_history: Annotated[List[Dict[str, Any]], operator.add]

    # canonical risk vectors
    ltm_scores: Dict[str, float]
    stm_scores: Dict[str, float]
    obs_scores: Dict[str, float]

    # evidence lists (reducer: concatenation)
    ltm_evidence: Annotated[List[str], operator.add]
    stm_evidence: Annotated[List[str], operator.add]
    obs_evidence: Annotated[List[str], operator.add]

    # override context or args
    override: Dict[str, Any]

    # candidate actions & metadata (overwrite for MVP)
    candidates: Dict[str, Any]

    # consequence of each action
    consequence: Dict[str, Any]

    violations: Annotated[List[str], operator.add]

    # Runtime informations
    runtime_replan_count: int
    runtime_rollback_count: int
    runtime_trajectory_length: int
    runtime_last_user_stm_scores: Dict[str, float]

    # session config injected by controller
    config: Dict[str, Any]

    # error for debug
    error: Optional[str]
