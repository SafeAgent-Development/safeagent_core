from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from modules.budget import call_budget_control
from modules.cost import compute_advantage_cost
from modules.override import override_generate
from modules.policy import safeagent_policy
from modules.risk_encoder import safeagent_risk_encoder
from modules.sync import (
    stm_scores_synchronize, stm_evidence_synchronize,
    ltm_scores_synchronize, ltm_evidence_synchronize
)
from modules.user_intent import user_intent
from modules.world_model import world_model

from states import SafeAgentWorldState


def build_safeagent_core_graph() -> CompiledStateGraph:
    """
    Build the SafeAgent core LangGraph.

    High-level flow for a single hook invocation:

      entry: call_budget_control
        ├─ risk_encoder             (in parallel)
        ├─ user_intent              (in parallel)
        └─ STM/LTM sync pipeline    (in parallel)
             call_budget_control
               → stm_scores_synchronize
               → stm_evidence_synchronize
               → ltm_scores_synchronize
               → ltm_evidence_synchronize
             ↓ join
        compute_advantage_cost
        ├─ world_model → policy      (branch 1: action evaluation & selection)
        └─ override_generate         (branch 2: content / function-call override)
        END

    Semantics:
      - call_budget_control:
          Update per-tool call_history and next allowed time, but do not gate.
      - risk_encoder:
          Run all configured encoders for the current hook and produce obs_scores / obs_evidence.
      - user_intent:
          Infer user_intent from the current observation (before_agent only).
      - STM/LTM sync pipeline:
          Propagate scores and evidence across observation → STM → LTM before
          the world model reasons about actions.
      - compute_advantage_cost:
          For the current hook, compute per-action STM advantages and multi-dimensional costs.
      - world_model:
          Given state + candidates (advantages/costs), predict consequences of each action.
      - policy:
          Apply hard thresholds and preference weights to pick a single action and
          decide whether LTM write is allowed.
      - override_generate:
          Optionally rewrite content or tool arguments based on obs_* signals.

      The final state (after policy and/or override_generate) is returned to the caller,
      who decides how to apply the chosen action and override.
    """

    graph = StateGraph(SafeAgentWorldState)

    # --- Node definitions ---

    # Entry: call budget accounting (no gating)
    graph.add_node("call_budget_control", call_budget_control)

    # Parallel branch: risk encoders
    graph.add_node("risk_encoder", safeagent_risk_encoder)

    # Parallel branch: user intent inference
    graph.add_node("user_intent", user_intent)

    # Advantage / cost computation for candidate actions
    graph.add_node("compute_advantage_cost", compute_advantage_cost)

    # Branch 1: world model + policy
    graph.add_node("world_model", world_model)
    graph.add_node("policy", safeagent_policy)

    # Branch 2: override generation (content / function call)
    graph.add_node("override_generate", override_generate)

    # STM / LTM synchronization pipeline
    graph.add_node("stm_scores_synchronize", stm_scores_synchronize)
    graph.add_node("stm_evidence_synchronize", stm_evidence_synchronize)
    graph.add_node("ltm_scores_synchronize", ltm_scores_synchronize)
    graph.add_node("ltm_evidence_synchronize", ltm_evidence_synchronize)

    # --- Edges: entry and parallel branches ---

    # Entry point
    graph.set_entry_point("call_budget_control")

    # Entry → risk encoders / user intent (parallel)
    graph.add_edge("call_budget_control", "risk_encoder")
    graph.add_edge("call_budget_control", "user_intent")

    # STM / LTM sync pipeline runs in parallel from entry
    graph.add_edge("call_budget_control", "stm_scores_synchronize")
    graph.add_edge("stm_scores_synchronize", "stm_evidence_synchronize")
    graph.add_edge("stm_evidence_synchronize", "ltm_scores_synchronize")
    graph.add_edge("ltm_scores_synchronize", "ltm_evidence_synchronize")

    # risk_encoder, user_intent, and ltm_evidence_synchronize all feed into
    # compute_advantage_cost (join point)
    graph.add_edge("ltm_evidence_synchronize", "compute_advantage_cost")
    graph.add_edge("risk_encoder", "compute_advantage_cost")
    graph.add_edge("user_intent", "compute_advantage_cost")

    # --- After compute_advantage_cost: two logical branches ---

    # Branch 1: world model → policy (choose action, decide LTM write)
    graph.add_edge("compute_advantage_cost", "world_model")
    graph.add_edge("world_model", "policy")

    # Branch 2: override generation
    graph.add_edge("compute_advantage_cost", "override_generate")

    # Both policy and override_generate are terminal for this graph.
    # The controller reads action / override / scores from the final state.
    graph.add_edge("policy", END)
    graph.add_edge("override_generate", END)

    return graph.compile()


# Global (or injected) checkpointer
checkpointer = MemorySaver()

# Main SafeAgent app with checkpointer enabled
safeagent_app = build_safeagent_core_graph().with_config(
    checkpointer=checkpointer
)
