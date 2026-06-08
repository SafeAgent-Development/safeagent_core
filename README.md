# SafeAgent Core

**Runtime safety core for LLM agents.**

SafeAgent Core is a LangGraph-powered safety runtime that evaluates agent hook events, encodes multi-dimensional risk signals, reasons about candidate safety actions, and returns a structured decision such as approve, override, reject, replan, block, or rewrite.

This repository is the **core runtime**. It is not the external controller or end-user application. An external controller is expected to call SafeAgent Core at agent lifecycle hook points and apply the returned action or override.

---

## What SafeAgent Core Does

SafeAgent Core processes one hook invocation at a time. For each invocation, it maintains a world state containing the current observation, short-term memory, long-term memory, risk scores, evidence, candidate actions, world-model consequences, and policy decision.

The core provides:

- **Risk encoding**
  - Runs configured encoders for the current hook and produces normalized risk scores plus evidence.

- **Short-term and long-term risk state**
  - Maintains STM/LTM scores and evidence so risk can persist across turns instead of being treated as a single isolated message.

- **Action advantage and cost modeling**
  - Computes candidate action effects over risk dimensions and estimates latency, utility, and user-experience cost.

- **World-model consequence estimation**
  - Uses an LLM world model to reason about the likely consequences of each candidate action.

- **Policy-based action selection**
  - Applies hard thresholds and preference-weighted ranking to choose a single action.

- **Override generation**
  - Produces safe rewrites for unsafe content or unsafe tool-call arguments.

- **Call budget tracking**
  - Records per-tool call frequency and computes when a tool call is next allowed.

- **MCP boundary**
  - Exposes a minimal MCP interface for registering sessions and executing SafeAgent steps.

---

## Core Concepts

### Hooks

SafeAgent Core is designed around agent lifecycle hooks.

- `before_agent` — user input before the agent starts reasoning
- `after_agent` — final agent output before it is returned to the user
- `before_model` — tool output before it is fed back into the model
- `after_model` — model output or tool-call plan before tools are executed
- `tool_wrapper` — a concrete tool/function call and its arguments

Each hook has its own configured encoders and candidate actions.

---

### Observation, STM, and LTM

SafeAgent separates risk state into three layers:

#### Observation (`obs_*`)

Risk signals from the current hook invocation only.

#### Short-Term Memory (`stm_*`)

Session-level risk state that captures recent behavior and decays over time.

#### Long-Term Memory (`ltm_*`)

Longer-term risk profile updated only when policy allows LTM writes.

This separation lets the system react quickly to new risks while avoiding uncontrolled long-term memory pollution.

---

### Risk Encoders

Encoders convert an observation into:

```json
{
  "scores": {
    "<risk_dimension>": 0.0
  },
  "evidence": [
    "short explanation of why this risk was detected"
  ]
}
```

Scores are floats in `[0,1]`.

Evidence is a list of short human-readable strings.

Encoders may be deterministic or LLM-based and are configured through Python entrypoints.

---

### Candidate Actions

SafeAgent Core does not directly execute user tasks.

Instead, it selects a safety action for the caller to apply.

Examples include:

#### Content / Agent Actions

- `APPROVE`
- `OVERRIDE`
- `REJECT`
- `REPLAN`
- `ROLLBACK`
- `TERMINATE`

#### Tool Actions

- `CALL_ALLOW`
- `CALL_REWRITE`
- `CALL_BLOCK`
- `CALL_JIT_APPROVAL`

The available action set depends on the current hook.

---

### Advantage and Cost

For each candidate action, SafeAgent computes:

#### Advantage

Estimated risk reduction (or increase) per risk dimension.

#### Cost

Estimated cost across:

- `latency`
- `utility`
- `ux`

These values are inputs to downstream reasoning rather than the final decision.

---

### World Model

The world model receives the current world state and candidate actions.

For each candidate action it predicts:

- `risk_control`
- `task_completion`
- `user_experience`

Each score is rated from **1–5**.

The world model can also identify workflow-level attacks such as:

- prompt injection chains
- memory poisoning
- backdoor triggers
- unsafe tool-use workflows

The world model does **not** choose the final action.

---

### Policy

The policy module selects the final action.

It performs two stages:

#### Hard Threshold Filtering

Removes unsafe candidate actions or blocks LTM writes when configured score thresholds are exceeded.

#### Preference-Based Ranking

Ranks remaining actions using world-model predictions and a configured preference mode:

- `safety_first`
- `task_first`
- `ux_first`

The result is:

- one selected action
- one LTM-write decision

---

### Override Generation

The override module creates safer alternatives for risky content or tool calls.

Depending on the hook, it may rewrite:

- user input
- agent output
- tool output
- function-call arguments

Override generation is independent from policy selection.

---

## Core Graph Flow

```text
                    ┌──────────────────────┐
                    │ call_budget_control  │
                    └──────────┬───────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
          ┌──────────────┐          ┌─────────────┐
          │ risk_encoder │          │ user_intent │
          └──────┬───────┘          └──────┬──────┘
                 └────────────┬────────────┘
                              ▼
                 ┌────────────────────────┐
                 │ compute_advantage_cost │
                 └────────────┬───────────┘
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
      ┌─────────────┐                  ┌───────────────────┐
      │ world_model │                  │ override_generate │
      └──────┬──────┘                  └─────────┬─────────┘
             ▼                                   │
      ┌─────────────┐                            │
      │   policy    │                            │
      └──────┬──────┘                            │
             └────────────────┬──────────────────┘
                              ▼
                 ┌────────────────────────┐
                 │ stm_scores_synchronize │
                 └────────────┬───────────┘
                              ▼
                 ┌────────────────────────┐
                 │ ltm_scores_synchronize │
                 └────────────┬───────────┘
                              ▼
                 ┌──────────────────────────┐
                 │ stm_evidence_synchronize │
                 └────────────┬─────────────┘
                              ▼
                 ┌──────────────────────────┐
                 │ ltm_evidence_synchronize │
                 └────────────┬─────────────┘
                              ▼
                             END
```

The graph uses a LangGraph checkpointer.

The external `session_id` is used as the LangGraph `thread_id`, allowing each session to maintain an independent world state.

---

## MCP Interface

SafeAgent Core exposes two MCP tools:

### `safeagent_register_session`

Registers a new session and initializes an empty world state.

Responsibilities:

1. Validate developer and runtime configuration.
2. Merge configuration layers.
3. Create an initial `SafeAgentWorldState`.
4. Persist it under `thread_id = session_id`.

---

### `safeagent_step`

Runs one SafeAgent graph execution for a session.

Responsibilities:

1. Validate the incoming hook request.
2. Load the previous world state.
3. Merge the new observation.
4. Execute the graph.
5. Return the selected action and related outputs.

Typical response:

```json
{
  "session_id": "<session_id>",
  "action": "APPROVE",
  "override": {},
  "allow_long_term_memory": true,
  "violations": [],
  "error": null
}
```

For `tool_wrapper` requests, call-budget enforcement may delay execution or convert an over-budget tool call into `CALL_BLOCK`, depending on policy configuration.

---

## Configuration

SafeAgent Core uses layered configuration.

### Server Configuration

Defines:

- LLM profiles
- encoder definitions
- hook-to-encoder mapping
- score profiles
- influence scopes

### Developer Configuration

Defines:

- action costs
- update profiles
- policy constraints
- developer restrictions

### Runtime Configuration

Defines per-session behavior such as:

- policy preference mode
- call budget settings
- threshold overrides

Configuration layers are validated and deep-merged during session registration.

---

## Encoder Configuration

Encoders are configured through Python entrypoints.

Each encoder declares the score dimensions it may emit.

Every score dimension must have a matching score profile.

Score profiles determine whether a dimension influences:

- `observation`
- `short_term_memory`
- `long_term_memory`

Example encoder categories include:

- unicode obfuscation
- canary leakage detection
- prompt injection
- policy violation
- memory poisoning
- backdoor triggers
- tool-plan risk
- task drift
- secret leakage
- tool-call argument risk

The hook configuration determines which encoders execute at each lifecycle stage.

---

## Output Contract

The primary outputs returned by SafeAgent Core are:

```json
{
  "action": "<selected_action>",
  "override": {},
  "allow_long_term_memory": true,
  "violations": [],
  "error": null
}
```

The external controller is responsible for applying the returned action:

- continue execution
- apply an override
- reject or block a request
- trigger replanning or rollback
- decide whether long-term memory should be written

SafeAgent Core provides the runtime safety decision layer; it does not own the external agent execution loop.