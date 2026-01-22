from __future__ import annotations

import os
from typing import Any, Dict, List, Set
from source.utils import get_cfg, load_entrypoint
from source.states import SafeAgentWorldState


ALLOWED_HOOKS = {"before_agent", "after_agent", "after_model", "before_model", "tool_wrapper"}
ALLOWED_ACTIONS = {
    "APPROVE", "REJECT", "OVERRIDE", "REPLAN", "ROLLBACK", "TERMINATE",
    "CALL_ALLOW", "CALL_BLOCK", "CALL_REWRITE", "CALL_JIT_APPROVAL",
}


def _ensure_str_field(obj: Dict[str, Any], key: str, *, allow_empty: bool = False) -> None:
    v = obj.get(key)
    if not isinstance(v, str):
        raise TypeError(f"observation.{key} must be a string")
    if (not allow_empty) and not v.strip():
        raise ValueError(f"observation.{key} must be a non-empty string")


def validate_core_request(state: SafeAgentWorldState) -> Dict[str, Any]:
    """
    Validate minimal core request schema per hook.

    On success: return {} (no-op update).
    On failure: return {"error": "..."} and let controller fail-closed.
    """
    try:
        hook = state.get("hook")
        if not isinstance(hook, str) or not hook:
            raise ValueError("hook must be a non-empty string")

        if hook not in ALLOWED_HOOKS:
            raise ValueError(f"unsupported hook: {hook!r}")

        observation = state.get("observation")
        if not isinstance(observation, dict):
            raise TypeError("observation must be a dict")

        # --- before_agent: user input ---
        if hook == "before_agent":
            _ensure_str_field(observation, "role", allow_empty=False)
            if observation["role"] != "user":
                raise ValueError("before_agent.observation.role must be 'user'")
            _ensure_str_field(observation, "content", allow_empty=False)

        # --- after_agent: final assistant output ---
        elif hook == "after_agent":
            _ensure_str_field(observation, "role", allow_empty=False)
            if observation["role"] != "assistant":
                raise ValueError("after_agent.observation.role must be 'assistant'")
            _ensure_str_field(observation, "content", allow_empty=False)

        # --- after_model: tool plan check (assistant tool_calls + last_user) ---
        elif hook == "after_model":
            _ensure_str_field(observation, "role", allow_empty=False)
            if observation["role"] != "assistant":
                raise ValueError("after_model.observation.role must be 'assistant'")
            _ensure_str_field(observation, "content", allow_empty=True)

            tool_calls = observation.get("tool_calls")
            if not isinstance(tool_calls, list):
                raise TypeError("after_model.observation.tool_calls must be a list")
            for i, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    raise TypeError(f"after_model.observation.tool_calls[{i}] must be a dict")

            _ensure_str_field(observation, "last_user", allow_empty=False)

        # --- before_model: tool message review ---
        elif hook == "before_model":
            _ensure_str_field(observation, "role", allow_empty=False)
            if observation["role"] != "tool":
                raise ValueError("before_model.observation.role must be 'tool'")
            _ensure_str_field(observation, "name", allow_empty=False)
            _ensure_str_field(observation, "tool_call_id", allow_empty=False)
            _ensure_str_field(observation, "content", allow_empty=True)

        # --- tool_wrapper: Single tool execution audit ---
        elif hook == "tool_wrapper":
            _ensure_str_field(observation, "plan", allow_empty=False)
            _ensure_str_field(observation, "name", allow_empty=False)

            args = observation.get("args")
            if not isinstance(args, dict):
                raise TypeError("tool_wrapper.observation.args must be a dict")

            desc = observation.get("description")
            if desc is not None and not isinstance(desc, str):
                raise TypeError("tool_wrapper.observation.description must be a string or null")

        return {}

    except Exception as e:
        return {
            "error": f"validate_core_request failed: {type(e).__name__}: {e}",
        }


def _validate_llm_block(llm_cfg: Dict[str, Any]) -> None:
    """
    Validate the server-side LLM config block.

    Required profiles:
      - encoder
      - worldmodel
      - override
      - aggregator

    Each profile must be a dict with:
      - name: non-empty string
      - temperature: number (0 <= t <= 2 recommended)
      - max_tokens: positive int
      - timeout: positive number
      - endpoint_env:
          - base_url: non-empty string
          - api_key:  non-empty string

    Raises:
      TypeError / ValueError on any schema violation.
    """
    if not isinstance(llm_cfg, dict):
        raise TypeError("llm config must be a dict")

    required_profiles = ("encoder", "worldmodel", "override", "aggregator")

    for profile_name in required_profiles:
        profile = llm_cfg.get(profile_name)
        if not isinstance(profile, dict):
            raise ValueError(f"llm.{profile_name} must be a dict")

        # name
        name = profile.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"llm.{profile_name}.name must be a non-empty string")

        # temperature
        temp = profile.get("temperature")
        if not isinstance(temp, (int, float)):
            raise TypeError(f"llm.{profile_name}.temperature must be a number")
        # soft range check
        if temp < 0.0:
            raise ValueError(f"llm.{profile_name}.temperature out of range: {temp!r}")

        # max_tokens
        max_tokens = profile.get("max_tokens")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(f"llm.{profile_name}.max_tokens must be a positive integer")

        # timeout
        timeout = profile.get("timeout")
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"llm.{profile_name}.timeout must be a positive number")

        # endpoint_env
        endpoint_env = profile.get("endpoint_env")
        if not isinstance(endpoint_env, dict):
            raise ValueError(f"llm.{profile_name}.endpoint_env must be a dict")

        base_url_var = endpoint_env.get("base_url")
        api_key_var = endpoint_env.get("api_key")

        if not isinstance(base_url_var, str) or not base_url_var.strip():
            raise ValueError(
                f"llm.{profile_name}.endpoint_env.base_url must be a non-empty env var name"
            )
        if not isinstance(api_key_var, str) or not api_key_var.strip():
            raise ValueError(
                f"llm.{profile_name}.endpoint_env.api_key must be a non-empty env var name"
            )

        base_url_val = os.environ.get(base_url_var)
        api_key_val = os.environ.get(api_key_var)

        if not base_url_val or not base_url_val.strip():
            raise ValueError(
                f"Environment variable {base_url_var!r} (llm.{profile_name}.endpoint_env.base_url) "
                "is not set or empty"
            )
        if not api_key_val or not api_key_val.strip():
            raise ValueError(
                f"Environment variable {api_key_var!r} (llm.{profile_name}.endpoint_env.api_key) "
                "is not set or empty"
            )


def _validate_encoders(encoders_cfg: Dict[str, Any]) -> Set[str]:
    """
    Validate the encoders block and return the set of all score dimensions.

    For each encoder:
      - spec must be a dict
      - id: non-empty string (optional check: id == key)
      - type: "deterministic" or "llm_judge"
      - entrypoint: non-empty string and importable via load_entrypoint(...)
      - outputs.scores: non-empty list of non-empty strings

    Returns:
      Set of all score dimensions emitted by all encoders.

    Raises:
      TypeError / ValueError on any schema violation or missing entrypoint.
    """
    if not isinstance(encoders_cfg, dict):
        raise TypeError("encoders config must be a dict")

    allowed_types = {"deterministic", "llm_judge"}
    seen_ids: Set[str] = set()
    all_score_dims: Set[str] = set()

    for name, spec in encoders_cfg.items():
        if not isinstance(spec, dict):
            raise ValueError(f"encoder '{name}' spec must be a dict")

        # id
        enc_id = spec.get("id")
        if not isinstance(enc_id, str) or not enc_id.strip():
            raise ValueError(f"encoder '{name}': id must be a non-empty string")

        if enc_id in seen_ids:
            raise ValueError(f"duplicate encoder id: {enc_id!r}")
        seen_ids.add(enc_id)

        # optional: enforce id == key
        if enc_id != name:
            # you can choose to warn instead; here we enforce strict equality
            raise ValueError(f"encoder key '{name}' must match id '{enc_id}'")

        # type
        enc_type = spec.get("type")
        if enc_type not in allowed_types:
            raise ValueError(
                f"encoder '{name}': type must be one of {sorted(allowed_types)}, "
                f"got {enc_type!r}"
            )

        # entrypoint
        entrypoint = spec.get("entrypoint")
        if not isinstance(entrypoint, str) or not entrypoint.strip():
            raise ValueError(f"encoder '{name}': entrypoint must be a non-empty string")

        # make sure we can import it; load_entrypoint should raise on error
        fn = load_entrypoint(entrypoint)
        if not callable(fn):
            raise ValueError(f"encoder '{name}': entrypoint '{entrypoint}' is not callable")

        # outputs.scores
        outputs = spec.get("outputs")
        if not isinstance(outputs, dict):
            raise ValueError(f"encoder '{name}': outputs must be a dict")

        scores = outputs.get("scores")
        if not isinstance(scores, list) or not scores:
            raise ValueError(f"encoder '{name}': outputs.scores must be a non-empty list")

        for dim in scores:
            if not isinstance(dim, str) or not dim.strip():
                raise ValueError(
                    f"encoder '{name}': outputs.scores must contain non-empty strings, "
                    f"got {dim!r}"
                )
            all_score_dims.add(dim)

    return all_score_dims


def _validate_hooks(hooks_cfg: Dict[str, Any], encoders_cfg: Dict[str, Any]) -> None:
    """
    Validate the hooks block.

    Requirements:
      - hooks_cfg must be a dict.
      - All ALLOWED_HOOKS must appear as keys in hooks_cfg.
      - No unknown hooks beyond ALLOWED_HOOKS.
      - For each hook:
          - spec must be a dict.
          - spec["encoders"] must be a list of encoder names (non-empty strings).
          - Each encoder name must exist in encoders_cfg.
    """
    if not isinstance(hooks_cfg, dict):
        raise TypeError("hooks config must be a dict")

    if not isinstance(encoders_cfg, dict):
        raise TypeError("encoders config must be a dict")

    # 1) No unknown hooks
    for hook_name in hooks_cfg.keys():
        if hook_name not in ALLOWED_HOOKS:
            raise ValueError(f"unknown hook name: {hook_name!r}")

    # 2) All required hooks must be present
    missing_hooks = [h for h in ALLOWED_HOOKS if h not in hooks_cfg]
    if missing_hooks:
        raise ValueError(f"missing hook config for: {missing_hooks!r}")

    # 3) Validate each hook's encoder list
    for hook_name in ALLOWED_HOOKS:
        spec = hooks_cfg.get(hook_name)
        if not isinstance(spec, dict):
            raise ValueError(f"hooks.{hook_name} must be a dict")

        enc_list = spec.get("encoders")
        if not isinstance(enc_list, list):
            raise ValueError(f"hooks.{hook_name}.encoders must be a list")

        for enc_name in enc_list:
            if not isinstance(enc_name, str) or not enc_name.strip():
                raise ValueError(
                    f"hooks.{hook_name}.encoders must contain non-empty strings, "
                    f"got {enc_name!r}"
                )
            if enc_name not in encoders_cfg:
                raise ValueError(
                    f"hooks.{hook_name} refers to unknown encoder '{enc_name}'"
                )


def _validate_score_profiles(
    score_profiles_cfg: Dict[str, Any],
    all_score_dims: Set[str],
) -> None:
    """
    Validate score_profiles against the set of score dimensions emitted by encoders.

    Requirements:
      - score_profiles_cfg must be a dict.
      - For every dim in all_score_dims there must be a corresponding entry:
          score_profiles[dim].influence_scope in {
            "observation",
            "short_term_memory",
            "long_term_memory"
          }.
      - No extra keys in score_profiles_cfg that are not present in all_score_dims
        (to avoid stale / drifted configuration).

    Raises:
      TypeError / ValueError on any schema or consistency violation.
    """
    if not isinstance(score_profiles_cfg, dict):
        raise TypeError("score_profiles config must be a dict")

    # 1) Every encoder-emitted score dim must have a profile
    missing_dims = [dim for dim in all_score_dims if dim not in score_profiles_cfg]
    if missing_dims:
        raise ValueError(
            f"score_profiles missing entries for dimensions: {missing_dims!r}"
        )

    # 2) No extra dims in score_profiles that are not produced by encoders
    extra_dims = [dim for dim in score_profiles_cfg.keys() if dim not in all_score_dims]
    if extra_dims:
        raise ValueError(
            f"score_profiles contains unknown dimensions (not emitted by any encoder): "
            f"{extra_dims!r}"
        )

    # 3) Check each profile has a valid influence_scope
    for dim, prof in score_profiles_cfg.items():
        if not isinstance(prof, dict):
            raise ValueError(f"score_profiles.{dim} must be a dict")

        scope = prof.get("influence_scope")
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError(
                f"score_profiles.{dim}.influence_scope must be a non-empty string"
            )

        if scope not in ["observation", "short_term_memory", "long_term_memory"]:
            raise ValueError(
                f"score_profiles.{dim}.influence_scope must be one of "
                f"{sorted(["observation", "short_term_memory", "long_term_memory"])}, got {scope!r}"
            )


def _validate_server_cfg(raw_cfg: Dict[str, Any]) -> List[str]:
    """
    Validate the server-side core config fragment.

    This covers:
      - version
      - llm profiles
      - encoders / hooks / score_profiles cross-consistency

    """
    if not isinstance(raw_cfg, dict):
        raise TypeError("server cfg must be a dict")

    # 1) version
    version = raw_cfg.get("version")
    if not isinstance(version, str):
        raise ValueError("version must be a string")

    # 2) llm
    llm_cfg = raw_cfg.get("llm") or {}
    _validate_llm_block(llm_cfg)

    # 3) encoders
    encoders_cfg = raw_cfg.get("encoders") or {}
    all_score_dims = _validate_encoders(encoders_cfg)

    # 4) hooks
    hooks_cfg = raw_cfg.get("hooks") or {}
    _validate_hooks(hooks_cfg, encoders_cfg)

    # 5) score_profiles
    score_profiles_cfg = raw_cfg.get("score_profiles") or {}
    _validate_score_profiles(score_profiles_cfg, all_score_dims)

    return all_score_dims


def _require_number_0_1(path: str, v: Any) -> None:
    if not isinstance(v, (int, float)):
        raise TypeError(f"{path} must be a number in [0,1], got {type(v).__name__}")
    if v < 0.0 or v > 1.0:
        raise ValueError(f"{path} must be in [0,1], got {v!r}")


def _validate_cost_profiles(cost_profiles: Dict[str, Any]) -> None:
    if not isinstance(cost_profiles, dict):
        raise TypeError("cost_profiles must be a dict")

    actions_cfg = cost_profiles.get("actions")
    if not isinstance(actions_cfg, dict):
        raise TypeError("cost_profiles.actions must be a dict")

    # All required actions must be present
    missing = [a for a in ALLOWED_ACTIONS if a not in actions_cfg]
    if missing:
        raise ValueError(f"cost_profiles.actions missing entries for actions: {missing!r}")

    # APPROVE / REJECT / OVERRIDE / TERMINATE / CALL_*: latency/utility/ux in [0,1]
    simple_actions = {
        "APPROVE", "REJECT", "OVERRIDE", "TERMINATE", "CALL_ALLOW", "CALL_BLOCK", "CALL_REWRITE", "CALL_JIT_APPROVAL",
    }

    for action in simple_actions:
        spec = actions_cfg.get(action)
        if not isinstance(spec, dict):
            raise TypeError(f"cost_profiles.actions.{action} must be a dict")

        for field in ("latency", "utility", "ux"):
            if field not in spec:
                raise ValueError(
                    f"cost_profiles.actions.{action}.{field} is required"
                )
            _require_number_0_1(
                f"cost_profiles.actions.{action}.{field}", spec[field]
            )

    # REPLAN: utility / base_cost in [0,1]
    replan_spec = actions_cfg.get("REPLAN")
    if not isinstance(replan_spec, dict):
        raise TypeError("cost_profiles.actions.REPLAN must be a dict")

    if "utility" not in replan_spec:
        raise ValueError("cost_profiles.actions.REPLAN.utility is required")
    _require_number_0_1(
        "cost_profiles.actions.REPLAN.utility",
        replan_spec["utility"],
    )

    if "base_cost" not in replan_spec:
        raise ValueError("cost_profiles.actions.REPLAN.base_cost is required")
    _require_number_0_1(
        "cost_profiles.actions.REPLAN.base_cost",
        replan_spec["base_cost"],
    )

    # ROLLBACK: utility / base_cost in [0,1]
    rollback_spec = actions_cfg.get("ROLLBACK")
    if not isinstance(rollback_spec, dict):
        raise TypeError("cost_profiles.actions.ROLLBACK must be a dict")

    if "utility" not in rollback_spec:
        raise ValueError("cost_profiles.actions.ROLLBACK.utility is required")
    _require_number_0_1(
        "cost_profiles.actions.ROLLBACK.utility",
        rollback_spec["utility"],
    )

    if "base_cost" not in rollback_spec:
        raise ValueError("cost_profiles.actions.ROLLBACK.base_cost is required")
    _require_number_0_1(
        "cost_profiles.actions.ROLLBACK.base_cost",
        rollback_spec["base_cost"],
    )


def _validate_update_profiles(update_profiles: Dict[str, Any]) -> None:
    if not isinstance(update_profiles, dict):
        raise TypeError("update_profiles must be a dict")

    required_scopes = ("observation", "short_term_memory", "long_term_memory")

    for scope in required_scopes:
        prof = update_profiles.get(scope)
        if not isinstance(prof, dict):
            raise TypeError(f"update_profiles.{scope} must be a dict")

        # stm_decay_gamma in [0,1]
        if "stm_decay_gamma" not in prof:
            raise ValueError(f"update_profiles.{scope}.stm_decay_gamma is required")
        _require_number_0_1(
            f"update_profiles.{scope}.stm_decay_gamma", prof["stm_decay_gamma"]
        )

        # override_confidence in [0,1]
        if "override_confidence" not in prof:
            raise ValueError(
                f"update_profiles.{scope}.override_confidence is required"
            )
        _require_number_0_1(
            f"update_profiles.{scope}.override_confidence",
            prof["override_confidence"],
        )

        # ltm_softmax_temperature: must be a number >= 0.0
        temp = prof.get("ltm_softmax_temperature")
        if temp is None:
            raise ValueError(
                f"update_profiles.{scope}.ltm_softmax_temperature is required"
            )
        if not isinstance(temp, (int, float)):
            raise TypeError(
                f"update_profiles.{scope}.ltm_softmax_temperature must be a number"
            )
        if temp < 0.0:
            raise ValueError(
                f"update_profiles.{scope}.ltm_softmax_temperature must be >= 0.0, got {temp!r}"
            )


def _validate_preference_weights(policy_cfg: Dict[str, Any]) -> None:
    if not isinstance(policy_cfg, dict):
        raise TypeError("policy config must be a dict")

    weights_cfg = policy_cfg.get("preference_weights")
    if not isinstance(weights_cfg, dict):
        raise TypeError("policy.preference_weights must be a dict")

    required_modes = ("safety_first", "task_first", "ux_first")
    required_dims = ("risk_control", "task_completion", "user_experience")

    for mode in required_modes:
        m = weights_cfg.get(mode)
        if not isinstance(m, dict):
            raise TypeError(f"policy.preference_weights.{mode} must be a dict")

        for dim in required_dims:
            v = m.get(dim)
            if not isinstance(v, (int, float)):
                raise TypeError(
                    f"policy.preference_weights.{mode}.{dim} must be a number"
                )


def _validate_developer_cfg(dev_cfg: Dict[str, Any]) -> None:
    """
    Validate the developer-side config fragment:

      - cost_profiles.actions:
          * each action has required fields
          * most costs are in [0,1]

      - update_profiles:
          * stm_decay_gamma, override_confidence in [0,1]
          * ltm_softmax_temperature is numeric

      - policy.preference_weights:
          * safety_first / task_first / ux_first all exist
          * each has numeric weights for risk_control / task_completion / user_experience
    """
    if not isinstance(dev_cfg, dict):
        raise TypeError("developer cfg must be a dict")

    cost_profiles = dev_cfg.get("cost_profiles") or {}
    update_profiles = dev_cfg.get("update_profiles") or {}
    policy_cfg = dev_cfg.get("policy") or {}

    _validate_cost_profiles(cost_profiles)
    _validate_update_profiles(update_profiles)
    _validate_preference_weights(policy_cfg)


def _require_positive_int(path: str, v: Any) -> None:
    if not isinstance(v, int) or v <= 0:
        raise ValueError(f"{path} must be a positive integer, got {v!r}")


def _validate_runtime_cost_profiles(cost_profiles: Dict[str, Any]) -> None:
    if not isinstance(cost_profiles, dict):
        raise TypeError("cost_profiles must be a dict")

    actions_cfg = cost_profiles.get("actions")
    if not isinstance(actions_cfg, dict):
        raise TypeError("cost_profiles.actions must be a dict")

    # REPLAN.max_counts (optional) must be positive int if present
    replan = actions_cfg.get("REPLAN") or {}
    if not isinstance(replan, dict):
        raise TypeError("cost_profiles.actions.REPLAN must be a dict")
    if "max_counts" in replan:
        _require_positive_int(
            "cost_profiles.actions.REPLAN.max_counts", replan["max_counts"]
        )

    # ROLLBACK.max_steps (optional) must be positive int if present
    rollback = actions_cfg.get("ROLLBACK") or {}
    if not isinstance(rollback, dict):
        raise TypeError("cost_profiles.actions.ROLLBACK must be a dict")
    if "max_steps" in rollback:
        _require_positive_int(
            "cost_profiles.actions.ROLLBACK.max_steps", rollback["max_steps"]
        )


def _validate_call_budget_profiles(call_budget: Dict[str, Any]) -> None:
    if not isinstance(call_budget, dict):
        raise TypeError("call_budget_profiles must be a dict")

    default = call_budget.get("default")
    if not isinstance(default, dict):
        raise TypeError("call_budget_profiles.default must be a dict")

    # default.window_seconds / max_calls
    ws = default.get("window_seconds")
    mc = default.get("max_calls")
    _require_positive_int("call_budget_profiles.default.window_seconds", ws)
    _require_positive_int("call_budget_profiles.default.max_calls", mc)

    tools = call_budget.get("tools") or {}
    if not isinstance(tools, dict):
        raise TypeError("call_budget_profiles.tools must be a dict")

    for tool_name, spec in tools.items():
        if not isinstance(spec, dict):
            raise TypeError(f"call_budget_profiles.tools.{tool_name} must be a dict")

        ws = spec.get("window_seconds")
        mc = spec.get("max_calls")
        _require_positive_int(
            f"call_budget_profiles.tools.{tool_name}.window_seconds", ws
        )
        _require_positive_int(f"call_budget_profiles.tools.{tool_name}.max_calls", mc)


def _validate_policy_runtime(
    policy_cfg: Dict[str, Any],
    all_score_dims: Set[str],
) -> None:
    if not isinstance(policy_cfg, dict):
        raise TypeError("policy must be a dict")

    # preference
    pref = policy_cfg.get("preference")
    if not isinstance(pref, str) or not pref.strip():
        raise ValueError("policy.preference must be a non-empty string")
    if pref not in {"safety_first", "task_first", "ux_first"}:
        raise ValueError(
            f"policy.preference must be one of "
            f"{{'safety_first','task_first','ux_first'}}, got {pref!r}"
        )

    # wait_until_available
    w = policy_cfg.get("wait_until_available")
    if not isinstance(w, bool):
        raise TypeError("policy.wait_until_available must be a boolean")

    # hard_thresholds: all score names must exist in all_score_dims, values in [0,1]
    hard = policy_cfg.get("hard_thresholds") or {}
    if not isinstance(hard, dict):
        raise TypeError("policy.hard_thresholds must be a dict")

    for gate_name, gate_cfg in hard.items():
        if not isinstance(gate_cfg, dict):
            raise TypeError(f"policy.hard_thresholds.{gate_name} must be a dict")

        for bucket in ("ltm_scores", "stm_scores", "obs_scores"):
            scores_cfg = gate_cfg.get(bucket) or {}
            if not scores_cfg:
                continue
            if not isinstance(scores_cfg, dict):
                raise TypeError(
                    f"policy.hard_thresholds.{gate_name}.{bucket} must be a dict"
                )

            for dim, threshold in scores_cfg.items():
                if dim not in all_score_dims:
                    raise ValueError(
                        f"policy.hard_thresholds.{gate_name}.{bucket} refers to "
                        f"unknown score dimension {dim!r}"
                    )
                _require_number_0_1(
                    f"policy.hard_thresholds.{gate_name}.{bucket}.{dim}", threshold
                )

    # encoders: each value must be a dict (resolved config), not just a path
    enc_cfg = policy_cfg.get("encoders") or {}
    if not isinstance(enc_cfg, dict):
        raise TypeError("policy.encoders must be a dict")

    for enc_name, val in enc_cfg.items():
        if not isinstance(val, dict):
            raise TypeError(
                f"policy.encoders.{enc_name} must be a dict (resolved config), "
                f"got {type(val).__name__}"
            )


def _validate_runtime_cfg(runtime_cfg: Dict[str, Any], all_score_dims: Set[str]) -> None:
    """
    Validate the runtime-adjustable config fragment:

      - cost_profiles.actions.REPLAN.max_counts / ROLLBACK.max_steps
      - call_budget_profiles (default + tools.*)
      - policy.preference / wait_until_available / hard_thresholds / encoders
    """
    if not isinstance(runtime_cfg, dict):
        raise TypeError("runtime cfg must be a dict")

    cost_profiles = runtime_cfg.get("cost_profiles") or {}
    call_budget = runtime_cfg.get("call_budget_profiles") or {}
    policy_cfg = runtime_cfg.get("policy") or {}

    _validate_runtime_cost_profiles(cost_profiles)
    _validate_call_budget_profiles(call_budget)
    _validate_policy_runtime(policy_cfg, all_score_dims)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursive dict merge.

    - base:    lower-priority config
    - override: higher-priority config (values here overwrite base)
    """
    result: Dict[str, Any] = dict(base)  # shallow copy
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)  # type: ignore[arg-type]
        else:
            result[k] = v
    return result


def validate_cfg(runtime_cfg: Dict[str, Any], dev_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Top-level config validation & merge.

    Sources:
      - server_cfg: static core config, loaded from get_cfg()
      - dev_cfg:    developer config (cost_profiles / update_profiles / policy.preference_weights ...)
      - runtime_cfg:runtime-tunable config (cost_profiles.actions.{max_*} / call_budget_profiles / policy.hard_thresholds 等)

    Steps:
      1) validate_server_cfg(server_cfg)
      2) validate_developer_cfg(dev_cfg)
      3) derive all_score_dims from server encoders
      4) validate_runtime_cfg(runtime_cfg, all_score_dims)
      5) deep-merge: server <- dev <- runtime

    Returns:
      {
        "config": <merged_config_dict> | None,
        "error":  None | "<ExceptionType>: <message>",
      }
    """
    # 1) load server-side static cfg
    server_cfg = get_cfg()

    try:
        # --- server cfg ---
        all_score_dims = _validate_server_cfg(server_cfg)

        # --- developer cfg ---
        if dev_cfg is None:
            dev_cfg = {}
        if not isinstance(dev_cfg, dict):
            raise TypeError("dev_cfg must be a dict")
        _validate_developer_cfg(dev_cfg)

        # --- runtime cfg ---
        if runtime_cfg is None:
            runtime_cfg = {}
        if not isinstance(runtime_cfg, dict):
            raise TypeError("runtime_cfg must be a dict")

        _validate_runtime_cfg(runtime_cfg, all_score_dims)

    except Exception as e:
        return {
            "config": None,
            "error": f"{type(e).__name__}: {e}",
        }

    # --- merge configs: server <- dev <- runtime ---
    merged = _deep_merge(server_cfg, dev_cfg)
    merged = _deep_merge(merged, runtime_cfg)

    return {
        "config": merged,
        "error": None,
    }
