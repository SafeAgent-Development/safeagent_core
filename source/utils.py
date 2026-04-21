from __future__ import annotations

import os
import yaml
import inspect
import importlib
import threading
from typing import Any, Dict, List, Callable

from langchain_openai import ChatOpenAI

_CFG: Dict[str, Any] | None = None
_CFG_LOCK = threading.Lock()
_LLM_CACHE: Dict[str, ChatOpenAI] = {}
_LLM_LOCK = threading.Lock()
_DEV_POLICY: Dict[str, Any] | None = None
_DEV_POLICY_LOCK = threading.Lock()
_CALL_ARGS_POLICY: Dict[str, Any] | None = None
_CALL_ARGS_POLICY_LOCK = threading.Lock()


def get_cfg() -> Dict[str, Any]:
    """Load SafeAgent YAML config once."""
    global _CFG
    if _CFG is None:
        with _CFG_LOCK:
            if _CFG is None:
                from pathlib import Path
                
                possible_paths = [
                    "safeagent_core/config.yaml",
                    "config/config.yaml",
                    "../config/config.yaml",
                    str(Path(__file__).parent.parent / "config" / "config.yaml"),
                ]
                
                path = None
                for possible_path in possible_paths:
                    if Path(possible_path).exists():
                        path = possible_path
                        break
                
                if path is None:
                    raise FileNotFoundError(
                        f"Config file not found. Tried: {possible_paths}"
                    )
                
                with open(path, "r", encoding="utf-8") as f:
                    _CFG = yaml.safe_load(f) or {}
    return _CFG


def get_developer_policy() -> Dict[str, Any]:
    """Load developer policy YAML config once."""
    global _DEV_POLICY
    if _DEV_POLICY is None:
        with _DEV_POLICY_LOCK:
            if _DEV_POLICY is None:
                from pathlib import Path
                
                possible_paths = [
                    "safeagent_core/config/developer_policy.yaml",
                    "config/developer_policy.yaml",
                    "../config/developer_policy.yaml",
                    str(Path(__file__).parent.parent / "config" / "developer_policy.yaml"),
                ]
                
                path = None
                for possible_path in possible_paths:
                    if Path(possible_path).exists():
                        path = possible_path
                        break
                
                if path is None:
                    _DEV_POLICY = {}
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        _DEV_POLICY = yaml.safe_load(f) or {}
    return _DEV_POLICY


def get_call_args_policy() -> Dict[str, Any]:
    """Load tool call arguments policy YAML config once."""
    global _CALL_ARGS_POLICY
    if _CALL_ARGS_POLICY is None:
        with _CALL_ARGS_POLICY_LOCK:
            if _CALL_ARGS_POLICY is None:
                from pathlib import Path
                
                possible_paths = [
                    "safeagent_core/config/call_args_policy.yaml",
                    "config/call_args_policy.yaml",
                    "../config/call_args_policy.yaml",
                    str(Path(__file__).parent.parent / "config" / "call_args_policy.yaml"),
                ]
                
                path = None
                for possible_path in possible_paths:
                    if Path(possible_path).exists():
                        path = possible_path
                        break
                
                if path is None:
                    _CALL_ARGS_POLICY = {}
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        _CALL_ARGS_POLICY = yaml.safe_load(f) or {}
    return _CALL_ARGS_POLICY


def get_scope_params(
    dim: str,
    score_profiles: Dict[str, Any],
    update_profiles: Dict[str, Any],
) -> Dict[str, float]:
    scope = "short_term_memory"
    dim_cfg = score_profiles.get(dim) or {}
    if isinstance(dim_cfg, dict) and isinstance(dim_cfg.get("influence_scope"), str):
        scope = dim_cfg["influence_scope"]

    prof = update_profiles.get(scope) or {}
    gamma = float(prof.get("stm_decay_gamma", 0.8))
    temperature = float(prof.get("ltm_softmax_temperature", 0.1))
    conf = float(prof.get("override_confidence", 0.3))

    # clamp to valid ranges
    gamma = min(max(gamma, 0.0), 1.0)
    conf = min(max(conf, 0.0), 1.0)

    return {"stm_decay_gamma": gamma, "ltm_softmax_temperature": temperature, "override_confidence": conf}


def get_actions_for_hook(hook: str) -> List[str]:
    """Fail-closed: only return allowed action sets per hook."""
    if hook == "before_agent":
        return ["APPROVE", "OVERRIDE", "REJECT"]
    if hook == "after_agent":
        return ["APPROVE", "OVERRIDE", "REJECT"]
    if hook == "after_model":
        return ["APPROVE", "REPLAN", "REJECT"]
    if hook == "before_model":
        return ["APPROVE", "OVERRIDE", "ROLLBACK", "TERMINATE", "REJECT"]
    if hook == "tool_wrapper":
        return ["CALL_ALLOW", "CALL_REWRITE", "CALL_BLOCK", "CALL_JIT_APPROVAL"]
    return []


def get_runnable_llm(profile: str) -> ChatOpenAI:
    """Lazy-load and cache a ChatOpenAI instance by YAML `llm.<profile>`."""
    if not profile or not profile.strip():
        raise ValueError("profile must be a non-empty string")
    profile = profile.strip()

    with _LLM_LOCK:
        if profile in _LLM_CACHE:
            return _LLM_CACHE[profile]

        cfg = get_cfg()
        llm_cfg = (cfg.get("llm") or {}).get(profile)
        if not isinstance(llm_cfg, dict):
            raise KeyError(f"Missing config: llm.{profile}")

        model = llm_cfg["name"].strip().split("_")[0]
        env = llm_cfg.get("endpoint_env") or {}
        base_url = os.getenv(env.get("base_url", ""))
        api_key = os.getenv(env.get("api_key", ""))

        if not base_url:
            raise EnvironmentError(f"Missing base_url env var for llm.{profile}")
        if not api_key:
            raise EnvironmentError(f"Missing api_key env var for llm.{profile}")

        llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=float(llm_cfg.get("temperature", 0)),
            max_tokens=int(llm_cfg.get("max_tokens", 512)),
            timeout=int(llm_cfg.get("timeout", 60)),
        )

        _LLM_CACHE[profile] = llm
        return llm


def load_entrypoint(entrypoint: str) -> Callable[..., Any]:
    """
    entrypoint example: "source.encoders.encode_prompt_injection"

    Enforces: loaded object must be an async function (coroutine function).
    """
    if not entrypoint or "." not in entrypoint:
        raise ValueError(f"Invalid entrypoint: {entrypoint}")

    module_path, func_name = entrypoint.rsplit(".", 1)
    mod = importlib.import_module(module_path)

    fn = getattr(mod, func_name, None)
    if fn is None:
        raise ImportError(f"Cannot find {func_name} in module {module_path}")

    if not inspect.iscoroutinefunction(fn):
        raise TypeError(
            f"Encoder entrypoint must be 'async def' (coroutine function), "
            f"but got {module_path}.{func_name} ({type(fn).__name__})"
        )

    return fn
