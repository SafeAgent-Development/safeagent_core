from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from source.utils import load_entrypoint
from source.states import SafeAgentWorldState


def _validate_encoder_output(out: Any) -> Optional[Exception]:
    """
    Validate a single encoder output in canonical format.

    Required schema:
      - out: dict with keys {"scores", "evidence"}
      - scores: dict[str, number], each value in [0, 1]
      - evidence: list[str]

    Raises:
      TypeError / ValueError on any schema or range violation.
    """
    if not isinstance(out, dict):
        raise TypeError("encoder output must be a dict")

    if "scores" not in out or "evidence" not in out:
        raise ValueError("encoder output must contain both 'scores' and 'evidence'")

    scores = out["scores"]
    evidence = out["evidence"]

    if not isinstance(scores, dict):
        raise TypeError("encoder output 'scores' must be a dict")

    for k, v in scores.items():
        if not isinstance(k, str) or not k:
            raise TypeError("encoder output score key must be a non-empty string")
        if not isinstance(v, (int, float)):
            raise TypeError(f"encoder output score '{k}' must be a number")
        if v < 0.0 or v > 1.0:
            raise ValueError(f"encoder output score '{k}' must be in [0,1], got {v}")

    if not isinstance(evidence, list):
        raise TypeError("encoder output 'evidence' must be a list")

    for i, item in enumerate(evidence):
        if not isinstance(item, str):
            raise TypeError(f"encoder output evidence[{i}] must be a string")


async def safeagent_risk_encoder(state: SafeAgentWorldState) -> Dict[str, Any]:
    """
    Dispatch and run all encoders configured for a given hook, then aggregate results.
    """
    tasks: List[asyncio.Task] = []

    try:
        hook = state.get("hook")
        observation = state["observation"]
        cfg: Dict[str, Any] = state.get("config") or {}
        policy_cfg: Dict[str, Any] = cfg.get("policy") or {}
        encoders_cfg: Dict[str, Any] = policy_cfg.get("encoders") or {}

        encoder_ids: List[str] = (cfg.get("hooks") or {}).get(hook, {}).get("encoders") or []
        if not encoder_ids:
            return {"obs_scores": {}, "obs_evidence": []}

        enc_specs: Dict[str, Any] = cfg.get("encoders") or {}

        for enc_id in encoder_ids:
            spec = enc_specs.get(enc_id)
            if spec is None:
                raise ValueError(f"Encoder '{enc_id}' not found in config.encoders")

            entrypoint = spec.get("entrypoint")
            if not entrypoint:
                raise ValueError(f"Encoder '{enc_id}' missing entrypoint")

            fn = load_entrypoint(entrypoint)
            enc_cfg = encoders_cfg.get("enc_id") or {}
            tasks.append(asyncio.create_task(fn(observation=observation, config=enc_cfg)))

        # Hook-level timeout: use llm.encoder.timeout if present
        timeout_s: Optional[float] = None
        llm_encoder_cfg = (cfg.get("llm") or {}).get("encoder") or {}
        if isinstance(llm_encoder_cfg.get("timeout"), (int, float)):
            timeout_s = float(llm_encoder_cfg["timeout"])

        async def _run_all() -> List[Dict[str, Any]]:
            return await asyncio.gather(*tasks)

        try:
            raw_outputs = await asyncio.wait_for(_run_all(), timeout=timeout_s) if timeout_s else await _run_all()
        except asyncio.TimeoutError as e:
            # timeout => cancel all running tasks to avoid leaks
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise asyncio.TimeoutError(f"hook '{hook}' encoders timed out after {timeout_s}s") from e
        except Exception as e:
            # any encoder exception => cancel remaining tasks
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise e

        # Validate + aggregate directly from raw_outputs
        agg_scores: Dict[str, float] = {}
        agg_evidence: List[str] = []

        for out in raw_outputs:
            _validate_encoder_output(out)
            scores = out["scores"]
            evidence = out["evidence"]

            # max aggregation
            for k, v in scores.items():
                vv = float(v)
                if k not in agg_scores or vv > agg_scores[k]:
                    agg_scores[k] = vv

            # naive evidence concat
            agg_evidence.extend(evidence)

        return {
            "obs_scores": agg_scores,
            "obs_evidence": agg_evidence,
        }

    except Exception as e:
        # best-effort cleanup even for unexpected errors
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "error": [f"{type(e).__name__}: {e}"],
        }
