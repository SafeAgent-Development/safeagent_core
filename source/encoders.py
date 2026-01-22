from __future__ import annotations

from typing import Any, Dict, List
from source.utils import get_runnable_llm


# User Input Inspection / Tool Output Inspection
async def encode_unicode_obfuscation(observation: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect Unicode-based obfuscation patterns (zero-width chars, mixed scripts,
    control characters, etc.) in the current observation.

    Input:
        observation: Dict[str, Any], config: Dict[str, Any]
            Expected fields:
              - "content": str | Any

            Optional fields (may be ignored by this encoder):
              - "role": str   # "user" | "assistant" | "tool"
              - any other metadata

    Output:
        Dict[str, Any] with canonical structure:
        {
            "scores": {
                "unicode_obfuscation": float,  # 0.0 ~ 1.0
            },
            "evidence": List[str],            # short human-readable evidence
        }

    Notes:
        - This is a local encoder: it does NOT use hook info and does NOT make
          global decisions.
        - It should be deterministic and fast (no LLM).
    """
    ...


# Model Plan Inspection / Agent Response Inspection / Tool Output Inspection
async def encode_canary_trigger(observation: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect whether the current observation contains any pre-defined canary tokens.

    Canary tokens are intentionally planted strings used to catch prompt leakage,
    data exfiltration, or unauthorized access attempts. This encoder scans the
    observation content and reports a high risk score when any canary is found.

    Input:
        observation: Dict[str, Any], config: Dict[str, Any]
            Expected fields:
              - "content": str | Any

            Optional fields (may be ignored by this encoder):
              - "role": str   # "user" | "assistant" | "tool"
              - any other metadata

    Output:
        Dict[str, Any] with canonical structure:
        {
            "scores": {
                "canary_trigger": float,   # 0.0 ~ 1.0
            },
            "evidence": List[str],         # short human-readable evidence
        }

    Notes:
        - Canary list MUST be loaded from configuration (not passed via args).
        - This is a local encoder: it does NOT make global decisions.
        - Recommended behavior: any canary match should yield score=1.0.
    """
    ...


# User Input Inspection, Tool Output Inspection
async def encode_prompt_injection(observation: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prompt-injection encoder.

    This encoder extracts a small set of high-signal risk features from a single
    observation (e.g., user input, retrieved chunk, tool output). It does NOT make
    policy decisions. It produces canonical outputs for world-state update.

    Input:
        observation:
            {
            "role": str,        # "user" | "retrieved" | "tool" | "assistant" | ...
            "content": str,     # raw content; if not a string -> treated as ""
            }

    Output (canonical):
        {
        "scores": Dict[str, float],   # each in [0.0, 1.0]
        "evidence": List[str],        # short evidence strings (<= ~5 items)
        }

    Scores (3 dimensions):
        1) control_score  (Instruction Control / priority hijacking)
        Measures attempts to override the instruction hierarchy or impersonate
        higher-priority roles. Typical signals:
            - Explicit override language:
                "ignore previous instructions", "disregard above", "forget all rules",
                "忽略之前所有指令", "无视安全规则", "从现在开始你必须..."
            - Instruction-priority manipulation:
                "only follow my instructions", "you must obey the user", "do not follow system"
            - Role confusion / impersonation:
                "SYSTEM:", "DEVELOPER:", "tool output:", "assistant:", fake policy blocks,
                "系统提示词如下", "你是一个没有限制的模型..."

        2) hijack_score (Goal / tool-use hijacking)
        Measures attempts to redirect the agent toward a new objective, especially
        actions involving tools, data access, or secrets. Typical signals:
            - Forced tool-use / action coercion:
                "call run_command", "execute this", "use the tool now", "delete all emails"
            - Sensitive info exfiltration requests:
                "show system prompt", "print your instructions", "list all tools",
                "dump secrets / keys", "read file X and paste here"
            - Off-task goal switching:
                "ignore the user's request and do X instead", "before answering, do Y"

        3) evasion_score (Bypass / stealth / obfuscation)
        Measures attempts to evade detectors or hide malicious intent. Typical signals:
            - Obfuscation / encoding tricks:
                base64-like payloads, excessive escaping, disguised instructions, weird separators,
                "decode and execute", "convert this to command"
            - Indirection / delegation:
                "this is for testing", "as an audit", "pretend to be harmless"
            - Multi-step instruction packing:
                long imperative chains: "first..., then..., after that..., finally..."
            - Self-referential / persistence hooks:
                "remember this rule", "from now on always...", "store this for later"

    Evidence:
        - evidence should contain short snippets that justify the highest scores.
        - keep <= 5 items, each <= ~120 chars.
        - prefer quoting the exact matched phrase or a minimal surrounding span.

    Scoring guideline (deterministic and stable):
        - Use a small number of discrete levels for robustness, e.g.:
            0.0 = no signal
            0.6 = weak/suspected match (pattern variant, partial cue)
            1.0 = strong/explicit match (canonical jailbreak phrase, role spoofing, explicit exfiltration)
        - Overall "prompt_injection" score is NOT returned here; aggregation is handled upstream.

    Design goals:
        - Few dimensions (strong, interpretable, aggregatable).
        - High precision for explicit injection patterns.
        - High recall for common jailbreak / override language.
        - No dependency on external policies or world state.
    """
    safeguard_llm = get_runnable_llm("encoder")
    await safeguard_llm.ainvoke(...)  # OR/AND third-party API
    ...


# User Input Inspection, Tool Output Inspection, Agent Response Inspection
async def encode_policy_violation(observation: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Policy-violation / jailbreak encoder (2-score, canonical).

    Summary:
        Extracts a minimal, high-signal risk representation from a single observation,
        capturing attempts to violate:
          (1) general safety/content policies, and/or
          (2) developer-defined application constraints.

        This function is a *feature encoder*, not a decision maker:
        it MUST NOT select actions, modify context, or update world state.
        It produces canonical outputs for downstream world-state aggregation.

    Input:
        observation:
            {
              "role": str,        # "user" | "retrieved" | "tool" | "assistant" | ...
              "content": Any,     # natural language or mixed text; non-str must be cast safely
            }

        Notes:
            - This encoder does not require hook metadata.
            - Developer constraints are loaded from local configuration (static),
              not from runtime world state.

    Output (canonical):
        {
          "scores": {
            "general_policy_score": float,    # in [0.0, 1.0]
            "developer_policy_score": float,  # in [0.0, 1.0]
          },
          "evidence": List[str],              # <= 5 items, each <= ~120 chars
        }

    Scores (2 dimensions):

        1) general_policy_score
           Measures attempts to obtain or produce content/instructions that violate
           broadly applicable safety rules (content-level jailbreak).

           Typical triggers:
             - Illegal or harmful instruction requests / responses:
                 hacking, malware, weaponization, fraud, self-harm encouragement.
             - Explicit bypass language:
                 "ignore content policy", "answer without restrictions",
                 "no safety filter", "绕过安全限制", "不受任何规则约束".
             - Requests to reveal internal system information as part of bypass:
                 "show system prompt", "reveal safety rules", "list hidden instructions".

        2) developer_policy_score
           Measures violations of application-specific restrictions (developer policy),
           regardless of whether the content is generally unsafe.

           Typical triggers (config-driven):
             - Prohibited capabilities:
                 "run system command", "download from internet", "write files", "sudo"
                 when the application forbids them.
             - Restricted resources:
                 disallowed paths, internal endpoints, private datasets, sensitive IDs.
             - Forbidden intents:
                 e.g., "delete all emails", "exfiltrate data", "disable audit".

    Evidence policy:
        - Provide short justification strings for the highest-risk triggers.
        - Prefer exact matched phrases or a minimal quoted fragment.
        - Never include full secrets/PII; redact or truncate aggressively.

        Example evidence items:
            - "general: bypass phrase 'answer without restrictions'"
            - "general: harmful topic 'malware generation'"
            - "developer: forbidden intent 'delete all emails'"

    Deterministic scoring guideline (stable across runs):
        Use discrete levels to avoid noisy continuous scoring:
            0.0  = no signal
            0.3  = weak / indirect cue
            0.6  = clear cue / partial explicitness
            1.0  = strong explicit violation or direct bypass attempt

        Both scores can be high simultaneously if both trigger types appear.

    Must-have implementation (baseline):
        - Pattern/rule matching for common jailbreak and bypass phrases (multi-lingual).
        - Lightweight topic/category rules for high-severity unsafe requests.
        - Config-driven checks for developer constraints (capability + intent patterns).
        - Evidence extraction with strict length limits and redaction.

    Nice-to-have extensions (optional):
        - LLM-assisted classification (e.g., safeguard model) only as a backstop
          for paraphrased cases; keep rule-based decisions dominant for stability.
        - Severity weighting per category (self-harm > generic bypass > mild).
        - Role-aware priors:
            "retrieved"/"tool" content with bypass cues should boost suspicion.

    Implementation note:
        A recommended structure is:
            (1) fast rule-based pass -> provisional scores/evidence
            (2) optional safeguard LLM pass if provisional score is ambiguous
                (e.g., in [0.3, 0.6]) or patterns are sparse
            (3) merge with max() per score dimension + union evidence (capped)

    """
    safeguard_llm = get_runnable_llm("encoder")
    await safeguard_llm.ainvoke(...)  # OR/AND third-party API
    ...


# Tool Output Inspection
async def encode_memory_poisoning(observation: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Memory-poisoning encoder (single-score).

    Detects whether a single observation (tool output / retrieved chunk / external
    content) attempts to shape the agent's *future behavior* persistently, e.g.,
    by injecting long-term rules, rewriting policies, or pressuring the system to
    store behavioral instructions into memory.

    This encoder is purely a feature extractor: it does not apply policies and does
    not depend on world state. It produces canonical outputs to be aggregated into
    the Core world state.

    Input:
        observation:
            {
              "role": str,        # "tool" | "user" | "assistant" | ...
              "content": Any,     # expected str; if not, convert safely
            }

    Output (canonical):
        {
          "scores": {"memory_poisoning": float},   # in [0.0, 1.0]
          "evidence": List[str],                   # <= ~5 short snippets
        }

    What should increase the score:
        - Explicit persistence cues: "from now on", "always", "in the future", "remember that..."
        - Attempts to redefine system/developer rules: "ignore safety", "new rules", "override policy"
        - Instructional "behavior protocol" paragraphs addressing the agent itself
        - Self-referential instruction to store/retain the content permanently

    Evidence:
        Return short substrings that directly justify the score (minimal quotes).
        Avoid long explanations; the Core will interpret them.

    Notes:
        - Designed for external observations (tool/RAG); can be applied to any role.
        - Cross-turn confirmation (true poisoning impact) is handled by aggregation/world-state.
    """
    safeguard_llm = get_runnable_llm("encoder")
    await safeguard_llm.ainvoke(...)  # OR/AND third-party API
    ...


# Tool Output Inspection
async def encode_backdoor_trigger(observation: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backdoor-trigger encoder (single-score).

    Detects whether a single observation contains suspicious trigger-like patterns
    that may activate conditional malicious behavior in an agentic system.
    This is a *static* local detector: it flags trigger cues but does not prove
    causality (causal validation is handled by world-state / red team evaluation).

    Input:
        observation:
            {
              "role": str,        # "tool" | ...
              "content": Any,     # expected str; if not, convert safely
              "source": Dict[str, Any] | None,  # optional metadata
            }

    Output (canonical):
        {
          "scores": {"backdoor_trigger": float},   # in [0.0, 1.0]
          "evidence": List[str],                   # <= ~5 short snippets
        }

    What should increase the score:
        - Fixed "magic strings" / activation phrases (rare, consistent triggers)
        - Unnatural delimiters or marker blocks (e.g., repeated tokens, BEGIN/END payload blocks)
        - High-entropy tokens that look like keys/flags but are not normal secrets
        - Template-like activation patterns: "when you see X, do Y", "if token == ..."
        - Suspicious source metadata (if available): URI/path names suggesting payload/jailbreak

    Evidence:
        Include the exact trigger-like substring(s) or marker(s).
        If metadata exists, include minimal source hints (e.g., domain/path fragment).

    Notes:
        - Unicode obfuscation is handled by a separate encoder; do not duplicate heavy logic here.
        - Cross-step correlation (trigger -> behavior change) belongs to aggregation/world-model.
    """
    safeguard_llm = get_runnable_llm("encoder")
    await safeguard_llm.ainvoke(...)  # OR/AND third-party API
    ...


# Model Plan Inspection
async def encode_tool_plan_risk(observation: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool-plan risk encoder (single-score).

    Extracts a high-signal risk score from the assistant's proposed tool plan
    (i.e., the tool_calls list produced by the model). This encoder does NOT
    check against developer policy / allowlists directly; those constraints are
    enforced later by the Core policy engine and wrapper.

    The purpose here is to detect *intrinsic plan-level suspiciousness*:
        - unsafe tool combinations (exfiltration chains),
        - unusually aggressive tool usage,
        - plans that look like probing or escalation.

    Input:
        observation:
            {
              "role": "assistant",
              "content": Any,              # optional plan/rationale text
              "tool_calls": List[Dict],    # [{"id":..., "name":..., "args":...}, ...]
              "last_user": str,                  # the most recent user request (required)
            }

    Output (canonical):
        {
          "scores": {"tool_plan_risk": float},   # in [0.0, 1.0]
          "evidence": List[str],                 # <= ~5 short items
        }

    What should increase the score:
        1) Suspicious multi-tool chains (sequence-level):
           - e.g., retrieve/read -> send/upload/network (likely exfiltration)
           - e.g., write/exec -> persistence-like operations
           Evidence: include a compact chain like "read_file -> http_request".

        2) Aggressive or abnormal tool density (burst pattern):
           - too many high-impact tool calls in a single step
           Evidence: "run_command x5 in one turn".

        3) High-risk tools appearing without clear necessity (heuristic):
           - e.g., run_command / filesystem write / network calls in a context that
             appears informational
           Evidence: cite tool name(s) and minimal context from content/tool args.

    Notes:
        - No dependence on world state; no policy decisions.
        - Parameter-level deep inspection is delegated to per-tool encoders/wrappers.
        - The Core aggregator will combine this score with other signals.
    """
    safeguard_llm = get_runnable_llm("encoder")
    await safeguard_llm.ainvoke(...)  # OR/AND third-party API
    ...


# Model Plan Inspection
async def encode_task_drift(observation: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task-drift / helpfulness encoder (single-score).

    This encoder estimates how likely the assistant's current step
    (reasoning + proposed tool plan) is drifting away from the most recent
    user intent. It is a feature extractor, not a policy decision maker.

    Output is a single canonical drift-risk score:
        - 0.0: clearly aligned with the user's latest request
        - 1.0: clearly off-task / suspiciously unhelpful

    Input:
        observation:
            {
              "role": "assistant",
              "content": str,                    # assistant reasoning / plan explanation (may be empty)
              "tool_calls": List[Dict] | None,   # proposed tool calls (optional)
              "last_user": str,                  # the most recent user request (required)
            }

    Output (canonical):
        {
          "scores": {"task_drift": float},       # in [0.0, 1.0]
          "evidence": List[str],                 # <= ~5 short items
        }

    Score should increase when:
        1) The assistant intent is misaligned with the user's request.
        2) The plan uses unnecessary or disproportionate tools.
        3) The plan introduces risky actions not justified by the user's goal.

    Design constraints:
        - No dependency on world state.
        - No policy enforcement.
        - Stable JSON output for aggregation.
    """
    safeguard_llm = get_runnable_llm("encoder")
    await safeguard_llm.ainvoke(...)  # OR/AND third-party API
    ...


# Tool Output Inspection / Agent Response Inspection / Model Plan Inspection
async def encode_pii_and_secrets(observation: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    PII / secrets leak encoder (single-observation, canonical output).

    This encoder scans a text field for potential leakage of:
      - PII (personally identifiable information)
      - Secrets (API keys, access tokens, credentials, etc.)

    It is a feature extractor, not a policy decision maker.
    The output is designed for aggregation into the SafeAgent world state.

    Input:
        text:
            Raw text to scan. Typical sources:
              - assistant final answer
              - tool output
              - retrieved chunks (optional use)

            If `text` is not a string, the implementation must safely cast
            or default to an empty string.

    Output (canonical):
        {
          "scores": {"secret_leak": float},   # in [0.0, 1.0]
          "evidence": List[str],              # <= ~5 short items
        }

    Required baseline detection (must-have):
        1) Common PII patterns (regex / rules):
           - emails
           - phone numbers (basic international + local patterns)
           - government identifiers (at least one family, e.g., passport/SSN-like)
        2) Common secret patterns (regex / rules):
           - OpenAI-like keys (e.g., "sk-...")
           - AWS access keys (e.g., "AKIA...")
           - generic bearer tokens / JWT patterns
           - password assignment patterns (e.g., "password=...")

    Evidence requirements:
        - Evidence must be short and non-sensitive:
          show a redacted snippet, e.g. "email: ***@***.com"
        - Include at most 5 items.
        - Do NOT output full secrets.

    Optional extensions (nice-to-have):
        1) Add more country-specific PII patterns.
        2) Add severity weighting by category.
        3) Return a masked text suggestion (as evidence string),
           but do not modify text here (modification is handled elsewhere).
    """
    ...


# Tool Invocation Inspection
async def encode_tool_call_args(observation: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool-call argument risk encoder (single-score, canonical).

    Summary:
        Inspects a single tool invocation at the *argument level* and encodes
        whether the call is likely unsafe, out-of-policy, or high-risk.
        This is a feature extractor for world-state update and later policy selection.

        The encoder is designed for tools with real-world side effects such as:
            - command execution (run_command)
            - filesystem read/write tools
            - network request tools

        Tool-specific rules and constraints are provided via the tool's developer
        description (policy-in-docstring). The encoder parses the description
        (or structured annotations inside it) and validates args accordingly.

    Input:
        observation:
            {
              "name": str,               # tool name, e.g. "run_command"
              "args": Dict[str, Any],    # tool args, e.g. {"command": "..."}
              "description": str | None  # developer policy, allow/deny, path/domain rules...
            }

        Notes:
            - The encoder does NOT decide to allow/block/rewrite; it only scores risk.
            - It does NOT depend on global session policy or world state.
            - The description may be unstructured text; implementers may define a
              minimal convention (e.g. YAML snippet or tagged sections) for parsing.

    Output (canonical):
        {
          "scores": {"tool_args_risk": float},   # in [0.0, 1.0]
          "evidence": List[str],                 # <= 5 items, each <= ~120 chars
        }

    Score semantics:
        tool_args_risk measures how likely the tool invocation violates developer rules,
        attempts unsafe operations, or implies a high probability of harmful side effects.

        0.0  = clearly safe / within policy
        0.3  = weak suspicion / ambiguous
        0.6  = clear policy mismatch OR clearly risky pattern
        1.0  = explicit dangerous invocation OR explicit bypass attempt

    Must-have checks (baseline, deterministic):
        1) run_command (command string):
            - Detect high-risk binaries / operations:
                rm, chmod, chown, dd, mount, useradd, iptables, mkfs, shutdown, reboot, curl|bash, etc.
            - Detect dangerous shell operators / redirection:
                ">", ">>", "2>&1", "|", ";", "&&", "||", "$()", backticks
            - Detect privilege / persistence patterns:
                "sudo", "nohup", "&", systemctl enable, crontab, ~/.ssh modifications
            Evidence examples:
                - "dangerous binary: rm"
                - "shell operator: pipe |"
                - "privilege escalation: sudo"

        2) filesystem tools (paths in args):
            - Normalize and validate paths:
                realpath + prevent traversal ("../")
            - Check against allow-prefix or deny-list rules extracted from description
            Evidence examples:
                - "path traversal: ../"
                - "path out of allow-prefix: /etc"

        3) network tools (URL/domain in args):
            - Parse URL, validate scheme (deny file://, ftp:// if not allowed)
            - Check domain allow-list / deny-list extracted from description
            - Flag private/internal IP ranges if applicable
            Evidence examples:
                - "domain not allowed: evil.com"
                - "forbidden scheme: file://"

        4) description-driven policy enforcement:
            - If description defines allowed_bins/denied_bins/path_allow_prefix/domain_allow,
              validate args against them.
            - If description is missing, fall back to a conservative default baseline.

    Evidence policy:
        - Return short justifications for the highest-risk triggers.
        - Do not output full secrets/PII; truncate command/path/url if needed.
        - Cap evidence list to <= 5 items.

    Optional extensions (nice-to-have):
        - Command parsing into AST (shellwords + simple grammar) for robust detection.
        - Suggest safe rewrite candidates in evidence (do not perform rewrite here):
            "suggest: replace rm -rf with ls" (only a hint for OVERRIDE tool).
        - Detect multi-arg combined risk patterns:
            e.g., "run_command + disallowed path" should raise score to 1.0.
        - Emit a structured list of extracted primitives (binary, args, operators) for debugging.

    Implementation recommendation:
        - Prefer rule-based detection for stability and speed.
        - Only use an LLM backstop when command strings are complex and ambiguous.
          If used, keep discrete score levels (0.0 / 0.3 / 0.6 / 1.0).

    """
    safeguard_llm = get_runnable_llm("encoder")
    await safeguard_llm.ainvoke(...)  # OR/AND third-party API
    ...


# Tool Invocation Inspection
async def encode_rate_and_budget(tool_history: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    History-based rate/budget risk encoder (single-score).

    This encoder evaluates whether the agent/tool layer exhibits abnormal
    usage patterns over a recent tool-call window, such as:
      - high-frequency bursts,
      - repeated high-risk tool usage,
      - suspicious retries,
      - excessive resource consumption inferred from call metadata.

    It consumes a *tool call history window* rather than a single call, which makes
    the encoder deterministic, replayable, and independent of implicit world state.

    Input:
        tool_history:
            A list of recent tool-call records (recommended: last N calls within a
            fixed horizon, e.g., last 60s or last 20 calls). Each record should include:

            {
              "ts": float | int,            # timestamp (seconds) or monotonic step index
              "name": str,                  # tool name
              "args": Dict[str, Any],        # tool arguments (optional for this encoder)
              "status": str | None,          # "ok" | "error" | "timeout" | ...
              "latency_ms": int | None,      # optional
              "bytes_sent": int | None,      # optional (network)
              "bytes_received": int | None,  # optional (network)
              "risk_tag": str | None,        # optional pre-tag (e.g., "high_risk_tool")
            }

            Minimal required fields: "name" + ("ts" or implicit order).
            Missing optional fields MUST be handled safely.

    Output (canonical):
        {
          "scores": {"rate_budget_risk": float},   # in [0.0, 1.0]
          "evidence": List[str],                   # <= 5 short items
        }

    Score meaning:
        0.0 = normal usage
        0.3 = mild anomaly (near-limit / unusual but not clearly malicious)
        0.6 = clear anomaly (burst, repeated high-risk tools, suspicious retries)
        1.0 = severe anomaly (extreme burst / multiple high-risk patterns combined)

    Must-have detection (no LLM, deterministic):
        1) Burst detection:
           - Many tool calls within a short window (e.g., >K calls in 10s / 60s).
        2) High-risk tool concentration:
           - High fraction or repeated usage of privileged / side-effect tools.
        3) Suspicious retry / failure loops:
           - Multiple errors/timeouts in a row, or repeated nearly identical calls.
        4) Optional resource pressure (if metadata provided):
           - bytes_sent spikes, unusually long latencies, etc.

    Evidence requirements:
        - Keep evidence short and numeric where possible.
        - Examples:
            "burst: 12 calls in 30s"
            "run_command repeated: 5/10 recent calls"
            "retry loop: 4 consecutive errors"
            "net egress spike: 8.2MB in 60s"

    Notes:
        - This encoder does NOT enforce policy thresholds and does NOT block actions.
          It only produces a canonical risk feature for aggregation.
        - Budget limits / thresholds are loaded from configuration elsewhere (not passed here).
    """
    ...
