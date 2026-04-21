"""
Tool Call Argument Validators

This module provides validation functions for different types of tool calls.
These validators are used by the encode_tool_call_args encoder to perform
rule-based risk checks on tool arguments.

Each validator function:
- Takes tool arguments and policy configuration
- Returns a risk score (0.0-1.0) and evidence list
- Is deterministic and fast (no LLM calls)
"""

import re
from typing import Dict, Any, List, Tuple


def check_run_command(args: Dict[str, Any], policy: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Check run_command arguments for risks.
    
    Args:
        args: Tool arguments dictionary
        policy: Policy configuration from call_args_policy.yaml
    
    Returns:
        Tuple of (risk_score, evidence_list)
    """
    risk = 0.0
    evidence = []
    
    command = args.get("command", "")
    if not isinstance(command, str):
        return risk, evidence
    
    command_lower = command.lower()
    
    denied_bins = policy.get("denied_bins", [])
    for bin_name in denied_bins:
        bin_pattern = r'\b' + re.escape(bin_name) + r'\b'
        if re.search(bin_pattern, command_lower):
            risk = max(risk, 1.0)
            evidence.append(f"dangerous binary: {bin_name} in command={command[:50]}")
            break
    
    dangerous_operators = policy.get("dangerous_operators", [])
    for op in dangerous_operators:
        if op in command:
            risk = max(risk, 0.6)
            evidence.append(f"shell operator: {op} in command={command[:50]}")
            break
    
    privilege_patterns = policy.get("privilege_patterns", [])
    for pattern in privilege_patterns:
        if pattern.lower() in command_lower:
            risk = max(risk, 1.0)
            evidence.append(f"privilege escalation: {pattern} in command={command[:50]}")
            break
    
    persistence_patterns = policy.get("persistence_patterns", [])
    for pattern in persistence_patterns:
        if pattern.lower() in command_lower:
            risk = max(risk, 1.0)
            evidence.append(f"persistence: {pattern} in command={command[:50]}")
            break
    
    dangerous_patterns = policy.get("dangerous_command_patterns", [])
    for pattern in dangerous_patterns:
        if re.search(pattern, command_lower):
            risk = max(risk, 1.0)
            evidence.append(f"dangerous pattern: {pattern[:20]} in command={command[:50]}")
            break
    
    suspicious_patterns = policy.get("suspicious_arg_patterns", [])
    for pattern in suspicious_patterns:
        if re.search(pattern, command_lower):
            risk = max(risk, 0.6)
            evidence.append(f"suspicious pattern: {pattern[:20]} in command={command[:50]}")
            break
    
    max_length = policy.get("max_command_length", 1000)
    if len(command) > max_length:
        risk = max(risk, 0.6)
        evidence.append(f"command too long: {len(command)} chars")
    
    allowed_bins = policy.get("allowed_bins", [])
    if allowed_bins:
        found_allowed = False
        for bin_name in allowed_bins:
            bin_pattern = r'\b' + re.escape(bin_name) + r'\b'
            if re.search(bin_pattern, command_lower):
                found_allowed = True
                break
        
        if not found_allowed:
            risk = max(risk, 1.0)
            evidence.append(f"binary not in allowlist: command={command[:50]}")
    
    return risk, evidence[:5]


def check_filesystem_tools(args: Dict[str, Any], policy: Dict[str, Any], tool_name: str) -> Tuple[float, List[str]]:
    """
    Check filesystem tool arguments for risks.
    
    Args:
        args: Tool arguments dictionary
        policy: Policy configuration from call_args_policy.yaml
        tool_name: Name of the tool (e.g., "write_file", "read_file")
    
    Returns:
        Tuple of (risk_score, evidence_list)
    """
    risk = 0.0
    evidence = []
    
    path = args.get("path", "")
    if not isinstance(path, str):
        return risk, evidence
    
    if "../" in path or "..\\" in path:
        risk = max(risk, 1.0)
        evidence.append(f"path traversal: ../ in path={path[:50]}")
    
    denied_prefixes = policy.get("path_deny_prefix", [])
    for prefix in denied_prefixes:
        if path.startswith(prefix):
            risk = max(risk, 1.0)
            evidence.append(f"path in denylist: {prefix} in path={path[:50]}")
            break
    
    allowed_prefixes = policy.get("path_allow_prefix", [])
    if allowed_prefixes:
        found_allowed = False
        for prefix in allowed_prefixes:
            if path.startswith(prefix):
                found_allowed = True
                break
        
        if not found_allowed:
            risk = max(risk, 0.6)
            evidence.append(f"path not in allowlist: path={path[:50]}")
    
    path_depth = len([p for p in path.split('/') if p])
    max_depth = policy.get("max_path_depth", 10)
    if path_depth > max_depth:
        risk = max(risk, 0.6)
        evidence.append(f"path too deep: {path_depth} levels")
    
    if tool_name == "write_file":
        content = args.get("content", "")
        if isinstance(content, str):
            max_size = policy.get("tool_specific_policies", {}).get("write_file", {}).get("max_file_size", 10485760)
            if len(content) > max_size:
                risk = max(risk, 0.6)
                evidence.append(f"file too large: {len(content)} bytes")
    
    return risk, evidence[:5]


def check_network_tools(args: Dict[str, Any], policy: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Check network tool arguments for risks.
    
    Args:
        args: Tool arguments dictionary
        policy: Policy configuration from call_args_policy.yaml
    
    Returns:
        Tuple of (risk_score, evidence_list)
    """
    risk = 0.0
    evidence = []
    
    url = args.get("url", "")
    if not isinstance(url, str):
        return risk, evidence
    
    url_lower = url.lower()
    
    denied_schemes = policy.get("denied_url_schemes", [])
    for scheme in denied_schemes:
        scheme_pattern = scheme + "://"
        if scheme_pattern in url_lower:
            risk = max(risk, 1.0)
            evidence.append(f"forbidden scheme: {scheme} in url={url[:50]}")
            break
    
    allowed_schemes = policy.get("allowed_url_schemes", [])
    if allowed_schemes:
        found_allowed = False
        for scheme in allowed_schemes:
            scheme_pattern = scheme + "://"
            if scheme_pattern in url_lower:
                found_allowed = True
                break
        
        if not found_allowed and "://" in url:
            risk = max(risk, 0.6)
            evidence.append(f"scheme not in allowlist: url={url[:50]}")
    
    denied_domains = policy.get("denied_domains", [])
    for domain in denied_domains:
        if domain.lower() in url_lower:
            risk = max(risk, 1.0)
            evidence.append(f"domain in denylist: {domain} in url={url[:50]}")
            break
    
    allowed_domains = policy.get("allowed_domains", [])
    if allowed_domains:
        found_allowed = False
        for domain in allowed_domains:
            if domain.lower() in url_lower:
                found_allowed = True
                break
        
        if not found_allowed and "://" in url:
            risk = max(risk, 0.6)
            evidence.append(f"domain not in allowlist: url={url[:50]}")
    
    internal_ranges = policy.get("internal_ip_ranges", [])
    for ip_range in internal_ranges:
        if ip_range in url_lower:
            risk = max(risk, 0.6)
            evidence.append(f"internal IP: {ip_range} in url={url[:50]}")
            break
    
    max_length = policy.get("max_url_length", 2000)
    if len(url) > max_length:
        risk = max(risk, 0.6)
        evidence.append(f"url too long: {len(url)} chars")
    
    return risk, evidence[:5]


def check_generic_tools(args: Dict[str, Any], policy: Dict[str, Any], tool_name: str) -> Tuple[float, List[str]]:
    """
    Check generic tool arguments for risks.
    
    Args:
        args: Tool arguments dictionary
        policy: Policy configuration from call_args_policy.yaml
        tool_name: Name of the tool
    
    Returns:
        Tuple of (risk_score, evidence_list)
    """
    risk = 0.0
    evidence = []
    
    for arg_name, arg_value in args.items():
        if isinstance(arg_value, str):
            if any(op in arg_value for op in ["|", ";", "&&", "||", "$()", "`"]):
                risk = max(risk, 0.6)
                evidence.append(f"shell operator in {arg_name}={arg_value[:50]}")
                break
            
            if "../" in arg_value or "..\\" in arg_value:
                risk = max(risk, 0.6)
                evidence.append(f"path traversal in {arg_name}={arg_value[:50]}")
                break
    
    max_args = policy.get("max_args", 20)
    if len(args) > max_args:
        risk = max(risk, 0.6)
        evidence.append(f"too many args: {len(args)}")
    
    tool_policies = policy.get("tool_specific_policies", {})
    if tool_name in tool_policies:
        tool_policy = tool_policies[tool_name]
        tool_max_args = tool_policy.get("max_args", max_args)
        if len(args) > tool_max_args:
            risk = max(risk, 0.6)
            evidence.append(f"too many args: {len(args)}")
    
    return risk, evidence[:5]
