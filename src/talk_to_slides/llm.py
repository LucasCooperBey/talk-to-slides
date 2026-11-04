"""Claude CLI helpers for the cleanup call."""
from __future__ import annotations

import json
import os
import subprocess

# Bound at import so a test that replaces subprocess.run is not treated as the real binary.
_REAL_SUBPROCESS_RUN = subprocess.run

CLI_EFFORTS = ("low", "medium", "high", "xhigh", "max")
CLI_MISSING_LINE = (
    "claude CLI not found: the report has slides and text but no summary; "
    "use --llm-backend api with ANTHROPIC_API_KEY, or --no-llm"
)


class ClaudeCliNotFound(RuntimeError):
    """The claude binary is not on PATH."""


def claude_cli_argv(model: str, system: str, effort: str | None = None) -> list[str]:
    from talk_to_slides.config import DENOISE_CLI_EFFORT

    if effort is None:
        effort = DENOISE_CLI_EFFORT
    if effort not in CLI_EFFORTS:
        raise ValueError(f"Unknown CLI effort: {effort}")
    # Never use --bare: it forces API-key auth and skips the logged-in claude CLI.
    return ["claude", "-p", "--safe-mode", "--tools", "", "--strict-mcp-config",
            "--no-session-persistence", "--output-format", "json", "--model", model,
            "--effort", effort, "--system-prompt", system]


def claude_cli_env(base: dict[str, str] | None = None) -> dict[str, str]:
    # Absent keys use the logged-in claude CLI. Dropping CLAUDECODE avoids nested-session
    # detection. The child must not inherit an interactive session's effort or thinking budget.
    env = dict(os.environ if base is None else base)
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE",
                "CLAUDE_EFFORT", "CLAUDE_CODE_EFFORT_LEVEL", "MAX_THINKING_TOKENS"):
        env.pop(key, None)
    return env


def run_claude_cli(user: str, system: str, model: str, *, effort: str | None = None, runner=None,
                   timeout: float = 1800.0) -> dict:
    import shutil

    real = runner is None
    if runner is None:
        runner = subprocess.run
    # Only the real subprocess path looks up the binary. A patched runner is a test double.
    if real and runner is _REAL_SUBPROCESS_RUN and shutil.which("claude") is None:
        raise ClaudeCliNotFound(CLI_MISSING_LINE)
    try:
        result = runner(claude_cli_argv(model, system, effort=effort), input=user, text=True,
                        capture_output=True, env=claude_cli_env(), timeout=timeout)
    except FileNotFoundError as exc:
        raise ClaudeCliNotFound(CLI_MISSING_LINE) from exc
    try:
        envelope = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"claude CLI returned invalid JSON (exit {result.returncode}): "
                           f"{result.stderr[:300]}") from e
    reply = envelope.get("result") if isinstance(envelope, dict) else None
    detail = " | ".join(part[:300] for part in (reply, result.stderr)
                        if isinstance(part, str) and part)
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed (exit {result.returncode}): {detail}")
    if (not isinstance(envelope, dict) or envelope.get("is_error")
            or envelope.get("subtype") != "success" or not isinstance(envelope.get("result"), str)):
        raise RuntimeError(f"claude CLI returned an unsuccessful response: {detail}")
    if envelope.get("stop_reason") == "max_tokens":
        raise RuntimeError("claude CLI returned stop_reason max_tokens")
    return envelope


def cli_usage_summary(usage) -> tuple[str, str, str]:
    if not isinstance(usage, dict):
        return "?", "?", "?"
    inputs = [usage.get(key) for key in
              ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")]
    inputs = [value for value in inputs if isinstance(value, int) and not isinstance(value, bool)]
    output = usage.get("output_tokens")
    details = usage.get("output_tokens_details")
    thinking = details.get("thinking_tokens") if isinstance(details, dict) else None
    return (str(sum(inputs)) if inputs else "?",
            str(output) if isinstance(output, int) and not isinstance(output, bool) else "?",
            str(thinking) if isinstance(thinking, int) and not isinstance(thinking, bool) else "?")
