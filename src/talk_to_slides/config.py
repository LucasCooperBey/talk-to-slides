"""Settings the extractor reads."""
import math
import os


def env_seconds(name: str, default: str, *, allow_zero: bool = False) -> float:
    raw = os.environ.get(name, default)
    try:
        seconds = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number of seconds, got {raw!r}") from exc
    if not math.isfinite(seconds) or seconds < 0 or (seconds == 0 and not allow_zero):
        rule = "finite and non-negative" if allow_zero else "finite and positive"
        raise ValueError(f"{name} must be {rule} seconds, got {raw!r}")
    return seconds


# Claude CLI alias. The CLI resolves it to the current Opus model.
DENOISE_CLI_MODEL = "opus"
# Measured on talks of 6 to 14 minutes: medium matched a higher effort's coverage
# at about a third of the output tokens, and low dropped slides.
DENOISE_CLI_EFFORT = "medium"
# Metered API model used when --llm-backend api is set.
DENOISE_API_MODEL = "claude-opus-5-5"
