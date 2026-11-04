"""WebVTT parsing and caption-file selection."""
from __future__ import annotations

import html
import html.entities
import json
import os
import re
from pathlib import Path

_VTT_TIME = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})")
_VTT_WORD_TIME = re.compile(r"<\d{2}:\d{2}:\d{2}\.\d{3}>")
_TAG = re.compile(r"<[^>]+>")
_CUE_ATTR = re.compile(r"(align|position):\S+")
_HTML_REF = re.compile(r"&(?:#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")


def _sec(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _decode_reference(match: re.Match) -> str:
    """Decode a numeric or fully named HTML reference. html.unescape alone falls back to the
    longest legacy name, so an unknown "&notes;" would become the not sign plus "es;"."""
    ref = match.group(0)
    if ref[1] == "#" or ref[1:] in html.entities.html5:
        return html.unescape(ref)
    return ref


def _rolling_vtt_tail(previous: list[str], current: list[str]) -> list[str]:
    """Remove the longest leading overlap with the previous cue's trailing lines."""
    for size in range(min(len(previous), len(current)), 0, -1):
        if current[:size] == previous[-size:]:
            return current[size:]
    return current


def parse_vtt(path: Path) -> list[dict]:
    """Return [{start,end,text}], preserving each newly typed rolling line's cue times."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    rolling = _VTT_WORD_TIME.search(raw) is not None
    cues: list[dict] = []
    cur_start = cur_end = None
    cur_lines: list[str] = []
    previous_lines: list[str] = []

    def append_cue() -> None:
        nonlocal previous_lines
        if cur_start is None:
            return
        lines = cur_lines
        if rolling and cur_lines:
            lines = _rolling_vtt_tail(previous_lines, cur_lines)
            previous_lines = cur_lines
        if lines:
            cues.append({"start": cur_start, "end": cur_end, "text": " ".join(lines)})

    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        m = _VTT_TIME.search(s)
        if m:
            append_cue()
            cur_start = _sec(m.group(1))
            cur_end = _sec(m.group(2))
            cur_lines = []
            continue
        if cur_start is None:
            continue
        s = _CUE_ATTR.sub("", _TAG.sub("", s)).strip()
        s = _HTML_REF.sub(_decode_reference, s)
        s = re.sub(r"\s+", " ", s.replace("\u00a0", " ")).strip()
        if s and (rolling or not cur_lines or s != cur_lines[-1]):
            cur_lines.append(s)
    append_cue()
    if rolling:
        return cues

    collapsed: list[dict] = []
    for c in cues:
        if collapsed:
            last = collapsed[-1]["text"]
            if c["text"] in last:
                continue
            if last in c["text"]:
                collapsed[-1] = c
                continue
        collapsed.append(c)
    return collapsed


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def pick_caption_file(directory: Path) -> tuple[Path, str] | None:
    """Prefer manual English, auto English, other manual, then a stable fallback."""
    candidates = {
        p.name[len("source."):-len(".vtt")]: p
        for p in directory.glob("source.*.vtt")
        if p.is_file() and p.name != "source..vtt"
    }
    if not candidates:
        return None
    info = _read_json(directory / "source.info.json")
    if not isinstance(info, dict):
        info = {}
    manual = info.get("subtitles")
    auto = info.get("automatic_captions")
    preferred = {"en": 0, "en-US": 1, "en-GB": 2}
    languages = []
    if isinstance(manual, dict):
        languages = sorted(
            (lang for lang in candidates if lang in manual),
            key=lambda lang: (
                preferred.get(lang, 3 if lang == "en" or lang.startswith("en-") else 4), lang,
            ),
        )
        if languages and (languages[0] == "en" or languages[0].startswith("en-")):
            return candidates[languages[0]], "manual"
    if isinstance(auto, dict) and "en" in auto and "en" in candidates:
        return candidates["en"], "auto"
    if languages:
        return candidates[languages[0]], "manual"
    fallback = min(candidates.values(), key=lambda p: (p.name != "source.en.vtt", p.name))
    return fallback, "unknown"


def write_transcript_from_vtt(vtt_path: Path, extract_dir: Path) -> Path:
    cues = parse_vtt(vtt_path)
    transcript = [
        {"start": round(float(c["start"]), 3), "end": round(float(c["end"]), 3), "text": c["text"]}
        for c in cues
        if c.get("text")
    ]
    out = extract_dir / "transcript.json"
    tmp = out.with_suffix(".json.tmp")
    try:
        with tmp.open("w") as output:
            output.write(json.dumps(transcript, indent=2) + "\n")
            output.flush()
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out
