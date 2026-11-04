"""WebVTT parsing and caption-file selection."""
from __future__ import annotations

import html
import html.entities
import json
import os
import re
from pathlib import Path

# Hours are two or more digits. Each side of the arrow may be HH:MM:SS.mmm or MM:SS.mmm.
_VTT_TS = r"(?:\d{2,}:)?\d{2}:\d{2}\.\d{3}"
_VTT_TIME = re.compile(rf"({_VTT_TS}) --> ({_VTT_TS})")
_VTT_WORD_TIME = re.compile(r"<\d{2}:\d{2}:\d{2}\.\d{3}>")
_TAG = re.compile(r"<[^>]+>")
_HTML_REF = re.compile(r"&(?:#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")
_METADATA_BLOCKS = ("NOTE", "STYLE", "REGION")


def _sec(t: str) -> float:
    parts = t.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _metadata_block(line: str) -> bool:
    """True for a NOTE, STYLE, or REGION block label, not for spoken text like NOTEbook."""
    for name in _METADATA_BLOCKS:
        if line == name or line.startswith(name + " ") or line.startswith(name + "\t"):
            return True
    return False


def _clean_cue_line(line: str) -> str:
    text = _TAG.sub("", line).strip()
    text = _HTML_REF.sub(_decode_reference, text)
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()


def _contains_word_run(haystack: str, needle: str) -> bool:
    """True when needle's words are a contiguous run of whole words in haystack."""
    hay = haystack.split()
    ned = needle.split()
    if not ned or len(ned) > len(hay):
        return False
    width = len(ned)
    return any(hay[index:index + width] == ned for index in range(len(hay) - width + 1))


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
    """Return [{start, end, text}], preserving each newly typed rolling line's cue times.

    Lines between a blank line (or the header) and the next timing are cue
    identifiers and are not cue text. A NOTE, STYLE, or REGION block is skipped
    through the next blank line. WEBVTT, Kind, and Language lines are skipped
    only before the first timing. Cue settings stay on the timing line. A
    non-rolling cue is dropped or replaces the previous cue only when its words
    are a contiguous run of whole words in the other cue.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    rolling = _VTT_WORD_TIME.search(raw) is not None
    cues: list[dict] = []
    cur_start = cur_end = None
    cur_lines: list[str] = []
    previous_lines: list[str] = []
    in_cue = False
    skipping_block = False

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
        stripped = line.strip()
        if skipping_block:
            if not stripped:
                skipping_block = False
            continue
        if not stripped:
            in_cue = False
            continue
        if not in_cue and _metadata_block(stripped):
            skipping_block = True
            continue
        timing = _VTT_TIME.search(stripped)
        if timing:
            append_cue()
            cur_start = _sec(timing.group(1))
            cur_end = _sec(timing.group(2))
            cur_lines = []
            in_cue = True
            continue
        if not in_cue:
            continue
        text = _clean_cue_line(stripped)
        if text and (rolling or not cur_lines or text != cur_lines[-1]):
            cur_lines.append(text)
    append_cue()
    if rolling:
        return cues

    collapsed: list[dict] = []
    for cue in cues:
        if collapsed:
            last = collapsed[-1]["text"]
            if _contains_word_run(last, cue["text"]):
                continue
            if _contains_word_run(cue["text"], last):
                collapsed[-1] = cue
                continue
        collapsed.append(cue)
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
