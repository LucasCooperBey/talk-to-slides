"""Fixes from the 2026-11-03 pre-push review of c22b6da.

source: 11-03 review gate WARN on c22b6da. Each test fails when its item is reverted.
"""
import json
import signal
from pathlib import Path
from unittest.mock import patch

from talk_to_slides.captions import parse_vtt
from talk_to_slides.extract import (
    _saved_extract,
    chunk_ranges,
    finalize_cmd,
    group_frames_by_slide,
    render_html,
    run_extract,
    source_fingerprint,
    transcribe_cmd,
    ts_to_seconds,
)

_CLEANUP = {
    "model": "opus",
    "effort": "medium",
    "backend": "cli",
    "at": "2026-10-01T00:00:00+00:00",
}
_PAID = {"summary": "Paid summary"}
_MIB = 1024 * 1024


def _cues(tmp_path: Path, text: str) -> list[dict]:
    path = tmp_path / "captions.vtt"
    path.write_text(text)
    return parse_vtt(path)


def _plant(directory: Path, source: str, data: dict, cleanup=None, fingerprint=None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    meta = {"source": source}
    if cleanup is not None:
        meta["cleanup"] = cleanup
    if fingerprint is not None:
        meta["source_fingerprint"] = fingerprint
    (directory / "data.json").write_text(json.dumps({
        "meta": meta,
        "data": data,
        "links": [],
        "slides": [],
        "chat_raw": [],
    }))


def _call_saved(directory: Path, source: str, fingerprint):
    try:
        return _saved_extract(str(directory), source, fingerprint)
    except Exception as exc:  # noqa: BLE001 - a revert must fail an assertion, not the runner
        return exc


def test_numeric_cue_ids_are_not_spoken_text(tmp_path):
    """source: 11-03 review gate WARN on c22b6da, item 1
    A numbered cue id is not appended to the previous cue.
    """
    cues = _cues(tmp_path, (
        "WEBVTT\n\n"
        "1\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "hello\n\n"
        "2\n"
        "00:00:01.000 --> 00:00:02.000\n"
        "there\n"
    ))
    assert cues == [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 1.0, "end": 2.0, "text": "there"},
    ], cues


def test_text_cue_id_is_not_spoken_text(tmp_path):
    """source: 11-03 review gate WARN on c22b6da, item 1
    A text cue id, including more than one line, is not cue text.
    """
    cues = _cues(tmp_path, (
        "WEBVTT\n\n"
        "intro\n"
        "part-one\n"
        "00:00:00.000 --> 00:00:01.500\n"
        "welcome\n\n"
        "chapter\n"
        "part-two\n"
        "00:00:01.500 --> 00:00:03.000\n"
        "next\n"
    ))
    assert cues == [
        {"start": 0.0, "end": 1.5, "text": "welcome"},
        {"start": 1.5, "end": 3.0, "text": "next"},
    ], cues


def test_note_blocks_between_cues_leak_nothing(tmp_path):
    """source: 11-03 review gate WARN on c22b6da, item 1
    A one-line NOTE, a two-line NOTE, and a NOTE that contains a timing leak nothing.
    Spoken text NOTEbook time is kept.
    """
    cues = _cues(tmp_path, (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "before\n\n"
        "NOTE one line comment\n\n"
        "NOTE\n"
        "two line comment\n\n"
        "NOTE\n"
        "00:00:08.000 --> 00:00:08.500\n"
        "hidden cue\n\n"
        "00:00:01.000 --> 00:00:02.000\n"
        "NOTEbook time\n\n"
        "00:00:02.000 --> 00:00:03.000\n"
        "after\n"
    ))
    assert [cue["text"] for cue in cues] == ["before", "NOTEbook time", "after"], cues


def test_style_and_region_blocks_before_the_first_cue_leak_nothing(tmp_path):
    """source: 11-03 review gate WARN on c22b6da, item 1
    A STYLE block and a REGION block before the first cue leak nothing, even with a timing inside.
    """
    cues = _cues(tmp_path, (
        "WEBVTT\n\n"
        "STYLE\n"
        "::cue { color: white; }\n"
        "00:00:07.000 --> 00:00:07.500\n"
        "style leaked\n\n"
        "REGION\n"
        "id:fred width:40%\n"
        "00:00:06.000 --> 00:00:06.500\n"
        "region leaked\n\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "hello\n"
    ))
    assert cues == [{"start": 0.0, "end": 1.0, "text": "hello"}], cues


def test_spoken_header_words_after_the_header_are_kept(tmp_path):
    """source: 11-03 review gate WARN on c22b6da, item 1
    Language, Kind, and WEBVTT are skipped only before the first timing.
    """
    cues = _cues(tmp_path, (
        "WEBVTT\n"
        "Kind: captions\n"
        "Language: en\n\n"
        "00:00:01.000 --> 00:00:02.000\n"
        "Language: French is next\n"
        "Kind: captions stay\n"
        "WEBVTT stays spoken\n"
    ))
    assert cues == [{
        "start": 1.0,
        "end": 2.0,
        "text": "Language: French is next Kind: captions stay WEBVTT stays spoken",
    }], cues


def test_spoken_cue_settings_are_kept(tmp_path):
    """source: 11-03 review gate WARN on c22b6da, item 1
    align, position, line, size, vertical, and region are settings only on the timing line.
    """
    cues = _cues(tmp_path, (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.000 align:start position:0% line:0 size:100% "
        "vertical:rl region:should-not-leak\n"
        "align:left your paragraphs\n"
    ))
    assert cues == [{"start": 0.0, "end": 1.0, "text": "align:left your paragraphs"}], cues


def test_minute_second_timings_parse(tmp_path):
    """source: 11-03 review gate WARN on c22b6da, item 1
    A file whose timings are only MM:SS.mmm returns those starts and ends.
    """
    cues = _cues(tmp_path, (
        "WEBVTT\n\n"
        "00:01.500 --> 00:02.250\n"
        "short\n\n"
        "01:00.000 --> 01:30.500\n"
        "minute\n"
    ))
    assert cues == [
        {"start": 1.5, "end": 2.25, "text": "short"},
        {"start": 60.0, "end": 90.5, "text": "minute"},
    ], cues


def test_mixed_minute_and_hour_timings_parse(tmp_path):
    """source: 11-03 review gate WARN on c22b6da, item 1
    Each side of the arrow may be MM:SS.mmm or HH:MM:SS.mmm, and hours may exceed two digits.
    """
    cues = _cues(tmp_path, (
        "WEBVTT\n\n"
        "00:01.000 --> 01:00:02.500\n"
        "mixed\n\n"
        "100:00:00.000 --> 00:01.250\n"
        "long hours\n"
    ))
    assert cues == [
        {"start": 1.0, "end": 3602.5, "text": "mixed"},
        {"start": 360000.0, "end": 1.25, "text": "long hours"},
    ], cues


def test_word_containment_keeps_no_and_drops_world(tmp_path):
    """source: 11-03 review gate WARN on c22b6da, item 1
    Containment compares whole words, so "no" survives "you know it" and "world" does not survive "hello world".
    """
    cues = _cues(tmp_path, (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "you know it\n\n"
        "00:00:01.000 --> 00:00:02.000\n"
        "no\n\n"
        "00:00:02.000 --> 00:00:03.000\n"
        "hello world\n\n"
        "00:00:03.000 --> 00:00:04.000\n"
        "world\n"
    ))
    assert cues == [
        {"start": 0.0, "end": 1.0, "text": "you know it"},
        {"start": 1.0, "end": 2.0, "text": "no"},
        {"start": 2.0, "end": 3.0, "text": "hello world"},
    ], cues


def _seconds(value):
    try:
        return ts_to_seconds(value)
    except Exception as exc:  # noqa: BLE001 - a revert must fail an assertion, not the runner
        return exc


def test_numeric_slide_time_does_not_crash_the_report():
    """source: 11-03 review gate WARN on c22b6da, item 2
    A finite number is seconds. A bool, a negative, a non-finite number, and a non-string are not.
    """
    assert _seconds(75) == 75
    assert _seconds(75.9) == 75
    assert _seconds(True) == -1
    assert _seconds(-1) == -1
    assert _seconds(float("nan")) == -1
    assert _seconds(None) == -1
    assert _seconds(["1:00"]) == -1

    frame = {"n": 1, "t": "1:20", "lines": ["shown"]}
    deck = [{"t": 75, "title": "At 75"}, {"t": "0:10", "title": "At 10"}]
    try:
        groups = group_frames_by_slide([frame], deck)
    except Exception as exc:  # noqa: BLE001 - a revert must fail an assertion, not the runner
        groups = exc
    assert not isinstance(groups, Exception), groups
    later = [group for group in groups if group["slide"]["title"] == "At 75"]
    assert later and [item["t"] for item in later[0]["frames"]] == ["1:20"], groups

    try:
        html = render_html({"slides": deck}, [frame], [], [], {})
    except Exception as exc:  # noqa: BLE001 - a revert must fail an assertion, not the runner
        html = str(exc)
    assert "At 75" in html, html[:500]


def _rejected_chunk(total, chunk):
    outcome = "unset"

    def handler(signum, frame):
        raise TimeoutError("chunk_ranges did not return")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, 0.3)
    try:
        try:
            chunk_ranges(total, chunk)
        except ValueError:
            outcome = "value"
        except TimeoutError:
            outcome = "looped"
        else:
            outcome = "returned"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert outcome == "value", outcome


def test_chunk_ranges_rejects_a_non_positive_chunk_and_keeps_a_normal_slice():
    """source: 11-03 review gate WARN on c22b6da, item 3
    chunk_s must be a finite number above 0. A normal slice is unchanged.
    """
    _rejected_chunk(10, 0)
    _rejected_chunk(10, -1)
    _rejected_chunk(10, float("nan"))
    assert chunk_ranges(10, 4) == [(0, 4), (4, 8), (8, 10)]


def test_chunk_option_rejects_zero(capsys):
    """source: 11-03 review gate WARN on c22b6da, item 3
    --chunk rejects 0, a negative number, and a non-finite number with exit 2.
    """
    for bad in ("0", "-1", "nan", "inf"):
        try:
            code = transcribe_cmd(["clip.wav", "--out", "unused", "--chunk", bad])
        except SystemExit as exc:
            code = exc.code
        err = capsys.readouterr().err
        assert code == 2, (bad, code, err)
        assert "chunk" in err, (bad, err)


def test_saved_extract_matches_the_same_bytes_and_keeps_a_legacy_name(tmp_path):
    """source: 11-03 review gate WARN on c22b6da, item 4
    The same name and the same bytes keep the prior. Different bytes drop it.
    A saved meta with no source_fingerprint still matches on the name.
    """
    same_a = tmp_path / "a" / "talk.mp4"
    same_b = tmp_path / "b" / "talk.mp4"
    other = tmp_path / "c" / "talk.mp4"
    for path, payload in ((same_a, b"same-bytes"), (same_b, b"same-bytes"), (other, b"other-bytes")):
        path.parent.mkdir()
        path.write_bytes(payload)
    same_fp = source_fingerprint(same_a)
    other_fp = source_fingerprint(other)
    assert same_fp == source_fingerprint(same_b)
    assert same_fp != other_fp

    kept = tmp_path / "kept"
    _plant(kept, "talk.mp4", _PAID, _CLEANUP, same_fp)
    assert _call_saved(kept, "talk.mp4", same_fp) == (_PAID, _CLEANUP)

    dropped = tmp_path / "dropped"
    _plant(dropped, "talk.mp4", _PAID, _CLEANUP, same_fp)
    assert _call_saved(dropped, "talk.mp4", other_fp) == (None, None)

    legacy = tmp_path / "legacy"
    _plant(legacy, "talk.mp4", _PAID, _CLEANUP)
    assert _call_saved(legacy, "talk.mp4", other_fp) == (_PAID, _CLEANUP)

    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / "data.json").write_text("{not json")
    assert _call_saved(junk, "talk.mp4", same_fp) == (None, None)


def test_run_extract_drops_a_same_name_cleanup_from_different_bytes(tmp_path):
    """source: 11-03 review gate WARN on c22b6da, item 4
    use_llm False does not reuse a cleanup whose recording has the same name and different bytes.
    """
    video_a = tmp_path / "old" / "talk.mp4"
    video_b = tmp_path / "new" / "talk.mp4"
    video_a.parent.mkdir()
    video_b.parent.mkdir()
    video_a.write_bytes(b"first recording")
    video_b.write_bytes(b"second recording")
    out = tmp_path / "out"
    _plant(out, "talk.mp4", _PAID, _CLEANUP, source_fingerprint(video_a))
    with patch("talk_to_slides.extract._probe_dims", return_value=(16, 16, 4.0)), \
            patch("talk_to_slides.extract._extract_crop_frames", return_value=([], 0)):
        run_extract(
            str(video_b), str(out), chat_fps=1, share_fps=1, chat_crop="", share_crop="",
            use_llm=False, model="", meta={"source": "talk.mp4"}, no_chat=True, ocr="none",
            log=lambda line: None,
        )
    saved = json.loads((out / "data.json").read_text())
    assert saved["data"] == {}
    assert "cleanup" not in saved["meta"]
    assert saved["meta"].get("source_fingerprint") == source_fingerprint(video_b)


def test_source_fingerprint_hashes_the_head_and_includes_the_size(tmp_path):
    """source: 11-03 review gate WARN on c22b6da, item 4
    Any byte of a short file changes the fingerprint. Past 1 MiB, only a head byte or the size does.
    """
    short = tmp_path / "short.bin"
    original = bytes(range(100))
    short.write_bytes(original)
    base = source_fingerprint(short)
    size, sep, digest = base.partition(":")
    assert sep == ":" and size == "100" and len(digest) == 64
    assert digest == digest.lower() and all(char in "0123456789abcdef" for char in digest)
    for index in (0, 99):
        changed = bytearray(original)
        changed[index] ^= 0xFF
        short.write_bytes(changed)
        assert source_fingerprint(short) != base

    long = tmp_path / "long.bin"
    head = b"a" * _MIB
    tail = b"b" * 10
    long.write_bytes(head + tail)
    base = source_fingerprint(long)
    assert base.split(":", 1)[0] == str(_MIB + 10)

    inside = bytearray(head + tail)
    inside[_MIB - 1] ^= 0xFF
    long.write_bytes(inside)
    assert source_fingerprint(long) != base

    after = bytearray(head + tail)
    after[_MIB] ^= 0xFF
    long.write_bytes(after)
    assert source_fingerprint(long) == base

    long.write_bytes(head + tail + b"c")
    assert source_fingerprint(long) != base


def test_render_only_ignores_a_bad_llm_backend(tmp_path):
    """source: 11-03 review gate WARN on c22b6da, item 5
    finalize --render-only does not resolve the model, and a bad backend still writes the report.
    """
    out = tmp_path / "out"
    out.mkdir()
    (out / "data.json").write_text(json.dumps({
        "meta": {"title": "Talk", "source": "talk.mp4"},
        "data": {"summary": "Cached summary"},
        "links": [],
        "slides": [],
        "chat_raw": [],
    }))

    def explode(*args, **kwargs):
        raise AssertionError("render-only resolved a model")

    with patch("talk_to_slides.extract.resolve_denoise_model", explode):
        try:
            code = finalize_cmd([
                "--out", str(out), "--render-only", "--llm-backend", "bogus",
            ])
        except (ValueError, SystemExit, AssertionError) as exc:
            code = exc
    assert code == 0, code
    report = out / "report.html"
    assert report.is_file()
    assert "Cached summary" in report.read_text()


def test_bad_llm_backend_fails_before_the_folder_lock(tmp_path):
    """source: 11-03 review gate WARN on c22b6da, item 5
    finalize without --render-only still rejects a bad backend before it takes the folder lock.
    """
    out = tmp_path / "out"
    out.mkdir()
    (out / "data.json").write_text(json.dumps({
        "meta": {}, "data": {"summary": "Cached summary"}, "slides": [], "chat_raw": [],
    }))
    entered = []

    def spy(*args, **kwargs):
        entered.append(True)
        raise TimeoutError("folder busy")

    with patch("talk_to_slides.folder_lock.folder_lock", spy):
        try:
            code = finalize_cmd(["--out", str(out), "--llm-backend", "bogus"])
        except SystemExit as exc:
            code = exc.code
        except ValueError:
            code = 1
    assert entered == [], entered
    assert code not in (0, None)
