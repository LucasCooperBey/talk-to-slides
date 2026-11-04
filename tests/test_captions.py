#!/usr/bin/env python3
"""Caption regression tests. Run: `.venv/bin/python eval/test_captions.py`."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "vtt"
sys.path.insert(0, str(ROOT))

from talk_to_slides.captions import parse_vtt, pick_caption_file  # noqa: E402
from talk_to_slides.extract import _extract_parser, _finalize_parser  # noqa: E402


def _expected(name):
    return json.loads((FIXTURES / f"{name}.expected.json").read_text())


def _assert_fixture(name):
    cues = parse_vtt(FIXTURES / f"{name}.vtt")
    actual = [{**c, "start": round(c["start"], 3), "end": round(c["end"], 3)} for c in cues]
    expected = _expected(name)
    assert actual == expected, f"{name}: got {actual!r}, expected {expected!r}"


def _parse_text(text):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "source.en.vtt"
        path.write_text(text)
        return parse_vtt(path)


def _caption_files(directory, languages):
    for lang in languages:
        (directory / f"source.{lang}.vtt").write_text("WEBVTT\n")


def _captions(directory):
    return subprocess.run(
        [sys.executable, "-m", "talk_to_slides.cli", "captions", str(directory)],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_rolling_fixture_matches_typed_lines():
    """source: real-failure 2026-10-15, rolling YouTube captions doubled transcript.json."""
    _assert_fixture("rolling-bread")


def test_manual_fixture_matches_decoded_text():
    """source: real-failure 2026-10-15, manual captions retained HTML entities."""
    _assert_fixture("manual-bread")


def test_plain_fixture_preserves_existing_cues():
    """source: interface 3 preserves non-rolling caption text and cue times."""
    _assert_fixture("plain-bread")


def test_fixtures_keep_three_caption_structures():
    """source: spec rule 8, rolling word times, manual entities, and plain cues stay distinct."""
    import re

    word_time = re.compile(r"<\d{2}:\d{2}:\d{2}\.\d{3}>")
    rolling = (FIXTURES / "rolling-bread.vtt").read_text()
    manual = (FIXTURES / "manual-bread.vtt").read_text()
    plain = (FIXTURES / "plain-bread.vtt").read_text()
    assert word_time.search(rolling)
    assert not word_time.search(manual) and "&nbsp;" in manual and "&amp;" in manual
    assert not word_time.search(plain)
    assert len(parse_vtt(FIXTURES / "plain-bread.vtt")) == 3
    for name in ("rolling-bread", "manual-bread", "plain-bread"):
        cues = parse_vtt(FIXTURES / f"{name}.vtt")
        saved = _expected(name)
        assert saved == [{**c, "start": round(c["start"], 3), "end": round(c["end"], 3)} for c in cues]
        assert saved, name


def test_rolling_has_no_repeated_prefix_and_halves_text():
    """source: real-failure 2026-10-15, rolling characters contained repeated lines.

    118 is the character count of rolling-bread.vtt when the rolling overlap is kept.
    Dedup has to stay at or under 55 percent of that, or this assertion does not bite.
    """
    cues = parse_vtt(FIXTURES / "rolling-bread.vtt")
    assert len(cues) > 1, cues
    for previous, current in zip(cues, cues[1:]):
        assert not current["text"].startswith(previous["text"]), (previous, current)
    chars = sum(len(c["text"]) for c in cues)
    assert 0 < chars <= 0.55 * 118, chars


def test_rolling_first_cue_keeps_zero_start():
    """source: real-failure 2026-10-15, containment collapse moved the first cue to 3.4s."""
    cues = parse_vtt(FIXTURES / "rolling-bread.vtt")
    assert cues and cues[0]["start"] == 0.0, cues[:1]


def test_manual_has_no_encoded_entities_or_nbsp():
    """source: real-failure 2026-10-15, manual cues contained encoded spaces."""
    raw = (FIXTURES / "manual-bread.vtt").read_text()
    assert "&nbsp;" in raw and "&amp;" in raw
    cues = parse_vtt(FIXTURES / "manual-bread.vtt")
    assert cues, cues
    for cue in cues:
        assert all(token not in cue["text"] for token in ("&nbsp;", "&amp;", "\u00a0")), cue


def test_both_modes_decode_entities_and_normalize_whitespace():
    """source: interface 4 decodes entities after stripping tags and normalizes each line."""
    for marker in ("", "<00:00:00.360>"):
        cues = _parse_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000 align:start position:0%\n"
            f"<c> A&nbsp;&amp;\t B</c>{marker}  &lt;literal&gt;\u00a0\n"
            "&#160;   \n"
        )
        assert cues == [{"start": 0.0, "end": 1.0, "text": "A & B <literal>"}], cues


def test_nonrolling_decodes_only_semicolon_terminated_entities():
    """source: push review of 21ce3a4, legacy entity decoding corrupted spoken ampersands."""
    cues = _parse_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n"
        "Q&notes a&section=2 AT&T &amp; &nbsp;x &#39;y&#39; &foo;\n"
    )
    assert cues == [
        {"start": 0.0, "end": 1.0, "text": "Q&notes a&section=2 AT&T & x 'y' &foo;"},
    ], cues


def test_rolling_decodes_only_semicolon_terminated_entities():
    """source: push review of 21ce3a4, rolling captions also corrupted spoken ampersands."""
    cues = _parse_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n"
        "Q&notes a&section=2 AT&T &amp; &nbsp;x &#39;y&#39; &foo;<00:00:00.360>\n"
    )
    assert cues == [
        {"start": 0.0, "end": 1.0, "text": "Q&notes a&section=2 AT&T & x 'y' &foo;"},
    ], cues


def test_rolling_overlap_uses_cleaned_lines():
    """source: interface 2 compares lines after entity decoding and whitespace normalization."""
    cues = _parse_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nA&nbsp;&amp; B<00:00:00.360>\n\n"
        "00:00:01.000 --> 00:00:02.000\nA\t &amp;  B\nC&#160; D\n"
    )
    assert cues == [
        {"start": 0.0, "end": 1.0, "text": "A & B"},
        {"start": 1.0, "end": 2.0, "text": "C D"},
    ], cues


def test_rolling_uses_previous_cue_even_when_not_emitted():
    """source: interface 2 compares with the immediately preceding cue, including dropped cues."""
    cues = _parse_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nalpha<00:00:00.360>\nbeta\n\n"
        "00:00:01.000 --> 00:00:02.000\nbeta\n\n"
        "00:00:02.000 --> 00:00:03.000\nalpha\nbeta\ngamma\n"
    )
    assert cues == [
        {"start": 0.0, "end": 1.0, "text": "alpha beta"},
        {"start": 2.0, "end": 3.0, "text": "alpha beta gamma"},
    ], cues


def test_rolling_removes_longest_overlap_and_keeps_repeated_lines_in_cue():
    """source: interface 2 drops only the longest complete line overlap between cues."""
    cues = _parse_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\na<00:00:00.360>\na\n\n"
        "00:00:01.000 --> 00:00:02.000\na\na\nc\n"
    )
    assert cues == [
        {"start": 0.0, "end": 1.0, "text": "a a"},
        {"start": 1.0, "end": 2.0, "text": "c"},
    ], cues


def test_unknown_semicolon_reference_stays_as_written():
    """source: architect review of the 21ce3a4 fixes, html.unescape turned "&notes;" into the not sign plus "es;"."""
    for suffix in ("", "<00:00:00.360>"):
        cues = _parse_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n"
            f"&notes; &ampersand; &lt;b&gt; &#x41;{suffix}\n"
        )
        assert cues == [
            {"start": 0.0, "end": 1.0, "text": "&notes; &ampersand; <b> A"},
        ], cues


def test_rolling_empty_cue_keeps_previous_lines():
    """source: push review of 21ce3a4, empty rolling cues must preserve previous lines."""
    cues = _parse_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello<00:00:00.360>\n\n"
        "00:00:01.000 --> 00:00:02.000\n&nbsp;\n\n"
        "00:00:02.000 --> 00:00:03.000\nhello\nworld\n"
    )
    assert cues == [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 2.0, "end": 3.0, "text": "world"},
    ], cues


def test_nonrolling_keeps_duplicate_line_and_containment_rules():
    """source: interface 3 retains legacy containment collapse without inline word timing tags."""
    cues = _parse_text(
        "WEBVTT\nKind: captions\nLanguage: en\n\n"
        "00:00:00.000 --> 00:00:01.000\n<c>hello</c>\nhello\n\n"
        "00:00:01.000 --> 00:00:02.000\nhello world\n\n"
        "00:00:02.000 --> 00:00:03.000\nworld\n"
    )
    assert cues == [{"start": 1.0, "end": 2.0, "text": "hello world"}], cues


def test_pick_manual_en_us_beats_auto_en():
    """source: interface 6 prefers available manual English captions over automatic English."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _caption_files(root, ["en", "en-US"])
        (root / "source.info.json").write_text(json.dumps({
            "subtitles": {"en-US": []}, "automatic_captions": {"en": []},
        }))
        assert pick_caption_file(root) == (root / "source.en-US.vtt", "manual")


def test_pick_manual_english_preference_order():
    """source: interface 6 orders manual en, en-US, en-GB, then other English languages."""
    languages = ["en", "en-US", "en-GB", "en-AU", "en-CA"]
    for index, expected in enumerate(languages):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _caption_files(root, reversed(languages[index:]))
            (root / "source.info.json").write_text(json.dumps({
                "subtitles": dict.fromkeys(languages, []),
            }))
            assert pick_caption_file(root) == (root / f"source.{expected}.vtt", "manual")


def test_pick_auto_en_when_only_auto_exists():
    """source: interface 6 labels available automatic English captions as auto."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _caption_files(root, ["de", "en"])
        (root / "source.info.json").write_text(json.dumps({"automatic_captions": {"en": []}}))
        assert pick_caption_file(root) == (root / "source.en.vtt", "auto")


def test_pick_auto_english_before_manual_nonenglish():
    """source: push review of 21ce3a4, manual German beat automatic English captions."""
    for auto, lang, kind in (({"en": []}, "en", "auto"), ({}, "de", "manual")):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _caption_files(root, ["de", "en"])
            (root / "source.info.json").write_text(json.dumps({
                "subtitles": {"de": []}, "automatic_captions": auto,
            }))
            selected = pick_caption_file(root)
            assert selected == (root / f"source.{lang}.vtt", kind), selected


def test_pick_three_letter_languages_do_not_outrank_auto_english():
    """source: push review of 99f7642, manual enm and eng were incorrectly ranked as English."""
    cases = [
        ({"eng": []}, {}, ["eng"], "eng", "manual"),
        ({"enm": []}, {"en": []}, ["enm", "en"], "en", "auto"),
        ({"eng": []}, {"en": []}, ["eng", "en"], "en", "auto"),
        ({"eng": [], "de": []}, {}, ["eng", "de"], "de", "manual"),
    ]
    for manual, auto, languages, lang, kind in cases:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _caption_files(root, languages)
            (root / "source.info.json").write_text(json.dumps({
                "subtitles": manual, "automatic_captions": auto,
            }))
            selected = pick_caption_file(root)
            assert selected == (root / f"source.{lang}.vtt", kind), selected


def test_pick_without_info_prefers_en_then_sorted_filename():
    """source: interface 6 makes unknown-caption selection deterministic without metadata."""
    for languages, expected in ((["en-US", "de", "en"], "en"), (["fr", "de"], "de")):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _caption_files(root, languages)
            assert pick_caption_file(root) == (root / f"source.{expected}.vtt", "unknown")


def test_pick_unreadable_or_non_dict_info_is_unknown():
    """source: interface 6 ignores unreadable metadata and non-dict subtitle mappings."""
    for info in (b"{broken", b"\xff", b"[]", b'{"subtitles": ["en"], "automatic_captions": "en"}'):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _caption_files(root, ["en-US", "en"])
            (root / "source.info.json").write_bytes(info)
            assert pick_caption_file(root) == (root / "source.en.vtt", "unknown")


def test_pick_empty_directory_has_no_candidate():
    """source: interface 6 distinguishes an empty download folder from a caption candidate."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        assert pick_caption_file(root) is None
        _caption_files(root, ["en"])
        assert pick_caption_file(root) == (root / "source.en.vtt", "unknown")


def test_pick_only_direct_source_caption_files():
    """source: interface 6 excludes nested files, non-source names, and directories."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "nested").mkdir()
        _caption_files(root / "nested", ["en"])
        (root / "source.en.vtt").mkdir()
        (root / "video.en.vtt").write_text("WEBVTT\n")
        assert pick_caption_file(root) is None
        _caption_files(root, ["fr"])
        assert pick_caption_file(root) == (root / "source.fr.vtt", "unknown")


def test_captions_subcommand_writes_expected_transcript():
    """source: interface 7 writes rolling captions to transcript.json and reports cue and char counts."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        shutil.copyfile(FIXTURES / "rolling-bread.vtt", root / "source.en.vtt")
        result = _captions(root)
        assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
        assert json.loads((root / "transcript.json").read_text()) == _expected("rolling-bread")
        assert result.stdout == f"captions: source.en.vtt (unknown), 3 cues, 29 chars -> {root}/transcript.json\n", result.stdout
        assert result.stderr == "", result.stderr


def test_captions_subcommand_empty_directory_exits_two():
    """source: interface 7 reports missing captions on stderr and exits 2 without writing a transcript."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        result = _captions(root)
        assert result.returncode == 2, (result.returncode, result.stdout, result.stderr)
        assert result.stdout == "", result.stdout
        assert result.stderr == f"captions: no source.*.vtt in {root}\n", result.stderr
        assert not (root / "transcript.json").exists()


def test_captions_subcommand_zero_cues_exits_three_without_transcript():
    """source: push review of 21ce3a4, captions_cmd exited 0 on an empty transcript."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "source.en.vtt").write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n   \t\n"
        )
        result = _captions(root)
        assert result.returncode == 3, (result.returncode, result.stdout, result.stderr)
        assert result.stdout == "", result.stdout
        assert result.stderr == (
            "captions: source.en.vtt (unknown) parsed to 0 cues; no transcript written\n"
        ), result.stderr
        assert not (root / "transcript.json").exists()


def test_captions_subcommand_zero_cues_preserves_existing_transcript():
    """source: push review of 99f7642, zero caption cues destroyed an existing transcript.json."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        original = b'[{"start": 0.0, "end": 1.0, "text": "Keep this transcript"}]\n'
        transcript = root / "transcript.json"
        transcript.write_bytes(original)
        (root / "source.en.vtt").write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n   \t\n"
        )
        result = _captions(root)
        assert result.returncode == 3, (result.returncode, result.stdout, result.stderr)
        assert result.stdout == "", result.stdout
        assert result.stderr == (
            "captions: source.en.vtt (unknown) parsed to 0 cues; no transcript written\n"
        ), result.stderr
        assert transcript.exists(), "zero cues deleted the existing transcript"
        assert transcript.read_bytes() == original


def test_finalize_effort_help_reads_current_config():
    """source: interface 8 prevents finalize help from retaining a hardcoded effort default."""
    with patch("talk_to_slides.config.DENOISE_CLI_EFFORT", "high"):
        help_text = " ".join(_finalize_parser().format_help().split())
    assert "claude CLI effort for the cleanup call (default high); ignored by --llm-backend api" in help_text


def test_extract_effort_help_reads_current_config():
    """source: interface 8 prevents extraction help from retaining a hardcoded effort default."""
    with patch("talk_to_slides.config.DENOISE_CLI_EFFORT", "high"):
        help_text = " ".join(_extract_parser().format_help().split())
    assert "claude CLI effort for the cleanup call (default high); ignored by --llm-backend api" in help_text


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
