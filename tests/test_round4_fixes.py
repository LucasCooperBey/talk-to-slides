"""Fixes from the 2026-11-03 pre-push review of 2026054.

source: 11-03 review gate WARN on 2026054.
"""
from pathlib import Path

from talk_to_slides.captions import parse_vtt


def _cues(tmp_path: Path, text: str) -> list[dict]:
    path = tmp_path / "captions.vtt"
    path.write_text(text)
    return parse_vtt(path)


def _vtt(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"


def _plain(tmp_path: Path, texts: list[str]) -> list[dict]:
    lines = ["WEBVTT", ""]
    for index, text in enumerate(texts):
        lines.extend([
            f"00:00:0{index}.000 --> 00:00:0{index + 1}.000",
            text,
            "",
        ])
    return _cues(tmp_path, _vtt(lines))


def test_space_only_line_keeps_rolling_cue_times(tmp_path):
    """source: 11-03 review gate WARN on 2026054, item 1
    A space-only line inside a rolling cue does not end it or drop the next text line.
    """
    cues = _cues(tmp_path, _vtt([
        "WEBVTT",
        "Kind: captions",
        "Language: en",
        "",
        "00:00:00.000 --> 00:00:02.000 align:start position:0%",
        " ",
        "hello<00:00:00.500><c> world</c>",
        "",
        "00:00:02.000 --> 00:00:04.000 align:start position:0%",
        "hello world",
        " ",
        "",
        "00:00:04.000 --> 00:00:06.000 align:start position:0%",
        "hello world",
        "how<00:00:04.500><c> are</c>",
        "",
        "00:00:06.000 --> 00:00:08.000 align:start position:0%",
        " ",
        "last<00:00:06.500><c> words</c>",
    ]))
    assert cues == [
        {"start": 0.0, "end": 2.0, "text": "hello world"},
        {"start": 4.0, "end": 6.0, "text": "how are"},
        {"start": 6.0, "end": 8.0, "text": "last words"},
    ], cues


def test_space_only_line_keeps_plain_cue_text(tmp_path):
    """source: 11-03 review gate WARN on 2026054, item 1
    A plain cue keeps the text line that follows a space-only line.
    """
    cues = _cues(tmp_path, _vtt([
        "WEBVTT",
        "",
        "00:00:00.000 --> 00:00:02.000",
        " ",
        "first line",
        "",
        "00:00:02.000 --> 00:00:04.000",
        "second line",
    ]))
    assert [cue["text"] for cue in cues] == ["first line", "second line"], cues


def test_tab_only_line_keeps_the_following_cue_text(tmp_path):
    """source: 11-03 review gate WARN on 2026054, item 1
    A tab-only line inside a cue adds no text and does not end the cue.
    """
    cues = _cues(tmp_path, _vtt([
        "WEBVTT",
        "",
        "00:00:00.000 --> 00:00:01.000",
        "\t",
        "kept words",
    ]))
    assert cues == [{"start": 0.0, "end": 1.0, "text": "kept words"}], cues


def test_whitespace_between_cues_does_not_keep_the_cue_id(tmp_path):
    """source: 11-03 review gate WARN on 2026054, item 1
    A whitespace-only line between a cue's empty line and the next cue id is still blank.
    """
    cues = _cues(tmp_path, _vtt([
        "WEBVTT",
        "",
        "1",
        "00:00:00.000 --> 00:00:01.000",
        "hello",
        "",
        " ",
        "2",
        "00:00:01.000 --> 00:00:02.000",
        "there",
    ]))
    assert [cue["text"] for cue in cues] == ["hello", "there"], cues


def test_note_block_ends_on_a_whitespace_only_line(tmp_path):
    """source: 11-03 review gate WARN on 2026054, item 1
    A whitespace-only line ends a NOTE block, and the following cue leaks no NOTE text.
    """
    cues = _cues(tmp_path, _vtt([
        "WEBVTT",
        "",
        "NOTE",
        "hidden comment",
        " ",
        "00:00:00.000 --> 00:00:01.000",
        "spoken",
    ]))
    assert [cue["text"] for cue in cues] == ["spoken"], cues
    assert all("hidden" not in cue["text"] and "comment" not in cue["text"] for cue in cues)


def test_trailing_period_replaces_the_shorter_cue(tmp_path):
    """source: 11-03 review gate WARN on 2026054, item 2
    hello world then hello world. keeps the longer raw text.
    """
    cues = _plain(tmp_path, ["hello world", "hello world."])
    assert [cue["text"] for cue in cues] == ["hello world."], cues
    assert cues == [{"start": 1.0, "end": 2.0, "text": "hello world."}], cues


def test_growing_caption_with_a_comma_replaces_the_previous_cue(tmp_path):
    """source: 11-03 review gate WARN on 2026054, item 2
    hello world then hello world, how are you collapses to the longer cue.
    """
    cues = _plain(tmp_path, ["hello world", "hello world, how are you"])
    assert [cue["text"] for cue in cues] == ["hello world, how are you"], cues


def test_shorter_unpunctuated_cue_does_not_replace_the_longer_one(tmp_path):
    """source: 11-03 review gate WARN on 2026054, item 2
    hello world. then hello world keeps the earlier, longer raw text.
    """
    cues = _plain(tmp_path, ["hello world.", "hello world"])
    assert [cue["text"] for cue in cues] == ["hello world."], cues
    assert cues == [{"start": 0.0, "end": 1.0, "text": "hello world."}], cues


def test_no_after_you_know_it_is_kept(tmp_path):
    """source: 11-03 review gate WARN on 2026054, item 2
    no is not a whole token of you know it, so both cues stay.
    """
    cues = _plain(tmp_path, ["you know it", "no"])
    assert [cue["text"] for cue in cues] == ["you know it", "no"], cues


def test_world_after_hello_world_is_dropped(tmp_path):
    """source: 11-03 review gate WARN on 2026054, item 2
    world is a whole token of hello world, so the later cue is dropped.
    """
    cues = _plain(tmp_path, ["hello world", "world"])
    assert [cue["text"] for cue in cues] == ["hello world"], cues


def test_inner_apostrophe_is_not_stripped(tmp_path):
    """source: 11-03 review gate WARN on 2026054, item 2
    don't stop then dont stop stays two cues because inner punctuation is kept.
    """
    cues = _plain(tmp_path, ["don't stop", "dont stop"])
    assert [cue["text"] for cue in cues] == ["don't stop", "dont stop"], cues
