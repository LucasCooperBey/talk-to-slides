#!/usr/bin/env python3
"""Tests for src/webinar_extract.py pure logic. `.venv/bin/python eval/test_webinar_extract.py`.

Covers the load-bearing pure helpers for dense webinar extraction:
  - stitch_scroll: reconstruct a continuous line stream from overlapping scroll windows
  - extract_urls: pull links out of noisy OCR text
  - avg_hash / hamming / frames_differ: cheap change-detection so we OCR only changed frames
"""
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from talk_to_slides.folder_lock import folder_lock
from talk_to_slides.extract import (  # noqa: E402
    stitch_scroll,
    extract_urls,
    avg_hash,
    hamming,
    frames_differ,
    chunk_ranges,
    fmt_ts,
    ts_to_seconds,
    group_frames_by_slide,
    render_html,
    denoise_acceptable,
    _clean_links,
    _esc,
    _safe_href,
    build_denoise_prompt,
    parse_denoise_reply,
    claude_cli_argv,
    claude_cli_env,
    cli_usage_summary,
    run_claude_cli,
    resolve_denoise_model,
    denoise_llm,
    run_extract,
    finalize_cmd,
    main,
    _DENOISE_SYSTEM,
    CLI_EFFORTS,
    _extract_parser,
    _finalize_parser,
)


# ---- stitch_scroll ------------------------------------------------------
def test_stitch_into_empty_returns_new():
    assert stitch_scroll([], ["a", "b", "c"]) == ["a", "b", "c"]


def test_stitch_overlap_appends_only_new_tail():
    assert stitch_scroll(["a", "b", "c"], ["b", "c", "d"]) == ["a", "b", "c", "d"]


def test_stitch_identical_window_adds_nothing():
    assert stitch_scroll(["a", "b", "c"], ["a", "b", "c"]) == ["a", "b", "c"]


def test_stitch_no_overlap_concatenates():
    assert stitch_scroll(["a", "b"], ["c", "d"]) == ["a", "b", "c", "d"]


def test_stitch_single_line_already_present():
    assert stitch_scroll(["a", "b", "c"], ["c"]) == ["a", "b", "c"]


def test_stitch_prefers_largest_overlap():
    # acc tail "a","b" matches new head "a","b" (k=2), not just the trailing "b"
    assert stitch_scroll(["x", "a", "b"], ["a", "b", "a", "b"]) == ["x", "a", "b", "a", "b"]


def test_stitch_overlap_is_whitespace_and_case_insensitive():
    # overlap matched via normalization; the ORIGINAL acc text is preserved
    out = stitch_scroll(["Hello  World"], ["hello world", "New line"])
    assert out == ["Hello  World", "New line"]


def test_stitch_ignores_blank_lines_in_new():
    assert stitch_scroll(["a", "b"], ["b", "", "c"]) == ["a", "b", "c"]


# ---- extract_urls -------------------------------------------------------
def test_extract_https_and_www():
    assert extract_urls("see https://foo.com/x and www.bar.io now") == [
        "https://foo.com/x",
        "www.bar.io",
    ]


def test_extract_strips_trailing_punctuation():
    assert extract_urls("visit https://x.com.") == ["https://x.com"]


def test_extract_dedupes_preserving_order():
    assert extract_urls("https://a.com then https://a.com again") == ["https://a.com"]


def test_extract_bare_domain_with_known_tld():
    assert extract_urls("go to acme.ai to sign up") == ["acme.ai"]


def test_extract_ignores_non_url_dotted_text():
    # "e.g" is not a TLD we recognize -> not a url
    assert extract_urls("e.g. something happened") == []


# ---- change detection ---------------------------------------------------
def test_hamming_counts_differing_bits():
    assert hamming(0b1010, 0b1000) == 1
    assert hamming(0, 0b111111) == 6


def test_frames_differ_threshold():
    assert frames_differ(0, 0, threshold=5) is False
    assert frames_differ(0, 0b111111, threshold=5) is True
    assert frames_differ(0, 0b11, threshold=5) is False  # 2 bits < threshold


def test_avg_hash_identical_images_match():
    from PIL import Image

    a = Image.new("L", (64, 64), 0)
    b = Image.new("L", (64, 64), 0)
    assert isinstance(avg_hash(a), int)
    assert avg_hash(a) == avg_hash(b)


def test_avg_hash_distinguishes_black_from_white():
    from PIL import Image

    black = Image.new("L", (64, 64), 0)
    white = Image.new("L", (64, 64), 255)
    assert avg_hash(black) != avg_hash(white)


# ---- audio chunking -----------------------------------------------------
def test_chunk_ranges_exact_multiple():
    assert chunk_ranges(960, 480) == [(0, 480), (480, 960)]


def test_chunk_ranges_with_remainder():
    assert chunk_ranges(1000, 480) == [(0, 480), (480, 960), (960, 1000)]


def test_chunk_ranges_shorter_than_chunk():
    assert chunk_ranges(300, 480) == [(0, 300)]


def test_chunk_ranges_zero_or_negative():
    assert chunk_ranges(0, 480) == []
    assert chunk_ranges(-5, 480) == []


def test_fmt_ts_minutes_seconds():
    assert fmt_ts(0) == "0:00"
    assert fmt_ts(65) == "1:05"
    assert fmt_ts(600) == "10:00"


def test_fmt_ts_hours():
    assert fmt_ts(3725) == "1:02:05"


# ---- ts_to_seconds (inverse of fmt_ts) ----------------------------------
def test_ts_to_seconds_mmss():
    assert ts_to_seconds("00:09") == 9
    assert ts_to_seconds("3:29") == 209
    assert ts_to_seconds("17:10") == 1030


def test_ts_to_seconds_hms():
    assert ts_to_seconds("1:02:05") == 3725


def test_ts_to_seconds_roundtrips_fmt_ts():
    for s in (0, 9, 65, 209, 1030, 3725):
        assert ts_to_seconds(fmt_ts(s)) == s


def test_ts_to_seconds_empty_or_bad_is_negative():
    assert ts_to_seconds("") == -1
    assert ts_to_seconds("abc") == -1
    assert ts_to_seconds(None) == -1


# ---- group_frames_by_slide (the alignment fix) --------------------------
def _raw(n, t):
    return {"n": n, "t": t, "lines": [], "thumb": f"thumb{n}"}


def test_group_empty_logical_returns_empty():
    # No de-noised deck -> caller falls back to flat render.
    assert group_frames_by_slide([_raw(1, "0:05")], []) == []


def test_group_buckets_frames_under_active_slide():
    # The regression: 4 raw frames spanning 0:00-1:00, 2 logical deck slides at 0:00 and 0:30.
    raw = [_raw(1, "0:05"), _raw(2, "0:20"), _raw(3, "0:40"), _raw(4, "0:55")]
    logical = [
        {"n": 1, "t": "0:00", "title": "Intro", "points": ["a"]},
        {"n": 2, "t": "0:30", "title": "Body", "points": ["b"]},
    ]
    groups = group_frames_by_slide(raw, logical)
    assert [g["slide"]["title"] for g in groups] == ["Intro", "Body"]
    assert [f["n"] for f in groups[0]["frames"]] == [1, 2]  # 0:05, 0:20 < 0:30
    assert [f["n"] for f in groups[1]["frames"]] == [3, 4]  # 0:40, 0:55 >= 0:30


def test_group_last_slide_gets_all_trailing_frames():
    # The exact bug shape: many trailing frames must NOT be left text-less; they bucket
    # under the last deck slide whose start-time precedes them.
    raw = [_raw(i, fmt_ts(i * 10)) for i in range(1, 11)]  # 0:10 .. 1:40
    logical = [
        {"n": 1, "t": "0:00", "title": "First", "points": []},
        {"n": 2, "t": "0:50", "title": "Last", "points": []},
    ]
    groups = group_frames_by_slide(raw, logical)
    assert len(groups[1]["frames"]) == 6  # frames at 0:50,1:00,1:10,1:20,1:30,1:40
    assert all(ts_to_seconds(f["t"]) >= 50 for f in groups[1]["frames"])
    # every frame is assigned somewhere (nothing dropped)
    assert sum(len(g["frames"]) for g in groups) == len(raw)


def test_group_frame_before_first_slide_buckets_under_first():
    raw = [_raw(1, "0:02")]
    logical = [{"n": 1, "t": "0:10", "title": "S1", "points": []}]
    groups = group_frames_by_slide(raw, logical)
    assert groups[0]["frames"] == raw


def test_group_forward_fills_missing_timecode():
    # A logical slide missing 't' inherits the prior slide's start (stable order kept).
    raw = [_raw(1, "0:05"), _raw(2, "0:40")]
    logical = [
        {"n": 1, "t": "0:00", "title": "A", "points": []},
        {"n": 2, "title": "B", "points": []},  # no 't'
    ]
    groups = group_frames_by_slide(raw, logical)
    assert [g["slide"]["title"] for g in groups] == ["A", "B"]
    # both slides share start 0 -> later slide wins ties, so both frames go to B
    assert sum(len(g["frames"]) for g in groups) == 2


def test_group_no_timecodes_distributes_positionally():
    # A deck with zero usable timecodes must NOT pile every frame under one slide.
    raw = [_raw(i, "") for i in range(1, 9)]  # 8 frames, no times
    logical = [
        {"n": 1, "title": "A", "points": []},
        {"n": 2, "title": "B", "points": []},
    ]
    groups = group_frames_by_slide(raw, logical)
    assert len(groups[0]["frames"]) == 4
    assert len(groups[1]["frames"]) == 4
    assert sum(len(g["frames"]) for g in groups) == len(raw)


# ---- render_html: gallery / lightbox / desc / empty sections -------------
def _frame(n, t, png="", thumb="data:image/jpeg;base64,THUMB"):
    return {"n": n, "t": t, "lines": [f"line{n}"], "png": png, "thumb": thumb}


def test_render_gallery_full_res_paths_and_all_thumbs():
    # Main/lightbox must reference the on-disk native-res PNG (report-relative),
    # and the strip must hold EVERY frame in the group, not just the extras.
    slides = [_frame(1, "0:05", png="/abs/out/slides/slide_01.png"),
              _frame(2, "0:31", png="/abs/out/slides/slide_02.png")]
    data = {"slides": [{"n": 1, "t": "0:05", "title": "Intro", "points": []}]}
    html = render_html(data, slides, [], [], {})
    assert 'data-full="slides/slide_01.png"' in html
    assert 'data-full="slides/slide_02.png"' in html
    assert html.count('class="gthumb') == 2
    assert 'class="gnav gprev"' in html and 'class="gnav gnext"' in html
    assert 'class="gcount">1/2<' in html


def test_render_no_arrows_or_counter_for_single_frame():
    slides = [_frame(1, "0:05", png="/abs/out/slides/slide_01.png")]
    data = {"slides": [{"n": 1, "t": "0:05", "title": "Solo", "points": []}]}
    html = render_html(data, slides, [], [], {})
    assert 'class="gnav gprev"' not in html and 'class="gcount"' not in html


def test_render_frame_without_png_falls_back_to_thumb():
    slides = [_frame(1, "0:05", png="", thumb="data:image/jpeg;base64,ZZZ")]
    data = {"slides": [{"n": 1, "t": "0:05", "title": "S", "points": []}]}
    html = render_html(data, slides, [], [], {})
    assert 'class="gmain" src="data:image/jpeg;base64,ZZZ"' in html


def test_render_lightbox_emitted_once_in_both_branches():
    slides = [_frame(1, "0:05", png="x/slides/slide_01.png")]
    grouped = render_html({"slides": [{"n": 1, "t": "0:05", "title": "S", "points": []}]},
                          slides, [], [], {})
    flat = render_html({}, slides, [], [], {})  # no de-noised deck -> flat branch
    assert grouped.count('id="lightbox"') == 1
    assert flat.count('id="lightbox"') == 1
    assert flat.count('class="gmain"') == 1  # flat frames get the zoomable gallery too


def test_render_desc_shown_only_when_present():
    slides = [_frame(1, "0:05", png="x/slides/slide_01.png")]
    with_desc = {"slides": [{"n": 1, "t": "0:05", "title": "APM", "points": [],
                             "desc": "Keeps the APM scoring mechanism on screen."}]}
    no_desc = {"slides": [{"n": 1, "t": "0:05", "title": "APM", "points": [], "desc": ""}]}
    assert "Keeps the APM scoring mechanism on screen." in render_html(with_desc, slides, [], [], {})
    assert 'class="desc"' not in render_html(no_desc, slides, [], [], {})


def test_render_hides_empty_chat_and_links_sections():
    # A --no-chat talk must not render a dead "0 lines" dropdown or empty tables,
    # and a no-LLM report must not paint an empty summary box or Takeaways section.
    html = render_html({}, [], [], [], {})
    assert "Raw scroll-stitched chat OCR" not in html
    assert "Chat log" not in html
    assert "<h2>Links" not in html
    assert "<h2>Takeaways</h2>" not in html
    assert "What's worth stealing" not in html
    assert "<span class=tag>Takeaway</span>" in html
    assert "class=sum" not in html


def test_render_llm_sections_appear_when_denoise_ran():
    html = render_html({"summary": "The gist.", "takeaways": []}, [], [], [], {})
    assert "The gist." in html
    assert "<h2>Takeaways</h2>" in html and "none extracted" in html
    assert "<span class=tag>Takeaway</span>" in html
    assert "What's worth stealing" not in html


def test_render_title_falls_back_to_the_recording_name():
    """source: 11-03 spec review; a report with no title showed the source tool's "Webinar Extract"."""
    html = render_html({}, [], [], [], {"source": "bread-talk.mp4"})
    assert "<title>bread-talk</title>" in html and "<h1>bread-talk<" in html
    assert "<title>Talk to slides</title>" in render_html({}, [], [], [], {})
    given = render_html({}, [], [], [], {"source": "bread-talk.mp4", "title": "Proofing"})
    assert "<title>Proofing</title>" in given
    assert "Webinar Extract" not in html + given


# ---- href policy for LLM-supplied links (untrusted webinar content) -------
def test_esc_escapes_quotes_for_attributes():
    assert _esc('a"b\'c') == "a&quot;b&#39;c"


def test_safe_href_policy():
    assert _safe_href("https://x.com/a?b=1") == "https://x.com/a?b=1"
    assert _safe_href("http://x.com") == "http://x.com"
    assert _safe_href("foo.com") == "https://foo.com"
    assert _safe_href("www.foo.com/path") == "https://www.foo.com/path"
    assert _safe_href("javascript:alert(1)") is None
    assert _safe_href("data:text/html,x") is None
    assert _safe_href('foo.com" onmouseover="alert(1)') is None
    assert _safe_href("") is None


def test_render_unsafe_link_renders_as_text_not_anchor():
    html = render_html({}, [], ["javascript:alert(1)", "foo.com"], [], {})
    assert 'href="javascript' not in html
    assert "<li>javascript:alert(1)</li>" in html          # plain text, no anchor
    assert '<a href="https://foo.com">foo.com</a>' in html  # bare domain promoted


def test_render_gmain_carries_b64_fallback_alongside_full_res():
    # Pins the Python half of the JS onerror-fallback invariant: the main img must keep
    # its b64 in data-fb, and strip thumbs must keep the b64 as their src (JS reads
    # t.src / data-fb as the recovery source when slides/*.png is missing).
    slides = [_frame(1, "0:05", png="/abs/out/slides/slide_01.png"),
              _frame(2, "0:31", png="/abs/out/slides/slide_02.png")]
    data = {"slides": [{"n": 1, "t": "0:05", "title": "Intro", "points": []}]}
    html = render_html(data, slides, [], [], {})
    assert 'data-fb="data:image/jpeg;base64,THUMB"' in html
    # Behavior, not markup order: an active thumb exists, and thumbs keep the b64 as src.
    assert 'class="gthumb on"' in html
    assert 'src="data:image/jpeg;base64,THUMB"' in html


def test_render_shows_chat_sections_when_populated():
    data = {"chat": [{"t": "10:01", "author": "A", "text": "hi"}]}
    html = render_html(data, [], ["https://x.com"], ["rawline"], {})
    assert "Chat log (1 messages)" in html
    assert "Raw scroll-stitched chat OCR (1 lines)" in html
    assert "<h2>Links (1)</h2>" in html


# ---- LLM type-drift + cache-acceptance hardening (Codex gate findings) ----
def test_render_survives_llm_type_drift():
    # Schema says desc/summary/points/title are strings; the model occasionally emits
    # lists/numbers. Render must degrade to joined/str'd text, never AttributeError.
    slides = [_frame(1, "0:05", png="x/slides/slide_01.png")]
    data = {
        "summary": ["part one.", "part two."],
        "slides": [{"n": 1, "t": "0:05", "title": 42,
                    "points": "single point not a list", "desc": ["APM", "scoring"]}],
    }
    html = render_html(data, slides, [], [], {})
    assert "part one. part two." in html
    assert "APM scoring" in html
    assert "<li>single point not a list</li>" in html  # str points must not iterate chars
    assert "42" in html


def test_denoise_acceptable_gates_partial_replies():
    cached = {"slides": [{"n": 1}], "summary": "s", "chat": [], "key_moments": []}
    assert denoise_acceptable({}, {}) is True                      # first run, no cache
    assert denoise_acceptable({"links": []}, cached) is False      # truthy-but-partial reply
    assert denoise_acceptable({}, cached) is False                 # unparseable reply
    assert denoise_acceptable({"slides": [{"n": 1}], "summary": "new"}, cached) is True
    # a fresh payload may ADD sections the cache lacks
    assert denoise_acceptable({"slides": [{"n": 1}], "summary": "s", "chat": [{"t": ""}]},
                              cached) is True


def test_denoise_acceptable_rejects_material_shrinkage():
    # A truncated-but-valid reply (1 slide where the cache has 17) must not clobber.
    cached = {"slides": [{"n": i} for i in range(17)],
              "key_moments": [{"t": "0:01"}] * 6, "summary": "s"}
    truncated = {"slides": [{"n": 1}], "key_moments": [{"t": "0:01"}], "summary": "new"}
    assert denoise_acceptable(truncated, cached) is False
    # Legitimate re-merge shrinkage (17 -> 15) is fine.
    ok = {"slides": [{"n": i} for i in range(15)],
          "key_moments": [{"t": "0:01"}] * 5, "summary": "new"}
    assert denoise_acceptable(ok, cached) is True
    # A cached summary replaced by whitespace is not substantive.
    assert denoise_acceptable({**ok, "summary": "  "}, cached) is False


def test_render_survives_non_string_links():
    html = render_html({}, [], [123, ["foo", ".com"], None, "bar.com"], [], {})
    assert "<h2>Links (3)</h2>" in html                    # None dropped, others coerced
    assert "<li>123</li>" in html                          # number -> plain text, no crash
    assert '<a href="https://bar.com">bar.com</a>' in html


def test_clean_links_join_safe():
    # finalize does "\n".join(links) as LLM input. A dirty cached list must never
    # TypeError there (it would get swallowed and wedge the cache into keep-stale-forever).
    cleaned = _clean_links([123, None, ["a", ".com"], "x.com", "  "])
    assert cleaned == ["123", "a .com", "x.com"]
    assert "\n".join(cleaned)  # join contract: always str entries


# ---- plan-backed de-noise ------------------------------------------------
def test_claude_cli_argv_preserves_plan_auth():
    """source: interface 3 requires safe CLI flags and forbids API-only --bare."""
    argv = claude_cli_argv("opus", "system text")
    assert argv == ["claude", "-p", "--safe-mode", "--tools", "", "--strict-mcp-config",
                    "--no-session-persistence", "--output-format", "json", "--model", "opus",
                    "--effort", "medium", "--system-prompt", "system text"]
    assert "--bare" not in argv


def test_claude_cli_env_strips_auth_without_mutation():
    """source: interface 4 removes billing/nesting variables only from a copy."""
    base = {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_AUTH_TOKEN": "test-token",
            "CLAUDECODE": "1", "KEEP_ME": "kept"}
    original = dict(base)
    assert claude_cli_env(base) == {"KEEP_ME": "kept"}
    assert base == original
    assert claude_cli_env({}) == {}
    with patch.dict(os.environ, base, clear=True):
        assert claude_cli_env() == {"KEEP_ME": "kept"}
        assert dict(os.environ) == original


def test_run_claude_cli_success_passes_stdin_and_clean_env():
    """source: interface 5 sends the unchanged prompt on stdin with plan auth."""
    calls = []
    envelope = {"subtype": "success", "is_error": False, "result": '{"summary": "ok"}',
                "usage": {"input_tokens": 12, "output_tokens": 7}}

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(envelope), stderr="")

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "KEEP_ME": "kept"}, clear=True):
        assert run_claude_cli("user\ntext", "system", "opus", runner=runner, timeout=2.5) == envelope
        with patch("subprocess.run", runner):
            assert run_claude_cli("user\ntext", "system", "opus") == envelope
    assert len(calls) == 2
    for (argv, kwargs), timeout in zip(calls, (2.5, 1800.0)):
        assert argv == claude_cli_argv("opus", "system")
        assert kwargs == {"input": "user\ntext", "text": True, "capture_output": True,
                          "env": {"KEEP_ME": "kept"}, "timeout": timeout}
        assert "ANTHROPIC_API_KEY" not in kwargs["env"]


def test_run_claude_cli_rejects_bad_responses():
    """source: interface 5 rejects every failed or malformed CLI envelope."""
    success = {"subtype": "success", "result": "reply"}
    detail = "failure detail " + "x" * 400
    cases = [
        (1, json.dumps(success), detail, detail[:300]),
        (1, json.dumps({**success, "result": detail}), "", detail[:300]),
        (0, "not JSON", detail, detail[:300]),
        (0, "[]", "invalid envelope", "invalid envelope"),
        (0, json.dumps({**success, "is_error": True, "result": detail}), "", detail[:300]),
        (0, json.dumps({**success, "subtype": "error_max_turns", "result": detail}), "", detail[:300]),
        (0, json.dumps({"result": "missing subtype"}), "", "missing subtype"),
        (0, json.dumps({**success, "result": []}), "wrong result type", "wrong result type"),
        (0, json.dumps({"subtype": "success"}), "missing result", "missing result"),
    ]
    for code, stdout, stderr, expected in cases:
        def runner(argv, **kwargs):
            return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)

        try:
            run_claude_cli("user", "system", "opus", runner=runner)
        except RuntimeError as e:
            assert expected in str(e)
            assert detail[:301] not in str(e)
        else:
            raise AssertionError(f"accepted invalid CLI response: exit={code}, stdout={stdout[:80]}")


def test_parse_denoise_reply_formats_and_non_objects():
    """source: interface 2 accepts JSON objects and returns {} for unusable replies."""
    expected = {"summary": "ok"}
    for raw in ('```json\n{"summary": "ok"}\n```', '```\n{"summary": "ok"}\n```',
                '{"summary": "ok"}', 'Here is the reply: {"summary": "ok"} Done.'):
        assert parse_denoise_reply(raw) == expected
    for raw in ("garbage", "broken {bad} JSON", '[{"summary": "ok"}]', '"text"', "null"):
        assert parse_denoise_reply(raw) == {}


def test_resolve_denoise_model_defaults_and_validation():
    """source: interface 6 uses backend-specific defaults and rejects unknown backends."""
    from talk_to_slides.config import DENOISE_API_MODEL, DENOISE_CLI_MODEL

    assert DENOISE_CLI_MODEL == "opus"
    for model in (None, ""):
        assert resolve_denoise_model("cli", model) == DENOISE_CLI_MODEL
        assert resolve_denoise_model("api", model) == DENOISE_API_MODEL
    for backend in ("cli", "api"):
        assert resolve_denoise_model(backend, "explicit-model") == "explicit-model"
    for model in (None, "explicit-model"):
        try:
            resolve_denoise_model("unknown", model)
        except ValueError:
            pass
        else:
            raise AssertionError("unknown backend accepted")


def test_build_denoise_prompt_preserves_system_and_sections():
    """source: interface 1 preserves metadata, raw OCR, transcript and schema wording."""
    chat = ["10:00 Alice: hi", "10:01 Bob: hello"]
    slides = [{"n": 1, "t": "00:05", "lines": ["Title", "Point"]}]
    meta = {"title": "Webinar"}
    transcript = [{"start": 5, "text": "Spoken words"}, {"start": 8, "text": ""}]
    before = json.dumps([chat, slides, meta, transcript])
    system, user = build_denoise_prompt(chat, slides, "https://example.com", meta, transcript)
    assert system == _DENOISE_SYSTEM
    expected_prefix = (
        'Webinar recording metadata: {"title": "Webinar"}\n\n'
        '=== RAW CHAT PANEL LINES (scroll-stitched, top-to-bottom, OCR-noisy) ===\n'
        '10:00 Alice: hi\n10:01 Bob: hello\n\n'
        '=== SHARED-SCREEN / SLIDE OCR (one block per detected slide change) ===\n'
        '[SLIDE 1 @ 00:05]\nTitle\nPoint\n\n'
        '=== SPOKEN-AUDIO TRANSCRIPT (whisper, timestamped) ===\n[0:05] Spoken words\n\n'
        '=== ALL DETECTED LINKS (raw) ===\nhttps://example.com\n\n'
        'Reconstruct the webinar and return ONLY JSON matching this schema:\n'
    )
    assert user.startswith(expected_prefix + '{"title": str, "presenter": str, "date": str,')
    # source: spec rule 6, takeaways are what a viewer could apply, with no audience pitch.
    assert '"takeaways": [str] (what a viewer could apply, in one line each)' in user
    assert "afp_takeaways" not in user
    assert "worth stealing" not in user
    assert json.dumps([chat, slides, meta, transcript]) == before
    for tr in (None, []):
        assert "=== SPOKEN-AUDIO TRANSCRIPT" not in build_denoise_prompt(chat, slides, "", meta, tr)[1]


def test_denoise_llm_cli_parses_reply_and_logs_usage():
    """source: interface 7 defaults to the plan, sends OCR/audio and logs CLI usage."""
    calls, logs = [], []
    expected = {"summary": "Clean summary"}
    slides = [{"n": 1, "t": "00:05", "lines": ["Slide text"]}]
    transcript = [{"start": 5, "text": "Spoken words"}]
    for usage, kwargs, tokens in (({"input_tokens": 12, "output_tokens": 7}, {"backend": "cli"}, "12 output_tokens=7"),
                                  ({}, {}, "? output_tokens=?")):
        def runner(argv, **options):
            calls.append((argv, options))
            envelope = {"subtype": "success", "result": json.dumps(expected)}
            if usage:
                envelope["usage"] = usage
            return SimpleNamespace(returncode=0, stdout=json.dumps(envelope), stderr="")

        with patch.dict(sys.modules, {"anthropic": None}), \
                patch("pathlib.Path.home", side_effect=AssertionError("read API key")):
            # source: spec rule 5 deletes the home-folder key loader.
            import talk_to_slides.extract as extract_mod
            assert not hasattr(extract_mod, "_load_env_key")
            assert denoise_llm(["hi"], slides, "", {}, "opus", transcript=transcript,
                               runner=runner, log=logs.append, **kwargs) == expected
        assert logs[-1] == f"  de-noise via claude CLI (plan): model=opus effort=medium input_tokens={tokens} thinking_tokens=?"
    assert len(calls) == 2
    for argv, options in calls:
        assert argv == claude_cli_argv("opus", _DENOISE_SYSTEM)
        assert "[SLIDE 1 @ 00:05]" in options["input"]
        assert "=== SPOKEN-AUDIO TRANSCRIPT" in options["input"]
        assert options["input"] == build_denoise_prompt(["hi"], slides, "", {}, transcript)[1]


def test_denoise_llm_rejects_unknown_backend():
    """source: interface 7 must reject an unsupported backend without running a model."""
    def runner(*args, **kwargs):
        raise AssertionError("unknown backend started a runner")

    try:
        denoise_llm([], [], "", {}, "opus", backend="unknown", runner=runner)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown backend accepted")


def test_extract_parser_defaults_to_cli_and_accepts_api():
    """source: interface 9 keeps extract arguments while making metered API opt-in."""
    parser = _extract_parser()
    args = parser.parse_args(["x.mp4"])
    assert args.llm_backend == "cli" and args.model is None
    assert args.chat_fps == 1.0 and args.share_fps == 0.33
    args = parser.parse_args(["x.mp4", "--llm-backend", "api", "--model", "explicit-model"])
    assert args.llm_backend == "api" and args.model == "explicit-model"
    assert "metered Anthropic API" in parser.format_help()
    assert "bills per token" in parser.format_help()


def test_finalize_parser_defaults_to_cli_and_accepts_api():
    """source: interface 9 keeps finalize arguments while making metered API opt-in."""
    parser = _finalize_parser()
    args = parser.parse_args(["--out", "d"])
    assert args.llm_backend == "cli" and args.model is None
    assert parser.prog == "talk-to-slides finalize"
    args = parser.parse_args(["--out", "d", "--llm-backend", "api", "--model", "explicit-model"])
    assert args.llm_backend == "api" and args.model == "explicit-model"
    assert "metered Anthropic API" in parser.format_help()
    assert "bills per token" in parser.format_help()


def test_main_resolves_and_passes_backend():
    """source: interface 9 passes the resolved default or explicit backend/model to extraction."""
    from talk_to_slides.config import DENOISE_API_MODEL

    for flags, backend, model in (([], "cli", "opus"),
                                   (["--llm-backend", "api"], "api", DENOISE_API_MODEL),
                                   (["--model", "explicit-model"], "cli", "explicit-model")):
        with patch.object(sys, "argv", ["webinar-extract", "x.mp4", *flags]), \
                patch("talk_to_slides.extract.run_extract", return_value={}) as extract, \
                redirect_stdout(StringIO()):
            assert main() == 0
        extract.assert_called_once()
        assert extract.call_args.kwargs["backend"] == backend
        assert extract.call_args.kwargs["model"] == model


def test_run_extract_passes_backend_and_writes_artifacts():
    """source: interface 8 forwards the backend while preserving extracted artifacts."""
    for kwargs, backend in (({}, "cli"), ({"backend": "api"}, "api")):
        logs = []
        data = {"summary": "Clean summary"}
        with TemporaryDirectory() as work, \
                patch("talk_to_slides.extract._probe_dims", return_value=(1280, 720, 60.0)), \
                patch("talk_to_slides.extract._extract_crop_frames", return_value=([], 0)), \
                patch("talk_to_slides.extract.denoise_llm", return_value=data) as denoise:
            result = run_extract("x.mp4", work, chat_fps=1.0, share_fps=0.33,
                                 chat_crop="", share_crop="", use_llm=True, model="chosen-model",
                                 meta={}, no_chat=True, log=logs.append, **kwargs)
            denoise.assert_called_once()
            assert denoise.call_args.kwargs["backend"] == backend
            assert denoise.call_args.args[4] == "chosen-model"
            assert f"  de-noising with chosen-model via {backend}…" in logs
            saved = json.loads((Path(work) / "data.json").read_text())
            assert saved["data"] == data
            assert (Path(work) / "report.html").read_text() == render_html(data, [], [], [], saved["meta"])
            assert result["html"] == str(Path(work) / "report.html")


def test_finalize_cli_writes_report_and_cache():
    """source: objective requires finalize to use plan auth and retain report/data formats."""
    calls = []
    data = {"summary": "Clean summary", "links": ["https://example.com"]}
    slides = [{"n": 1, "t": "00:05", "lines": ["Slide text"], "png": ""}]
    transcript = [{"start": 5, "end": 10, "text": "Spoken words"}]
    meta = {"title": "Webinar"}

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        envelope = {"subtype": "success", "result": json.dumps(data),
                    "usage": {"input_tokens": 12, "output_tokens": 7}}
        return SimpleNamespace(returncode=0, stdout=json.dumps(envelope), stderr="")

    with TemporaryDirectory() as work, patch("subprocess.run", runner), \
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True), \
            patch.dict(sys.modules, {"anthropic": None}), \
            patch("pathlib.Path.home", side_effect=AssertionError("read API key")):
        # source: spec rule 5 deletes the home-folder key loader. The env key is set,
        # so a restored loader that returns early would not touch Path.home.
        import talk_to_slides.extract as extract_mod
        assert not hasattr(extract_mod, "_load_env_key")
        out = Path(work)
        (out / "data.json").write_text(json.dumps({"meta": meta, "slides": slides, "chat_raw": ["hi"]}))
        (out / "transcript.json").write_text(json.dumps(transcript))
        logs = StringIO()
        with redirect_stdout(logs):
            assert finalize_cmd(["--out", work]) == 0
        assert len(calls) == 1
        assert calls[0][0] == claude_cli_argv("opus", _DENOISE_SYSTEM)
        assert "ANTHROPIC_API_KEY" not in calls[0][1]["env"]
        assert "=== SPOKEN-AUDIO TRANSCRIPT" in calls[0][1]["input"]
        assert "de-noising with opus via cli" in logs.getvalue()
        assert "model=opus effort=medium input_tokens=12 output_tokens=7 thinking_tokens=?" in logs.getvalue()
        saved = json.loads((out / "data.json").read_text())
        assert saved == {
            "meta": {**meta, "cleanup": {"model": "opus", "effort": "medium", "backend": "cli",
                                         "at": saved["meta"]["cleanup"]["at"]}},
            "data": data, "links": data["links"], "slides": slides, "chat_raw": ["hi"]}
        assert (out / "report.html").read_text() == render_html(
            data, [{**s, "thumb": ""} for s in slides], data["links"], ["hi"], meta, transcript=transcript)


def test_claude_cli_env_strips_inherited_effort_without_mutation():
    """source: real-failure 2026-10-15, finalize inherited CLAUDE_EFFORT=max and
    MAX_THINKING_TOKENS=32000 from Claude Code and spent 22,628 output tokens
    where medium spent 7,394.
    """
    base = {"CLAUDE_EFFORT": "max", "MAX_THINKING_TOKENS": "32000", "KEEP_ME": "kept"}
    original = dict(base)
    assert claude_cli_env(base) == {"KEEP_ME": "kept"}
    assert base == original
    with patch.dict(os.environ, base, clear=True):
        assert claude_cli_env() == {"KEEP_ME": "kept"}
        assert dict(os.environ) == original


def test_claude_cli_env_strips_effort_level_override():
    """source: design-rule 2026-10-15 push review of f1e5191, the claude binary also
    reads CLAUDE_CODE_EFFORT_LEVEL, which could bring back an inherited effort.
    """
    base = {"CLAUDE_CODE_EFFORT_LEVEL": "max", "KEEP_ME": "kept"}
    assert claude_cli_env(base) == {"KEEP_ME": "kept"}
    assert base == {"CLAUDE_CODE_EFFORT_LEVEL": "max", "KEEP_ME": "kept"}


def test_cli_usage_summary_counts_cached_input_and_handles_missing():
    """source: real-failure 2026-10-15, the log read input_tokens=2 for an
    18,488-token prompt because it omitted cached input.
    """
    usage = {"input_tokens": 2, "cache_creation_input_tokens": 2381,
             "cache_read_input_tokens": 649, "output_tokens": 4,
             "output_tokens_details": {"thinking_tokens": 0}}
    original = json.dumps(usage)
    assert cli_usage_summary(usage) == ("3032", "4", "0")
    assert json.dumps(usage) == original
    for missing in ({}, None, [], "usage", 12):
        assert cli_usage_summary(missing) == ("?", "?", "?")
    for details in (None, [], "details", 12):
        assert cli_usage_summary({**usage, "output_tokens_details": details}) == ("3032", "4", "?")
    assert cli_usage_summary({"cache_read_input_tokens": 0}) == ("0", "?", "?")


def test_cli_usage_summary_ignores_bools_and_non_ints():
    """source: interface 6 counts only integer token values, excluding booleans."""
    for value in (True, False, "2", 2.5, None, [], {}):
        usage = {"input_tokens": value, "cache_creation_input_tokens": value,
                 "cache_read_input_tokens": value, "output_tokens": value,
                 "output_tokens_details": {"thinking_tokens": value}}
        assert cli_usage_summary(usage) == ("?", "?", "?")
        assert cli_usage_summary({**usage, "cache_creation_input_tokens": 3}) == ("3", "?", "?")


def test_claude_cli_argv_effort_override_and_validation():
    """source: interface 3 pins cleanup effort explicitly and rejects unknown values."""
    from talk_to_slides.config import DENOISE_CLI_EFFORT

    assert DENOISE_CLI_EFFORT == "medium"
    argv = claude_cli_argv("opus", "s", effort="low")
    assert argv[-4:] == ["--effort", "low", "--system-prompt", "s"]
    with patch("talk_to_slides.config.DENOISE_CLI_EFFORT", "high"):
        assert claude_cli_argv("opus", "s")[-4:] == ["--effort", "high", "--system-prompt", "s"]
        assert claude_cli_argv("opus", "s", effort="low")[-4:] == ["--effort", "low", "--system-prompt", "s"]
    try:
        claude_cli_argv("opus", "s", effort="turbo")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown effort accepted")


def test_effort_parsers_defaults_choices_and_validation():
    """source: interface 9 keeps defaults unset and exposes all supported CLI efforts."""
    assert CLI_EFFORTS == ("low", "medium", "high", "xhigh", "max")
    for parser, args in ((_extract_parser(), ["x.mp4"]),
                         (_finalize_parser(), ["--out", "d"])):
        assert parser.parse_args(args).effort is None
        for effort in CLI_EFFORTS:
            assert parser.parse_args([*args, "--effort", effort]).effort == effort
        assert ("claude CLI effort for the cleanup call (default medium); ignored by --llm-backend api"
                in " ".join(parser.format_help().split()))
        with redirect_stderr(StringIO()):
            try:
                parser.parse_args([*args, "--effort", "turbo"])
            except SystemExit as e:
                assert e.code == 2
            else:
                raise AssertionError("unknown effort accepted by parser")


def test_finalize_cli_effort_and_clean_env():
    """source: real-failure 2026-10-15, finalize must use medium by default and
    honor --effort without inheriting the parent session's effort or thinking budget.
    """
    for flags, effort in (([], "medium"), (["--effort", "low"], "low")):
        calls = []

        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            envelope = {"subtype": "success", "result": '{"summary": "ok"}',
                        "usage": {"input_tokens": 2, "cache_creation_input_tokens": 2381,
                                  "cache_read_input_tokens": 649, "output_tokens": 4,
                                  "output_tokens_details": {"thinking_tokens": 0}}}
            return SimpleNamespace(returncode=0, stdout=json.dumps(envelope), stderr="")

        with TemporaryDirectory() as work, patch("subprocess.run", runner), \
                patch.dict(os.environ, {"CLAUDE_EFFORT": "max", "MAX_THINKING_TOKENS": "32000",
                                        "KEEP_ME": "kept"}, clear=True), \
                patch.dict(sys.modules, {"anthropic": None}), \
                redirect_stdout(StringIO()) as logs:
            (Path(work) / "data.json").write_text("{}")
            assert finalize_cmd(["--out", work, *flags]) == 0
            assert len(calls) == 1
            argv, kwargs = calls[0]
            assert argv[-4:] == ["--effort", effort, "--system-prompt", _DENOISE_SYSTEM]
            assert kwargs["env"] == {"KEEP_ME": "kept"}
            assert os.environ["CLAUDE_EFFORT"] == "max"
            assert os.environ["MAX_THINKING_TOKENS"] == "32000"
            assert (f"  de-noise via claude CLI (plan): model=opus effort={effort} "
                    "input_tokens=3032 output_tokens=4 thinking_tokens=0") in logs.getvalue().splitlines()
            assert json.loads((Path(work) / "data.json").read_text())["data"] == {"summary": "ok"}


def test_main_passes_effort():
    """source: interface 9 forwards default None and explicit effort to run_extract."""
    for flags, effort in (([], None), (["--effort", "low"], "low")):
        with TemporaryDirectory() as work, \
                patch.object(sys, "argv", ["webinar-extract", "x.mp4", "--out", work, *flags]), \
                patch("talk_to_slides.extract.run_extract", return_value={}) as extract, \
                redirect_stdout(StringIO()):
            assert main() == 0
        extract.assert_called_once()
        assert extract.call_args.kwargs["effort"] == effort


def test_run_extract_passes_effort():
    """source: interface 8 forwards effort from extraction to the cleanup call."""
    for kwargs, effort in (({}, None), ({"effort": "low"}, "low")):
        with TemporaryDirectory() as work, \
                patch("talk_to_slides.extract._probe_dims", return_value=(1280, 720, 60.0)), \
                patch("talk_to_slides.extract._extract_crop_frames", return_value=([], 0)), \
                patch("talk_to_slides.extract.denoise_llm", return_value={}) as denoise:
            run_extract("x.mp4", work, chat_fps=1.0, share_fps=0.33,
                        chat_crop="", share_crop="", use_llm=True, model="opus",
                        meta={}, no_chat=True, log=lambda message: None, **kwargs)
            denoise.assert_called_once()
            assert denoise.call_args.kwargs["effort"] == effort


def test_run_claude_cli_rejects_max_tokens():
    """source: design-rule 2026-10-15 push review of 6cbe59b rejects truncated replies."""
    def runner(argv, **kwargs):
        envelope = {"subtype": "success", "result": '{"summary":', "stop_reason": "max_tokens"}
        return SimpleNamespace(returncode=0, stdout=json.dumps(envelope), stderr="")

    try:
        run_claude_cli("user", "system", "opus", runner=runner)
    except RuntimeError as e:
        assert "stop_reason max_tokens" in str(e)
    else:
        raise AssertionError("accepted a reply truncated by max_tokens")


def test_run_claude_cli_reports_result_before_stderr():
    """source: design-rule 2026-10-15 push review of 6cbe59b puts the real error
    before stderr warnings, retaining up to 300 characters of each.
    """
    for code, is_error, subtype in ((0, True, "success"), (1, False, "success"),
                                    (0, False, "error_max_turns")):
        for reply, warning in (("real error", "a warning"),
                                ("real error " + "x" * 400, "a warning " + "y" * 400)):
            def runner(argv, **kwargs):
                envelope = {"subtype": subtype, "is_error": is_error, "result": reply}
                return SimpleNamespace(returncode=code, stdout=json.dumps(envelope), stderr=warning)

            try:
                run_claude_cli("user", "system", "opus", runner=runner)
            except RuntimeError as e:
                assert str(e).endswith(f"{reply[:300]} | {warning[:300]}")
            else:
                raise AssertionError("accepted an error envelope")


def test_denoise_llm_cli_effort_and_cached_usage():
    """source: interfaces 6 and 7 resolve the config effort, forward overrides,
    and log total input, output and thinking tokens from the CLI envelope.
    """
    for effort, expected in ((None, "high"), ("low", "low")):
        calls, logs = [], []

        def runner(argv, **kwargs):
            calls.append(argv)
            envelope = {"subtype": "success", "result": '{"summary": "ok"}',
                        "usage": {"input_tokens": 2, "cache_creation_input_tokens": 2381,
                                  "cache_read_input_tokens": 649, "output_tokens": 4,
                                  "output_tokens_details": {"thinking_tokens": 0}}}
            return SimpleNamespace(returncode=0, stdout=json.dumps(envelope), stderr="")

        with patch("talk_to_slides.config.DENOISE_CLI_EFFORT", "high"):
            assert denoise_llm([], [], "", {}, "opus", effort=effort,
                               runner=runner, log=logs.append) == {"summary": "ok"}
        assert len(calls) == 1
        assert calls[0][-4:] == ["--effort", expected, "--system-prompt", _DENOISE_SYSTEM]
        assert logs == [f"  de-noise via claude CLI (plan): model=opus effort={expected} "
                        "input_tokens=3032 output_tokens=4 thinking_tokens=0"]


def test_finalize_accepted_reply_records_cleanup_provenance():
    """source: accepted cleanup records the resolved model, effort, backend, and UTC time."""
    for flags, backend, effort in (([], "cli", "high"), (["--effort", "low"], "cli", "low"),
                                   (["--llm-backend", "api"], "api", ""),
                                   (["--llm-backend", "api", "--effort", "low"], "api", "low")):
        with TemporaryDirectory() as work, \
                patch("talk_to_slides.config.DENOISE_CLI_EFFORT", "high"), \
                patch("talk_to_slides.extract.denoise_llm", return_value={"summary": "Fresh"}) as denoise, \
                redirect_stdout(StringIO()):
            path = Path(work) / "data.json"
            path.write_text(json.dumps({"meta": {"title": "Talk", "cleanup": {"model": "old"}},
                                        "data": {"summary": "Cached"}}))
            before = datetime.now(timezone.utc)
            assert finalize_cmd(["--out", work, "--model", "chosen-model", *flags]) == 0
            after = datetime.now(timezone.utc)
            saved = json.loads(path.read_text())
            cleanup = saved["meta"]["cleanup"]
            stamp = datetime.fromisoformat(cleanup["at"])
            assert before <= stamp <= after
            assert stamp.utcoffset() == timezone.utc.utcoffset(None)
            assert cleanup == {"model": "chosen-model", "effort": effort,
                               "backend": backend, "at": cleanup["at"]}
            assert saved["data"] == {"summary": "Fresh"}
            assert saved["meta"]["title"] == "Talk"
            assert (Path(work) / "report.html").exists()
            denoise.assert_called_once()
            assert denoise.call_args.args[4] == "chosen-model"
            assert denoise.call_args.kwargs["backend"] == backend


def test_finalize_cached_fallback_preserves_cleanup_provenance():
    """source: failures, unparseable replies, and rejected partial replies retain cached provenance."""
    cleanup = {"model": "previous", "effort": "low", "backend": "cli", "at": "2026-10-14T00:00:00+00:00"}
    for prior in ({"cleanup": cleanup}, {}, {"cleanup": None}):
        for cached in ({"summary": "Cached", "slides": [{"n": 1, "title": "Old slide"}]}, {}):
            replies = (RuntimeError("unavailable"), {}) if not cached else (
                RuntimeError("unavailable"), {}, {"links": []})
            for reply in replies:
                kwargs = {"side_effect": reply} if isinstance(reply, Exception) else {"return_value": reply}
                with TemporaryDirectory() as work, \
                        patch("talk_to_slides.extract.denoise_llm", **kwargs) as denoise, \
                        redirect_stdout(StringIO()):
                    path = Path(work) / "data.json"
                    meta = {"title": "Talk", **prior}
                    path.write_text(json.dumps({"meta": meta, "data": cached}))
                    assert finalize_cmd(["--out", work, "--model", "replacement", "--effort", "max"]) == 0
                    saved = json.loads(path.read_text())
                    assert saved["meta"] == meta
                    assert saved["data"] == cached
                    assert (Path(work) / "report.html").exists()
                    denoise.assert_called_once()


def test_finalize_render_only_preserves_cleanup_provenance():
    """source: render-only cannot attribute cached slides to a model that did not clean them."""
    cleanup = {"model": "previous", "effort": "medium", "backend": "cli", "at": "2026-10-14T00:00:00+00:00"}
    for prior in ({"cleanup": cleanup}, {}, {"cleanup": None}):
        with TemporaryDirectory() as work, \
                patch("talk_to_slides.extract.denoise_llm") as denoise, redirect_stdout(StringIO()):
            path = Path(work) / "data.json"
            meta = {"title": "Talk", **prior}
            cached = {"summary": "Cached"}
            path.write_text(json.dumps({"meta": meta, "data": cached}))
            assert finalize_cmd(["--out", work, "--render-only", "--model", "replacement"]) == 0
            saved = json.loads(path.read_text())
            assert saved["meta"] == meta and saved["data"] == cached
            assert (Path(work) / "report.html").exists()
            denoise.assert_not_called()


def test_finalize_busy_lock_leaves_data_unchanged_without_reading():
    """source: interface 6 exits 3 with FINALIZE BUSY before reading or writing, including render-only."""
    assert _finalize_parser().parse_args(["--out", "unused"]).lock_timeout == 600
    for flags in ([], ["--render-only"]):
        with TemporaryDirectory() as work:
            out = Path(work)
            path = out / "data.json"
            path.write_text(json.dumps({"meta": {}, "data": {"summary": "Cached"}}))
            output, errors = StringIO(), StringIO()
            with folder_lock(out, 0):
                before = {p.name: p.read_bytes() for p in out.iterdir()}
                with patch.object(Path, "read_text", side_effect=AssertionError("read while busy")), \
                        patch("talk_to_slides.extract.denoise_llm") as denoise, \
                        redirect_stdout(output), redirect_stderr(errors):
                    assert finalize_cmd(["--out", work, "--lock-timeout", "0.2", *flags]) == 3
                assert {p.name: p.read_bytes() for p in out.iterdir()} == before
            assert errors.getvalue() == f"FINALIZE BUSY: {out}\n"
            assert output.getvalue() == ""
            denoise.assert_not_called()


def test_finalize_holds_lock_from_read_through_replace_including_render_only():
    """source: interface 6 serializes the entire finalize read and replace against compaction."""
    for flags in ([], ["--render-only"]):
        with TemporaryDirectory() as work:
            out = Path(work)
            path = out / "data.json"
            path.write_text(json.dumps({"meta": {}, "data": {"summary": "Cached"}}))
            read, replace = Path.read_text, os.replace
            events = []

            def assert_locked():
                try:
                    with folder_lock(out, 0):
                        raise AssertionError("finalize left the folder unlocked")
                except TimeoutError as exc:
                    assert str(exc) == f"folder busy: {out}"

            def checked_read(target, *args, **kwargs):
                if target == path:
                    assert_locked()
                    events.append("read")
                return read(target, *args, **kwargs)

            def checked_replace(source, target):
                assert Path(target) == path
                assert_locked()
                result = replace(source, target)
                assert_locked()
                events.append("replace")
                return result

            with patch.object(Path, "read_text", checked_read), \
                    patch.object(os, "replace", checked_replace), \
                    patch("talk_to_slides.extract.denoise_llm", return_value={"summary": "Fresh"}), \
                    redirect_stdout(StringIO()):
                assert finalize_cmd(["--out", work, *flags]) == 0
            assert events == ["read", "replace"]
            with folder_lock(out, 0):
                assert json.loads(path.read_text())["data"] == {
                    "summary": "Cached" if flags else "Fresh"}


def test_finalize_releases_lock_when_data_is_unparseable():
    """source: interface 5 releases the shared lock when finalize raises while reading data.json."""
    with TemporaryDirectory() as work:
        out = Path(work)
        path = out / "data.json"
        path.write_text("{unparseable")
        try:
            finalize_cmd(["--out", work, "--render-only"])
        except json.JSONDecodeError:
            pass
        else:
            raise AssertionError("unparseable data.json did not raise")
        with folder_lock(out, 0):
            assert path.read_text() == "{unparseable"


def test_extract_records_partial_ffmpeg_sampling_before_dedup():
    """source: 2026-10-15 check=False hid ffmpeg failure and zero or partial sampled frames."""
    with TemporaryDirectory() as work:
        out = Path(work) / "output"
        sampled = []
        reader = SimpleNamespace(skipped=0, verified_empty=0)

        def ffmpeg(argv, **kwargs):
            assert argv[0] == "ffmpeg" and kwargs == {"check": False}
            pattern = argv[-1]
            count = 5 if "share_%05d" in pattern else 3
            sampled.append(Path(pattern).parent)
            for index in range(count):
                Path(pattern.replace("%05d", f"{index:05d}")).write_bytes(b"frame")
            return SimpleNamespace(returncode=1 if count == 5 else 0)

        def read_empty(path):
            assert Path(path).read_bytes() == b"frame"
            reader.verified_empty += 1
            return []

        with patch("talk_to_slides.extract._probe_dims", return_value=(1080, 1920, 20.0)), \
                patch("subprocess.run", side_effect=ffmpeg) as run, \
                patch("talk_to_slides.ocr_vision._get_default_worker", return_value=reader), \
                patch("talk_to_slides.extract.dedup_changed", side_effect=lambda paths, **kw: paths[:2]), \
                patch("talk_to_slides.extract.ocr_lines", side_effect=read_empty):
            result = run_extract("fake.mp4", str(out), chat_fps=1, share_fps=0.25,
                                 chat_crop="crop=in_w:in_h", share_crop="crop=in_w:in_h",
                                 use_llm=False, model="unused", meta={}, log=lambda line: None)
        meta = json.loads((out / "data.json").read_text())["meta"]
        assert meta["ffmpeg_rc"] == 1 and meta["share_sampled"] == 5 and meta["share_fps"] == 0.25
        assert meta["ocr_frames"] == 2 and meta["ocr_empty"] == 4
        assert run.call_count == 2 and result["slides"] == 0
        assert sampled[0] == sampled[1] and sampled[0].parent == out
        assert sampled[0].name.startswith(".webextract-") and not sampled[0].exists()
        assert not list(out.glob(".webextract-*"))


def test_extract_work_folder_is_local_and_removed_after_failure():
    """source: interface 8 puts temporary extraction work inside out_dir and removes it in finally."""
    with TemporaryDirectory() as work:
        out = Path(work) / "output"
        folders = []

        def fail_probe(video):
            folders.extend(out.glob(".webextract-*"))
            assert len(folders) == 1 and folders[0].is_dir()
            raise RuntimeError("probe failed")

        with patch("talk_to_slides.extract._probe_dims", side_effect=fail_probe):
            try:
                run_extract("fake.mp4", str(out), chat_fps=1, share_fps=0.25,
                            chat_crop="", share_crop="", use_llm=False, model="unused",
                            meta={}, no_chat=True, log=lambda line: None)
            except RuntimeError as exc:
                assert str(exc) == "probe failed"
            else:
                raise AssertionError("probe failure did not propagate")
        assert len(folders) == 1 and not folders[0].exists()
        assert not list(out.glob(".webextract-*"))


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
