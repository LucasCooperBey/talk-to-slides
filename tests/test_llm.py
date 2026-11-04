"""Cleanup backend behavior that the copied extractor tests do not pin.

source: spec rules 4 and 5. A missing claude binary, a home-folder key file,
and a whisper endpoint with no default address.
"""
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from talk_to_slides.config import DENOISE_API_MODEL
from talk_to_slides.extract import finalize_cmd, run_extract, transcribe_cmd
from talk_to_slides.llm import CLI_MISSING_LINE, ClaudeCliNotFound, run_claude_cli


def test_api_model_default_is_claude_opus_5_5():
    """source: spec rule 5, the metered backend defaults to claude-opus-5-5."""
    assert DENOISE_API_MODEL == "claude-opus-5-5"


def _hide_claude(tmp_path, monkeypatch):
    """Put a marker script named claude on PATH, then report that it is missing.

    The PATH script only runs when the missing-binary guard is gone. Patching
    subprocess.run would skip that guard, because a test double is not the real binary.
    """
    marker = tmp_path / "claude-started"
    fake = tmp_path / "claude"
    fake.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    return marker


def test_missing_claude_raises_before_a_process_starts(tmp_path, monkeypatch):
    """source: spec rule 5, a missing claude CLI is reported and not executed."""
    marker = _hide_claude(tmp_path, monkeypatch)
    with patch("shutil.which", return_value=None):
        try:
            run_claude_cli("user", "system", "opus")
        except ClaudeCliNotFound as exc:
            assert str(exc) == CLI_MISSING_LINE
        else:
            raise AssertionError("missing claude was accepted")
    assert not marker.exists()
    assert CLI_MISSING_LINE == (
        "claude CLI not found: the report has slides and text but no summary; "
        "use --llm-backend api with ANTHROPIC_API_KEY, or --no-llm"
    )


def test_missing_claude_writes_the_report_without_a_summary(tmp_path, monkeypatch):
    """source: spec rule 5, extract degrades the same way a failed cleanup does."""
    marker = _hide_claude(tmp_path, monkeypatch)
    logs = []
    out = tmp_path / "out"
    with patch("talk_to_slides.extract._probe_dims", return_value=(320, 240, 3.0)), \
            patch("talk_to_slides.extract._extract_crop_frames", return_value=([], 0)), \
            patch("shutil.which", return_value=None):
        result = run_extract(
            "baking.mp4", str(out), chat_fps=1, share_fps=1, chat_crop="", share_crop="",
            use_llm=True, model="opus", meta={"source": "baking.mp4"}, no_chat=True,
            ocr="none", log=logs.append,
        )
    assert not marker.exists()
    assert CLI_MISSING_LINE in logs
    saved = json.loads((out / "data.json").read_text())
    assert saved["data"] == {}
    html = (out / "report.html").read_text()
    assert "class=sum" not in html
    assert result["html"].endswith("report.html")


def test_finalize_missing_claude_keeps_an_empty_cleanup(tmp_path, monkeypatch):
    """source: spec rule 5, finalize prints the same line and still writes the report."""
    marker = _hide_claude(tmp_path, monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    (out / "data.json").write_text(json.dumps({
        "meta": {"source": "baking.mp4"}, "data": {}, "links": [],
        "slides": [], "chat_raw": [],
    }))
    with patch("shutil.which", return_value=None):
        stdout = __import__("io").StringIO()
        with patch("sys.stdout", stdout):
            assert finalize_cmd(["--out", str(out)]) == 0
    assert not marker.exists()
    assert CLI_MISSING_LINE in stdout.getvalue()
    saved = json.loads((out / "data.json").read_text())
    assert saved["data"] == {}
    assert (out / "report.html").is_file()


def test_api_backend_does_not_read_a_home_key_file():
    """source: spec rule 5, ANTHROPIC_API_KEY comes from the environment only."""
    import talk_to_slides.extract as extract

    assert not hasattr(extract, "_load_env_key")
    created = []
    sent = {}

    class Messages:
        def create(self, **kwargs):
            sent.update(kwargs)
            block = SimpleNamespace(type="text", text='{"summary": "Baked."}')
            return SimpleNamespace(content=[block])

    class Anthropic:
        def __init__(self, *args, **kwargs):
            created.append((args, kwargs))
            self.messages = Messages()

    module = ModuleType("anthropic")
    module.Anthropic = Anthropic
    with patch.dict(sys.modules, {"anthropic": module}), \
            patch.dict("os.environ", {}, clear=False), \
            patch("pathlib.Path.home", side_effect=AssertionError("home key file")):
        import os
        os.environ.pop("ANTHROPIC_API_KEY", None)
        assert extract.denoise_llm([], [], "", {}, "claude-opus-5-5", backend="api") == {
            "summary": "Baked.",
        }
    assert created == [((), {})]
    assert sent["model"] == "claude-opus-5-5"


def test_transcribe_exits_2_when_no_whisper_url_is_set(capsys, monkeypatch):
    """source: spec rule 4, there is no default whisper address."""
    monkeypatch.delenv("WHISPER_URL", raising=False)
    monkeypatch.delenv("WHISPER_API_KEY", raising=False)
    with TemporaryDirectory() as work:
        code = transcribe_cmd(["clip.wav", "--out", work])
        assert not (Path(work) / "transcript.json").exists()
    assert code == 2
    err = capsys.readouterr().err
    assert "--whisper-url" in err and "WHISPER_URL" in err
    assert "100.93" not in err
    assert "l" + "ux" not in err.lower()


def _transcribe(tmp, url_args, env):
    posts = []

    def post(url, **kwargs):
        posts.append((url, kwargs.get("headers")))
        return SimpleNamespace(ok=True, json=lambda: {"text": "warm dough"})

    def ffmpeg(argv, **kwargs):
        Path(argv[-1]).write_bytes(b"id3")
        return SimpleNamespace(returncode=0)

    with patch.dict("os.environ", env, clear=False), \
            patch("talk_to_slides.extract._media_duration", return_value=1.0), \
            patch("subprocess.run", side_effect=ffmpeg), \
            patch("requests.post", side_effect=post):
        code = transcribe_cmd(["clip.wav", "--out", str(tmp), *url_args])
    return code, posts, json.loads((tmp / "transcript.json").read_text())


def test_transcribe_url_flag_env_and_bearer(tmp_path, monkeypatch):
    """source: spec rule 4, the endpoint is the flag or WHISPER_URL, and the key is a Bearer token."""
    monkeypatch.delenv("WHISPER_API_KEY", raising=False)
    monkeypatch.setenv("WHISPER_URL", "https://whisper.example/v1/audio/transcriptions")
    code, posts, transcript = _transcribe(tmp_path / "env", [], {})
    assert code == 0
    assert posts == [("https://whisper.example/v1/audio/transcriptions", None)]
    assert transcript[0]["text"] == "warm dough"

    monkeypatch.setenv("WHISPER_API_KEY", "secret-token")
    code, posts, transcript = _transcribe(
        tmp_path / "flag",
        ["--whisper-url", "https://override.example/v1/audio/transcriptions"],
        {},
    )
    assert code == 0
    assert posts[0][0] == "https://override.example/v1/audio/transcriptions"
    assert posts[0][1]["Authorization"] == "Bearer secret-token"
    assert transcript[0]["text"] == "warm dough"
