"""Fixes from the 2026-11-03 push review.

source: 11-03 review gate REJECT on 5597e4c. Each test fails when its item is reverted.
"""
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from talk_to_slides.extract import (
    extract_cmd,
    render_html,
    run_cmd,
    run_extract,
)
from talk_to_slides.folder_lock import folder_lock
from talk_to_slides.ocr import ocr_lines

CLEANUP = {
    "model": "opus",
    "effort": "medium",
    "backend": "cli",
    "at": "2026-10-01T00:00:00+00:00",
}
PAID = {"summary": "Paid summary", "slides": [{"title": "Old"}]}
TESSERACT_MISSING = "tesseract not found on PATH; install it, or pass --ocr auto or --ocr none"


def _plant(out: Path, source: str, data: dict, cleanup=None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    meta = {"source": source}
    if cleanup is not None:
        meta["cleanup"] = cleanup
    (out / "data.json").write_text(json.dumps({
        "meta": meta,
        "data": data,
        "links": [],
        "slides": [],
        "chat_raw": [],
    }))


def _extract(out: Path, *, source: str = "talk.mp4", use_llm: bool = False,
             denoise=None, log=None, ocr: str = "none", frames=None):
    logs = []

    def crops(video, out_dir, fps, crop, prefix):
        return list(frames or []), 0

    kwargs = dict(
        chat_fps=1, share_fps=1, chat_crop="", share_crop="",
        use_llm=use_llm, model="opus", meta={"source": source},
        no_chat=True, ocr=ocr, log=logs.append if log is None else log,
    )
    with patch("talk_to_slides.extract._probe_dims", return_value=(16, 16, 4.0)), \
            patch("talk_to_slides.extract._extract_crop_frames", side_effect=crops), \
            patch("talk_to_slides.extract.denoise_llm", side_effect=denoise):
        run_extract("talk.mp4", str(out), **kwargs)
    return logs


_REAL_WHICH = shutil.which


def _which_tesseract(name, *args, **kwargs):
    if name == "tesseract":
        return "/usr/bin/tesseract"
    return _REAL_WHICH(name, *args, **kwargs)


def test_no_llm_keeps_same_source_cleanup_and_drops_a_different_source(tmp_path):
    """source: 11-03 review gate REJECT on 5597e4c, item 1
    A no-model re-extract keeps a same-source cleanup and drops a different source.
    """
    same = tmp_path / "same"
    _plant(same, "talk.mp4", PAID, CLEANUP)
    _extract(same)
    saved = json.loads((same / "data.json").read_text())
    assert saved["data"] == PAID
    assert saved["meta"]["cleanup"] == CLEANUP

    other = tmp_path / "other"
    _plant(other, "talk.mp4", PAID, CLEANUP)
    _extract(other, source="other.mp4")
    saved = json.loads((other / "data.json").read_text())
    assert saved["data"] == {}
    assert "cleanup" not in saved["meta"]

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "data.json").write_text("{not json")
    _extract(broken)
    saved = json.loads((broken / "data.json").read_text())
    assert saved["data"] == {}

    empty = tmp_path / "empty"
    _plant(empty, "talk.mp4", {}, CLEANUP)
    _extract(empty)
    saved = json.loads((empty / "data.json").read_text())
    assert saved["data"] == {}
    assert "cleanup" not in saved["meta"]


def test_llm_reply_keeps_prior_when_incomplete_and_replaces_when_complete(tmp_path):
    """source: 11-03 review gate REJECT on 5597e4c, item 1
    An empty model reply keeps the cached de-noise; a complete reply replaces it.
    """
    kept = tmp_path / "kept"
    _plant(kept, "talk.mp4", PAID, CLEANUP)
    logs = _extract(kept, use_llm=True, denoise=lambda *args, **kwargs: {})
    saved = json.loads((kept / "data.json").read_text())
    assert saved["data"] == PAID
    assert saved["meta"]["cleanup"] == CLEANUP
    assert "  LLM reply incomplete vs cache - keeping cached de-noise" in logs

    fresh = {"summary": "Fresh summary", "slides": [{"title": "New"}]}
    replaced = tmp_path / "replaced"
    _plant(replaced, "talk.mp4", PAID, CLEANUP)
    logs = _extract(replaced, use_llm=True, denoise=lambda *args, **kwargs: fresh)
    saved = json.loads((replaced / "data.json").read_text())
    assert saved["data"] == fresh
    assert "cleanup" not in saved["meta"]
    assert "  LLM reply incomplete vs cache - keeping cached de-noise" not in logs


def test_run_render_only_reuses_saved_cleanup_or_exits_when_missing(tmp_path, monkeypatch, capsys):
    """source: 11-03 review gate REJECT on 5597e4c, item 1
    run --render-only re-renders a saved cleanup and exits 2 when nothing is saved.
    """
    def boom(*args, **kwargs):
        raise AssertionError("extract or captions ran during render-only")

    monkeypatch.setattr("talk_to_slides.extract.run_extract", boom)
    monkeypatch.setattr("talk_to_slides.extract.attach_captions", boom)
    video = tmp_path / "talk.mp4"
    ready = tmp_path / "ready"
    _plant(ready, "talk.mp4", {"summary": "Saved summary text"})
    assert run_cmd([str(video), "--out", str(ready), "--render-only"]) == 0
    assert "Saved summary text" in (ready / "report.html").read_text()

    missing = tmp_path / "missing"
    assert run_cmd([str(video), "--out", str(missing), "--render-only"]) == 2
    err = capsys.readouterr().err
    assert err.splitlines()[-1] == (
        f"run --render-only: no data.json in {missing} to re-render; "
        "run without --render-only first"
    )
    assert not missing.exists()


def test_run_no_llm_keeps_a_planted_cleanup(tmp_path):
    """source: 11-03 review gate REJECT on 5597e4c, item 1
    run --no-llm over a folder that already holds a cleanup does not wipe it.
    """
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"video")
    out = tmp_path / "out"
    _plant(out, "talk.mp4", PAID, CLEANUP)
    with patch("talk_to_slides.extract._probe_dims", return_value=(16, 16, 4.0)), \
            patch("talk_to_slides.extract._extract_crop_frames", return_value=([], 0)):
        assert run_cmd([str(video), "--out", str(out), "--no-llm", "--ocr", "none"]) == 0
    saved = json.loads((out / "data.json").read_text())
    assert saved["data"]["summary"] == "Paid summary"
    assert saved["meta"]["cleanup"] == CLEANUP
    assert "Paid summary" in (out / "report.html").read_text()


def test_extract_lock_times_out_before_probe_and_commands_report_busy(tmp_path, capsys):
    """source: 11-03 review gate REJECT on 5597e4c, item 2
    Extraction waits on the folder lock and a busy folder exits 3.
    """
    out = tmp_path / "out"
    out.mkdir()
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"video")
    probed = []

    def probe(video_path):
        probed.append(video_path)
        return (16, 16, 4.0)

    with folder_lock(out, 0), \
            patch("talk_to_slides.extract._probe_dims", side_effect=probe):
        with pytest.raises(TimeoutError):
            run_extract(
                str(video), str(out), chat_fps=1, share_fps=1, chat_crop="", share_crop="",
                use_llm=False, model="", meta={"source": "talk.mp4"}, no_chat=True,
                ocr="none", lock_timeout=0.1, log=lambda line: None,
            )
    assert probed == []

    with folder_lock(out, 0):
        assert extract_cmd([
            str(video), "--out", str(out), "--lock-timeout", "0.1", "--no-llm", "--ocr", "none",
        ]) == 3
    assert f"EXTRACT BUSY: {out}" in capsys.readouterr().err

    with folder_lock(out, 0):
        assert run_cmd([
            str(video), "--out", str(out), "--lock-timeout", "0.1", "--no-llm", "--ocr", "none",
        ]) == 3
    assert f"EXTRACT BUSY: {out}" in capsys.readouterr().err
    assert probed == []


def test_tesseract_frame_failures_are_skipped_and_missing_binary_exits(tmp_path, capsys):
    """source: 11-03 review gate REJECT on 5597e4c, item 3
    One failed tesseract frame is skipped, and a missing binary exits 2 before probing.
    """
    with patch("talk_to_slides.ocr.subprocess.run",
               side_effect=subprocess.TimeoutExpired(["tesseract"], 1)):
        assert ocr_lines("frame-2.png", "tesseract") == []

    frames = []
    for name, color in (("a.png", "white"), ("b.png", "black"), ("c.png", "red")):
        path = tmp_path / name
        Image.new("RGB", (32, 32), color).save(path)
        frames.append(str(path))

    def run_frames(fail):
        calls = {"n": 0}

        def fake_run(argv, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                if fail == "timeout":
                    raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 60))
                return subprocess.CompletedProcess(argv, 1, stdout="", stderr="nope")
            return subprocess.CompletedProcess(argv, 0, stdout="Proof\n", stderr="")

        out = tmp_path / fail
        logs = []
        # Solid colors share an average hash, so keep every planted frame and
        # let the second of the three be the one tesseract fails.
        with patch("shutil.which", side_effect=_which_tesseract), \
                patch("talk_to_slides.ocr.subprocess.run", side_effect=fake_run), \
                patch("talk_to_slides.extract.dedup_changed",
                      side_effect=lambda paths, **kwargs: list(paths)):
            _extract(out, ocr="tesseract", frames=frames, log=logs.append)
        saved = json.loads((out / "data.json").read_text())
        assert saved["meta"]["ocr_skipped"] == 1, saved["meta"]
        assert saved["meta"]["ocr_frames"] == 3
        assert "  ocr skipped 1 frames" in logs
        return saved

    run_frames("timeout")
    run_frames("nonzero")

    video = tmp_path / "talk.mp4"
    video.write_bytes(b"video")
    probed = []

    def probe(video_path):
        probed.append(video_path)
        raise AssertionError("probed")

    def which(name, *args, **kwargs):
        if name == "tesseract":
            return None
        return _REAL_WHICH(name, *args, **kwargs)

    with patch("shutil.which", side_effect=which), \
            patch("talk_to_slides.extract._probe_dims", side_effect=probe):
        extract_out = tmp_path / "extract-out"
        assert extract_cmd([
            str(video), "--out", str(extract_out), "--ocr", "tesseract", "--no-llm",
        ]) == 2
        assert TESSERACT_MISSING in capsys.readouterr().err
        assert not extract_out.exists()

        run_out = tmp_path / "run-out"
        assert run_cmd([
            str(video), "--out", str(run_out), "--ocr", "tesseract", "--no-llm",
        ]) == 2
        assert TESSERACT_MISSING in capsys.readouterr().err
        assert not run_out.exists()
    assert probed == []


def test_old_slide_pngs_are_removed_and_other_files_stay(tmp_path):
    """source: 11-03 review gate REJECT on 5597e4c, item 4
    A new extract deletes slide_NN.png and leaves every other file in slides/.
    """
    out = tmp_path / "out"
    slides = out / "slides"
    slides.mkdir(parents=True)
    (slides / "slide_09.png").write_bytes(b"old")
    (slides / "notes.txt").write_text("keep")
    frame = tmp_path / "frame.png"
    Image.new("RGB", (16, 16), "white").save(frame)
    _extract(out, frames=[str(frame)])
    assert not (slides / "slide_09.png").exists()
    assert (slides / "notes.txt").read_text() == "keep"
    assert (slides / "slide_01.png").is_file()


def test_render_survives_bad_shapes_and_escapes_gallery_attributes():
    """source: 11-03 review gate REJECT on 5597e4c, item 5
    A bad data.json shape still renders, and gallery attributes are escaped.
    """
    bare = [{"n": 1, "lines": ["kept"], "png": "slides/a.png"}]
    for payload in (
        {"chat": None},
        {"takeaways": None},
        {"chat": ["x"]},
        {"key_moments": [1, "a"]},
        {"slides": ["s", {"t": "00:01"}]},
    ):
        html = render_html(payload, bare, [], [], {})
        assert "<html>" in html

    evil_png = 'shot.png" onerror="alert(1)'
    evil_thumb = 'thumb" onload="alert(1)'
    frames = [
        {"n": 1, "lines": ["a"], "png": evil_png, "thumb": evil_thumb},
        {"n": 2, "t": "00:02", "lines": ["b"], "png": evil_png, "thumb": "data:image/jpeg;base64,QQ"},
    ]
    html = render_html(
        {"slides": ["s", {"t": "00:01", "title": "Deck", "points": []}]},
        frames, None, [], {},
    )
    assert "Deck" in html
    assert "&quot;" in html
    assert '" on' not in html


def test_bad_llm_backend_fails_before_extract(tmp_path, monkeypatch):
    """source: 11-03 review gate REJECT on 5597e4c, item 6
    A bad model setting fails before the long extract pass.
    """
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"video")
    out = tmp_path / "out"
    calls = []

    def explode(*args, **kwargs):
        calls.append("extract")
        raise AssertionError("run_extract ran")

    monkeypatch.setattr("talk_to_slides.extract.run_extract", explode)
    with pytest.raises(SystemExit) as caught:
        run_cmd([str(video), "--out", str(out), "--llm-backend", "nope"])
    assert caught.value.code == 2
    assert calls == []
    assert not out.exists()

    def reject(backend, model):
        calls.append("resolve")
        raise ValueError(f"Unknown LLM backend: {backend}")

    monkeypatch.setattr("talk_to_slides.extract.resolve_denoise_model", reject)
    with pytest.raises(ValueError, match="Unknown LLM backend: cli"):
        run_cmd([str(video), "--out", str(out), "--llm-backend", "cli"])
    assert calls == ["resolve"]
