"""Command shape for layout, run, and portable output paths.

source: spec rules 1, 2, 6, and 7.
"""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from talk_to_slides.cli import main
from talk_to_slides.extract import (
    CHAT_CROP,
    FULL_FRAME_CROP,
    SHARE_CROP,
    _extract_parser,
    finalize_cmd,
    resolve_layout,
    run_extract,
)


def _layout(argv):
    return resolve_layout(_extract_parser().parse_args(argv))


def test_default_layout_is_full_frame_without_chat():
    """source: spec rule 2, the default is the whole frame and no chat pass."""
    args = _extract_parser().parse_args(["baking.mp4", "--full-frame"])
    assert args.full_frame is True
    assert _layout(["baking.mp4"]) == (True, CHAT_CROP, FULL_FRAME_CROP)
    assert _layout(["baking.mp4", "--full-frame"]) == (True, CHAT_CROP, FULL_FRAME_CROP)


def test_zoom_chat_layout_uses_the_side_crops():
    """source: spec rule 2, zoom-chat selects the chat and share crops."""
    assert _layout(["baking.mp4", "--layout", "zoom-chat"]) == (False, CHAT_CROP, SHARE_CROP)
    assert _layout(["baking.mp4", "--layout", "zoom-chat", "--no-chat"]) == (
        True, CHAT_CROP, SHARE_CROP,
    )
    assert _layout(["baking.mp4", "--layout", "zoom-chat", "--full-frame"]) == (
        True, CHAT_CROP, FULL_FRAME_CROP,
    )


def test_explicit_crops_replace_the_layout_crops():
    """source: spec rule 2, --chat-crop and --share-crop win over the layout."""
    assert _layout(["baking.mp4", "--chat-crop", "crop=chat", "--share-crop", "crop=share"]) == (
        True, "crop=chat", "crop=share",
    )
    assert _layout([
        "baking.mp4", "--layout", "zoom-chat", "--chat-crop", "crop=chat", "--share-crop", "crop=share",
    ]) == (False, "crop=chat", "crop=share")


def test_extract_command_passes_the_default_layout_and_output_dir(tmp_path, monkeypatch):
    """source: spec rules 2 and 7, the CLI default is full frame under ./talk-to-slides-out."""
    monkeypatch.chdir(tmp_path)
    seen = {}

    def fake_extract(video, out_dir, **kwargs):
        seen["video"] = video
        seen["out"] = out_dir
        seen["kwargs"] = kwargs
        return {"slides": 0}

    monkeypatch.setattr(sys, "argv", ["talk-to-slides", "baking.mp4", "--no-llm"])
    with patch("talk_to_slides.extract.run_extract", side_effect=fake_extract):
        assert main() == 0
    assert seen["kwargs"]["no_chat"] is True
    assert seen["kwargs"]["share_crop"] == FULL_FRAME_CROP
    assert seen["kwargs"]["ocr"] == "auto"
    assert seen["kwargs"]["use_llm"] is False
    assert seen["out"] == str(tmp_path / "talk-to-slides-out" / "baking")
    assert seen["kwargs"]["meta"]["source"] == "baking.mp4"
    assert "source_path" not in seen["kwargs"]["meta"]


def test_run_chains_extract_captions_and_finalize(tmp_path, monkeypatch):
    """source: spec rule 1, run extracts with no model, then captions, then finalize."""
    video = tmp_path / "baking.mp4"
    video.write_bytes(b"video")
    out = tmp_path / "out"
    seen = {}

    def fake_extract(video_path, out_dir, **kwargs):
        seen["use_llm"] = kwargs["use_llm"]
        seen["no_chat"] = kwargs["no_chat"]
        seen["share_crop"] = kwargs["share_crop"]
        seen["chat_crop"] = kwargs["chat_crop"]
        return {"slides": 0}

    def fake_captions(video_path, out_dir, captions_file):
        seen["captions"] = captions_file
        seen["caption_out"] = str(out_dir)
        return 0

    def fake_finalize(argv):
        seen["finalize"] = argv
        return 0

    monkeypatch.setattr(sys, "argv", [
        "talk-to-slides", "run", str(video), "--out", str(out),
        "--captions", "notes.vtt", "--layout", "zoom-chat", "--no-llm",
    ])
    with patch("talk_to_slides.extract.run_extract", side_effect=fake_extract), \
            patch("talk_to_slides.extract.attach_captions", side_effect=fake_captions), \
            patch("talk_to_slides.extract.finalize_cmd", side_effect=fake_finalize):
        assert main() == 0
    assert seen["use_llm"] is False
    assert seen["no_chat"] is False
    assert seen["chat_crop"] == CHAT_CROP
    assert seen["share_crop"] == SHARE_CROP
    assert seen["captions"] == "notes.vtt"
    assert seen["caption_out"] == str(out)
    assert "--render-only" in seen["finalize"]
    assert seen["finalize"][seen["finalize"].index("--out") + 1] == str(out)


def test_run_stops_when_captions_have_no_cues(tmp_path, monkeypatch):
    """source: spec rule 1, a caption file with no cues does not continue into finalize."""
    video = tmp_path / "baking.mp4"
    video.write_bytes(b"video")
    called = []
    monkeypatch.setattr(sys, "argv", [
        "talk-to-slides", "run", str(video), "--out", str(tmp_path / "out"), "--no-llm",
    ])
    with patch("talk_to_slides.extract.run_extract", return_value={"slides": 0}), \
            patch("talk_to_slides.extract.attach_captions", return_value=3), \
            patch("talk_to_slides.extract.finalize_cmd", side_effect=lambda argv: called.append(argv)):
        assert main() == 3
    assert called == []


def test_extract_stores_png_relative_to_the_output_folder(tmp_path):
    """source: spec rule 7, slide png values are slides/slide_NN.png and source_path is absent."""
    frame = tmp_path / "frame.png"
    Image.new("RGB", (16, 16), "white").save(frame)
    out = tmp_path / "out"
    with patch("talk_to_slides.extract._probe_dims", return_value=(16, 16, 5.0)), \
            patch("talk_to_slides.extract._extract_crop_frames", return_value=([str(frame)], 0)):
        run_extract(
            str(tmp_path / "baking.mp4"), str(out), chat_fps=1, share_fps=1,
            chat_crop="", share_crop="", use_llm=False, model="",
            meta={"source": "baking.mp4", "title": "", "presenter": "", "date": ""},
            no_chat=True, ocr="none", log=lambda line: None,
        )
    saved = json.loads((out / "data.json").read_text())
    assert saved["slides"][0]["png"] == "slides/slide_01.png"
    assert (out / "slides" / "slide_01.png").is_file()
    assert saved["meta"]["source"] == "baking.mp4"
    assert "source_path" not in saved["meta"]
    assert saved["slides"][0]["lines"] == []


def test_finalize_resolves_a_relative_png_against_out(tmp_path):
    """source: spec rule 7, finalize finds slides/slide_01.png under --out."""
    out = tmp_path / "out"
    (out / "slides").mkdir(parents=True)
    Image.new("RGB", (32, 32), "white").save(out / "slides" / "slide_01.png")
    (out / "data.json").write_text(json.dumps({
        "meta": {"source": "baking.mp4", "title": "Bread"},
        "data": {},
        "links": [],
        "slides": [{"n": 1, "t": "00:05", "lines": ["Proof"], "png": "slides/slide_01.png"}],
        "chat_raw": [],
    }))
    assert finalize_cmd(["--out", str(out), "--render-only"]) == 0
    html = (out / "report.html").read_text()
    assert "data:image/jpeg;base64," in html
    assert "00:05" in html
    saved = json.loads((out / "data.json").read_text())
    assert saved["slides"][0]["png"] == "slides/slide_01.png"
    assert "source_path" not in json.dumps(saved)


def test_publish_command_is_gone():
    """source: spec rule 6, there is no publish command, port, or blueprint mention."""
    import talk_to_slides.cli as cli
    import talk_to_slides.extract as extract

    assert not hasattr(extract, "publish_cmd")
    source = Path(extract.__file__).read_text() + Path(cli.__file__).read_text()
    assert "publish_cmd" not in source
    # The old viewer port, checked in two parts so this file does not contain it.
    assert "87" + "88" not in source
    assert "blueprint" not in source.lower()
