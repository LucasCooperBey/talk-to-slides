"""OCR backend selection, tesseract, and the no-text path.

source: spec rule 3. These fail when auto never looks for a backend, tesseract
is not invoked as a binary, or empty text is sent through text dedup.
"""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from talk_to_slides.extract import run_extract
from talk_to_slides.ocr import OCR_NONE_WARNING, ocr_lines, select_backend
from talk_to_slides.ocr_vision import OcrWorker


def _png(path, color):
    Image.new("RGB", (32, 32), color).save(path)


def test_select_backend_auto_order_and_explicit_choices():
    """source: spec rule 3, auto is Vision, then tesseract, then none."""
    with patch("talk_to_slides.ocr._vision_imports", return_value=True):
        assert select_backend("auto") == "vision"
    with patch("talk_to_slides.ocr._vision_imports", return_value=False), \
            patch("talk_to_slides.ocr.shutil.which", return_value="/usr/bin/tesseract"):
        assert select_backend("auto") == "tesseract"
    with patch("talk_to_slides.ocr._vision_imports", return_value=False), \
            patch("talk_to_slides.ocr.shutil.which", return_value=None):
        assert select_backend("auto") == "none"
    assert select_backend("vision") == "vision"
    assert select_backend("tesseract") == "tesseract"
    assert select_backend("none") == "none"
    try:
        select_backend("other")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown OCR choice accepted")


def test_tesseract_runs_the_binary_with_stdout_and_timeout():
    """source: spec rule 3, tesseract is the binary, not a Python wrapper."""
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["timeout"] = kwargs["timeout"]
        return SimpleNamespace(returncode=0, stdout=" Proof \n\n crumb \n")

    with patch.dict("os.environ", {"TALK_TO_SLIDES_OCR_TIMEOUT": "12"}), \
            patch("talk_to_slides.ocr.subprocess.run", side_effect=fake_run):
        assert ocr_lines("slide.png", "tesseract") == ["Proof", "crumb"]
    assert seen["argv"] == ["tesseract", "slide.png", "stdout"]
    assert seen["timeout"] == 12.0

    def failed(argv, **kwargs):
        return SimpleNamespace(returncode=1, stdout="ignored")

    with patch("talk_to_slides.ocr.subprocess.run", side_effect=failed):
        assert ocr_lines("slide.png", "tesseract") == []


def test_none_keeps_every_visual_frame_and_skips_text_dedup():
    """source: spec rule 3, empty text must not collapse every frame into one slide."""
    with TemporaryDirectory() as work:
        root = Path(work)
        frames = []
        for name, color in (("a.png", "white"), ("b.png", "black")):
            path = root / name
            _png(path, color)
            frames.append(str(path))

        def crops(video, out_dir, fps, crop, prefix):
            return frames, 0

        def dedup_must_not_run(*args, **kwargs):
            raise AssertionError("dedup_changed_texts ran with no OCR text")

        with patch("talk_to_slides.extract._probe_dims", return_value=(32, 32, 10.0)), \
                patch("talk_to_slides.extract._extract_crop_frames", side_effect=crops), \
                patch("talk_to_slides.extract.dedup_changed_texts", side_effect=dedup_must_not_run):
            result = run_extract(
                "baking.mp4", str(root / "out"), chat_fps=1, share_fps=1,
                chat_crop="", share_crop="", use_llm=False, model="",
                meta={"source": "baking.mp4"}, no_chat=True, ocr="none",
                log=lambda line: None,
            )
        saved = json.loads((root / "out" / "data.json").read_text())
        assert result["slides"] == 2
        assert saved["meta"]["ocr"] == "none"
        assert [slide["lines"] for slide in saved["slides"]] == [[], []]
        assert [slide["png"] for slide in saved["slides"]] == [
            "slides/slide_01.png", "slides/slide_02.png",
        ]


def test_auto_none_warns_once_and_explicit_none_does_not():
    """source: spec rule 3, the warning names both ways to get slide text, and only for auto."""
    with TemporaryDirectory() as work:
        root = Path(work)
        frame = root / "frame.png"
        _png(frame, "white")
        logs = []

        def crops(video, out_dir, fps, crop, prefix):
            return [str(frame)], 0

        common = dict(
            chat_fps=1, share_fps=1, chat_crop="", share_crop="", use_llm=False,
            model="", meta={"source": "baking.mp4"}, no_chat=True, log=logs.append,
        )
        with patch("talk_to_slides.ocr._vision_imports", return_value=False), \
                patch("talk_to_slides.ocr.shutil.which", return_value=None), \
                patch("talk_to_slides.extract._probe_dims", return_value=(32, 32, 4.0)), \
                patch("talk_to_slides.extract._extract_crop_frames", side_effect=crops):
            run_extract("baking.mp4", str(root / "auto"), **common, ocr="auto")
            assert logs.count(OCR_NONE_WARNING) == 1
            assert "ocrmac" in OCR_NONE_WARNING and "tesseract" in OCR_NONE_WARNING
            logs.clear()
            run_extract("baking.mp4", str(root / "explicit"), **common, ocr="none")
            assert OCR_NONE_WARNING not in logs


def test_ocr_lines_none_is_empty_and_unknown_backend_raises():
    """source: spec rule 3, none stores no text and a bad backend is refused."""
    assert ocr_lines("missing.png", "none") == []
    try:
        ocr_lines("missing.png", "other")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown OCR backend accepted")


def test_vision_child_starts_with_no_cwd():
    """source: spec rule 3, the Vision child is the installed module and has no cwd."""
    import subprocess

    seen = {}
    real = subprocess.Popen

    def tracked(argv, **kwargs):
        seen["argv"] = list(argv)
        seen["kwargs"] = kwargs
        return real(argv, **kwargs)

    script = (
        "import json, sys\n"
        "print(json.dumps({'ready': True}), flush=True)\n"
        "for raw in sys.stdin:\n"
        "    req = json.loads(raw)\n"
        "    print(json.dumps({'id': req['id'], 'lines': ['ok']}), flush=True)\n"
    )
    worker = OcrWorker(argv=[sys.executable, "-c", script], timeout=2, first_timeout=2)
    try:
        with patch("talk_to_slides.ocr_vision.subprocess.Popen", tracked):
            assert worker.ocr("frame.png") == ["ok"]
    finally:
        worker.close()
    assert seen["argv"][0] == sys.executable
    assert "cwd" not in seen["kwargs"]
    default = OcrWorker()
    try:
        assert default.argv == [sys.executable, "-m", "talk_to_slides.ocr_vision"]
    finally:
        default.close()
