"""One synthetic recording through talk-to-slides run.

source: spec rule 9. Fails when run does not keep three distinct slides.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

SLIDES = (
    "Slide one: proof",
    "Slide two: crumb",
    "Slide three: crust",
)


def _require_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required for the end-to-end recording")


def _run(argv):
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result


def _still(path, text, top):
    # The three lines sit in different thirds. An 8x8 average hash treats
    # same-position text as one slide, so the words have to move.
    image = Image.new("RGB", (1280, 720), "white")
    ImageDraw.Draw(image).text(
        (80, top), text, fill="black", font=ImageFont.load_default(size=140),
    )
    image.save(path)


def _video(folder: Path) -> Path:
    parts = []
    for index, text in enumerate(SLIDES, 1):
        image = folder / f"still_{index}.png"
        part = folder / f"part_{index}.mp4"
        _still(image, text, (40, 280, 520)[index - 1])
        # -t is an output limit so each still is 5 seconds. Lossless so identical
        # frames stay inside the visual-hash threshold. This ffmpeg build has no
        # drawtext filter, so the text is drawn first.
        _run([
            "ffmpeg", "-y", "-loop", "1", "-i", str(image), "-t", "5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "1", "-crf", "0",
            str(part),
        ])
        parts.append(part)
    listing = folder / "parts.txt"
    listing.write_text("".join(f"file '{part}'\n" for part in parts))
    video = folder / "baking.mp4"
    _run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
        "-c", "copy", str(video),
    ])
    (folder / "source.en.vtt").write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:05.000\n"
        "Slide one proof\n\n"
        "00:00:05.000 --> 00:00:10.000\n"
        "Slide two crumb\n\n"
        "00:00:10.000 --> 00:00:15.000\n"
        "Slide three crust\n"
    )
    return video


def test_run_writes_three_slides_from_a_synthetic_recording(tmp_path):
    """source: spec rule 9, run --no-llm keeps three slides, their times, and OCR text."""
    _require_ffmpeg()
    video = _video(tmp_path)
    out = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable, "-m", "talk_to_slides.cli", "run", str(video),
            "--out", str(out), "--no-llm", "--share-fps", "1",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    saved = json.loads((out / "data.json").read_text())
    slides = saved["slides"]
    assert len(slides) == 3, slides
    pngs = sorted((out / "slides").glob("*.png"))
    assert len(pngs) == 3
    report = (out / "report.html").read_text()
    times = [slide["t"] for slide in slides]
    assert len(set(times)) == 3, times
    for stamp in times:
        assert stamp in report
    assert saved["meta"]["source"] == "baking.mp4"
    assert "source_path" not in saved["meta"]
    if saved["meta"].get("ocr") != "none":
        joined = "\n".join(line for slide in slides for line in slide["lines"]).lower()
        for word in ("proof", "crumb", "crust"):
            assert word in joined, (word, joined, saved["meta"].get("ocr"))
