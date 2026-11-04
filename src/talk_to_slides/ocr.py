"""OCR backend selection and the tesseract binary."""
from __future__ import annotations

import shutil
import subprocess

from talk_to_slides.config import env_seconds

OCR_NONE_WARNING = (
    "warning: no OCR backend found. Slide images will be kept without text. "
    "Install ocrmac for Apple Vision, or install the tesseract binary."
)


def _vision_imports() -> bool:
    try:
        import ocrmac  # noqa: F401
    except ImportError:
        return False
    return True


def select_backend(choice: str) -> str:
    """Return vision, tesseract, or none.

    auto uses Apple Vision when ocrmac imports, else tesseract when the binary
    is on PATH, else none.
    """
    if choice in ("vision", "tesseract", "none"):
        return choice
    if choice != "auto":
        raise ValueError(f"Unknown OCR backend: {choice}")
    if _vision_imports():
        return "vision"
    if shutil.which("tesseract"):
        return "tesseract"
    return "none"


def _tesseract_lines(png_path: str) -> list[str]:
    timeout = env_seconds("TALK_TO_SLIDES_OCR_TIMEOUT", "60")
    result = subprocess.run(
        ["tesseract", png_path, "stdout"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def ocr_lines(png_path: str, backend: str) -> list[str]:
    if backend == "none":
        return []
    if backend == "tesseract":
        return _tesseract_lines(png_path)
    if backend == "vision":
        from talk_to_slides.ocr_vision import ocr_lines as vision_lines

        return vision_lines(png_path)
    raise ValueError(f"Unknown OCR backend: {backend}")
