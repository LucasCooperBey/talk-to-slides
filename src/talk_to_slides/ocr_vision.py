"""Apple Vision OCR isolated in a persistent child with per-frame deadlines."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from talk_to_slides.config import env_seconds


CANARY_TEXT = "CANARY 4817"
CANARY_TOKEN = "4817"
CANARY_TTL = 30.0


def make_canary_png(path: str) -> str:
    """Write known text for checking whether Vision can still read images."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1200, 300), "white")
    ImageDraw.Draw(image).text((60, 80), CANARY_TEXT, fill="black",
                              font=ImageFont.load_default(size=120))
    image.save(path, format="PNG")
    return path


def vision_ocr_lines(png_path: str) -> list[str]:
    """OCR an image with Apple Vision, returning text lines ordered top-to-bottom.

    Vision reports bottom-origin y in [0,1]; we sort descending so output reads top-down.
    """
    from ocrmac import ocrmac

    res = ocrmac.OCR(png_path, recognition_level="accurate").recognize()
    rows = sorted(res, key=lambda r: -r[2][1])  # bbox = (x, y, w, h), y bottom-origin
    return [txt for txt, _conf, _bbox in rows if txt.strip()]


class _AttemptFailed(Exception):
    """A dead or unresponsive child can be replaced and retried."""


class OcrWorker:
    def __init__(self, argv: list[str] | None = None, timeout: float | None = None, log=None,
                 *, first_timeout: float | None = None,
                 reap_timeout: float = 10.0, wedged_cooldown: float = 300.0,
                 canary_path: str | None = None, canary_ttl: float = CANARY_TTL):
        self.argv = argv if argv is not None else [sys.executable, "-m", "talk_to_slides.ocr_vision"]
        self.timeout = timeout if timeout is not None else env_seconds("TALK_TO_SLIDES_OCR_TIMEOUT", "60")
        self.first_timeout = (first_timeout if first_timeout is not None
                              else env_seconds("TALK_TO_SLIDES_OCR_FIRST_TIMEOUT", "180"))
        self.reap_timeout = reap_timeout
        self.wedged_cooldown = wedged_cooldown
        self.canary_path = canary_path
        self.canary_ttl = canary_ttl
        self.log = log if log is not None else lambda line: print(line, file=sys.stderr, flush=True)
        self.skipped = 0
        self.verified_empty = 0
        self._canary_ok_at = float("-inf")
        self._canary_dir = None
        self._consecutive_skips = 0
        self._wedged_until = 0.0
        self._lock = threading.Lock()
        self._process = None
        self._request_id = 0
        self._stdout = b""
        self._stderr = b""
        self._stderr_eof = False

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _AttemptFailed(f"timed out after {self.timeout:g}s")
        return remaining

    def _read_stderr(self) -> None:
        try:
            chunk = os.read(self._process.stderr.fileno(), 65536)
        except BlockingIOError:
            return
        if chunk:
            self._stderr = (self._stderr + chunk)[-8192:]
        else:
            self._stderr_eof = True

    def _drain_stderr(self) -> None:
        # Drain diagnostics after exit, including import errors without a newline.
        while not self._stderr_eof:
            ready, _, _ = select.select([self._process.stderr], [], [], 0)
            if not ready:
                break
            self._read_stderr()

    def _stop(self) -> None:
        self._canary_ok_at = float("-inf")
        child = self._process
        if child is None:
            return
        if child.poll() is None:
            child.kill()
        try:
            child.wait(timeout=self.reap_timeout)
        except subprocess.TimeoutExpired:
            self.log(f"ocr: child {child.pid} did not exit {self.reap_timeout:g}s after SIGKILL; abandoned it")
        else:
            self._drain_stderr()
        for pipe in (child.stdin, child.stdout, child.stderr):
            pipe.close()
        self._process = None

    def _exit_reason(self, deadline: float) -> str:
        child = self._process
        try:
            child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            self._stop()
            return "worker closed its output but did not exit"
        self._drain_stderr()
        lines = [line for line in self._stderr.decode("utf-8", errors="replace").splitlines()
                 if line.strip()]
        reason = f"worker exited with {child.returncode}"
        return f"{reason}: {lines[-1][:200]}" if lines else reason

    def _read_message(self, deadline: float) -> dict:
        while True:
            remaining = self._remaining(deadline)
            if b"\n" in self._stdout:
                line, self._stdout = self._stdout.split(b"\n", 1)
                try:
                    reply = json.loads(line)
                except ValueError as exc:
                    raise _AttemptFailed("bad reply line") from exc
                if not isinstance(reply, dict):
                    raise _AttemptFailed("bad reply line")
                return reply
            child = self._process
            readers = [child.stdout]
            if not self._stderr_eof:
                readers.append(child.stderr)
            ready, _, _ = select.select(readers, [], [], remaining)
            for pipe in ready:
                if pipe is child.stderr:
                    self._read_stderr()
                else:
                    chunk = os.read(pipe.fileno(), 65536)
                    if not chunk:
                        raise _AttemptFailed(self._exit_reason(deadline))
                    self._stdout += chunk

    def _start(self) -> None:
        self._canary_ok_at = float("-inf")
        self._served = False
        self._stdout = self._stderr = b""
        self._stderr_eof = False
        deadline = time.monotonic() + max(self.timeout, self.first_timeout)
        try:
            self._process = subprocess.Popen(
                self.argv,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=0,
            )
            for pipe in (self._process.stdin, self._process.stdout, self._process.stderr):
                os.set_blocking(pipe.fileno(), False)
            if self._read_message(deadline) != {"ready": True}:
                raise _AttemptFailed("missing ready line")
        except (OSError, ValueError, _AttemptFailed) as exc:
            self._stop()
            lines = self._stderr.decode("utf-8", errors="replace").strip().splitlines()
            reason = lines[-1] if lines else str(exc)
            raise RuntimeError(f"ocr worker failed to start: {reason}") from exc

    def _send(self, path: str, request_id: int, deadline: float) -> None:
        data = memoryview((json.dumps({"id": request_id, "path": path}) + "\n").encode())
        pipe = self._process.stdin
        while data:
            _, ready, _ = select.select([], [pipe], [], self._remaining(deadline))
            if ready:
                try:
                    data = data[os.write(pipe.fileno(), data):]
                except BlockingIOError:
                    continue
                except BrokenPipeError as exc:
                    raise _AttemptFailed(self._exit_reason(deadline)) from exc

    def check_canary(self) -> bool:
        with self._lock:
            return self._check_canary_locked()

    def _check_canary_locked(self) -> bool:
        if self._process is None:
            self._start()
        if self.canary_path is None:
            self._canary_dir = tempfile.mkdtemp(prefix="vc-ocr-canary-")
            self.canary_path = make_canary_png(os.path.join(self._canary_dir, "canary.png"))
        self._request_id += 1
        timeout = self.timeout if self._served else max(self.timeout, self.first_timeout)
        deadline = time.monotonic() + timeout
        try:
            try:
                self._send(self.canary_path, self._request_id, deadline)
                while True:
                    reply = self._read_message(deadline)
                    if reply.get("id") == self._request_id:
                        break
            finally:
                self._served = True
        except _AttemptFailed:
            self._stop()
            return False
        lines = reply.get("lines")
        if ("error" in reply or not isinstance(lines, list)
                or not all(isinstance(line, str) for line in lines)
                or CANARY_TOKEN not in " ".join(lines)):
            return False
        self._canary_ok_at = time.monotonic()
        return True

    def ocr(self, path: str) -> list[str]:
        with self._lock:
            if self._consecutive_skips >= 3 and time.monotonic() < self._wedged_until:
                raise RuntimeError("ocr: 3 frames in a row failed; Vision looks wedged")
            reasons = []
            for attempt in range(2):
                if self._process is None:
                    self._start()
                self._request_id += 1
                timeout = self.timeout if self._served else max(self.timeout, self.first_timeout)
                deadline = time.monotonic() + timeout
                try:
                    try:
                        self._send(path, self._request_id, deadline)
                        while True:
                            reply = self._read_message(deadline)
                            if reply.get("id") == self._request_id:
                                break
                    finally:
                        self._served = True
                    if "error" in reply:
                        raise RuntimeError(f"ocr failed for {Path(path).name}: {reply['error']}")
                    lines = reply.get("lines")
                    if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
                        raise _AttemptFailed("bad reply line")
                    if not any(line.strip() for line in lines):
                        if (time.monotonic() - self._canary_ok_at > self.canary_ttl
                                and not self._check_canary_locked()):
                            raise _AttemptFailed("no text and the canary failed")
                        self.verified_empty += 1
                except _AttemptFailed as exc:
                    reasons.append(str(exc))
                    self._stop()
                    continue
                self._consecutive_skips = 0
                return lines
            self.skipped += 1
            self._consecutive_skips += 1
            self.log(f"ocr: skipped {Path(path).name}: {reasons[0]}, then {reasons[1]}")
            if self._consecutive_skips >= 3:
                self._wedged_until = time.monotonic() + self.wedged_cooldown
                raise RuntimeError("ocr: 3 frames in a row failed; Vision looks wedged")
            return []

    def close(self) -> None:
        with self._lock:
            self._stop()
            if self._canary_dir is not None:
                shutil.rmtree(self._canary_dir)
                self._canary_dir = None
                self.canary_path = None


_default_worker: OcrWorker | None = None
_default_lock = threading.Lock()


def _get_default_worker() -> OcrWorker:
    global _default_worker
    with _default_lock:
        if _default_worker is None:
            _default_worker = OcrWorker()
            atexit.register(_default_worker.close)
        return _default_worker


def ocr_lines(png_path: str) -> list[str]:
    return _get_default_worker().ocr(png_path)


def _child_loop() -> None:
    def watch_parent():
        parent_pid = os.getppid()
        while True:
            time.sleep(1)
            if os.getppid() != parent_pid:
                os._exit(1)

    threading.Thread(target=watch_parent, daemon=True).start()
    with os.fdopen(os.dup(1), "w", encoding="utf-8") as protocol:
        os.dup2(2, 1)
        from ocrmac import ocrmac  # noqa: F401; import must succeed before ready

        def write(message):
            print(json.dumps(message), file=protocol, flush=True)

        write({"ready": True})
        for raw in sys.stdin:
            request = json.loads(raw)
            try:
                lines = vision_ocr_lines(request["path"])
            except Exception as exc:
                write({"id": request["id"], "error": f"{type(exc).__name__}: {exc}"})
            else:
                write({"id": request["id"], "lines": lines})


if __name__ == "__main__":
    _child_loop()
