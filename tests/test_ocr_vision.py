#!/usr/bin/env python3
"""Offline regressions for bounded Apple Vision OCR and extract progress."""
from __future__ import annotations

import functools
import json
import os
import select
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TIMEOUT = 0.3
START_TIMEOUT = 5.0

FAKE = r'''
import json, os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print(json.dumps({"ready": True}), flush=True)
for raw in sys.stdin:
    request = json.loads(raw)
    path = request["path"]
    if path.startswith(("hang-once:", "crash-once:")):
        marker = Path(path.split(":", 1)[1])
        if not marker.exists():
            marker.write_text(str(os.getpid()))
            if path.startswith("crash-once:"):
                sys.exit(1)
            while True:
                time.sleep(60)
        lines = ["recovered"]
    elif path == "hang":
        while True:
            time.sleep(60)
    elif path == "crash":
        sys.exit(1)
    elif path == "error":
        print(json.dumps({"id": request["id"], "error": "ValueError: bad image"}), flush=True)
        continue
    else:
        lines = [path.removeprefix("ok:")]
    print(json.dumps({"id": request["id"], "lines": lines}), flush=True)
'''

MOCK_VISION = r'''
import os, runpy, sys, types
class OCR:
    def __init__(self, path, recognition_level):
        assert recognition_level == "accurate"
        self.path = path
    def recognize(self):
        os.write(1, b"native Vision noise\n")
        print("Python ocrmac noise", flush=True)
        if self.path == "error":
            raise ValueError("bad image")
        return [("bottom", .9, (0, .1, 1, .1)),
                ("  ", .9, (0, .7, 1, .1)),
                ("top", .9, (0, .9, 1, .1))]
module = types.ModuleType("ocrmac")
def attribute(name):
    if name == "ocrmac":
        os.write(1, b"ocrmac import noise\n")
        return types.SimpleNamespace(OCR=OCR)
    raise AttributeError(name)
module.__getattr__ = attribute
sys.modules["ocrmac"] = module
runpy.run_module("talk_to_slides.ocr_vision", run_name="__main__")
'''


def _bounded(test):
    @functools.wraps(test)
    def run():
        def expired(signum, frame):
            raise AssertionError("test exceeded its 15s hard limit")

        previous = signal.signal(signal.SIGALRM, expired)
        signal.alarm(15)
        try:
            return test()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
    return run


@contextmanager
def _fake(*, timeout=TIMEOUT, first_timeout=None, argv=None, canary_path=None,
          start_timeout=START_TIMEOUT):
    from talk_to_slides import ocr_vision as vision_ocr

    children, logs = [], []
    spawn = subprocess.Popen

    def tracked(*args, **kwargs):
        child = spawn(*args, **kwargs)
        children.append(child)
        return child

    worker = vision_ocr.OcrWorker(
        argv=argv if argv is not None else [sys.executable, "-c", FAKE],
        timeout=timeout, log=logs.append, canary_path=canary_path,
        first_timeout=timeout if first_timeout is None else first_timeout,
        start_timeout=start_timeout,
    )
    try:
        with patch.object(vision_ocr.subprocess, "Popen", tracked):
            yield worker, children, logs
    finally:
        # Also reap children when a timeout/cleanup mutation breaks the worker.
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=2)
        worker.close()


def _assert_dead(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    raise AssertionError(f"child {pid} is still alive")


def _assert_error(call, expected):
    try:
        call()
    except RuntimeError as exc:
        assert str(exc) == expected, str(exc)
    else:
        raise AssertionError(f"did not raise {expected!r}")


@_bounded
def test_persistent_child_serves_two_frames():
    """source: 2026-10-15 Vision OCR hang in webinar-extract; reuse one healthy child."""
    with _fake() as (worker, children, logs):
        assert children == [], "child must start lazily"
        assert worker.ocr("ok:first") == ["first"]
        pid = children[0].pid
        assert worker.ocr("ok:second") == ["second"]
        assert len(children) == 1 and children[0].pid == pid
        assert children[0].poll() is None
        assert worker.skipped == 0 and logs == []


@_bounded
def test_hang_once_retries_and_reaps_first_child():
    """source: 2026-10-15 Vision OCR hang in webinar-extract; a stuck frame can recover."""
    with TemporaryDirectory() as work, _fake() as (worker, children, logs):
        marker = Path(work) / "once"
        started = time.monotonic()
        assert worker.ocr(f"hang-once:{marker}") == ["recovered"]
        assert time.monotonic() - started < 2 * TIMEOUT + 1
        assert len(children) == 2
        assert int(marker.read_text()) == children[0].pid
        assert children[0].returncode == -signal.SIGKILL
        _assert_dead(children[0].pid)
        assert worker.skipped == 0 and logs == []


@_bounded
def test_hang_skips_then_next_frame_uses_new_child():
    """source: 2026-10-15 Vision OCR hang in webinar-extract; skip after two deadlines."""
    with _fake() as (worker, children, logs):
        started = time.monotonic()
        assert worker.ocr("hang") == []
        assert 2 * TIMEOUT <= time.monotonic() - started < 2 * TIMEOUT + 1
        assert worker.skipped == 1
        reason = f"timed out after {TIMEOUT:g}s"
        assert logs == [f"ocr: skipped hang: {reason}, then {reason}"], logs
        assert len(children) == 2
        for child in children:
            assert child.returncode == -signal.SIGKILL
            _assert_dead(child.pid)
        assert worker.ocr("ok:next") == ["next"]
        assert len(children) == 3 and children[-1].poll() is None


@_bounded
def test_crash_recovers_once_and_repeated_crashes_skip():
    """source: 2026-10-15 Vision OCR hang in webinar-extract; replace dead Vision too."""
    with TemporaryDirectory() as work, _fake() as (worker, children, logs):
        assert worker.ocr(f"crash-once:{work}/once") == ["recovered"]
        assert len(children) == 2 and children[0].returncode == 1
        _assert_dead(children[0].pid)
        assert worker.ocr("crash") == []
        assert logs == ["ocr: skipped crash: worker exited with 1, then worker exited with 1"]
        assert worker.skipped == 1


@_bounded
def test_vision_error_propagates_without_retry():
    """source: 2026-10-15 Vision OCR hang in webinar-extract; preserve Vision exceptions."""
    with _fake() as (worker, children, logs):
        _assert_error(lambda: worker.ocr("error"), "ocr failed for error: ValueError: bad image")
        assert len(children) == 1 and worker.skipped == 0 and logs == []


@_bounded
def test_start_failure_and_missing_ready_fail_loudly():
    """source: 2026-10-15 Vision OCR hang in webinar-extract; startup cannot silently skip."""
    cases = [
        ("import sys; print('first', file=sys.stderr); print('import failed', file=sys.stderr); sys.exit(1)",
         "import failed", START_TIMEOUT),
        ("import sys; sys.exit(1)", "worker exited with 1", START_TIMEOUT),
        ("import time; time.sleep(60)", f"timed out after {TIMEOUT:g}s", TIMEOUT),
    ]
    for script, reason, start_timeout in cases:
        with _fake(argv=[sys.executable, "-c", script], start_timeout=start_timeout) as (worker, children, logs):
            _assert_error(lambda: worker.ocr("ok:hello"), f"ocr worker failed to start: {reason}")
            assert len(children) == 1 and worker.skipped == 0 and logs == []
            _assert_dead(children[0].pid)


@_bounded
def test_three_consecutive_skips_abort():
    """source: 2026-10-15 Vision OCR hang in webinar-extract; stop when Vision looks wedged."""
    with _fake() as (worker, children, logs):
        assert worker.ocr("hang") == []
        assert worker.ocr("hang") == []
        _assert_error(lambda: worker.ocr("hang"),
                      "ocr: 3 frames in a row failed; Vision looks wedged")
        reason = f"timed out after {TIMEOUT:g}s"
        assert logs == [f"ocr: skipped hang: {reason}, then {reason}"] * 3
        assert worker.skipped == 3 and len(children) == 6


@_bounded
def test_success_resets_consecutive_skip_count():
    """source: 2026-10-15 Vision OCR hang in webinar-extract; abort only consecutive skips."""
    with _fake() as (worker, children, logs):
        assert worker.ocr("crash") == []
        assert worker.ocr("crash") == []
        assert worker.ocr("ok:reset") == ["reset"]
        assert worker.ocr("crash") == []
        assert worker.skipped == 3 and len(logs) == 3


@_bounded
def test_close_reaps_child_and_is_idempotent():
    """source: 2026-10-15 Vision OCR hang in webinar-extract; close must reap Vision."""
    with _fake() as (worker, children, logs):
        assert worker.ocr("ok:hello") == ["hello"]
        worker.close()
        assert children[0].returncode == -signal.SIGKILL
        _assert_dead(children[0].pid)
        worker.close()


@_bounded
def test_stop_abandons_unreaped_child_and_next_frame_succeeds():
    """source: push review of b27e6df; waiting after SIGKILL must remain bounded."""
    from talk_to_slides import ocr_vision as vision_ocr

    class Pipe:
        closed = False

        def close(self):
            self.closed = True

    class UnreapedChild:
        pid = 123456

        def __init__(self):
            self.stdin, self.stdout, self.stderr = Pipe(), Pipe(), Pipe()
            self.wait_timeouts = []

        def poll(self):
            return None

        def kill(self):
            pass

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            raise subprocess.TimeoutExpired("stand-in", timeout)

    logs = []
    worker = vision_ocr.OcrWorker(
        argv=[sys.executable, "-c", FAKE], timeout=TIMEOUT,
        log=logs.append, reap_timeout=0.2,
    )
    child = UnreapedChild()
    worker._process = child
    try:
        with patch.object(vision_ocr.select, "select",
                          side_effect=AssertionError("must not drain an abandoned child")):
            started = time.monotonic()
            worker._stop()
            assert time.monotonic() - started < 2
        assert child.wait_timeouts == [0.2], child.wait_timeouts
        assert logs == [f"ocr: child {child.pid} did not exit 0.2s after SIGKILL; abandoned it"], logs
        assert all(pipe.closed for pipe in (child.stdin, child.stdout, child.stderr))
        assert worker._process is None
        assert worker.ocr("ok:next") == ["next"]
        assert worker.skipped == 0
        pid = worker._process.pid
        worker.close()
        _assert_dead(pid)
    finally:
        if worker._process is not child:
            worker.close()


@_bounded
def test_bad_reply_retries_on_fresh_child():
    """source: push review of b27e6df; malformed JSON must retry on a fresh child."""
    fake = FAKE.replace('    path = request["path"]',
                        '    marker = Path(sys.argv[1])\n'
                        '    if not marker.exists():\n'
                        '        marker.write_text(str(os.getpid()))\n'
                        '        print("not json", flush=True)\n'
                        '        continue\n'
                        '    path = request["path"]')
    with TemporaryDirectory() as work:
        marker = Path(work) / "once"
        with _fake(argv=[sys.executable, "-c", fake, str(marker)]) as (worker, children, logs):
            assert worker.ocr("ok:recovered") == ["recovered"]
            assert len(children) == 2
            assert int(marker.read_text()) == children[0].pid
            assert children[0].returncode == -signal.SIGKILL
            _assert_dead(children[0].pid)
            assert children[1].poll() is None
            assert worker.skipped == 0 and logs == []


@_bounded
def test_repeated_bad_replies_skip_and_reap_children():
    """source: push review of b27e6df; malformed or non-object JSON must count as failed attempts."""
    for reply in ("not json", "null", "[]", '"text"', "123", "true"):
        fake = FAKE.replace('print(json.dumps({"id": request["id"], "lines": lines}), flush=True)',
                            f'print({reply!r}, flush=True)')
        with _fake(argv=[sys.executable, "-c", fake]) as (worker, children, logs):
            assert worker.ocr("ok:broken.png") == [], reply
            assert worker.skipped == 1
            assert logs == ["ocr: skipped ok:broken.png: bad reply line, then bad reply line"], logs
            assert len(children) == 2 and worker._process is None
            for child in children:
                assert child.returncode == -signal.SIGKILL
                _assert_dead(child.pid)


@_bounded
def test_bad_first_line_fails_start_loudly():
    """source: push review of b27e6df; malformed first JSON line must fail startup with its reason."""
    script = "import time; print('not json', flush=True); time.sleep(60)"
    with _fake(argv=[sys.executable, "-c", script]) as (worker, children, logs):
        _assert_error(lambda: worker.ocr("ok:hello"), "ocr worker failed to start: bad reply line")
        assert len(children) == 1 and worker._process is None
        assert worker.skipped == 0 and logs == []
        _assert_dead(children[0].pid)


@_bounded
def test_webinar_routes_through_default_child():
    """source: 2026-10-15 Vision OCR hang in webinar-extract; bound both caller entry points."""
    from talk_to_slides import ocr_vision as vision_ocr, extract as webinar_extract

    # Fail safely if a regression calls Vision directly in this process.
    with _fake() as (worker, children, logs), \
            patch.object(vision_ocr, "_default_worker", worker), \
            patch.dict(sys.modules, {"ocrmac": None}):
        assert webinar_extract.ocr_lines("ok:hello") == ["hello"]
        assert len(children) == 1


@_bounded
def test_main_flushes_first_progress_line_while_running():
    """source: 2026-10-15 Vision OCR hang in webinar-extract; killed runs lost buffered logs."""
    script = """
import sys, time
from unittest.mock import patch
from talk_to_slides import extract as webinar_extract
with patch.object(webinar_extract, '_probe_dims', side_effect=lambda path: time.sleep(30)):
    webinar_extract.main()
"""
    with TemporaryDirectory() as work:
        out = str(Path(work).resolve() / "out")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
        env["TMPDIR"] = work
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(Path(work) / "missing.mp4"),
             "--out", out, "--no-llm"],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        )
        try:
            ready, _, _ = select.select([child.stdout], [], [], 10)
            assert ready, "first progress line was buffered for 10s"
            assert child.stdout.readline().decode().rstrip() == f"talk-to-slides -> {out}"
            assert child.poll() is None, "extract must still be blocked in _probe_dims"
        finally:
            child.kill()
            child.communicate(timeout=2)


@_bounded
def test_child_protocol_isolates_noise_sorts_lines_and_exits_at_eof():
    """source: 2026-10-15 Vision OCR hang in webinar-extract; protect the child JSON protocol."""
    requests = [{"id": 1, "path": "ok"}, {"id": 2, "path": "error"}]
    child = subprocess.Popen([sys.executable, "-c", MOCK_VISION], cwd=ROOT,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = child.communicate(
            "".join(json.dumps(request) + "\n" for request in requests).encode(), timeout=3,
        )
        assert child.returncode == 0, stderr.decode()
        assert [json.loads(line) for line in stdout.splitlines()] == [
            {"ready": True}, {"id": 1, "lines": ["top", "bottom"]},
            {"id": 2, "error": "ValueError: bad image"},
        ], stdout
        assert b"native Vision noise" in stderr and b"Python ocrmac noise" in stderr
        assert b"ocrmac import noise" in stderr
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=2)


@_bounded
def test_extract_counts_only_current_pass_skips():
    """source: 2026-10-15 Vision OCR hang in webinar-extract; persist losses from both passes."""
    from talk_to_slides import ocr_vision as vision_ocr, extract as webinar_extract

    with TemporaryDirectory() as work, _fake() as (worker, children, logs), \
            patch.object(vision_ocr, "_default_worker", worker), \
            patch.object(webinar_extract, "_probe_dims", return_value=(900, 160, 10.0)), \
            patch.object(webinar_extract, "_extract_crop_frames", side_effect=[(["hang"], 0), (["hang"], 0)]), \
            patch.object(webinar_extract, "dedup_changed", side_effect=lambda frames, **kwargs: frames):
        worker.skipped = 7
        webinar_extract.run_extract("missing.mp4", work, chat_fps=1, share_fps=1,
                                    chat_crop="", share_crop="", use_llm=False, model="",
                                    meta={}, log=logs.append)
        saved = json.loads((Path(work) / "data.json").read_text())
        assert saved["meta"]["ocr_skipped"] == 2, saved["meta"]
        assert logs.count("  ocr skipped 2 frames") == 1, logs


@_bounded
def test_extract_records_zero_skips_without_starting_vision():
    """source: 2026-10-15 Vision OCR hang in webinar-extract; distinguish no lost frames."""
    from talk_to_slides import ocr_vision as vision_ocr, extract as webinar_extract

    with TemporaryDirectory() as work, _fake() as (worker, children, logs), \
            patch.object(vision_ocr, "_default_worker", worker), \
            patch.object(webinar_extract, "_probe_dims", return_value=(900, 160, 10.0)), \
            patch.object(webinar_extract, "_extract_crop_frames", return_value=([], 0)):
        worker.skipped = 7
        webinar_extract.run_extract("missing.mp4", work, chat_fps=1, share_fps=1,
                                    chat_crop="", share_crop="", use_llm=False, model="",
                                    meta={}, log=logs.append)
        saved = json.loads((Path(work) / "data.json").read_text())
        assert saved["meta"]["ocr_skipped"] == 0
        assert not any(line.startswith("  ocr skipped") for line in logs)
        assert children == []


@_bounded
def test_concurrent_requests_and_mismatched_ids():
    """source: 2026-10-15 Vision OCR hang in webinar-extract; serialize and match replies."""
    fake = FAKE.replace('    path = request["path"]',
                        '    print(json.dumps({"id": -1, "lines": ["wrong"]}), flush=True)\n'
                        '    path = request["path"]')
    with ThreadPoolExecutor(max_workers=4) as pool:
        with _fake(argv=[sys.executable, "-c", fake]) as (worker, children, logs):
            results = list(pool.map(worker.ocr, [f"ok:{i}" for i in range(12)]))
            assert results == [[str(i)] for i in range(12)], results
            assert len(children) == 1 and worker.skipped == 0


@_bounded
def test_default_timeout_command_and_atexit_cleanup():
    """source: 2026-10-15 Vision OCR hang in webinar-extract; defaults and exit must be safe."""
    from talk_to_slides import ocr_vision as vision_ocr

    with patch.dict(os.environ, {"TALK_TO_SLIDES_OCR_TIMEOUT": "0.75"}):
        worker = vision_ocr.OcrWorker()
        try:
            assert worker.timeout == 0.75
            assert worker.argv == [sys.executable, "-m", "talk_to_slides.ocr_vision"]
        finally:
            worker.close()
    with patch.dict(os.environ, {}, clear=True):
        worker = vision_ocr.OcrWorker()
        try:
            assert worker.timeout == 60.0
        finally:
            worker.close()
    with TemporaryDirectory() as work:
        marker = Path(work) / "pid"
        fake = FAKE.replace('print(json.dumps({"ready": True}), flush=True)',
                            f'Path({str(marker)!r}).write_text(str(os.getpid()))\n'
                            'print(json.dumps({"ready": True}), flush=True)')
        script = (
            "import functools, sys\nfrom talk_to_slides import ocr_vision as vision_ocr\n"
            f"vision_ocr.OcrWorker = functools.partial(vision_ocr.OcrWorker, "
            f"argv=[sys.executable, '-c', {fake!r}], timeout={TIMEOUT})\n"
            "assert vision_ocr.ocr_lines('ok:hello') == ['hello']\n"
        )
        child = subprocess.Popen([sys.executable, "-c", script], cwd=ROOT,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            stdout, stderr = child.communicate(timeout=3)
            assert child.returncode == 0, stderr.decode()
            _assert_dead(int(marker.read_text()))
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=2)
            if marker.exists():
                try:
                    os.kill(int(marker.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


@_bounded
def test_wedged_cooldown_blocks_attempts_then_rearms_and_recovers():
    """source: 10-15 pp2p review; wedged Vision must stop consuming per-frame deadlines."""
    from talk_to_slides import ocr_vision as vision_ocr

    constructor = functools.partial(vision_ocr.OcrWorker, wedged_cooldown=0.5, reap_timeout=0.1)
    expected = "ocr: 3 frames in a row failed; Vision looks wedged"
    with patch.object(vision_ocr, "OcrWorker", constructor), \
         _fake() as (worker, children, logs):
        assert worker.ocr("hang") == []
        assert worker.ocr("hang") == []
        _assert_error(lambda: worker.ocr("hang"), expected)
        assert worker.skipped == 3 and len(children) == 6 and len(logs) == 3
        with patch.object(worker, "_send", wraps=worker._send) as send_request:
            started = time.monotonic()
            _assert_error(lambda: worker.ocr("hang"), expected)
            assert time.monotonic() - started < worker.timeout / 3
            send_request.assert_not_called()
        assert worker.skipped == 3 and len(children) == 6 and len(logs) == 3

        time.sleep(0.51)
        _assert_error(lambda: worker.ocr("hang"), expected)
        assert worker.skipped == 4 and len(children) == 8 and len(logs) == 4
        _assert_error(lambda: worker.ocr("ok:too-soon"), expected)
        assert worker.skipped == 4 and len(children) == 8 and len(logs) == 4
        time.sleep(0.51)
        assert worker.ocr("ok:recovered") == ["recovered"]
        assert worker._consecutive_skips == 0 and len(children) == 9


@_bounded
def test_malformed_lines_retry_fresh_child_then_skip():
    """source: 10-15 pp2p review; missing or non-string OCR lines must be retried safely."""
    for fields in ({}, {"lines": None}, {"lines": "abc"}, {"lines": [1, 2]}):
        fake = (
            "import json, sys\n"
            "print(json.dumps({'ready': True}), flush=True)\n"
            "for raw in sys.stdin:\n"
            "    request = json.loads(raw)\n"
            f"    reply = dict({fields!r}, id=request['id'])\n"
            "    print(json.dumps(reply), flush=True)\n"
        )
        with _fake(argv=[sys.executable, "-c", fake]) as (worker, children, logs):
            worker.reap_timeout = 0.1
            assert worker.ocr("frame.png") == [], fields
            assert worker.skipped == 1 and len(children) == 2, fields
            assert children[0].pid != children[1].pid
            assert logs == ["ocr: skipped frame.png: bad reply line, then bad reply line"], logs
            for child in children:
                _assert_dead(child.pid)


@_bounded
def test_child_watchdog_exits_during_vision_after_parent_sigkill():
    """source: 10-15 pp2p review; parent death must kill a child blocked inside Vision."""
    child_code = (
        "import os, sys, time, types\n"
        "sys.modules['ocrmac'] = types.SimpleNamespace(ocrmac=object())\n"
        "import talk_to_slides.ocr_vision as v\n"
        "def recognize(path):\n"
        "    os.write(2, b'recognizing\\n')\n"
        "    time.sleep(600)\n"
        "    return []\n"
        "v.vision_ocr_lines = recognize\n"
        "v._child_loop()\n"
    )
    with TemporaryDirectory() as work:
        marker = Path(work) / "pid"
        parent_code = (
            "import json, subprocess, sys, time\nfrom pathlib import Path\n"
            f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}], "
            "stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n"
            f"Path({str(marker)!r}).write_text(str(child.pid))\n"
            "assert json.loads(child.stdout.readline()) == {'ready': True}\n"
            "child.stdin.write(b'{\"id\": 1, \"path\": \"frame.png\"}\\n')\n"
            "child.stdin.flush()\n"
            "assert child.stderr.readline() == b'recognizing\\n'\n"
            "print(child.pid, flush=True)\n"
            "time.sleep(600)\n"
        )
        deadline = time.monotonic() + 4.5
        parent = subprocess.Popen([sys.executable, "-c", parent_code], cwd=ROOT,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            ready, _, _ = select.select([parent.stdout], [], [], 2)
            assert ready, "intermediate parent did not report a recognizing child"
            raw = parent.stdout.readline()
            assert raw.strip().isdigit(), raw
            pid = int(raw)
            assert pid == int(marker.read_text())
            parent.kill()
            parent.wait(timeout=0.2)
            assert parent.returncode == -signal.SIGKILL
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.02)
            _assert_dead(pid)
        finally:
            if parent.poll() is None:
                parent.kill()
            parent.communicate(timeout=0.2)
            if marker.exists():
                try:
                    os.kill(int(marker.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


@_bounded
def test_crash_skip_retains_last_stderr_line():
    """source: 10-15 pp2p review; retrying a crashed child must preserve its traceback."""
    fake = (
        "import sys\n"
        "print('{\"ready\": true}', flush=True)\n"
        "sys.stdin.readline()\n"
        "sys.stderr.write('Traceback (most recent call last):\\nMemoryError: boom\\n\\n')\n"
        "sys.stderr.flush()\n"
        "sys.exit(1)\n"
    )
    with _fake(argv=[sys.executable, "-c", fake]) as (worker, children, logs):
        assert worker.ocr("frame.png") == []
        reason = "worker exited with 1: MemoryError: boom"
        assert logs == [f"ocr: skipped frame.png: {reason}, then {reason}"], logs
        assert len(children) == 2 and worker.skipped == 1
        for child in children:
            assert child.returncode == 1
            _assert_dead(child.pid)


@_bounded
def test_closed_stdout_without_exit_has_explicit_reason():
    """source: 10-15 pp2p review; stdout EOF from a live child must not report None."""
    fake = (
        "import os, sys, time\n"
        "print('{\"ready\": true}', flush=True)\n"
        "sys.stdin.readline()\n"
        "os.close(1)\n"
        "time.sleep(600)\n"
    )
    with _fake(argv=[sys.executable, "-c", fake]) as (worker, children, logs):
        worker.reap_timeout = 0.1
        assert worker.ocr("frame.png") == []
        reason = "worker closed its output but did not exit"
        assert logs == [f"ocr: skipped frame.png: {reason}, then {reason}"], logs
        assert "None" not in logs[0]
        assert len(children) == 2 and worker.skipped == 1
        for child in children:
            _assert_dead(child.pid)


@_bounded
def test_empty_reply_requires_passing_canary():
    """source: 2026-10-15 5V8tP5jDZ8U lost 88 visual frames silently; verify empty OCR."""
    with _fake(canary_path="ok:CANARY 4817") as (worker, children, logs):
        assert worker.verified_empty == 0
        assert worker._canary_ok_at == float("-inf")
        with patch.object(worker, "_send", wraps=worker._send) as requests:
            assert worker.ocr("ok:") == [""]
        assert [call.args[0] for call in requests.call_args_list] == ["ok:", "ok:CANARY 4817"]
        assert len({call.args[1] for call in requests.call_args_list}) == 2
        assert worker.verified_empty == 1 and worker.skipped == 0 and logs == []
        assert len(children) == 1


@_bounded
def test_empty_reply_reuses_canary_until_ttl_expires():
    """source: canary interface caches success for 30 seconds on the same child."""
    from talk_to_slides import ocr_vision as vision_ocr

    assert vision_ocr.CANARY_TTL == 30.0
    with _fake(canary_path="ok:CANARY 4817") as (worker, children, logs), \
            patch.object(worker, "_send", wraps=worker._send) as requests:
        assert worker.canary_ttl == vision_ocr.CANARY_TTL
        assert worker.ocr("ok:") == [""]
        assert worker.ocr("ok:") == [""]
        assert [call.args[0] for call in requests.call_args_list] == [
            "ok:", "ok:CANARY 4817", "ok:",
        ]
        worker._canary_ok_at = time.monotonic() - worker.canary_ttl - 1
        assert worker.ocr("ok:") == [""]
        assert [call.args[0] for call in requests.call_args_list] == [
            "ok:", "ok:CANARY 4817", "ok:", "ok:", "ok:CANARY 4817",
        ]
        assert worker.verified_empty == 3 and worker.skipped == 0 and logs == []
        assert len(children) == 1


@_bounded
def test_failing_canary_retries_fresh_child_then_counts_skip():
    """source: 2026-10-15 silent Vision failure must be retried and counted as OCR loss."""
    with _fake(canary_path="ok:") as (worker, children, logs), \
            patch.object(worker, "_send", wraps=worker._send) as requests:
        assert worker.ocr("ok:") == []
        assert len(requests.call_args_list) == 4
        assert len(children) == 2 and children[0].pid != children[1].pid
        for child in children:
            assert child.returncode == -signal.SIGKILL
            _assert_dead(child.pid)
        assert worker.skipped == 1 and worker.verified_empty == 0
        assert worker._canary_ok_at == float("-inf")
        reason = "no text and the canary failed"
        assert logs == [f"ocr: skipped ok:: {reason}, then {reason}"], logs


@_bounded
def test_three_failing_canary_frames_abort_as_wedged():
    """source: canary failures use the existing three-consecutive-skips abort."""
    with _fake(canary_path="ok:") as (worker, children, logs):
        assert worker.ocr("ok:") == []
        assert worker.ocr("ok:") == []
        _assert_error(lambda: worker.ocr("ok:"), "ocr: 3 frames in a row failed; Vision looks wedged")
        assert worker.skipped == 3 and worker.verified_empty == 0
        assert len(children) == 6
        reason = "no text and the canary failed"
        assert logs == [f"ocr: skipped ok:: {reason}, then {reason}"] * 3, logs


@_bounded
def test_nonempty_reply_never_sends_canary():
    """source: canary interface leaves non-empty OCR replies unchanged."""
    with _fake(canary_path="hang") as (worker, children, logs), \
            patch.object(worker, "_send", wraps=worker._send) as requests:
        assert worker.ocr("ok:readable") == ["readable"]
        assert [call.args[0] for call in requests.call_args_list] == ["ok:readable"]
        assert worker.verified_empty == 0 and worker.skipped == 0 and logs == []


@_bounded
def test_replacement_child_must_pass_canary_again():
    """source: canary success belongs to one child and cannot survive its replacement."""
    with TemporaryDirectory() as work, \
            _fake(canary_path="ok:CANARY 4817") as (worker, children, logs), \
            patch.object(worker, "_send", wraps=worker._send) as requests:
        assert worker.ocr("ok:") == [""]
        crash = f"crash-once:{work}/once"
        assert worker.ocr(crash) == ["recovered"]
        assert worker._canary_ok_at == float("-inf")
        assert worker.ocr("ok:") == [""]
        assert [call.args[0] for call in requests.call_args_list] == [
            "ok:", "ok:CANARY 4817", crash, crash, "ok:", "ok:CANARY 4817",
        ]
        assert len(children) == 2 and worker.verified_empty == 2
        assert worker.skipped == 0 and logs == []


@_bounded
def test_check_canary_returns_false_for_error_and_hang():
    """source: canary errors and timeouts mean failed health checks, not Vision exceptions."""
    for path in ("error", "hang"):
        with _fake(canary_path=path) as (worker, children, logs):
            assert worker.check_canary() is False
            assert len(children) == 1
            assert worker.skipped == 0 and worker.verified_empty == 0 and logs == []
            assert worker._canary_ok_at == float("-inf")
            if path == "hang":
                assert worker._process is None
                _assert_dead(children[0].pid)


@_bounded
def test_check_canary_matches_request_id_and_requires_string_lines():
    """source: canary replies require their own id and a string list containing token 4817."""
    cases = [
        ({"lines": ["CANARY", "4817"]}, True),
        ({"lines": []}, False),
        ({"lines": ["CANARY"]}, False),
        ({"lines": "4817"}, False),
        ({"lines": [4817]}, False),
        ({"lines": ["4817", None]}, False),
        ({}, False),
        ({"error": "bad image", "lines": ["4817"]}, False),
    ]
    for fields, expected in cases:
        fake = FAKE.replace(
            '    print(json.dumps({"id": request["id"], "lines": lines}), flush=True)',
            f'    print(json.dumps({{"id": -1, "lines": {[] if expected else ["4817"]!r}}}), flush=True)\n'
            f'    print(json.dumps(dict({fields!r}, id=request["id"])), flush=True)',
        )
        with _fake(argv=[sys.executable, "-c", fake], canary_path="ok:CANARY 4817") as (worker, children, logs):
            before = time.monotonic()
            assert worker.check_canary() is expected, fields
            if expected:
                assert before <= worker._canary_ok_at <= time.monotonic()
            else:
                assert worker._canary_ok_at == float("-inf")
            assert worker.skipped == 0 and worker.verified_empty == 0 and logs == []


@_bounded
def test_check_canary_stops_child_after_crash_or_bad_reply():
    """source: a failed canary protocol attempt stops the child and returns False."""
    bad = FAKE.replace('print(json.dumps({"id": request["id"], "lines": lines}), flush=True)',
                       'print("not json", flush=True)')
    for path, fake in (("crash", FAKE), ("ok:CANARY 4817", bad)):
        with _fake(argv=[sys.executable, "-c", fake], canary_path=path) as (worker, children, logs):
            assert worker.check_canary() is False
            assert len(children) == 1 and worker._process is None
            _assert_dead(children[0].pid)


@_bounded
def test_check_canary_preserves_start_failure():
    """source: child startup RuntimeError must still propagate from check_canary."""
    with _fake(argv=[sys.executable, "-c", "import sys; sys.exit(1)"],
               canary_path="ok:CANARY 4817") as (worker, children, logs):
        _assert_error(worker.check_canary, "ocr worker failed to start: worker exited with 1")
        assert len(children) == 1 and worker._process is None


@_bounded
def test_empty_list_and_whitespace_replies_are_verified():
    """source: every reply without a non-blank string needs a passing canary."""
    for lines in ([], ["", " \t"]):
        fake = FAKE.replace('        lines = [path.removeprefix("ok:")]',
                            f'        lines = {lines!r} if path == "frame" else [path.removeprefix("ok:")]')
        with _fake(argv=[sys.executable, "-c", fake], canary_path="ok:CANARY 4817") as (worker, children, logs):
            assert worker.ocr("frame") == lines
            assert worker.verified_empty == 1 and worker.skipped == 0 and logs == []


@_bounded
def test_make_canary_png_writes_visible_rgb_text():
    """source: canary interface requires a 1200x300 white RGB PNG with black CANARY 4817."""
    from PIL import Image
    from talk_to_slides import ocr_vision as vision_ocr

    assert vision_ocr.CANARY_TEXT == "CANARY 4817"
    assert vision_ocr.CANARY_TOKEN == "4817"
    with TemporaryDirectory() as work:
        path = str(Path(work) / "canary.png")
        assert vision_ocr.make_canary_png(path) == path
        with Image.open(path) as image:
            assert image.format == "PNG" and image.size == (1200, 300) and image.mode == "RGB"
            assert image.getpixel((0, 0)) == (255, 255, 255)
            pixels = image.load()
            assert any(pixels[x, y] == (0, 0, 0) for y in range(80, 250) for x in range(60, 1100))


@_bounded
def test_owned_canary_is_lazy_and_removed_on_close():
    """source: worker-generated canary directories are lazy, reusable, and owned until close."""
    from talk_to_slides import ocr_vision as vision_ocr

    with TemporaryDirectory() as work, _fake() as (worker, children, logs), \
            patch.object(vision_ocr, "make_canary_png", wraps=vision_ocr.make_canary_png) as make_png, \
            patch.object(vision_ocr.tempfile, "tempdir", work):
        assert worker.ocr("ok:readable") == ["readable"]
        make_png.assert_not_called()
        worker.check_canary()
        make_png.assert_called_once()
        path = Path(make_png.call_args.args[0])
        assert path.is_file() and path.parent.name.startswith("vc-ocr-canary-")
        worker.check_canary()
        make_png.assert_called_once()
        worker.close()
        assert not path.parent.exists()
        assert worker._canary_ok_at == float("-inf")
        worker.close()


@_bounded
def test_supplied_canary_is_never_created_or_removed():
    """source: callers retain ownership of canary_path, including when that path is missing."""
    from talk_to_slides import ocr_vision as vision_ocr

    with TemporaryDirectory() as work, \
            patch.object(vision_ocr, "make_canary_png", side_effect=AssertionError("caller owns canary")):
        for exists in (False, True):
            path = Path(work) / f"canary-{exists}.png"
            if exists:
                path.write_bytes(b"caller-owned canary")
            with _fake(canary_path=str(path)) as (worker, children, logs):
                assert worker.check_canary() is False
                worker.close()
            assert path.exists() is exists
            if exists:
                assert path.read_bytes() == b"caller-owned canary"


@_bounded
def test_extract_records_current_pass_frames_and_verified_empty():
    """source: extract meta records share OCR frames and verified empties from share and chat."""
    from talk_to_slides import ocr_vision as vision_ocr, extract as webinar_extract

    with TemporaryDirectory() as work, \
            _fake(canary_path="ok:CANARY 4817") as (worker, children, logs), \
            patch.object(vision_ocr, "_default_worker", worker), \
            patch.object(webinar_extract, "_probe_dims", return_value=(900, 160, 10.0)), \
            patch.object(webinar_extract, "_extract_crop_frames", side_effect=[(["ok:"], 0), (["ok:", "crash"], 0)]), \
            patch.object(webinar_extract, "dedup_changed", side_effect=lambda frames, **kwargs: frames):
        worker.verified_empty = 7
        worker.skipped = 4
        webinar_extract.run_extract("missing.mp4", work, chat_fps=1, share_fps=1,
                                    chat_crop="", share_crop="", use_llm=False, model="",
                                    meta={}, log=logs.append)
        saved = json.loads((Path(work) / "data.json").read_text())
        assert saved["meta"]["ocr_frames"] == 2, saved["meta"]
        assert saved["meta"]["ocr_empty"] == 2, saved["meta"]
        assert saved["meta"]["ocr_skipped"] == 1, saved["meta"]


@_bounded
def test_extract_records_zero_frames_and_empty_without_vision():
    """source: even zero-frame extracts record ocr_frames 0 and ocr_empty 0 for this run."""
    from talk_to_slides import ocr_vision as vision_ocr, extract as webinar_extract

    with TemporaryDirectory() as work, _fake() as (worker, children, logs), \
            patch.object(vision_ocr, "_default_worker", worker), \
            patch.object(webinar_extract, "_probe_dims", return_value=(900, 160, 10.0)), \
            patch.object(webinar_extract, "_extract_crop_frames", return_value=([], 0)), \
            patch.object(webinar_extract, "dedup_changed", side_effect=lambda frames, **kwargs: frames):
        worker.verified_empty = 7
        webinar_extract.run_extract("missing.mp4", work, chat_fps=1, share_fps=1,
                                    chat_crop="", share_crop="", use_llm=False, model="",
                                    meta={"ocr_frames": 99, "ocr_empty": 99}, log=logs.append)
        saved = json.loads((Path(work) / "data.json").read_text())
        assert saved["meta"]["ocr_frames"] == 0, saved["meta"]
        assert saved["meta"]["ocr_empty"] == 0, saved["meta"]
        assert children == []


SLOW_FAKE = FAKE.replace(
    'print(json.dumps({"ready": True}), flush=True)',
    'time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 0)\n'
    'print(json.dumps({"ready": True}), flush=True)',
).replace(
    'for raw in sys.stdin:', 'for count, raw in enumerate(sys.stdin):',
).replace(
    '    path = request["path"]',
    '    path = request["path"]\n'
    '    if path.startswith(("slow:", "slow-first:")):\n'
    '        kind, delay, path = path.split(":", 2)\n'
    '        if kind == "slow" or count == 0:\n'
    '            time.sleep(float(delay))',
)


@_bounded
def test_cold_first_frame_gets_long_deadline_without_skip():
    """source: 2026-10-15 cold Vision took 86.9s, warm 0.2s; taSmwcqdkQk timed out twice."""
    with _fake(timeout=0.3, first_timeout=1.5,
               argv=[sys.executable, "-c", SLOW_FAKE]) as (worker, children, logs):
        started = time.monotonic()
        assert worker.ocr("slow-first:0.6:ok:first") == ["first"]
        assert 0.6 <= time.monotonic() - started < 1.5
        assert worker._served is True
        assert worker.ocr("slow-first:0.6:ok:warm") == ["warm"]
        assert len(children) == 1 and children[0].poll() is None
        assert worker.skipped == 0 and logs == []


@_bounded
def test_cold_second_slow_frame_retries_with_fresh_first_deadline():
    """source: each child's first request gets the long deadline; later requests keep timeout."""
    with _fake(timeout=0.3, first_timeout=1.5,
               argv=[sys.executable, "-c", SLOW_FAKE]) as (worker, children, logs):
        assert worker.ocr("slow:0.6:ok:first") == ["first"]
        started = time.monotonic()
        assert worker.ocr("slow:0.6:ok:second") == ["second"]
        assert 0.9 <= time.monotonic() - started < 1.5
        assert len(children) == 2 and children[0].pid != children[1].pid
        assert children[0].returncode == -signal.SIGKILL
        _assert_dead(children[0].pid)
        assert children[1].poll() is None and worker._served is True
        assert worker.skipped == 0 and logs == []


@_bounded
def test_cold_hang_uses_first_deadline_on_both_attempts_then_skips():
    """source: first_timeout must bound a cold hang while preserving two attempts and skip logs."""
    with _fake(timeout=0.3, first_timeout=0.6) as (worker, children, logs):
        started = time.monotonic()
        assert worker.ocr("hang") == []
        assert 1.2 <= time.monotonic() - started < 2.2
        assert len(children) == 2 and worker._process is None
        assert worker._served is True and worker.skipped == 1
        reason = "timed out after 0.3s"
        assert logs == [f"ocr: skipped hang: {reason}, then {reason}"], logs
        for child in children:
            assert child.returncode == -signal.SIGKILL
            _assert_dead(child.pid)


@_bounded
def test_cold_delayed_ready_leaves_full_first_request_deadline():
    """source: ocrmac import delays ready; startup and the first request each get the long deadline."""
    with _fake(timeout=0.3, first_timeout=1.5,
               argv=[sys.executable, "-c", SLOW_FAKE, "0.6"]) as (worker, children, logs):
        worker._start()
        assert worker._served is False
        assert worker.ocr("slow-first:0.6:ok:first") == ["first"]
        assert worker._served is True
        assert len(children) == 1 and children[0].poll() is None
        assert worker.skipped == 0 and logs == []


@_bounded
def test_cold_canary_first_is_long_then_warm_then_reset_on_restart():
    """source: a canary consumes the child's first-request deadline and a replacement resets it."""
    with _fake(timeout=0.3, first_timeout=1.5, canary_path="slow:0.6:ok:CANARY 4817",
               argv=[sys.executable, "-c", SLOW_FAKE]) as (worker, children, logs):
        assert worker.check_canary() is True
        assert worker._served is True and len(children) == 1
        started = time.monotonic()
        assert worker.check_canary() is False
        assert 0.3 <= time.monotonic() - started < 0.6
        assert worker._served is True and worker._process is None
        _assert_dead(children[0].pid)
        assert worker.check_canary() is True
        assert len(children) == 2 and children[1].poll() is None
        assert worker.skipped == 0 and logs == []


@_bounded
def test_cold_empty_reply_consumes_first_deadline_before_canary():
    """source: _served becomes True as soon as the first reply is read, including empty OCR."""
    with _fake(timeout=0.3, first_timeout=1.5, canary_path="slow:0.6:ok:CANARY 4817",
               argv=[sys.executable, "-c", SLOW_FAKE]) as (worker, children, logs):
        assert worker.ocr("ok:") == []
        assert len(children) == 2 and worker.skipped == 1 and worker.verified_empty == 0
        assert worker._served is True and worker._process is None
        reason = "no text and the canary failed"
        assert logs == [f"ocr: skipped ok:: {reason}, then {reason}"], logs
        for child in children:
            _assert_dead(child.pid)


@_bounded
def test_cold_error_reply_consumes_first_deadline_for_frame_and_canary():
    """source: the first reply consumes the long deadline even when it reports an error."""
    for canary in (False, True):
        with _fake(timeout=0.3, first_timeout=1.5, canary_path="error",
                   argv=[sys.executable, "-c", SLOW_FAKE]) as (worker, children, logs):
            if canary:
                assert worker.check_canary() is False
            else:
                _assert_error(lambda: worker.ocr("error"), "ocr failed for error: ValueError: bad image")
            assert worker._served is True
            assert worker.ocr("slow:0.6:ok:recovered") == ["recovered"]
            assert len(children) == 2
            _assert_dead(children[0].pid)
            assert children[1].poll() is None and worker.skipped == 0 and logs == []


@_bounded
def test_cold_first_timeout_never_shortens_normal_timeout():
    """source: startup, first OCR and first canary use max(timeout, first_timeout)."""
    for canary in (False, True):
        with _fake(timeout=1.5, first_timeout=0.3, canary_path="slow:0.6:ok:CANARY 4817",
                   argv=[sys.executable, "-c", SLOW_FAKE, "0.6"]) as (worker, children, logs):
            if canary:
                assert worker.check_canary() is True
            else:
                assert worker.ocr("slow-first:0.6:ok:first") == ["first"]
            assert worker._served is True
            assert len(children) == 1 and children[0].poll() is None
            assert worker.skipped == 0 and logs == []


@_bounded
def test_cold_first_timeout_defaults_to_environment_and_allows_override():
    """source: first_timeout=None reads TALK_TO_SLIDES_OCR_FIRST_TIMEOUT; an explicit value takes precedence."""
    from talk_to_slides import ocr_vision as vision_ocr

    with patch.dict(os.environ, {"TALK_TO_SLIDES_OCR_FIRST_TIMEOUT": "1.25"}):
        for kwargs, expected in (({}, 1.25), ({"first_timeout": None}, 1.25),
                                 ({"first_timeout": 0.6}, 0.6)):
            worker = vision_ocr.OcrWorker(argv=[sys.executable, "-c", FAKE], **kwargs)
            try:
                assert worker.first_timeout == expected
                assert worker._process is None
            finally:
                worker.close()


@_bounded
def test_cold_first_timeout_defaults_to_180_without_environment():
    """source: TALK_TO_SLIDES_OCR_FIRST_TIMEOUT defaults to 180 seconds when unset."""
    from talk_to_slides import ocr_vision as vision_ocr

    with patch.dict(os.environ, {}, clear=True):
        worker = vision_ocr.OcrWorker(argv=[sys.executable, "-c", FAKE])
        try:
            assert worker.first_timeout == 180.0
            assert worker.timeout == 60.0 and worker._process is None
        finally:
            worker.close()


def _run() -> int:
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
