"""Fixes for the Vision start deadline and punctuation-only caption cues.

source: 11-03 CI run 41763995096 and review gate WARN on 419c556.
"""
import functools
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from talk_to_slides.captions import parse_vtt


SLOW_READY = r'''
import json, sys, time
time.sleep(1.0)
print(json.dumps({"ready": True}), flush=True)
for raw in sys.stdin:
    request = json.loads(raw)
    path = request["path"]
    lines = [path.removeprefix("ok:")]
    print(json.dumps({"id": request["id"], "lines": lines}), flush=True)
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


def _assert_dead(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    raise AssertionError(f"child {pid} is still alive")


@contextmanager
def _slow_ready_worker(**kwargs):
    """Reap every stand-in child, including when the worker misses one."""
    from talk_to_slides import ocr_vision as vision_ocr

    children = []
    spawn = subprocess.Popen

    def tracked(*args, **pop_kwargs):
        child = spawn(*args, **pop_kwargs)
        children.append(child)
        return child

    worker = vision_ocr.OcrWorker(
        argv=[sys.executable, "-c", SLOW_READY],
        log=lambda line: None,
        **kwargs,
    )
    try:
        with patch.object(vision_ocr.subprocess, "Popen", tracked):
            yield worker, children
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=2)
        worker.close()


def _serve(worker):
    try:
        return worker.ocr("ok:x")
    except RuntimeError as exc:
        raise AssertionError(str(exc)) from exc


@_bounded
def test_start_timeout_lets_a_slow_ready_child_serve():
    """source: 11-03 CI run 41763995096 and review gate WARN on 419c556, item 1
    A child that sleeps 1.0s before ready serves when start_timeout is 5.
    """
    with _slow_ready_worker(timeout=0.3, first_timeout=0.3, start_timeout=5) as (worker, children):
        assert _serve(worker) == ["x"]
        assert len(children) == 1 and children[0].poll() is None
        pid = children[0].pid
    _assert_dead(pid)


@_bounded
def test_omitted_start_timeout_uses_the_larger_deadline():
    """source: 11-03 CI run 41763995096 and review gate WARN on 419c556, item 1
    With no start_timeout, startup waits the larger of timeout and first_timeout.
    """
    with _slow_ready_worker(timeout=0.3, first_timeout=3) as (worker, children):
        assert worker.start_timeout is None
        assert _serve(worker) == ["x"]
        assert len(children) == 1 and children[0].poll() is None
        pid = children[0].pid
    _assert_dead(pid)


@_bounded
def test_short_start_timeout_fails_with_the_ready_message():
    """source: 11-03 CI run 41763995096 and review gate WARN on 419c556, item 1
    start_timeout=0.3 still fails a 1.0s ready with the 0.3s message.
    """
    expected = "ocr worker failed to start: timed out after 0.3s"
    with _slow_ready_worker(timeout=0.3, first_timeout=0.3, start_timeout=0.3) as (worker, children):
        try:
            worker.ocr("ok:x")
        except RuntimeError as exc:
            assert str(exc) == expected, str(exc)
        else:
            raise AssertionError(f"did not raise {expected!r}")
        assert len(children) == 1
        pid = children[0].pid
    _assert_dead(pid)


def _plain(tmp_path: Path, texts: list[str]) -> list[dict]:
    lines = ["WEBVTT", ""]
    for index, text in enumerate(texts):
        lines.extend([
            f"00:00:0{index}.000 --> 00:00:0{index + 1}.000",
            text,
            "",
        ])
    path = tmp_path / "captions.vtt"
    path.write_text("\n".join(lines) + "\n")
    return parse_vtt(path)


def test_punctuation_only_cues_stay_distinct(tmp_path):
    """source: 11-03 CI run 41763995096 and review gate WARN on 419c556, item 2
    A music note followed by an ellipsis keeps both cues.
    """
    cues = _plain(tmp_path, ["♪", "..."])
    assert [cue["text"] for cue in cues] == ["♪", "..."], cues


def test_exact_punctuation_repeat_is_dropped(tmp_path):
    """source: 11-03 CI run 41763995096 and review gate WARN on 419c556, item 2
    An exact repeat of a punctuation-only cue is dropped.
    """
    cues = _plain(tmp_path, ["♪", "♪"])
    assert [cue["text"] for cue in cues] == ["♪"], cues


def test_punctuation_only_cue_after_words_is_kept(tmp_path):
    """source: 11-03 CI run 41763995096 and review gate WARN on 419c556, item 2
    hello world. then ... keeps both cues.
    """
    cues = _plain(tmp_path, ["hello world.", "..."])
    assert [cue["text"] for cue in cues] == ["hello world.", "..."], cues


def test_longer_raw_text_still_replaces_equal_tokens(tmp_path):
    """source: 11-03 CI run 41763995096 and review gate WARN on 419c556, item 2
    hello world then hello world. still keeps the longer raw text.
    """
    cues = _plain(tmp_path, ["hello world", "hello world."])
    assert [cue["text"] for cue in cues] == ["hello world."], cues
    assert cues == [{"start": 1.0, "end": 2.0, "text": "hello world."}], cues
