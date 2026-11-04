"""Command line for talk-to-slides."""
from __future__ import annotations


def main() -> int:
    """Dispatch extract, run, captions, transcribe, and finalize."""
    import sys

    from talk_to_slides import extract

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    if len(sys.argv) > 1 and sys.argv[1] == "transcribe":
        return extract.transcribe_cmd(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "captions":
        return extract.captions_cmd(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "finalize":
        return extract.finalize_cmd(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        return extract.run_cmd(sys.argv[2:])
    return extract.extract_cmd(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
