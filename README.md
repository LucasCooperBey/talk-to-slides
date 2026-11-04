# talk-to-slides

talk-to-slides reads a local talk recording and writes one folder of slide images, the text on each slide, and a short HTML report. When a Claude model is available, the report also includes a cleaned summary, key moments, and takeaways.

## Requirements

Python 3.11 or newer, plus ffmpeg and ffprobe. Slide text uses Apple Vision on macOS, the `tesseract` binary on Linux, or `--ocr none` to keep the images without text.

## Install

```
uv tool install git+https://github.com/LucasCooperBey/talk-to-slides
```

## Quick start

```
talk-to-slides run recording.mp4 --out talk-out
```

`run` extracts the slides, reads a `source.*.vtt` file beside the video when one is there, and writes `report.html`, `data.json`, and `slides/`. Pass `--captions notes.vtt` to name a caption file. If `--out` is omitted, the folder is `./talk-to-slides-out/<video file name without its extension>`. `run` makes one model call, at the finalize step (see Model step); add `--no-llm` to skip it.

## Commands

`talk-to-slides VIDEO --out DIR` samples the recording and writes the slide folder. The default reads the whole frame and skips a chat pass. `--layout zoom-chat` uses a side-chat crop and a shared-screen crop. `--no-llm` skips the cleanup call. `--ocr` chooses `auto`, `vision`, `tesseract`, or `none`.

`talk-to-slides captions DIR` reads a `source.*.vtt` file in that folder and writes `transcript.json`.

`talk-to-slides transcribe AUDIO... --out DIR` sends audio to an OpenAI-compatible `/v1/audio/transcriptions` endpoint. Set `--whisper-url` or `WHISPER_URL`. There is no default address. When `WHISPER_API_KEY` is set, it is sent as a Bearer token.

`talk-to-slides finalize --out DIR` reads `data.json` and, when present, `transcript.json`, and writes the report. `--render-only` rebuilds the HTML from the saved cleanup without calling a model.

`talk-to-slides run VIDEO --out DIR` runs extract without a model, then captions, then finalize, in one process.

## Model step

The default cleanup uses the `claude` command, the Claude Code CLI, while you are logged in. If that command is not on PATH, the report still has the slides and their text, without the summary. `--llm-backend api` uses the optional `api` extra (`anthropic`) and reads `ANTHROPIC_API_KEY` from the environment. `--no-llm` skips the cleanup. One cleanup call took about 11,000 input and 7,000 output tokens on talks of 6 to 14 minutes.

Use it only on recordings you have the right to process and whose speakers agreed to it.

## License

MIT
