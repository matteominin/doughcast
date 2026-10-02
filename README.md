# Doughcast

Doughcast is a local-first assistant for a pizzeria. It will turn Italian voice notes into confirmed daily records and, once enough history exists, forecast demand two days ahead for dough preparation.

The voice slice is implemented locally: faster-whisper transcribes Italian audio and Ollama/Gemma returns schema-constrained JSON for owner confirmation. Uploaded audio is read from a temporary path by the API layer and is not persisted by the voice module.

## Setup

```sh
make setup
make test
make eval-extraction
```

Private data belongs in `data/private/`, which is ignored by git. The core path is intended to run locally after model downloads; no cloud service is part of the planned runtime path.

Voice notes require `ffmpeg` for webm, ogg, and m4a decoding. Set `WHISPER_MODEL` (default `small`) and `OLLAMA_MODEL` (default `gemma4:e4b`) through the environment. `make eval-extraction` is deterministic and text-only: it skips model calls and writes `reports/extraction_eval.json`. The current report covers 25 cases, including five adversarial null cases.

Production extraction uses Ollama's `format` JSON schema, retries one invalid/unavailable response, validates `DailyRecord`, and returns Italian warnings for implausible arithmetic or missing sold-out times. Recalled notes use `mode=recalled`, return only the explicitly named days, and mark records as `source="recalled"`.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The contracts mirror AGENTS.md §5.

## Licenses to verify

| Component | License/status |
| --- | --- |
| Gemma 4 | TO VERIFY on the model card |
| faster-whisper and Whisper weights | TO VERIFY for code and weights |
| TabPFN | Open-weight; checkpoint and non-commercial terms TO VERIFY |
| Public restaurant dataset | Dataset-card license and attribution TO VERIFY |

Do not describe the project as fully open source until these terms are verified.

## Limits and open questions

The public benchmark will be a proxy dataset of restaurant visitors, not pizza sales. Real friend data, permissions, recipe percentages, and final model versions are still to be supplied by the human. No real audio is included. The real faster-whisper and Ollama runtime checks are blocked until model weights, Ollama, and ffmpeg are installed; the default recommended Whisper size is `small` for local CPU usability, with `medium` to be tried when hardware permits.

## Commits after the deadline

TODO: record any commits made after 2026-10-05 06:59 UTC.