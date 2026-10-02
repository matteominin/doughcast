# Architecture

Doughcast is planned as a local-first pipeline:

```mermaid
flowchart LR
    Voice[Italian voice note] --> Transcribe[Local Whisper]
    Transcribe --> Extract[Local Gemma JSON extraction]
    Extract --> Confirm[Owner confirmation]
    Confirm --> Store[SQLite daily_records]
    Store --> Features[Leakage-safe features]
    Features --> Forecast[Local TabPFN forecast]
    Forecast --> Dough[Dough-ball recommendation]
```

`src/dough/voice/transcribe.py` loads faster-whisper lazily, forces Italian decoding, and supplies pizzeria vocabulary as an initial prompt. `src/dough/voice/extract.py` sends the transcript to a local Ollama client with a JSON schema, retries once, validates with `DailyRecord`, and returns Italian warnings. `mode="recalled"` uses an array schema and marks returned records as recalled. `src/dough/voice/evaluate.py` is deliberately model-free so `make eval-extraction` is deterministic and offline.

The API layer owns temporary upload cleanup; voice functions accept a path but do not persist audio. The real model path requires locally installed faster-whisper weights, Ollama/Gemma, and ffmpeg for compressed containers.

## Contracts

- `DailyRecord` uses nullable fields and validates weather and `HH:MM` sold-out times.
- `Forecast` contains p10/p50/p90 demand estimates and a dough recommendation.
- SQLite upserts records by ISO date and can filter or export them as CSV.