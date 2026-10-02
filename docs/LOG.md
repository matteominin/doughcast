# Development log

## 2026-10-02 UTC

- A0 created the prescribed package, data, reports, docs, static, evaluation, and test layout.
- Added Pydantic contracts, env-backed configuration, SQLite upsert/range/CSV persistence, and focused tests.
- Left forecasting, voice, extraction, API behavior, and UI as importable stubs by design.
- Open: verify model tags/licenses, implement remaining agents, and run the full pipeline.
- A2 implemented lazy faster-whisper Italian transcription with pizzeria vocabulary, schema-constrained Ollama extraction, validation warnings, recalled mode, and a deterministic text-only evaluation path.
- `make eval-extraction` passes 25 cases: exact-match 1.00 and hallucination rate 0.00 in `reports/extraction_eval.json`. Mocked voice tests pass 8 with one slow integration test skipped.
- Live Whisper/Ollama checks remain open because model weights, Ollama, and ffmpeg are not available in this environment.