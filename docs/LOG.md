# Development log

## 2026-10-02 UTC

- A0 created the prescribed package, data, reports, docs, static, evaluation, and test layout.
- Added Pydantic contracts, env-backed configuration, SQLite upsert/range/CSV persistence, and focused tests.
- Left forecasting, voice, extraction, API behavior, and UI as importable stubs by design.
- Open: verify model tags/licenses, implement remaining agents, and run the full pipeline.
- A2 implemented lazy faster-whisper Italian transcription with pizzeria vocabulary, schema-constrained Ollama extraction, validation warnings, recalled mode, and a deterministic text-only evaluation path.
- `make eval-extraction` passes 25 cases: exact-match 1.00 and hallucination rate 0.00 in `reports/extraction_eval.json`. Mocked voice tests pass 8 with one slow integration test skipped.
- Live Whisper/Ollama checks remain open because model weights, Ollama, and ffmpeg are not available in this environment.
- A1 implemented the public dataset tidy loader/inspection report, strict
	horizon-shifted features, seasonal and weekday baselines, LightGBM quantiles,
	lazy local TabPFN quantiles, rolling-origin metrics, and dough conversion.
- Local tests pass (`14 passed, 1 skipped`). `make benchmark` and `make
	cold-start` write blocked reports because `datasets` on Python 3.14 raises
	`Pickler._batch_setitems() takes 2 positional arguments but 3 were given`
	while loading the Hub dataset. No proxy or TabPFN result is claimed.
- Synthetic rolling-origin evaluation passes for deterministic baselines;
	LightGBM remains blocked by missing macOS `libomp.dylib`.
- A1 follow-up rerun: focused data/forecast tests pass (`7 passed`). Fixed
	evaluation so pinball loss uses the unrounded service estimate and model
	metrics remain aligned when one optional model fails. The public benchmark
	and cold-start targets remain blocked by the Python 3.14 Hugging Face
	`Pickler._batch_setitems` incompatibility; no external result was claimed.