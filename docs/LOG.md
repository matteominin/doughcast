# Development log

## 2026-10-02 UTC

- A0 created the prescribed package, data, reports, docs, static, evaluation, and test layout.
- Added Pydantic contracts, env-backed configuration, SQLite upsert/range/CSV persistence, and focused tests.
- Left forecasting, voice, extraction, API behavior, and UI as importable stubs by design.
- A2 implemented lazy faster-whisper Italian transcription with pizzeria vocabulary, schema-constrained Ollama extraction, validation warnings, recalled mode, and a deterministic text-only evaluation path.
- `make eval-extraction` passes 25 cases: exact-match 1.00 and hallucination rate 0.00 in `reports/extraction_eval.json`.
- A1 implemented the public dataset tidy loader/inspection report, strict horizon-shifted features, seasonal and weekday baselines, LightGBM quantiles, lazy local TabPFN quantiles, rolling-origin metrics, and dough conversion.
- Implemented `hf_hub_download` parquet fallback in `src/dough/data/hf_loader.py` to resolve Python 3.14 Hugging Face `datasets` pickler incompatibility.
- Added `HistGradientBoostingRegressor` fallback to `LightGBMQuantileModel` when native `libomp.dylib` is unavailable on macOS ARM.
- Ran `make benchmark`: 60 rolling-origin test windows evaluated on CPU. TabPFN achieved 85.0% interval coverage for 80% uncertainty bounds.
- Ran `make cold-start`: evaluated history lengths 7 through 180 days; set `MIN_TRAIN_ROWS=28`. Generated learning curve plot `reports/cold_start_learning_curve.png`.
- Implemented `make offline-check`: created socket-level network denial guard in `src/dough/offline_check.py` to verify Store -> Features -> Models -> Recommendation -> API workflow 100% offline.
- Added audio upload size limit (25MB), content-type checking, and extension validation to `POST /api/voice` in `src/dough/api.py` with full pytest coverage.
- All 21 tests pass in `make test`, `make eval-extraction` passes 25 cases, `make offline-check` passes 100% offline.