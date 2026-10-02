# Decisions

## 2026-10-02: Keep the A0 scaffold dependency-light at runtime

The project declares the decided ML and API dependencies in `pyproject.toml`, but the scaffold does not import heavy model packages. This lets later agents work against stable contracts without loading models during tests. A framework swap or contract change requires orchestrator review.

## 2026-10-02: Direct Parquet streaming for Hugging Face datasets on Python 3.14

Python 3.14 standard library `pickle` changed `Pickler._batch_setitems()`, causing Hugging Face `datasets` / `dill` to fail during hashing. To ensure the public benchmark runs seamlessly across all Python 3.11+ versions, `load_public_dataset` streams `restaurant/train-00000-of-00001.parquet` directly via `huggingface_hub` when `load_dataset` raises an exception.

## 2026-10-02: Native Scikit-Learn fallback for LightGBM Quantile Regression

On macOS systems without C OpenMP libraries (`libomp.dylib`), `lightgbm` import raises an `OSError`. `LightGBMQuantileModel` dynamically falls back to `sklearn.ensemble.HistGradientBoostingRegressor(loss="quantile")`, preserving exact p10/p50/p90 quantile regression capability without crashing.

## 2026-10-02: Socket-level network denial guard for `make offline-check`

`make offline-check` patches `socket.socket.connect` to block non-localhost outgoing network requests while running the complete end-to-end core path (Store -> Features -> Models -> Recommendation -> API). This guarantees that the application runs 100% offline without network leaks.

## 2026-10-02: Keep extraction evaluation deterministic

The extraction evaluation uses a small deterministic text parser rather than making Ollama calls. This makes `make eval-extraction` reproducible and network-independent while mocked tests cover the production structured-output path. A live-model benchmark was rejected because model versions and local weights would make the report nondeterministic.

## 2026-10-02: Retry invalid local-model output once

Ollama extraction retries exactly once, then returns an empty nullable record with an Italian warning. Silently accepting malformed JSON or inventing fallback values was rejected because the owner must confirm every extracted record.

## 2026-10-02: Store JSON lists as JSON text in SQLite

`DailyRecord.events` is serialized as JSON rather than a delimiter-separated string so empty strings and future event text remain unambiguous. A normalized child table was rejected as unnecessary for the current one-record-per-day contract.

## 2026-10-02: Keep live forecasting TabPFN-only

Feature and baseline models share an interface for evaluation, but the product readiness gate must not silently substitute a baseline for TabPFN. This keeps the live claim aligned with the contract; benchmark baselines remain useful as comparators.

## 2026-10-02: Shift all demand features by the forecast horizon

For a target date `t`, demand-derived features use observations no newer than `t - HORIZON_DAYS`. This prevents a two-day-ahead forecast from reading sales that would not yet be known when dough is prepared.

## 2026-10-02: Score quantiles before rounding dough quantities

Pinball loss is computed from the continuous service-level estimate, while business waste and stock-out metrics use the separately rounded dough-ball quantity. Scoring the rounded quantity would mix an operational rounding rule into a probabilistic forecast metric and could hide small quantile errors.