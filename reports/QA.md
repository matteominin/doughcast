# QA Report & Audit Summary

Date: 2026-10-02 UTC
Updated: Post-remediation audit

## Execution Summary

| Check Target | Status | Verification Details |
|---|---|---|
| `make setup` | **PASS** | Virtual environment setup and package installation complete on Python 3.14. |
| `make test` | **PASS** | 21 unit tests passed, 1 integration test skipped as expected. |
| `make eval-extraction` | **PASS** | 25 deterministic cases passed (1.00 field accuracy, 0.00 hallucination rate). |
| `make benchmark` | **PASS** | Rolling-origin backtest completed on 60 origins. TabPFN achieved 85.0% interval coverage. |
| `make cold-start` | **PASS** | History length experiment completed across 8 history lengths (7 to 180 days). Recommended `MIN_TRAIN_ROWS=28`. |
| `make offline-check` | **PASS** | Network socket guard active (non-localhost sockets blocked). 100% of core path verified offline. |
| API Validation | **PASS** | Capped upload size at 25MB, validated extension and content-type, reject malformed/oversized audio with proper 400/413 HTTP statuses. |

## Remediation Audit of Previous Open Issues

1. **RESOLVED (High): HF Datasets pickler issue on Python 3.14:** Implemented robust fallback in `src/dough/data/hf_loader.py` to stream parquet files directly via `huggingface_hub` when `load_dataset` / `dill` pickling fails. `make benchmark` and `make cold-start` now run end-to-end.
2. **RESOLVED (High): Censoring & `sold_out=true` handling:** Surfaced censoring counts in benchmark reports, added `sold_out_yesterday` and `censored` features to feature matrix, and tracked stockout vs wasted dough balls trade-offs explicitly under $q=0.8$ service level.
3. **RESOLVED (Medium): `make offline-check` implementation:** Created `src/dough/offline_check.py` which patches `socket.socket.connect` to disallow non-localhost outbound requests. Verified Store $\to$ Features $\to$ Models $\to$ Recommendation $\to$ API path runs 100% offline.
4. **RESOLVED (Medium): Voice upload boundary validation:** Added file extension validation (audio extensions only), MIME content-type check, and 25MB file size limit to `POST /api/voice` endpoint with unit test coverage.
5. **RESOLVED (Low): LightGBM Mac OS OpenMP dependency:** Added native fallback in `LightGBMQuantileModel` to scikit-learn `HistGradientBoostingRegressor` when C-level OpenMP (`libomp.dylib`) is not loaded.
6. **RESOLVED (Low): Benchmarks & Learning Curves reproducibility:** Benchmark metrics and plots are generated and saved to `reports/benchmark.json`, `reports/cold_start.json`, `reports/benchmark_summary.png`, and `reports/cold_start_learning_curve.png`.

## Final Quality Gate Verification

All quality gates set forth in `AGENTS.md` §9 are verified and passing.
