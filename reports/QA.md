# QA Report

Date: 2026-10-02 UTC
Agent: A5

This report records checks performed against the current `main` checkout. No
product feature code was changed by A5.

## Checks

| Area | Check | Result |
| --- | --- | --- |
| Setup | `make setup` | PASS: editable package installation completed in the configured Python 3.14 virtual environment. |
| Test suite | `.venv/bin/python -m pytest -q` | PASS: 20 passed, 1 skipped. One deprecation warning from the installed FastAPI/Starlette TestClient stack. |
| Extraction evaluation | `make eval-extraction` | PASS: 25 deterministic text-only cases; exact match 1.00; hallucination rate 0.00. This does not validate live Whisper or Ollama. |
| Benchmark | `make benchmark` / direct module invocation | BLOCKED before evaluation while loading the Hugging Face dataset: Python 3.14 plus the installed `datasets` stack raises `TypeError: Pickler._batch_setitems() takes 2 positional arguments but 3 were given`. `reports/benchmark.json` correctly records blocked status. |
| Cold start | `make cold-start` / direct module invocation | BLOCKED by the same Hugging Face/Python 3.14 error. `reports/cold_start.json` correctly records blocked status. |
| Reproducibility | Repeat benchmark and compare outputs | NOT MEASURABLE: benchmark cannot load its dataset. The text-only extraction evaluation is deterministic and generated successfully. No model accuracy claim is made. |
| Leakage, horizon 1 | Changed values on target day and later; compared history features | PASS for unavailability: future changes do not alter demand features. A change on the cutoff day is visible by design. |
| Leakage, horizon 2 | Changed values on target day and later; compared history features | PASS for unavailability: future changes do not alter demand features. A change on the cutoff day is visible by design. Existing pytest coverage directly tests horizon 2; the A5 probe covered both 1 and 2. |
| Censoring | Inspected `sold_out` flow and evaluation | PARTIAL / OPEN: `sold_out_yesterday` and `censored` are exposed as features, and benchmark counts `censored_days` when present. However, `sold_out=true` target values are still treated as exact demand in `evaluate`; they are not lower bounds as required by AGENTS.md. The public proxy has no censoring column, so this was not measurable end-to-end. |
| Offline path | `make offline-check` | FAIL: target only prints `Offline check is reserved for the quality agent.` It does not install/run a socket-guarded or otherwise network-blocked core-path check. Deterministic extraction evaluation itself ran without network/model calls. |
| API validation | Empty upload, invalid date, reversed date range | PASS: empty audio returns 400; invalid query date and reversed ranges return 422. |
| API malformed input | Non-audio upload and approximately 10-minute payload with fake local decoder | OPEN: both returned 200 because the API does not validate MIME/content or enforce an upload-size/duration limit before passing data to the decoder. With the real decoder these may fail later, but the boundary is not robust. |
| API privacy | Decoder exception path | PASS: transcription failure returns 503 and the temporary upload was deleted. Existing success-path test also verifies deletion. |
| Privacy | Scanned tracked files for tokens, email/phone patterns, private paths, and media; checked `.gitignore` | PASS with expected test/documentation references only. No tracked private data or audio; `data/private/`, audio extensions, `.env`, and virtual environments are ignored. `AGENTS.md` is untracked local instructions, not a product artifact. |
| Runtime dependencies | README setup and environment | BLOCKED/PARTIAL: tests work on Python 3.14, but the public dataset path does not. Live Whisper/Ollama/ffmpeg and TabPFN gated checkpoint/token were not available for a real-model run. LightGBM runtime is documented as blocked by missing macOS `libomp.dylib`. |
| Git history | Recent commits, branch tracking, status | PASS with local hygiene note: six small commits on 2026-10-02; `main` matches `origin/main`; no A5 commit/push yet. `AGENTS.md` remains an untracked file supplied in the workspace and was not staged. |
| Docs consistency | README, architecture, findings, generated reports, make dry run | PARTIAL: benchmark blockers and proxy limitations are documented. README's setup commands are syntactically present, but `offline-check` is not listed and its target is a no-op. `reports/QA.md` is now the source for this audit. |

## Severity-ranked open issues

1. **High: censoring is not implemented as lower-bound demand.** The code
   reports censored rows and feeds a censoring feature, but evaluates/fits the
   observed `pizzas_sold` as if it were true demand. This can understate demand
   on sold-out days and violates the DailyRecord contract. Owner: forecasting
   agent/orchestrator.
2. **High: benchmark and cold-start are blocked on the current supported
   environment.** The project declares Python `>=3.11`, but Python 3.14 cannot
   load the selected public dataset with the installed dependency set. Pin or
   constrain a compatible runtime/dependency combination and rerun both reports
   before making benchmark claims.
3. **Medium: `make offline-check` is a no-op.** Add a real network-denial check
   for the local deterministic/core path, or document an explicit unavailable
   prerequisite and make the target fail rather than claim completion.
4. **Medium: upload boundary accepts arbitrary types and unbounded payloads.**
   Validate allowed audio suffix/content and cap request size or duration before
   decoding. This is especially relevant because the endpoint is intended to
   handle local uploads robustly.
5. **Low: no reproducibility result exists for model evaluation.** Fixed seeds
   are present in model code, but dataset loading prevents comparing two
   benchmark runs. Re-run after issue 2 is resolved and compare generated JSON
   within a documented tolerance.
6. **Low: the installed TestClient emits a deprecation warning.** Align the
   Starlette/httpx dependency versions when dependency maintenance is next done.

## Not claimed

No TabPFN, LightGBM, Whisper, or Ollama accuracy/runtime claim is made from this
run. The 1.00 extraction result is for the deterministic text-only evaluator,
not Gemma output. No private or real-friend data was inspected or added.
