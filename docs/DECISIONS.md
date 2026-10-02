# Decisions

## 2026-10-02: Keep the A0 scaffold dependency-light at runtime

The project declares the decided ML and API dependencies in `pyproject.toml`, but the scaffold does not import heavy model packages. This lets later agents work against stable contracts without loading models during tests. A framework swap or contract change requires orchestrator review.

## 2026-10-02: Keep extraction evaluation deterministic

The extraction evaluation uses a small deterministic text parser rather than making Ollama calls. This makes `make eval-extraction` reproducible and network-independent while mocked tests cover the production structured-output path. A live-model benchmark was rejected because model versions and local weights would make the report nondeterministic.

## 2026-10-02: Retry invalid local-model output once

Ollama extraction retries exactly once, then returns an empty nullable record with an Italian warning. Silently accepting malformed JSON or inventing fallback values was rejected because the owner must confirm every extracted record.

## 2026-10-02: Store JSON lists as JSON text in SQLite

`DailyRecord.events` is serialized as JSON rather than a delimiter-separated string so empty strings and future event text remain unambiguous. A normalized child table was rejected as unnecessary for the current one-record-per-day contract.