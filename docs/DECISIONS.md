# Decisions

## 2026-10-02: Keep the A0 scaffold dependency-light at runtime

The project declares the decided ML and API dependencies in `pyproject.toml`, but the scaffold does not import heavy model packages. This lets later agents work against stable contracts without loading models during tests. A framework swap or contract change requires orchestrator review.

## 2026-10-02: Store JSON lists as JSON text in SQLite

`DailyRecord.events` is serialized as JSON rather than a delimiter-separated string so empty strings and future event text remain unambiguous. A normalized child table was rejected as unnecessary for the current one-record-per-day contract.