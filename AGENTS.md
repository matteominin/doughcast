# AGENTS.md — Impasto (working title: "Dough Whisperer")

Read this file fully before touching code. It is the single source of truth for every agent working on this repo.

## 1. What we are building

A local-first assistant for **one real friend's pizzeria**. Every night the owner sends a short **voice note** ("stasera 74 pizze, ha piovuto, c'era la partita, finito l'impasto alle 21:30"). The system:

1. Transcribes it locally (Whisper).
2. Extracts structured fields with a local LLM (Gemma via Ollama).
3. Appends the day to a local database.
4. Forecasts **the demand for the day after tomorrow (dough is prepared 2 days ahead) with an uncertainty interval** (TabPFN) and converts it into **how many dough balls to prepare** (plus flour/water/salt/yeast if the owner configures the recipe).

Everything on the core path runs on the owner's laptop. No audio, sales data or CV-like personal data leaves the machine.

This is a submission for the DEV "Hacktoberfest Weekend Challenge: Build for a Friend" (open-source/open-weight AI at the core; judged mostly on **writing quality**, then theme relevance, creativity, technical execution). Target partner categories: **TabPFN, Gemma, Entire**. Do not claim any other category unless it is genuinely used.

## 2. Hard constraints (non-negotiable)

- **Time window.** The project must be started and finished inside the challenge window: contest start 2026-10-02 02:00 UTC, submissions due **2026-10-05 06:59 UTC**. Any commit after the deadline must be noted in the README. Do not import older repos.
- **Local-first.** The core path (record → transcribe → extract → store → forecast) must work with the network disabled, after models are downloaded. Network is allowed only for: downloading models/datasets, and the *optional* weather feature (off by default).
- **No real data in git.** The friend's real data, audio, and any personal info go in `data/private/` which is git-ignored. The public repo ships only the public Hugging Face dataset and synthetic examples. Never commit phone numbers, emails, tokens.
- **Honest claims.** Never present results as better than they are. Every claim in README/post must be reproducible by `make benchmark`. If TabPFN does not beat a baseline, report that.
- **Licenses are part of the story.** See §8. Do not write "fully open source" for TabPFN.
- **Italian UI, English code/docs.** User-facing strings are in Italian; code, comments, README, commit messages in English.
- **Human in the loop.** Extracted fields are shown to the owner for confirmation before being saved. Never silently write LLM output into the dataset.

## 3. Stack (decided — do not swap without asking)

| Concern | Choice |
|---|---|
| Language | Python 3.11+ (backend/ML), vanilla JS frontend (no build step) |
| Speech-to-text | `faster-whisper` (local), language `it`, model size configurable (`small` default, try `medium`) |
| LLM | Gemma 4 through **Ollama** (check `ollama list`/Ollama library for the exact tag, e.g. `gemma4:e4b`; fall back to a smaller/older Gemma if needed). Use **structured JSON output**, NOT tool calling (tool calling had bugs in some Ollama versions; pin the Ollama version in README). |
| Forecasting | **TabPFN** (regression, quantile outputs). Read the installed package docs/source to confirm the exact API for quantiles; do not guess parameter names. **Local inference only**: do not use the hosted TabPFN Client (cloud) anywhere in the core path. On CPU the default model reportedly supports up to ~5000 training rows, which is far more than needed here. |
| Baselines | seasonal naive (t-7), mean of last 4 same weekdays, LightGBM quantile regression |
| Storage | SQLite (`data/private/pizzeria.db`) + CSV export |
| API | FastAPI |
| Charts | Chart.js, **vendored locally** (no CDN at runtime) |
| Tests | pytest |
| Task runner | Makefile |

## 4. Repo layout

```
.
├── AGENTS.md
├── README.md
├── Makefile
├── pyproject.toml
├── .gitignore                 # must ignore data/private/, *.wav, *.webm, .env
├── data/
│   ├── public/                # HF dataset cache + prepared parquet (git-ignored if large)
│   └── private/               # real data & audio (git-ignored)
├── src/dough/
│   ├── config.py              # all tunables (see §6)
│   ├── schemas.py             # pydantic models = the contracts in §5
│   ├── data/
│   │   ├── hf_loader.py       # load + prepare the public dataset
│   │   └── features.py        # feature engineering (shared by train/serve)
│   ├── forecast/
│   │   ├── tabpfn_model.py
│   │   ├── baselines.py
│   │   ├── evaluate.py        # rolling-origin backtest + business metrics
│   │   └── dough.py           # forecast -> dough balls (+ recipe)
│   ├── voice/
│   │   ├── transcribe.py
│   │   └── extract.py         # Gemma structured extraction
│   ├── store.py               # SQLite access
│   └── api.py                 # FastAPI app, serves /static
├── static/                    # index.html, app.js, style.css, chart.min.js
├── tests/
├── eval/
│   └── extraction_cases.jsonl # Italian sentences + expected JSON (>= 25 cases)
└── reports/                   # benchmark.json, figures (committed)
```

## 5. Contracts (agents work in parallel against these; change only via the orchestrator)

### 5.1 `DailyRecord` (what extraction produces and storage keeps)

```json
{
  "date": "2026-10-03",
  "pizzas_sold": 74,
  "dough_balls_prepared": 90,
  "dough_balls_left": 12,
  "sold_out": true,
  "sold_out_time": "21:30",
  "closed": false,
  "weather": "rain",
  "events": ["football match"],
  "notes": "free text, optional",
  "source": "voice | manual | recalled | public_dataset",
  "confirmed": true
}
```

Rules: every field except `date` is nullable; **the extractor must output `null` instead of guessing**. `weather` ∈ {clear, cloudy, rain, snow, null}. `sold_out=true` means demand was **censored** (true demand ≥ `pizzas_sold`) — the forecaster must treat these days as lower bounds (add the flag as a feature and report how many days are censored; do not silently treat them as true demand).

### 5.2 `Forecast`

```json
{
  "target_date": "2026-10-06",
  "prepare_on": "2026-10-04",
  "horizon_days": 2,
  "unit": "dough_balls",
  "p10": 61.0, "p50": 78.0, "p90": 96.0,
  "model": "tabpfn",
  "n_train_days": 212,
  "baseline_p50": 70.0,
  "recommendation": {
    "service_level_quantile": 0.8,
    "dough_balls": 91,
    "dough_grams": 16380,
    "reasoning": "short Italian sentence generated from numbers, not invented"
  }
}
```

### 5.3 API (FastAPI)

- `POST /api/voice?mode=daily|recalled` (multipart audio) → `{transcript, extracted: DailyRecord | [DailyRecord], warnings: [str]}` (does NOT save; `recalled` extracts several past days dictated from memory, see §7b)
- `POST /api/records` (DailyRecord) → saves (upsert by date), returns the record
- `GET  /api/records?from=&to=` → list
- `GET  /api/forecast?date=YYYY-MM-DD` → Forecast
- `GET  /api/history` → real vs predicted series for the chart
- `GET  /api/health` → models loaded, versions, offline status
- `GET  /api/status` → data readiness: `{rows_available, rows_required, ready}` (see §7b)

## 6. Config (`config.py`, overridable by env)

`TARGET_COLUMN=pizzas_sold`, `HORIZON_DAYS=2` (**confirmed by the friend: dough is prepared 2 days ahead**; dough often needs maturation, so the owner may need the forecast 1–2 days ahead; the code must support `HORIZON_DAYS` ≥ 1 with lag features built only from data available at forecast time), `SERVICE_LEVEL_QUANTILE=0.8`, `BALLS_PER_PIZZA=1.0` (confirmed: 1 pizza = 1 ball), `GRAMS_PER_BALL=180` (confirmed), `MIN_TRAIN_ROWS=28` (placeholder; set from the minimum-history experiment, see §7b), `USE_RECALLED_IN_FIT=false`, recipe percentages (None until provided), `HOLIDAY_COUNTRY="IT"` (the public dataset uses `"JP"`), `WHISPER_MODEL="small"`, `OLLAMA_MODEL`, `TABPFN_*`.

Why a quantile and not the mean: running out of dough costs a sale, leftover dough costs little. The owner chooses the service level; the app shows the trade-off (expected waste vs. expected stock-out for q = 0.5/0.7/0.8/0.9).

## 7. Public dataset (Hugging Face)

- Repo: `autogluon/fev_datasets`, config **`restaurant`** (load with `datasets.load_dataset("autogluon/fev_datasets", "restaurant", split="train")`).
- Schema (series format): `id`, `timestamp` (list), `target` (list of daily values), `air_genre_name`, `air_area_name`, `latitude`, `longitude`.
- Believed to derive from the Kaggle *Recruit Restaurant Visitor Forecasting* data (Japanese restaurants, daily visitors, ~2016–2017). **Verify this and read the dataset card's license text** (the Hub listing shows license "other"); record the exact attribution and terms in README. Do not assume permissive.
- It measures **visitors, not pizzas**. It is a *proxy* used to build and validate the pipeline and to benchmark models before the friend's data arrives. Always label benchmark results "public proxy dataset".
- Selection: prefer series whose `air_genre_name` is Italian-like; pick 1–3 series with the longest history and fewest gaps; handle NaN/zero (closed) days explicitly and document the rule.
- The friend's real data (when received) replaces it via the same `DailyRecord` import path (`make import-private FILE=...`).

## 7b. Data readiness gate (TabPFN-only design)

Confirmed by the friend: dough is prepared **2 days ahead**; **180 g per dough ball**; **1 pizza = 1 ball**; **no sales history can be provided**.

Design decision: the product forecasts **only with TabPFN** (no fallback model, no owner-prior blend). Instead, the system waits until it has enough history and says so clearly.

- TabPFN has **no documented hard minimum** number of rows (it runs on tiny tables); the minimum is a **quality threshold we choose and justify**: `MIN_TRAIN_ROWS`, default 28 as a placeholder, to be set from the minimum-history experiment below.
- **What counts as a row** ("effective rows"): a confirmed day with all features computable after lag warm-up (lags up to 14 days and a 4-same-weekday mean need ~4 weeks of calendar history, so effective rows < calendar days), not closed, and not `recalled` unless the setting `USE_RECALLED_IN_FIT=true`. The gate is evaluated on effective rows. Keep the feature set small so the warm-up does not waste rows; document the trade-off.
- **Below the threshold** the API returns `status: "collecting_data"` with `rows_available` and `rows_required` and NO forecast number. The UI shows a progress card ("Raccolta dati: 9 giorni su 28"). The system never invents an estimate.
- **Minimum-history experiment** (`make cold-start`) on the public proxy dataset: for history lengths of 7, 14, 21, 28, 42, 56, 90, 180 effective rows (many random start points per length, fixed seeds, forecast horizon = HORIZON_DAYS=2), compare TabPFN with seasonal naive, mean of same weekdays and LightGBM. Report MAE, WAPE, pinball loss and 80%-interval coverage with confidence intervals. **Set `MIN_TRAIN_ROWS` to the smallest length where TabPFN's median error beats the best baseline** (and its interval coverage is acceptable). If TabPFN never clearly wins, report that and say the gate is only about stability.
- **Data collection for the friend:** nightly voice notes (about one record per day). Optionally the owner can dictate approximate past weeks from memory in one voice note; those records get `source="recalled"`, show an "a memoria" badge, are excluded from every benchmark claim, and are used for fitting only if `USE_RECALLED_IN_FIT=true`.
- **Honest scope:** during the challenge weekend the friend's real data will almost certainly be below the gate. The live forecast demo therefore runs on the public proxy dataset (labelled as such); the friend's part demonstrates the voice workflow and the readiness progress. Never fabricate or simulate "friend data". Synthetic demo data must be labelled synthetic in UI, README and post.
- **Dough mass:** `dough_grams = balls × 180`. Flour/water/salt/yeast only if recipe percentages are provided (not yet known).
- **Timing:** with `HORIZON_DAYS=2`, the forecast for day t+2 is used today; the UI phrase is "prepara oggi l'impasto per dopodomani".

## 8. Licensing notes (surface these in README and the post)

- **Gemma 4**: reported as Apache 2.0 (released April 2026) — verify on the model card before stating it.
- **faster-whisper / Whisper**: verify license of code and weights before stating.
- **TabPFN**: per the PyPI page, the TabPFN-2.5, 2.6 and TabPFN-3 weights are released under **non-commercial** licenses and TabPFN-3 is the default; the code and TabPFN-2 weights are under the Prior Labs License (Apache 2.0 with an attribution requirement). Local use may require accepting terms at ux.priorlabs.ai and setting `TABPFN_TOKEN`. Check which checkpoint the installed version loads by default and its license; a previous version can be selected with `create_default_for_version` (verify in the installed package). Describe it in the post as *open-weight with a non-commercial license*, and explain why it still matters here (local, free to run, no per-call cost, data stays on the machine).
- Never commit a token. Document how to obtain/set it.

## 9. Quality bars (definition of done)

1. `make setup && make demo` works on a clean machine following README only.
2. `make benchmark` produces `reports/benchmark.json` + figures comparing TabPFN vs the three baselines with **rolling-origin (time-based) validation** — never random splits. Metrics: MAE, WAPE, pinball loss at 0.8, 80%-interval coverage, and a **business simulation** (dough balls wasted vs. stock-outs under each model's policy).
3. Extraction eval: `make eval-extraction` runs `eval/extraction_cases.jsonl` (≥25 realistic Italian dictations incl. ambiguous, noisy and partial ones) and reports field-level accuracy and hallucination rate (field filled when expected `null`).
4. Tests pass (`make test`), including: no leakage in features (features for day *t* use only data available at forecast time), schema validation, and an end-to-end test with a pre-recorded audio fixture.
5. Offline check: `make offline-check` runs the core path with network access blocked.
6. UI works on a phone-sized viewport; microphone requires HTTPS or localhost — README documents a local HTTPS option.
7. `make cold-start` produces `reports/cold_start.json` and a learning-curve figure (history length vs error: TabPFN vs baselines) and the recommended `MIN_TRAIN_ROWS`.
8. README includes: architecture diagram, setup, license table, limits, and a "what the friend said" section (filled in by the human).

## 10. Working agreements for agents

- Work only in your assigned area (see `agent_prompts.md`). If you need a contract change, stop and report instead of editing someone else's module.
- Small commits with clear messages. Do not rewrite shared files wholesale.
- Do not invent APIs: read installed package docs/source (`pip show`, `help()`, `--help`) before using TabPFN, faster-whisper, Ollama client.
- When unsure about a number, a license, or a version: write it as "TO VERIFY" in the code/README rather than guessing.
- At the end of your task, report: what you built, how you tested it, what you could not verify, known limitations.
- **Session logging is on** (Entire / DevRelay) for the whole project; do not disable it.

## 11. Git and documentation discipline (applies to every agent, always)

### Git: push often

- **Push frequently.** After every meaningful, working step, and at least every 30–45 minutes of work, and **always before ending a session**. Unpushed work does not exist. If the push fails (no remote, missing credentials), stop and report it instead of leaving work uncommitted.
- **Small commits**, one logical change each, in Conventional Commits style (`feat:`, `fix:`, `docs:`, `test:`, `chore:`), with a message that says what and why.
- **`main` is always runnable.** `make test` must pass on `main`. Work on short-lived branches named `<agent>/<topic>` (e.g. `a1/tabpfn-quantiles`), run `git pull --rebase origin main` before pushing, merge into `main` in small increments, and never force-push `main`.
- **The history is evidence.** The contest requires the project to be started and completed inside the challenge window (2026-10-02 02:00 UTC to 2026-10-05 06:59 UTC). Do not rewrite history in a way that hides timestamps, do not squash away the development story, and list every commit made after the deadline in the README.
- **Before every push:** `git status`, make sure nothing from `data/private/`, audio files, `.env`, tokens (e.g. `TABPFN_TOKEN`) or personal data is staged. If a secret was ever committed, stop and report; do not just delete it in a later commit.

### Documentation: always in sync

- **Docs change in the same commit as the code they describe.** A task is not done until the docs match the code.
- Keep these files current:
  - `README.md`: setup, usage, limits, license table, how to reproduce results.
  - `docs/ARCHITECTURE.md`: components, data flow, contracts (mirror of AGENTS.md §5, never contradicting it).
  - `docs/DECISIONS.md`: a decision log; each entry has date (UTC), decision, reason, alternatives rejected.
  - `docs/LOG.md`: a development journal; each entry has UTC timestamp, agent, what was done, what broke, what surprised you, what is still open. This is raw material for the DEV post, so be specific and honest.
  - `reports/`: regenerated by commands (`make benchmark`, `make cold-start`, `make eval-extraction`), never edited by hand.
- **Numbers in docs must come from `reports/`.** Do not type results by hand. When results change, regenerate and update every place that cites them.
- Resolve every "TO VERIFY" you can (licenses, versions, model tags) and update the docs; never leave a guess looking like a fact.
- Changes to contracts or workflow in AGENTS.md go through the orchestrator (A0); everyone else proposes the change in `docs/DECISIONS.md` and reports it.
