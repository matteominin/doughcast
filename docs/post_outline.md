# DEV Post Outline

This is an outline, not final post prose. All proxy-dataset claims must remain
labelled as proxy results, and all personal details require the human's review
and permission.

## What I Built

- Describe Doughcast as a local-first assistant for one pizzeria: Italian
  voice note, local transcription, structured extraction, owner confirmation,
  SQLite storage, and a two-day-ahead dough workflow.
- State the confirmed operating facts from the project brief: one pizza per
  dough ball, 180 g per ball, and dough prepared two days ahead.
- Exact checked-in evaluation numbers: 25 deterministic text-only cases,
  exact match `1.00`, hallucination rate `0.00`.
  Reference: [reports/extraction_eval.json](../reports/extraction_eval.json).
- Explain that these numbers are not Gemma or Whisper accuracy numbers.
- TODO: human writes the friend's story, name or pseudonym, permission, and
  why this problem matters to them.

## Demo

- Show the flow: record, inspect extracted fields, confirm, save, and view the
  readiness/forecast state.
- Be explicit that a real model demo requires local Whisper/Ollama/ffmpeg and a
  usable TabPFN checkpoint.
- State the current reproducibility result: `make benchmark` is `blocked` by
  `Pickler._batch_setitems() takes 2 positional arguments but 3 were given`
  while loading the public visitor proxy dataset.
  Reference: [reports/benchmark.json](../reports/benchmark.json).
- TODO: human adds screenshots, demo video link, hardware details, and the
  friend's reaction.

## Code

- Point to the local pipeline in `src/dough/voice/`, `src/dough/data/`,
  `src/dough/forecast/`, `src/dough/store.py`, and `src/dough/api.py`.
- Point to the architecture diagram in [README.md](../README.md) and
  [docs/ARCHITECTURE.md](ARCHITECTURE.md).
- Explain that the live product is TabPFN-only after the readiness gate, while
  baselines are evaluation comparators and not production fallbacks.
- Mention the owner-confirmation boundary and temporary upload cleanup.
- TODO: human adds the repository URL, final branch/commit link, and any code
  excerpt or screenshot.

## How I Built It

- Describe the local stack: Python/FastAPI, SQLite, faster-whisper, Ollama,
  Gemma structured JSON output, TabPFN quantiles, and vanilla JavaScript.
- Explain horizon-shifted features and the readiness gate; do not claim that
  the gate threshold was validated by a successful cold-start experiment.
- Exact QA numbers: `20 passed, 1 skipped`; extraction evaluation passed; the
  benchmark and cold-start runs were blocked before model evaluation.
  Reference: [reports/QA.md](../reports/QA.md).
- Surface the QA censoring gap: `sold_out=true` is exposed/countable, but the
  current evaluation still treats the observed sales value as exact demand
  instead of a lower bound.
- Surface the other QA gaps: `make offline-check` is a no-op, and upload type
  and duration limits are not enforced.
- TODO: human adds what broke during the build, the actual model versions and
  hardware, and the personal session narrative.

## Why Does Open Innovation Matter?

- Discuss local inference as a practical privacy property for a small business:
  audio and sales records stay on the owner's machine after model downloads.
- Discuss the value of inspectable, replaceable components and an explicit
  human confirmation step.
- State the TabPFN caveat plainly: its active checkpoint/license and token
  requirements remain unverified here, so it must not be presented as fully
  open source.
- Do not claim benchmark superiority: [reports/FINDINGS.md](../reports/FINDINGS.md)
  records no TabPFN accuracy result because the benchmark is blocked.
- TODO: human adds their own view of open innovation and why it mattered to the
  friend, without inventing a quote.

## My Agent Session

- Summarize the staged agent workflow: scaffold, voice, forecasting, API,
  frontend, QA, and documentation.
- Cite the current QA evidence: setup and tests passed in the tested
  environment; the dataset-dependent benchmark was blocked; live model checks
  were not available.
  Reference: [reports/QA.md](../reports/QA.md).
- TODO: human adds the session log link or screenshots, what they personally
  changed, and any story about debugging or collaboration.

## Prize Categories

- List only the categories genuinely used: **TabPFN**, **Gemma**, and
  **Entire**.
- TabPFN: local quantile adapter and intended demand forecast; no accuracy
  claim until the blocked benchmark can run.
- Gemma: local structured extraction through Ollama; the checked-in evaluator
  is deterministic and does not measure Gemma output.
- Entire: session logging is part of the project workflow; TODO: human verifies
  the final public-facing attribution and link.
- TODO: human confirms eligibility wording, links, and any partner-specific
  submission requirements.

## Evidence and TODO Checklist

- `reports/extraction_eval.json`: 25 cases, exact match `1.00`, hallucination
  rate `0.00`.
- `reports/benchmark.json`: status `blocked`; no forecast metrics.
- `reports/FINDINGS.md`: Python 3.14/datasets blocker, TabPFN verification gap,
  and LightGBM `libomp.dylib` issue.
- `reports/QA.md`: 20 passed, 1 skipped; censoring, offline-check, and upload
  validation gaps.
- TODO: replace all human-only placeholders before publishing.