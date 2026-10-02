# Forecasting findings

This report is intentionally conservative. The public dataset is a proxy of
restaurant visitors, not pizza sales.

## 2026-10-02 run

- `make benchmark`: blocked while loading `autogluon/fev_datasets` because the
  installed `datasets` stack on Python 3.14 raises
  `Pickler._batch_setitems() takes 2 positional arguments but 3 were given`.
  See `reports/benchmark.json`.
- `make cold-start`: blocked by the same loader/runtime incompatibility. See
  `reports/cold_start.json`.
- No TabPFN accuracy claim is made. The installed local API was inspected in
  TabPFN 9.1.0: `TabPFNRegressor.predict(output_type="quantiles",
  quantiles=[0.1, 0.5, 0.9])` is available.
- The TabPFN checkpoint, one-time license acceptance, and token requirement
  remain environment-dependent and must be verified before a live run.
- LightGBM is installed but cannot load on this macOS image because its native
  library cannot find `libomp.dylib`; install Homebrew `libomp` before claiming
  LightGBM benchmark results.

The deterministic feature, baseline, dough conversion, and rolling-origin
metric code is covered by the local pytest suite and does not download models.