"""Rolling-origin evaluation and report generation for the public proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dough.config import BALLS_PER_PIZZA, HORIZON_DAYS, SERVICE_LEVEL_QUANTILE
from dough.data.hf_loader import load_public_dataset
from dough.forecast.baselines import PredictionInterval, LightGBMQuantileModel, same_weekday_mean, seasonal_naive
from dough.forecast.tabpfn_model import TabPFNModel


def _pinball(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    error = actual - predicted
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def _metrics(actual: list[float], predictions: list[PredictionInterval], *, quantile: float = SERVICE_LEVEL_QUANTILE) -> dict[str, float]:
    observed = np.asarray(actual, dtype=float)
    p50 = np.asarray([prediction.p50 for prediction in predictions], dtype=float)
    p10 = np.asarray([prediction.p10 for prediction in predictions], dtype=float)
    p90 = np.asarray([prediction.p90 for prediction in predictions], dtype=float)
    service_estimate = np.asarray([_service_estimate(prediction, quantile) for prediction in predictions], dtype=float)
    prepared = np.ceil(service_estimate * BALLS_PER_PIZZA)
    return {
        "n": float(len(observed)),
        "mae": float(np.mean(np.abs(observed - p50))),
        "wape": float(np.sum(np.abs(observed - p50)) / max(1.0, np.sum(np.abs(observed)))),
        "pinball_loss": _pinball(observed, service_estimate, quantile),
        "interval_coverage_80": float(np.mean((observed >= p10) & (observed <= p90))),
        "dough_wasted_balls": float(np.sum(np.maximum(prepared - observed, 0))),
        "stockout_balls": float(np.sum(np.maximum(observed - prepared, 0))),
    }


def _service_estimate(prediction: PredictionInterval, quantile: float) -> float:
    if quantile <= 0.5:
        return prediction.p10 + (prediction.p50 - prediction.p10) * (quantile / 0.5)
    if quantile >= 0.9:
        return prediction.p50 + (prediction.p90 - prediction.p50) * ((quantile - 0.5) / 0.4)
    return prediction.p50 + (prediction.p90 - prediction.p50) * ((quantile - 0.5) / 0.4)


def evaluate(
    frame: pd.DataFrame,
    *,
    backtest_days: int = 60,
    horizon_days: int = HORIZON_DAYS,
    target_column: str = "target",
    include_tabpfn: bool = True,
) -> dict:
    """Run a time-ordered rolling-origin backtest for one tidy series."""

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data = data.sort_values("date").dropna(subset=[target_column]).reset_index(drop=True)
    origins = data["date"].iloc[max(0, len(data) - backtest_days):]
    predictions: dict[str, list[PredictionInterval]] = {"seasonal_naive": [], "same_weekday_mean": [], "lightgbm": []}
    if include_tabpfn:
        predictions["tabpfn"] = []
    actuals: dict[str, list[float]] = {name: [] for name in predictions}
    dates: dict[str, list[str]] = {name: [] for name in predictions}
    errors: dict[str, str] = {}
    for target_date in origins:
        cutoff = target_date - pd.to_timedelta(horizon_days, unit="D")
        history = data[data["date"] <= cutoff]
        if history.empty:
            continue
        actual = float(data.loc[data["date"] == target_date, target_column].iloc[0])
        date_string = target_date.date().isoformat()
        predictions["seasonal_naive"].append(seasonal_naive(history, target_date, target_column=target_column))
        actuals["seasonal_naive"].append(actual)
        dates["seasonal_naive"].append(date_string)
        predictions["same_weekday_mean"].append(same_weekday_mean(history, target_date, target_column=target_column))
        actuals["same_weekday_mean"].append(actual)
        dates["same_weekday_mean"].append(date_string)
        try:
            prediction = LightGBMQuantileModel(horizon_days=horizon_days, target_column=target_column).fit(history).predict(target_date)
        except Exception as error:
            errors["lightgbm"] = f"{type(error).__name__}: {error}"
        else:
            predictions["lightgbm"].append(prediction)
            actuals["lightgbm"].append(actual)
            dates["lightgbm"].append(date_string)
        if include_tabpfn:
            try:
                prediction = TabPFNModel(horizon_days=horizon_days, target_column=target_column).fit(history).predict(target_date)
            except Exception as error:
                errors["tabpfn"] = f"{type(error).__name__}: {error}"
            else:
                predictions["tabpfn"].append(prediction)
                actuals["tabpfn"].append(actual)
                dates["tabpfn"].append(date_string)
    result = {"dataset": "public proxy dataset (visitors, not pizzas)", "horizon_days": horizon_days, "dates": sorted(set().union(*dates.values())), "errors": errors, "models": {}}
    for name, values in predictions.items():
        if values:
            result["models"][name] = _metrics(actuals[name], values)
    if "sold_out" in data:
        result["censored_days"] = int(data["sold_out"].fillna(False).astype(bool).sum())
    return result


def _write_blocked(path: Path, operation: str, error: Exception) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "blocked", "operation": operation, "reason": f"{type(error).__name__}: {error}", "dataset": "public proxy dataset (visitors, not pizzas)"}, indent=2) + "\n", encoding="utf-8")


def run_benchmark(output_path: str | Path = "reports/benchmark.json") -> dict:
    path = Path(output_path)
    try:
        frame = load_public_dataset()
        series = frame[frame["id"] == frame["id"].iloc[0]].rename(columns={"target": "target"})
        result = evaluate(series, target_column="target")
    except Exception as error:
        _write_blocked(path, "benchmark", error)
        return {"status": "blocked", "reason": f"{type(error).__name__}: {error}"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_cold_start(output_path: str | Path = "reports/cold_start.json") -> dict:
    """Run the configured history-length experiment when the proxy is available."""
    path = Path(output_path)
    try:
        frame = load_public_dataset()
        series = frame[frame["id"] == frame["id"].iloc[0]].copy()
        lengths = [7, 14, 21, 28, 42, 56, 90, 180]
        result = {"status": "blocked", "dataset": "public proxy dataset (visitors, not pizzas)", "history_lengths": lengths, "n_series_rows": len(series), "note": "TabPFN cold-start execution requires the local checkpoint and is intentionally not auto-downloaded by this command."}
    except Exception as error:
        _write_blocked(path, "cold-start", error)
        return {"status": "blocked", "reason": f"{type(error).__name__}: {error}"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["benchmark", "cold-start"])
    args = parser.parse_args()
    result = run_benchmark() if args.operation == "benchmark" else run_cold_start()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()