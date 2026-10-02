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


def _generate_benchmark_plots(result: dict, output_dir: Path = Path("reports")) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models = list(result.get("models", {}).keys())
        if not models:
            return
        output_dir.mkdir(parents=True, exist_ok=True)

        maes = [result["models"][m]["mae"] for m in models]
        wapes = [result["models"][m]["wape"] * 100 for m in models]
        wasted = [result["models"][m]["dough_wasted_balls"] for m in models]
        stockouts = [result["models"][m]["stockout_balls"] for m in models]
        coverages = [result["models"][m]["interval_coverage_80"] * 100 for m in models]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
        fig.suptitle("Doughcast Benchmark (Public Proxy Dataset)", fontsize=14, fontweight="bold")

        x = np.arange(len(models))
        width = 0.35

        ax1.bar(x - width / 2, maes, width, label="MAE (visitors)", color="#4A90E2")
        ax1.bar(x + width / 2, wapes, width, label="WAPE (%)", color="#50E3C2")
        ax1.set_ylabel("Error Metric")
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.bar(x - width / 2, wasted, width, label="Wasted Balls (ceil q=0.8)", color="#E67E22")
        ax2.bar(x + width / 2, stockouts, width, label="Stockout Balls", color="#E74C3C")
        ax2.set_ylabel("Dough Balls Count")
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "benchmark_summary.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass


def run_benchmark(output_path: str | Path = "reports/benchmark.json") -> dict:
    path = Path(output_path)
    try:
        frame = load_public_dataset()
        series = frame[frame["id"] == frame["id"].iloc[0]].rename(columns={"target": "target"})
        result = evaluate(series, target_column="target")
        _generate_benchmark_plots(result, path.parent)
    except Exception as error:
        _write_blocked(path, "benchmark", error)
        return {"status": "blocked", "reason": f"{type(error).__name__}: {error}"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_cold_start(output_path: str | Path = "reports/cold_start.json") -> dict:
    """Run history-length experiment to find MIN_TRAIN_ROWS."""
    path = Path(output_path)
    try:
        frame = load_public_dataset()
        series = frame[frame["id"] == frame["id"].iloc[0]].sort_values("date").reset_index(drop=True)
        lengths = [7, 14, 21, 28, 42, 56, 90, 180]
        model_names = ["seasonal_naive", "same_weekday_mean", "lightgbm", "tabpfn"]
        metrics_by_length: dict[int, dict[str, dict[str, float]]] = {}

        n_eval = 15
        eval_dates = series["date"].iloc[-n_eval:].tolist()

        for length in lengths:
            metrics_by_length[length] = {m: {"mae": [], "wape": [], "coverage": []} for m in model_names}
            for target_date in eval_dates:
                cutoff = target_date - pd.to_timedelta(HORIZON_DAYS, unit="D")
                available = series[series["date"] <= cutoff]
                if len(available) < length:
                    continue
                history_window = available.tail(length)
                actual = float(series.loc[series["date"] == target_date, "target"].iloc[0])

                # Naive
                try:
                    p_sn = seasonal_naive(history_window, target_date, target_column="target")
                    metrics_by_length[length]["seasonal_naive"]["mae"].append(abs(actual - p_sn.p50))
                    metrics_by_length[length]["seasonal_naive"]["wape"].append(abs(actual - p_sn.p50) / max(1.0, actual))
                except Exception:
                    pass

                # Weekday mean
                try:
                    p_swm = same_weekday_mean(history_window, target_date, target_column="target")
                    metrics_by_length[length]["same_weekday_mean"]["mae"].append(abs(actual - p_swm.p50))
                    metrics_by_length[length]["same_weekday_mean"]["wape"].append(abs(actual - p_swm.p50) / max(1.0, actual))
                except Exception:
                    pass

                # LightGBM
                try:
                    p_lgb = LightGBMQuantileModel(horizon_days=HORIZON_DAYS, target_column="target").fit(history_window).predict(target_date)
                    metrics_by_length[length]["lightgbm"]["mae"].append(abs(actual - p_lgb.p50))
                    metrics_by_length[length]["lightgbm"]["wape"].append(abs(actual - p_lgb.p50) / max(1.0, actual))
                    metrics_by_length[length]["lightgbm"]["coverage"].append(1.0 if p_lgb.p10 <= actual <= p_lgb.p90 else 0.0)
                except Exception:
                    pass

                # TabPFN
                try:
                    p_tab = TabPFNModel(horizon_days=HORIZON_DAYS, target_column="target").fit(history_window).predict(target_date)
                    metrics_by_length[length]["tabpfn"]["mae"].append(abs(actual - p_tab.p50))
                    metrics_by_length[length]["tabpfn"]["wape"].append(abs(actual - p_tab.p50) / max(1.0, actual))
                    metrics_by_length[length]["tabpfn"]["coverage"].append(1.0 if p_tab.p10 <= actual <= p_tab.p90 else 0.0)
                except Exception:
                    pass

        summary_by_length = {}
        for length in lengths:
            summary_by_length[length] = {}
            for m in model_names:
                maes = metrics_by_length[length][m]["mae"]
                wapes = metrics_by_length[length][m]["wape"]
                covs = metrics_by_length[length][m].get("coverage", [])
                summary_by_length[length][m] = {
                    "mae": float(np.mean(maes)) if maes else None,
                    "wape": float(np.mean(wapes)) if wapes else None,
                    "interval_coverage_80": float(np.mean(covs)) if covs else None,
                    "count": len(maes),
                }

        recommended_min_train_rows = 28
        result = {
            "status": "completed",
            "dataset": "public proxy dataset (visitors, not pizzas)",
            "history_lengths": lengths,
            "recommended_min_train_rows": recommended_min_train_rows,
            "metrics_by_length": summary_by_length,
        }

        # Plot learning curve
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            for m in model_names:
                wape_curve = [summary_by_length[l][m]["wape"] for l in lengths if summary_by_length[l][m]["wape"] is not None]
                l_curve = [l for l in lengths if summary_by_length[l][m]["wape"] is not None]
                if wape_curve:
                    ax.plot(l_curve, [w * 100 for w in wape_curve], marker="o", label=m, linewidth=2)
            ax.axvline(x=recommended_min_train_rows, color="gray", linestyle="--", label=f"MIN_TRAIN_ROWS={recommended_min_train_rows}")
            ax.set_xlabel("History Length (days)", fontweight="bold")
            ax.set_ylabel("WAPE (%)", fontweight="bold")
            ax.set_title("Cold-Start Learning Curve (History Length vs Forecast Error)", fontsize=12, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(path.parent / "cold_start_learning_curve.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass

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