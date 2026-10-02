from datetime import date

import pandas as pd

from dough.data.features import build_features, effective_rows
from dough.forecast.baselines import same_weekday_mean, seasonal_naive
from dough.forecast.dough import recommendation_from_forecast, tradeoff_table
from dough.forecast.evaluate import _metrics, evaluate
from dough.schemas import Forecast, Recommendation


def _history(days: int = 70) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    return pd.DataFrame({"date": dates, "target": range(days)})


def test_features_do_not_read_inside_horizon():
    history = _history()
    target = date(2026, 3, 12)
    before = build_features(history, [target], horizon_days=2, target_column="target", country="IT", include_target=False)
    changed = history.copy()
    changed.loc[changed["date"] == pd.Timestamp("2026-03-11"), "target"] = 999999
    after = build_features(changed, [target], horizon_days=2, target_column="target", country="IT", include_target=False)
    demand_columns = ["lag_7", "lag_14", "same_weekday_mean_4", "rolling_7_mean"]
    assert before[demand_columns].equals(after[demand_columns])


def test_effective_rows_excludes_warmup_rows():
    features = build_features(_history(), horizon_days=2, target_column="target", country="IT")
    effective = effective_rows(features, "target")
    assert len(effective) < len(features)
    assert effective["date"].min() >= pd.Timestamp("2026-01-31")


def test_baselines_use_only_past_observations():
    history = _history()
    target = date(2026, 3, 12)
    assert seasonal_naive(history, target, target_column="target").p50 == 63
    assert same_weekday_mean(history, target, target_column="target").p50 == 52.5


def test_dough_recommendation_rounds_balls_and_mass():
    forecast = Forecast(
        target_date="2026-03-12",
        prepare_on="2026-03-10",
        horizon_days=2,
        p10=60,
        p50=78,
        p90=96,
        model="tabpfn",
        n_train_days=50,
        recommendation=Recommendation(service_level_quantile=0.8, dough_balls=0, dough_grams=0, reasoning=""),
    )
    result = recommendation_from_forecast(forecast)
    assert result.recommendation.dough_balls == 89
    assert result.recommendation.dough_grams == 16020
    assert len(tradeoff_table(60, 78, 96)) == 4


def test_evaluation_is_time_ordered_and_reports_business_metrics():
    result = evaluate(_history(90), backtest_days=10, horizon_days=2, target_column="target", include_tabpfn=False)
    assert len(result["dates"]) == 10
    assert result["dates"] == sorted(result["dates"])
    assert "mae" in result["models"]["seasonal_naive"]
    assert "dough_wasted_balls" in result["models"]["same_weekday_mean"]


def test_pinball_uses_unrounded_service_estimate():
    metrics = _metrics([12], [type("Prediction", (), {"p10": 0, "p50": 8, "p90": 13})()], quantile=0.8)
    assert metrics["pinball_loss"] == 0.2