"""Deterministic baselines and a local LightGBM quantile regressor."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from dough.config import HORIZON_DAYS, TARGET_COLUMN
from dough.data.features import build_features, effective_rows


@dataclass(frozen=True)
class PredictionInterval:
    p10: float
    p50: float
    p90: float

    def ordered(self) -> "PredictionInterval":
        values = sorted(max(0.0, value) for value in (self.p10, self.p50, self.p90))
        return PredictionInterval(*values)


def _normalise(history: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    frame = history.copy()
    if "date" not in frame:
        frame = frame.reset_index(names="date")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if target_column not in frame and "target" in frame:
        frame[target_column] = frame["target"]
    if target_column not in frame:
        raise ValueError(f"history must contain {target_column!r} or 'target'")
    return frame.sort_values("date").dropna(subset=[target_column])


def _value_at(history: pd.DataFrame, target_date: date | str, offset_days: int, target_column: str) -> float:
    frame = _normalise(history, target_column).set_index("date")
    timestamp = pd.Timestamp(target_date).normalize() - pd.to_timedelta(offset_days, unit="D")
    value = frame[target_column].get(timestamp, np.nan)
    return float(value) if pd.notna(value) else float(frame[target_column].tail(1).iloc[0])


def seasonal_naive(history: pd.DataFrame, target_date: date | str, *, target_column: str = TARGET_COLUMN) -> PredictionInterval:
    value = _value_at(history, target_date, 7, target_column)
    return PredictionInterval(value, value, value)


def same_weekday_mean(history: pd.DataFrame, target_date: date | str, *, target_column: str = TARGET_COLUMN, count: int = 4) -> PredictionInterval:
    frame = _normalise(history, target_column)
    target = pd.Timestamp(target_date).normalize()
    values = frame[(frame["date"] < target) & (frame["date"].dt.dayofweek == target.dayofweek)][target_column].tail(count)
    value = float(values.mean()) if not values.empty else float(frame[target_column].tail(1).iloc[0])
    return PredictionInterval(value, value, value)


class LightGBMQuantileModel:
    """LightGBM p10/p50/p90 model using the shared feature builder."""

    def __init__(self, *, horizon_days: int = HORIZON_DAYS, target_column: str = TARGET_COLUMN, random_state: int = 0):
        self.horizon_days = horizon_days
        self.target_column = target_column
        self.random_state = random_state
        self.models: dict[float, object] = {}
        self.feature_columns: list[str] = []
        self.history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame) -> "LightGBMQuantileModel":
        use_lgb = True
        try:
            from lightgbm import LGBMRegressor
            # Test simple instantiation to ensure C libraries like libomp can load
            _ = LGBMRegressor(verbosity=-1)
        except (ImportError, OSError):
            use_lgb = False
            from sklearn.ensemble import HistGradientBoostingRegressor

        self.history = _normalise(history, self.target_column)
        features = build_features(self.history, horizon_days=self.horizon_days, target_column=self.target_column)
        train = effective_rows(features, self.target_column)
        self.feature_columns = [column for column in train.columns if column not in {"date", self.target_column} and pd.api.types.is_numeric_dtype(train[column])]
        if len(train) < 8 or not self.feature_columns:
            raise ValueError("LightGBM needs at least 8 complete historical feature rows")
        X, y = train[self.feature_columns], train[self.target_column]
        for quantile in (0.1, 0.5, 0.9):
            if use_lgb:
                model = LGBMRegressor(objective="quantile", alpha=quantile, n_estimators=100, learning_rate=0.05, num_leaves=15, verbosity=-1, random_state=self.random_state)
            else:
                model = HistGradientBoostingRegressor(loss="quantile", quantile=quantile, max_iter=100, learning_rate=0.05, max_leaf_nodes=15, random_state=self.random_state)
            model.fit(X, y)
            self.models[quantile] = model
        return self

    def predict(self, target_date: date | str) -> PredictionInterval:
        if self.history is None or not self.models:
            raise RuntimeError("fit must be called before predict")
        row = build_features(self.history, [target_date], horizon_days=self.horizon_days, target_column=self.target_column, include_target=False)
        if row[self.feature_columns].isna().any(axis=None):
            raise ValueError("target date does not have enough leakage-safe history")
        values = [float(np.asarray(self.models[q].predict(row[self.feature_columns])).reshape(-1)[0]) for q in (0.1, 0.5, 0.9)]
        return PredictionInterval(*values).ordered()


def lightgbm_quantile(history: pd.DataFrame, target_date: date | str, *, horizon_days: int = HORIZON_DAYS, target_column: str = TARGET_COLUMN) -> PredictionInterval:
    return LightGBMQuantileModel(horizon_days=horizon_days, target_column=target_column).fit(history).predict(target_date)