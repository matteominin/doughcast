"""Leakage-safe calendar and demand-history features."""

from __future__ import annotations

from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd
import holidays

from dough.config import HOLIDAY_COUNTRY, HORIZON_DAYS, TARGET_COLUMN


def _holiday_features(dates: pd.Series, country: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = range(int(dates.dt.year.min()), int(dates.dt.year.max()) + 1)
    holiday_dates = sorted(holidays.country_holidays(country, years=years))
    if not holiday_dates:
        zeros = np.zeros(len(dates), dtype=float)
        return zeros, zeros, zeros
    values = np.array([d.date() for d in dates], dtype=object)
    distances = np.array([min((value - holiday).days for holiday in holiday_dates) for value in values], dtype=float)
    return np.isin(values, holiday_dates).astype(float), np.maximum(-distances, 0), np.maximum(distances, 0)


def _normalise(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    result = frame.copy()
    if "date" not in result:
        if isinstance(result.index, pd.DatetimeIndex):
            result = result.reset_index(names="date")
        else:
            raise ValueError("history must contain a date column or DatetimeIndex")
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    if target_column not in result and "target" in result:
        result[target_column] = result["target"]
    if target_column not in result:
        raise ValueError(f"history must contain {target_column!r} or 'target'")
    return result.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def build_features(history: pd.DataFrame, target_dates: Iterable[date | str] | None = None, *, horizon_days: int = HORIZON_DAYS, target_column: str = TARGET_COLUMN, country: str = HOLIDAY_COUNTRY, include_target: bool = True, include_weather: bool = False) -> pd.DataFrame:
    """Build rows whose demand features use observations no newer than ``t - horizon_days``."""

    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")
    source = _normalise(history, target_column)
    dates = source["date"] if target_dates is None else pd.to_datetime(list(target_dates)).normalize()
    result = pd.DataFrame({"date": dates}).sort_values("date").reset_index(drop=True)
    result["day_of_week"] = result.date.dt.dayofweek
    result["month"] = result.date.dt.month
    result["day_of_month"] = result.date.dt.day
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(float)
    holiday, days_from, days_to = _holiday_features(result.date, country)
    result["is_holiday"], result["days_from_holiday"], result["days_to_holiday"] = holiday, days_from, days_to
    values = source.set_index("date")[target_column].astype(float)
    cutoff_dates = result["date"] - pd.to_timedelta(horizon_days, unit="D")
    for lag in (7, 14):
        result[f"lag_{lag}"] = [values.get(cutoff - pd.to_timedelta(lag, unit="D")) for cutoff in cutoff_dates]
    same_weekday_values = [values.reindex([cutoff - pd.to_timedelta(7 * offset, unit="D") for offset in range(1, 5)]) for cutoff in cutoff_dates]
    result["same_weekday_mean_4"] = [series.mean() if series.notna().sum() == 4 else np.nan for series in same_weekday_values]
    result["rolling_7_mean"] = [values.loc[:cutoff].tail(7).mean() if len(values.loc[:cutoff].tail(7)) == 7 else np.nan for cutoff in cutoff_dates]
    if "sold_out" in source:
        sold_out = source.set_index("date")["sold_out"].astype(float)
        result["sold_out_yesterday"] = [sold_out.get(cutoff, np.nan) for cutoff in cutoff_dates]
        result["censored"] = result["sold_out_yesterday"].fillna(0)
    if include_weather and "weather" in source:
        weather = source.set_index("date")["weather"]
        result["weather"] = [weather.get(cutoff) for cutoff in cutoff_dates]
    if include_target:
        result[target_column] = result["date"].map(source.set_index("date")[target_column])
    return result


def effective_rows(features: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Return rows with complete history and a known target for readiness checks."""
    required = ["lag_7", "lag_14", "same_weekday_mean_4", "rolling_7_mean", target_column]
    return features.dropna(subset=[column for column in required if column in features]).copy()