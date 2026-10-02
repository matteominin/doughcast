"""Lazy local TabPFN regression adapter.

TabPFN is imported only when a model is constructed, so feature tests and the
below-readiness product path do not require downloaded weights.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from dough.config import HORIZON_DAYS, TARGET_COLUMN, TABPFN_VERSION
from dough.data.features import build_features, effective_rows
from dough.forecast.baselines import PredictionInterval


class TabPFNModel:
    def __init__(self, *, horizon_days: int = HORIZON_DAYS, target_column: str = TARGET_COLUMN, version: str = TABPFN_VERSION, device: str = "auto", random_state: int = 0):
        self.horizon_days = horizon_days
        self.target_column = target_column
        self.version = version
        self.device = device
        self.random_state = random_state
        self.model = None
        self.history: pd.DataFrame | None = None
        self.feature_columns: list[str] = []

    def _new_regressor(self):
        from tabpfn import TabPFNRegressor

        kwargs = {"device": self.device, "random_state": self.random_state, "show_progress_bar": False}
        try:
            if self.version in {"default", ""}:
                return TabPFNRegressor(**kwargs)
            from tabpfn.constants import ModelVersion

            try:
                model_version = ModelVersion(self.version)
            except ValueError as error:
                raise ValueError(f"Unsupported TabPFN version {self.version!r}") from error
            return TabPFNRegressor.create_default_for_version(model_version, **kwargs)
        except Exception as error:
            raise RuntimeError(f"TabPFN initialization failed (requires TABPFN_TOKEN or local license): {error}") from error

    def fit(self, history: pd.DataFrame) -> "TabPFNModel":
        self.history = history.copy()
        features = build_features(self.history, horizon_days=self.horizon_days, target_column=self.target_column)
        train = effective_rows(features, self.target_column)
        self.feature_columns = [column for column in train.columns if column not in {"date", self.target_column} and pd.api.types.is_numeric_dtype(train[column])]
        if len(train) < 2 or not self.feature_columns:
            raise ValueError("TabPFN needs at least two complete historical feature rows")
        try:
            self.model = self._new_regressor()
            self.model.fit(train[self.feature_columns], train[self.target_column])
        except Exception as error:
            raise RuntimeError(f"TabPFN fit failed: {error}") from error
        return self

    def predict(self, target_date: date | str) -> PredictionInterval:
        if self.model is None or self.history is None:
            raise RuntimeError("fit must be called before predict")
        row = build_features(self.history, [target_date], horizon_days=self.horizon_days, target_column=self.target_column, include_target=False)
        if row[self.feature_columns].isna().any(axis=None):
            raise ValueError("target date does not have enough leakage-safe history")
        outputs = self.model.predict(row[self.feature_columns], output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
        values = [float(np.asarray(output).reshape(-1)[0]) for output in outputs]
        return PredictionInterval(*values).ordered()