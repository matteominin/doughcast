"""Offline verification script for Doughcast.

Blocks network access to non-localhost addresses and verifies that the core
path (store -> feature engineering -> forecasting -> recommendation -> API)
functions completely offline without network calls.
"""

from __future__ import annotations

import socket
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

# Patch socket to block non-localhost outgoing network connections
_orig_connect = socket.socket.connect


def _offline_connect(self, address):
    host = str(address[0])
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError(f"Network access attempted during offline check: {address}")
    return _orig_connect(self, address)


socket.socket.connect = _offline_connect


def run_offline_check() -> bool:
    print("[offline-check] Network guard active (non-localhost sockets blocked).")

    from dough.api import create_app
    from dough.config import HORIZON_DAYS
    from dough.data.features import build_features, effective_rows
    from dough.forecast.baselines import LightGBMQuantileModel, same_weekday_mean, seasonal_naive
    from dough.forecast.dough import recommendation_from_forecast
    from dough.schemas import DailyRecord, Forecast
    from dough.store import RecordStore

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "pizzeria_offline.db"
        store = RecordStore(db_path)

        # 1. Insert 60 days of records locally
        start_date = date(2026, 7, 1)
        for i in range(60):
            current_date = start_date + timedelta(days=i)
            pizzas = 50 + (i % 7) * 5 + (i % 3) * 2
            sold_out = (i % 5 == 0)
            store.upsert(
                DailyRecord(
                    date=current_date,
                    pizzas_sold=pizzas,
                    dough_balls_prepared=pizzas + (5 if not sold_out else 0),
                    dough_balls_left=0 if sold_out else 5,
                    sold_out=sold_out,
                    closed=False,
                    weather="clear" if i % 2 == 0 else "rain",
                    source="manual",
                    confirmed=True,
                )
            )

        records = store.list_records()
        assert len(records) == 60, f"Expected 60 records, got {len(records)}"

        # 2. Build features offline
        import pandas as pd

        df = pd.DataFrame([{"date": r.date, "pizzas_sold": r.pizzas_sold, "sold_out": r.sold_out} for r in records])
        feats = build_features(df, horizon_days=HORIZON_DAYS)
        eff = effective_rows(feats, "pizzas_sold")
        assert len(eff) > 0, "Effective rows calculation failed offline"

        # 3. Fit baseline models offline
        target_date = date(2026, 8, 28)
        history = df[df["date"] <= target_date - timedelta(days=HORIZON_DAYS)]
        sn_pred = seasonal_naive(history, target_date, target_column="pizzas_sold")
        swm_pred = same_weekday_mean(history, target_date, target_column="pizzas_sold")
        lgb_pred = LightGBMQuantileModel(horizon_days=HORIZON_DAYS, target_column="pizzas_sold").fit(history).predict(target_date)

        assert sn_pred.p50 > 0
        assert swm_pred.p50 > 0
        assert lgb_pred.p50 > 0 and lgb_pred.p10 <= lgb_pred.p50 <= lgb_pred.p90

        # 4. Generate recommendation offline
        forecast_obj = Forecast(
            target_date=target_date,
            prepare_on=target_date - timedelta(days=HORIZON_DAYS),
            horizon_days=HORIZON_DAYS,
            p10=lgb_pred.p10,
            p50=lgb_pred.p50,
            p90=lgb_pred.p90,
            model="lightgbm_offline",
            n_train_days=len(history),
            baseline_p50=sn_pred.p50,
            recommendation={"service_level_quantile": 0.8, "dough_balls": 0, "dough_grams": 0, "reasoning": ""},
        )
        rec = recommendation_from_forecast(forecast_obj)
        assert rec.recommendation.dough_balls > 0
        assert rec.recommendation.dough_grams == rec.recommendation.dough_balls * 180

        # 5. API smoke test offline
        client = TestClient(create_app(store=store))
        health_resp = client.get("/api/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["offline"] is True

        records_resp = client.get("/api/records")
        assert records_resp.status_code == 200
        assert len(records_resp.json()["records"]) == 60

    print("[offline-check] SUCCESS: Core path (Store -> Features -> Models -> Recommendation -> API) verified 100% offline.")
    return True


if __name__ == "__main__":
    try:
        run_offline_check()
    except Exception as e:
        print(f"[offline-check] FAILED: {e}", file=sys.stderr)
        sys.exit(1)
