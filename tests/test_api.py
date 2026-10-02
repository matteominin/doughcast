from datetime import date, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dough.api import create_app
from dough.forecast.baselines import PredictionInterval
from dough.schemas import DailyRecord
from dough.store import RecordStore


def _client(tmp_path, **dependencies):
    return TestClient(create_app(store=RecordStore(tmp_path / "records.db"), **dependencies))


def test_records_validate_and_upsert_as_confirmed(tmp_path):
    client = _client(tmp_path)
    invalid = client.post("/api/records", json={"date": "2026-10-02", "weather": "storm"})
    assert invalid.status_code == 422

    payload = {"date": "2026-10-02", "pizzas_sold": 60}
    assert client.post("/api/records", json=payload).json()["confirmed"] is True
    payload["pizzas_sold"] = 74
    response = client.post("/api/records", json=payload)
    assert response.status_code == 200
    assert response.json()["pizzas_sold"] == 74

    listed = client.get("/api/records", params={"from": "2026-10-02", "to": "2026-10-02"})
    assert [record["pizzas_sold"] for record in listed.json()["records"]] == [74]


def test_status_excludes_unconfirmed_closed_and_recalled_rows(tmp_path, monkeypatch):
    monkeypatch.setattr("dough.api.MIN_TRAIN_ROWS", 50)
    store = RecordStore(tmp_path / "records.db")
    store.upsert(
        DailyRecord(date=date(2025, 12, 31), pizzas_sold=99, confirmed=False, source="manual")
    )
    client = TestClient(create_app(store=store))
    for offset in range(70):
        source = "recalled" if offset == 1 else "manual"
        payload = {
            "date": (date(2026, 1, 1) + timedelta(days=offset)).isoformat(),
            "pizzas_sold": 50 + offset,
            "source": source,
        }
        assert client.post("/api/records", json=payload).status_code == 200

    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json()["rows_available"] < response.json()["rows_required"]
    assert response.json()["ready"] is False


class _FakeForecaster:
    def fit(self, history):
        self.history = history
        return self

    def predict(self, target_date):
        return PredictionInterval(61, 78, 96)


def test_forecast_uses_fake_after_readiness_gate(tmp_path, monkeypatch):
    monkeypatch.setattr("dough.api.MIN_TRAIN_ROWS", 5)
    client = _client(tmp_path, forecaster=_FakeForecaster())
    for offset in range(70):
        response = client.post(
            "/api/records",
            json={
                "date": (date(2026, 1, 1) + timedelta(days=offset)).isoformat(),
                "pizzas_sold": 50 + offset,
                "source": "manual",
            },
        )
        assert response.status_code == 200

    response = client.get("/api/forecast", params={"date": "2026-03-20"})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "tabpfn"
    assert body["target_date"] == "2026-03-20"
    assert body["recommendation"]["dough_balls"] == 89


def test_voice_removes_temporary_audio_after_processing(tmp_path):
    paths = []

    def transcriber(path):
        paths.append(path)
        return type("Transcript", (), {"text": "stasera 74 pizze"})()

    def extractor(transcript, reference_date, mode):
        return type(
            "Extraction",
            (),
            {
                "extracted": {"date": reference_date.isoformat(), "pizzas_sold": 74, "source": "voice", "confirmed": False},
                "warnings": [],
            },
        )()

    client = _client(tmp_path, transcriber=transcriber, extractor=extractor)
    response = client.post("/api/voice", files={"audio": ("note.webm", b"fake audio", "audio/webm")})
    assert response.status_code == 200
    assert response.json()["extracted"]["pizzas_sold"] == 74
    assert paths and not Path(paths[0]).exists()


def test_recalled_voice_is_returned_but_not_confirmed(tmp_path):
    def transcriber(path):
        return type("Transcript", (), {"text": "lunedì 45, martedì 40"})()

    def extractor(transcript, reference_date, mode):
        return type(
            "Extraction",
            (),
            {
                "extracted": [
                    {"date": "2026-09-28", "pizzas_sold": 45},
                    {"date": "2026-09-29", "pizzas_sold": 40},
                ],
                "warnings": [],
            },
        )()

    client = _client(tmp_path, transcriber=transcriber, extractor=extractor)
    response = client.post(
        "/api/voice?mode=recalled",
        files={"audio": ("note.webm", b"fake audio", "audio/webm")},
    )
    assert response.status_code == 200
    assert all(item["source"] == "recalled" and item["confirmed"] is False for item in response.json()["extracted"])
    assert client.get("/api/records").json()["records"] == []


def test_voice_rejects_invalid_extension_and_oversized_audio(tmp_path):
    client = _client(tmp_path)
    res_bad_ext = client.post("/api/voice", files={"audio": ("script.exe", b"fake audio", "application/octet-stream")})
    assert res_bad_ext.status_code == 400

    res_oversized = client.post("/api/voice", files={"audio": ("large.webm", b"0" * (26 * 1024 * 1024), "audio/webm")})
    assert res_oversized.status_code == 413
