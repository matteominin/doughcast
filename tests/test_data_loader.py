import pandas as pd

from dough.data.hf_loader import inspect_dataset, select_series


def test_series_selection_prefers_italian_and_writes_inspection(tmp_path):
    frame = pd.DataFrame(
        {
            "id": ["short", "short", "italian", "italian", "long", "long", "long"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02", "2026-01-03"]),
            "target": [1, 2, 3, 4, 5, 6, 7],
            "air_genre_name": ["Cafe"] * 2 + ["Italian"] * 2 + ["Cafe"] * 3,
            "air_area_name": [None] * 7,
            "latitude": [None] * 7,
            "longitude": [None] * 7,
        }
    )
    selected = select_series(frame, n_series=1)
    assert set(selected["id"]) == {"italian"}
    report = inspect_dataset(selected, report_path=tmp_path / "dataset_summary.md")
    assert report.exists()
    assert "public proxy" in report.read_text(encoding="utf-8")