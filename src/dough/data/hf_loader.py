"""Load and inspect the public restaurant visitor proxy dataset.

The Hugging Face dataset is intentionally loaded lazily. Importing Doughcast
must remain offline-safe; only an explicit call to ``load_public_dataset`` may
contact the Hub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

DATASET_NAME = "autogluon/fev_datasets"
DATASET_CONFIG = "restaurant"


def _series_rows(dataset: Any) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for item in dataset:
        timestamps = pd.to_datetime(item["timestamp"], errors="coerce")
        targets = pd.to_numeric(item["target"], errors="coerce")
        if len(timestamps) != len(targets):
            raise ValueError(f"Series {item.get('id', '<unknown>')} has mismatched timestamps/target")
        rows.append(pd.DataFrame({
            "id": item.get("id"), "date": timestamps, "target": targets,
            "air_genre_name": item.get("air_genre_name"), "air_area_name": item.get("air_area_name"),
            "latitude": item.get("latitude"), "longitude": item.get("longitude"),
        }))
    if not rows:
        return pd.DataFrame(columns=["id", "date", "target"])
    result = pd.concat(rows, ignore_index=True)
    result["date"] = result["date"].dt.normalize()
    return result.sort_values(["id", "date"]).reset_index(drop=True)


def _selection_score(group: pd.DataFrame) -> tuple[int, int, int]:
    genre = str(group["air_genre_name"].dropna().iloc[0]).lower() if group["air_genre_name"].notna().any() else ""
    italian = int("italian" in genre or "pizza" in genre)
    gaps = int(group["date"].sort_values().diff().dt.days.sub(1).clip(lower=0).sum())
    return italian, len(group), -gaps


def select_series(frame: pd.DataFrame, n_series: int = 3) -> pd.DataFrame:
    """Select up to three long, low-gap series, preferring Italian-like genres."""

    if frame.empty:
        return frame.copy()
    groups = sorted((group for _, group in frame.groupby("id", sort=False)), key=_selection_score, reverse=True)
    return pd.concat(groups[: max(1, n_series)], ignore_index=True).sort_values(["id", "date"]).reset_index(drop=True)


def inspect_dataset(frame: pd.DataFrame, dataset_info: Any = None, report_path: str | Path = "reports/dataset_summary.md") -> Path:
    """Write an auditable summary of frequency, gaps, missing values, and selection."""

    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Public dataset inspection", "",
        f"Dataset: `{DATASET_NAME}`, configuration `{DATASET_CONFIG}`.",
        "This is a public proxy of restaurant visitors, not pizza sales.", "", "## Series summary", "",
        "| id | rows | start | end | median frequency (days) | gaps | zeros | NaNs | genre |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for series_id, group in frame.groupby("id", sort=True):
        ordered = group.sort_values("date")
        deltas = ordered["date"].diff().dt.days.dropna()
        gaps = int(deltas.sub(1).clip(lower=0).sum())
        genre = str(ordered["air_genre_name"].dropna().iloc[0]) if ordered["air_genre_name"].notna().any() else ""
        lines.append(f"| {series_id} | {len(group)} | {ordered.date.min().date()} | {ordered.date.max().date()} | {deltas.median() if not deltas.empty else 'n/a'} | {gaps} | {int((ordered.target == 0).sum())} | {int(ordered.target.isna().sum())} | {genre} |")
    lines.extend(["", "## Selection rule", "", "Up to three series are selected by Italian-like genre first, then row count, then fewest calendar gaps.", "", "## Dataset card license and attribution", ""])
    license_text = getattr(dataset_info, "license", None) if dataset_info is not None else None
    description = getattr(dataset_info, "description", None) if dataset_info is not None else None
    lines.append(f"License field: {license_text or 'TO VERIFY: dataset card was not available during inspection.'}")
    if description:
        lines.extend(["", "Dataset card description:", "", description.strip()])
    lines.extend(["", "The exact Hub card terms must be reviewed before redistribution; the current metadata does not establish a permissive license."])
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def load_public_dataset(*, cache_dir: str | Path | None = None, n_series: int = 3, report_path: str | Path = "reports/dataset_summary.md") -> pd.DataFrame:
    """Download/load, tidy, inspect, and select the public restaurant series."""

    from datasets import load_dataset
    kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", **kwargs)
    frame = _series_rows(dataset)
    selected = select_series(frame, n_series=n_series)
    inspect_dataset(selected, dataset_info=getattr(dataset, "info", None), report_path=report_path)
    return selected