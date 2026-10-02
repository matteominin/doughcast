"""Small SQLite persistence layer for daily records."""

import csv
import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Iterable

from .schemas import DailyRecord


class RecordStore:
    """Persist DailyRecord values in a local SQLite database."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        if str(self.database_path) != ":memory:":
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS daily_records (
                    date TEXT PRIMARY KEY,
                    pizzas_sold INTEGER,
                    dough_balls_prepared INTEGER,
                    dough_balls_left INTEGER,
                    sold_out INTEGER,
                    sold_out_time TEXT,
                    closed INTEGER,
                    weather TEXT,
                    events TEXT,
                    notes TEXT,
                    source TEXT,
                    confirmed INTEGER
                )"""
            )

    def upsert(self, record: DailyRecord) -> DailyRecord:
        values = record.model_dump()
        values["date"] = record.date.isoformat()
        values["events"] = json.dumps(values["events"], ensure_ascii=True) if values["events"] is not None else None
        for key in ("sold_out", "closed", "confirmed"):
            if values[key] is not None:
                values[key] = int(values[key])
        columns = list(values)
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(f"{column}=excluded.{column}" for column in columns if column != "date")
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO daily_records ({', '.join(columns)}) VALUES ({placeholders}) "
                f"ON CONFLICT(date) DO UPDATE SET {updates}",
                [values[column] for column in columns],
            )
        return record

    def list_records(self, from_date: date | None = None, to_date: date | None = None) -> list[DailyRecord]:
        query = "SELECT * FROM daily_records"
        clauses: list[str] = []
        parameters: list[str] = []
        if from_date is not None:
            clauses.append("date >= ?")
            parameters.append(from_date.isoformat())
        if to_date is not None:
            clauses.append("date <= ?")
            parameters.append(to_date.isoformat())
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY date"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._record_from_row(row) for row in rows]

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> DailyRecord:
        values = dict(row)
        values["date"] = date.fromisoformat(values["date"])
        values["events"] = json.loads(values["events"]) if values["events"] is not None else None
        for key in ("sold_out", "closed", "confirmed"):
            if values[key] is not None:
                values[key] = bool(values[key])
        return DailyRecord.model_validate(values)

    def export_csv(self, output_path: str | Path, records: Iterable[DailyRecord] | None = None) -> Path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        records = list(self.list_records() if records is None else records)
        fieldnames = list(DailyRecord.model_fields)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                values = record.model_dump(mode="json")
                values["events"] = json.dumps(values["events"], ensure_ascii=True) if values["events"] is not None else ""
                writer.writerow(values)
        return destination