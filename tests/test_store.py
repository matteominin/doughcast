from datetime import date

from dough.schemas import DailyRecord
from dough.store import RecordStore


def test_store_upsert_filter_and_csv_export(tmp_path):
    store = RecordStore(tmp_path / "pizzeria.db")
    first = DailyRecord(
        date=date(2026, 10, 2), pizzas_sold=60, events=["football match"], sold_out=False, confirmed=True
    )
    store.upsert(first)
    store.upsert(first.model_copy(update={"pizzas_sold": 74}))
    store.upsert(DailyRecord(date=date(2026, 10, 3), closed=True))

    records = store.list_records(from_date=date(2026, 10, 2), to_date=date(2026, 10, 2))
    assert len(records) == 1
    assert records[0].pizzas_sold == 74
    assert records[0].events == ["football match"]

    csv_path = store.export_csv(tmp_path / "records.csv")
    assert csv_path.read_text(encoding="utf-8").count("\n") == 3