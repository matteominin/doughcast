from datetime import date

import pytest
from pydantic import ValidationError

from dough.schemas import DailyRecord


def test_daily_record_accepts_nullable_contract_fields():
    record = DailyRecord(date=date(2026, 10, 3), pizzas_sold=74, weather="rain", sold_out_time="21:30")
    assert record.pizzas_sold == 74
    assert record.events is None


def test_daily_record_rejects_invalid_weather_and_time():
    with pytest.raises(ValidationError):
        DailyRecord(date="2026-10-03", weather="windy")
    with pytest.raises(ValidationError):
        DailyRecord(date="2026-10-03", sold_out_time="9:30")